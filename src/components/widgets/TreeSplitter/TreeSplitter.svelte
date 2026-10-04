<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import * as S from './sim';

  interface Props {
    title?: string;
    /** 'split': one threshold on x. 'gain': thresholds on x and y compete. 'grow': the whole tree. 'impurity': Gini vs entropy. 'depth': train vs validation accuracy. */
    mode?: S.Mode;
    criterion?: S.Criterion;
    /** Deepest level of the tree in mode 'grow'. */
    maxDepth?: number;
  }
  let { title = '', mode = 'split', criterion = 'gini', maxDepth = 3 }: Props = $props();
  const init = untrack(() => ({ mode, crit: criterion, md: maxDepth, title }));
  const isTree = init.mode === 'split' || init.mode === 'gain' || init.mode === 'grow';

  let crit = $state<S.Criterion>(init.crit), md = $state(init.md), minLeaf = $state(1);
  let story = $state.raw<S.Story | null>(isTree ? S.buildStory(init.mode as S.Story['mode'], init.crit, init.md) : null);
  let study = $state.raw<S.Study | null>(init.mode === 'depth' ? S.depthStudy(1, init.crit) : null);
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let ui = $state<{ vals: { k: string; v: string }[]; say: string; p: number; d: number }>({ vals: [], say: '', p: 0, d: 0 });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const T = $derived(init.mode === 'impurity' ? S.T_IMP : init.mode === 'depth' ? S.T_DEPTH : story!.T);
  const MK = $derived(init.mode === 'impurity' ? S.IMP_MARKERS : init.mode === 'depth' ? study!.markers : story!.markers);
  const nm = $derived(crit === 'gini' ? 'Gini' : 'entropy');
  $effect(() => { playing; speed; wake(); });

  const restart = () => { t = 0; playing = !reducedMotion(); wake(); };
  function setCrit(v: S.Criterion) {
    crit = v;
    if (story) { story = S.buildStory(story.mode, v, md); restart(); }
    if (study) { study = S.depthStudy(minLeaf, v); wake(); }
  }
  function setDepth(v: number) { md = v; story = S.buildStory('grow', crit, v); restart(); }
  function setMinLeaf(v: number) { minLeaf = v; study = S.depthStudy(v, crit); wake(); }
  const seekTo = (to: number) => { t = to; playing = false; wake(); };

  const f3 = (v: number) => f2(v, 3);
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const FEAT = S.FEATURES;

  /** 0, step, 2·step, … up to max, with a step from a short list so labels stay readable. */
  function niceTicks(max: number): number[] {
    const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1].find((s) => max / s <= 4) ?? 1;
    return Array.from({ length: Math.floor(max / step + 1e-9) + 1 }, (_, i) => +(i * step).toFixed(4));
  }

  function narrate(f: S.Frame, st: S.Story): string {
    const E = st.events[f.event], node = st.nodes[E.node], w = E.sweeps[E.win], name = st.crit === 'gini' ? 'Gini' : 'entropy';
    const feat = FEAT[w.feature], k = f.event + 1, m = st.events.length;
    if (f.stage === 'intro') {
      return `All ${node.n} points start in one node: ${node.counts[0]} of class 0 and ${node.counts[1]} of class 1. Its ${name} impurity is ${f3(node.impurity)}${st.mode === 'grow' ? '. The tree will try every threshold on both features, node by node.' : '.'}`;
    }
    if (f.stage === 'sweep') {
      if (st.mode === 'split') return 'A candidate threshold slides along x. The curve records the average impurity of the two groups it would create. Lower is better.';
      if (st.mode === 'gain') return 'Each feature gets its own candidate threshold. The curves record the information gain, the drop in impurity. Higher is better.';
      return `Split ${k} of ${m}: a node with ${node.n} points tries every threshold on both features. The curves record the information gain.`;
    }
    if (f.stage === 'snap') return st.mode === 'split' ? 'The threshold settles where the children are purest, which is also where the gain is largest.' : 'The threshold settles on the largest gain. The other feature\'s line fades.';
    if (f.stage === 'done' && st.mode === 'grow') {
      const leaves = S.leavesOf(st.tree).length;
      return `The tree is finished: ${leaves} leaves, ${pct(S.accuracy(st.tree, S.TRAIN.X, S.TRAIN.y))} of the training points are classified correctly.`;
    }
    const others = E.sweeps.filter((_, i) => i !== E.win);
    if (st.mode === 'gain' && others[0]) return `${feat} wins: its best gain is ${f3(w.best.gain)}, against ${f3(others[0].best.gain)} for ${FEAT[others[0].feature]}. The node splits at ${feat} ≤ ${f2(w.best.thr)}.`;
    return `${st.mode === 'grow' ? `Split ${k}: ` : 'Split: '}${feat} ≤ ${f2(w.best.thr)}, gain ${f3(w.best.gain)}. The children hold ${node.left!.n} and ${node.right!.n} points.`;
  }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const ctxOf = () => stage.ctx;

    /** Shrink the font until `text` fits `maxW`. */
    function fit(text: string, maxW: number, px: number, weight = 500) {
      const ctx = ctxOf();
      let s = px;
      ctx.font = stage.mono(s, weight);
      while (s > 8 && ctx.measureText(text).width > maxW) { s -= 0.5; ctx.font = stage.mono(s, weight); }
    }

    type Sq = { px0: number; py0: number; s: number; gx: (v: number) => number; gy: (v: number) => number };
    const square = (px0: number, py0: number, s: number): Sq => ({ px0, py0, s, gx: scale(0, 1, px0, px0 + s), gy: scale(0, 1, py0 + s, py0) });

    function frameAxes(q: Sq) {
      const { ctx, c } = stage;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(q.px0, q.py0, q.s, q.s);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [0, 0.5, 1]) ctx.fillText(String(v), q.gx(v), q.py0 + q.s + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, 0.5, 1]) ctx.fillText(String(v), q.px0 - 5, q.gy(v));
      ctx.font = stage.body(12, true); ctx.textBaseline = 'bottom'; ctx.textAlign = 'right';
      ctx.fillText('x', q.px0 + q.s, q.py0 + q.s + 30);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('y', q.px0 - 26, q.py0 - 3);
    }
    const tint = (q: Sq, lo: number[], hi: number[], cls: number, a: number) => {
      const { ctx, c } = stage;
      ctx.globalAlpha = a; ctx.fillStyle = cls ? c.pos.hex : c.neg.hex;
      ctx.fillRect(q.gx(lo[0]), q.gy(hi[1]), q.gx(hi[0]) - q.gx(lo[0]), q.gy(lo[1]) - q.gy(hi[1]));
      ctx.globalAlpha = 1;
    };
    /** The segment where node `n` is cut, inside its own region. */
    const cutLine = (q: Sq, n: S.TreeNode, thr = n.threshold!, f = n.feature!): [number, number, number, number] =>
      f === 0 ? [q.gx(thr), q.gy(n.lo[1]), q.gx(thr), q.gy(n.hi[1])] : [q.gx(n.lo[0]), q.gy(thr), q.gx(n.hi[0]), q.gy(thr)];
    function points(q: Sq, X: number[][], y: number[], mask: Uint8Array | null, wrong: ((i: number) => boolean) | null) {
      const { ctx, c } = stage;
      for (let i = 0; i < X.length; i++) {
        const x = q.gx(X[i][0]), v = q.gy(X[i][1]), bad = wrong?.(i) ?? false;
        ctx.globalAlpha = mask && !mask[i] ? 0.3 : 1;
        ctx.fillStyle = y[i] ? c.pos.hex : c.neg.hex; ctx.strokeStyle = bad ? c.text.hex : c.raised.hex; ctx.lineWidth = bad ? 2 : 1.5;
        ctx.beginPath();
        if (y[i]) ctx.arc(x, v, 4.6, 0, 6.2832); else { ctx.moveTo(x, v - 5.6); ctx.lineTo(x + 5.6, v); ctx.lineTo(x, v + 5.6); ctx.lineTo(x - 5.6, v); ctx.closePath(); }
        ctx.fill(); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // ------------------------------------------------------------ tree modes
    const layTree = () => {
      const W = stage.w, H = stage.h;
      if (W >= 540) {
        const leftW = Math.round(W * 0.58), s = Math.min(leftW - 40, Math.round(H * 0.56));
        return { s, px0: 32, py0: 6, ct: 6 + s + 46, cb: H - 34, tree: { x: leftW + 10, y: 26, w: W - leftW - 14, h: H - 32 }, cap: 4 };
      }
      const s = Math.min(W - 48, 340), ct = 6 + s + 46, cb = ct + 96;
      return { s, px0: 34, py0: 6, ct, cb, tree: { x: 4, y: cb + 64, w: W - 8, h: Math.max(60, H - cb - 68) }, cap: cb + 44 };
    };

    function drawTree(box: { x: number; y: number; w: number; h: number }, cap: number, st: S.Story, f: S.Frame) {
      const { ctx, c } = stage;
      ctx.font = stage.body(12); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('The tree (upper branch: yes)', box.x, cap);
      const cols = st.maxDepth + 1, colW = box.w / cols, nodeW = Math.min(150, colW * 0.78), rowH = box.h / st.nRows, nodeH = Math.min(54, rowH - 6), two = nodeH >= 30;
      const nx = (n: S.TreeNode) => box.x + n.depth * colW, ny = (n: S.TreeNode) => box.y + (st.row[n.id] + 0.5) * rowH;
      ctx.lineWidth = 1.5; ctx.strokeStyle = c.muted.hex;
      for (const n of st.nodes) {
        if (!n.left) continue;
        for (const ch of [n.left, n.right!]) {
          const a = S.nodeAlpha(st, f, ch.id);
          if (a <= 0) continue;
          const x0 = nx(n) + nodeW, x1 = nx(ch), xm = (x0 + x1) / 2;
          ctx.globalAlpha = a; ctx.beginPath(); ctx.moveTo(x0, ny(n)); ctx.lineTo(xm, ny(n)); ctx.lineTo(xm, ny(ch)); ctx.lineTo(x1, ny(ch)); ctx.stroke();
        }
      }
      for (const n of st.nodes) {
        const a = S.nodeAlpha(st, f, n.id);
        if (a <= 0) continue;
        const split = S.isSplit(st, f, n.id), active = f.node === n.id && (f.stage === 'intro' || f.stage === 'sweep' || f.stage === 'snap');
        const x = nx(n), y = ny(n) - nodeH / 2;
        ctx.globalAlpha = a;
        ctx.fillStyle = c.raised.hex; ctx.beginPath(); ctx.roundRect(x, y, nodeW, nodeH, 4); ctx.fill();
        if (!split) { ctx.globalAlpha = a * 0.25; ctx.fillStyle = n.pred ? c.pos.hex : c.neg.hex; ctx.beginPath(); ctx.roundRect(x, y, nodeW, nodeH, 4); ctx.fill(); ctx.globalAlpha = a; }
        ctx.strokeStyle = active ? c.accent.hex : c.rule.hex; ctx.lineWidth = active ? 2.4 : 1.2; ctx.stroke();
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = c.text.hex;
        const head = split ? `${FEAT[n.feature!]} ≤ ${f2(n.threshold!)}` : two ? `class ${n.pred}` : `n=${n.n}`;
        fit(head, nodeW - 8, nodeH > 40 ? 14 : 11, 650);
        ctx.fillText(head, x + nodeW / 2, y + nodeH / (two ? 3 : 2));
        if (two) { const sub = `n=${n.n} · ${f2(n.impurity)}`; fit(sub, nodeW - 8, nodeH > 40 ? 12 : 10); ctx.fillStyle = c.muted.hex; ctx.fillText(sub, x + nodeW / 2, y + (nodeH * 2) / 3 + 1); }
      }
      ctx.globalAlpha = 1;
    }

    function renderTree(st: S.Story) {
      const { ctx, c } = stage, L = layTree(), f = S.frameAt(t, st), q = square(L.px0, L.py0, L.s);
      const E = st.events[f.event], node = st.nodes[E.node], scanning = f.stage === 'intro' || f.stage === 'sweep' || f.stage === 'snap';
      frameAxes(q);
      // partition: current leaves tinted by their majority class
      const single = E.sweeps.length === 1 && (f.stage === 'sweep' || f.stage === 'snap');
      for (const n of st.nodes) {
        const a = S.nodeAlpha(st, f, n.id);
        if (a <= 0) continue;
        const split = S.isSplit(st, f, n.id);
        if (split && st.eventOf[n.id] !== f.splitCount - 1) continue;
        if (single && n.id === node.id) continue;
        tint(q, n.lo, n.hi, n.pred, 0.17 * (split ? 1 - f.reveal : a));
      }
      if (single) {
        const sf = f.sweeps[0];
        tint(q, node.lo, [sf.thr, node.hi[1]], sf.ev.cL[1] > sf.ev.cL[0] ? 1 : 0, sf.ev.nL ? 0.24 : 0);
        tint(q, [sf.thr, node.lo[1]], node.hi, sf.ev.cR[1] >= sf.ev.cR[0] && sf.ev.nR ? 1 : 0, sf.ev.nR ? 0.24 : 0);
      }
      if (scanning) {
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
        ctx.strokeRect(q.gx(node.lo[0]) + 0.5, q.gy(node.hi[1]) + 0.5, q.gx(node.hi[0]) - q.gx(node.lo[0]) - 1, q.gy(node.lo[1]) - q.gy(node.hi[1]) - 1);
        ctx.setLineDash([]);
      }
      // cuts already made
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 2.2; ctx.lineCap = 'round';
      for (let i = 0; i < f.splitCount; i++) {
        const n = st.nodes[st.events[i].node], [x0, y0, x1, y1] = cutLine(q, n);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }
      ctx.lineCap = 'butt';
      let mask: Uint8Array | null = null;
      if (f.stage === 'sweep' || f.stage === 'snap') { mask = new Uint8Array(S.TRAIN.X.length); for (const i of node.idx) mask[i] = 1; }
      points(q, S.TRAIN.X, S.TRAIN.y, mask, null);
      // candidate thresholds
      if (scanning) {
        for (const sf of f.sweeps) {
          if (sf.alpha <= 0) continue;
          const col = sf.feature === 0 ? c.fwd.hex : c.accent.hex, [x0, y0, x1, y1] = cutLine(q, node, sf.thr, sf.feature);
          ctx.globalAlpha = sf.alpha; ctx.strokeStyle = col; ctx.lineWidth = 2.6; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
          ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(x0, y0, 4.5, 0, 6.2832); ctx.fill(); ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }

      // curve beneath: impurity (split) or information gain (gain, grow) against the threshold
      const kind = st.mode === 'split' ? 'impurity' : 'gain', val = (w: number) => (kind === 'impurity' ? w : node.impurity - w);
      const yMax = kind === 'impurity' ? node.impurity * 1.12 : Math.max(0.02, ...E.sweeps.map((s) => s.best.gain)) * 1.25;
      const cx = scale(0, 1, L.px0, L.px0 + L.s), cy = scale(0, yMax, L.cb, L.ct);
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText(kind === 'impurity' ? `Average ${st.crit === 'gini' ? 'Gini' : 'entropy'} of the two children` : `Information gain (${st.crit === 'gini' ? 'Gini' : 'entropy'})`, L.px0 - 26, L.ct - 10);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(L.px0, L.ct, L.s, L.cb - L.ct);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      const ticks = niceTicks(yMax), dec = ticks.every((v) => Math.abs(v * 10 - Math.round(v * 10)) < 1e-9) ? 1 : 2;
      for (const v of ticks) { ctx.fillText(f2(v, dec), L.px0 - 5, cy(v)); if (v > 0) { ctx.beginPath(); ctx.moveTo(L.px0, cy(v)); ctx.lineTo(L.px0 + L.s, cy(v)); ctx.stroke(); } }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [0, 0.5, 1]) ctx.fillText(String(v), cx(v), L.cb + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('threshold', L.px0 + L.s, L.cb + 32);
      if (kind === 'impurity') {
        ctx.setLineDash([4, 4]); ctx.strokeStyle = c.muted.hex; ctx.beginPath(); ctx.moveTo(L.px0, cy(node.impurity)); ctx.lineTo(L.px0 + L.s, cy(node.impurity)); ctx.stroke(); ctx.setLineDash([]);
        ctx.font = stage.body(11.5, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(`impurity before the split ${f3(node.impurity)}`, L.px0 + L.s - 4, cy(node.impurity) - 3);
      }
      ctx.save(); ctx.beginPath(); ctx.rect(L.px0, L.ct - 2, L.s, L.cb - L.ct + 4); ctx.clip();
      E.sweeps.forEach((sw, i) => {
        const sf = f.sweeps[i];
        if (sf.traced <= 0) return;
        const col = sw.feature === 0 ? c.fwd.hex : c.accent.hex, a = i === E.win ? 1 : f.stage === 'sweep' || f.stage === 'intro' ? 1 : 0.35;
        const last = Math.floor(sf.traced * (S.G - 1));
        ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath();
        for (let j = 0; j <= last; j++) { const X = cx(sw.lo + ((sw.hi - sw.lo) * j) / (S.G - 1)), Y = cy(val(sw.curve[j])); if (j) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
        if (sf.traced < 1) ctx.lineTo(cx(sf.thr), cy(val(sf.ev.weighted)));
        ctx.stroke();
        if (sf.alpha > 0) {
          ctx.globalAlpha = sf.alpha; ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(cx(sf.thr), cy(val(sf.ev.weighted)), 5.5, 0, 6.2832); ctx.fill(); ctx.stroke();
        }
        if (i === E.win && f.stage !== 'sweep' && f.stage !== 'intro') {
          ctx.globalAlpha = 1; ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.setLineDash([3, 3]);
          ctx.beginPath(); ctx.moveTo(cx(sf.thr), cy(val(sf.ev.weighted))); ctx.lineTo(cx(sf.thr), L.cb); ctx.stroke(); ctx.setLineDash([]);
        }
        ctx.globalAlpha = 1;
      });
      ctx.restore();
      drawTree(L.tree, L.cap, st, f);
    }

    // ------------------------------------------------------------ impurity mode
    function renderImpurity() {
      const { ctx, c, w: W, h: H } = stage, { main, side } = splitPanels(W, H, W < 540 ? 0.5 : 0.4), im = S.impurityAt(t);
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('100 points in one node', main.x + 4, main.y + 2);
      const g = Math.max(60, Math.min(main.w - 16, main.h - 34)), cell = g / 10, x0 = main.x + (main.w - g) / 2, y0 = main.y + 26;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, g, g);
      const r = cell * 0.3;
      for (let i = 0; i < S.BAG; i++) {
        const X = x0 + (i % 10) * cell + cell / 2, Y = y0 + Math.floor(i / 10) * cell + cell / 2, one = S.BAG_RANK[i] < im.k;
        ctx.fillStyle = one ? c.pos.hex : c.neg.hex; ctx.beginPath();
        if (one) ctx.arc(X, Y, r, 0, 6.2832); else { ctx.moveTo(X, Y - r * 1.2); ctx.lineTo(X + r * 1.2, Y); ctx.lineTo(X, Y + r * 1.2); ctx.lineTo(X - r * 1.2, Y); ctx.closePath(); }
        ctx.fill();
      }
      // curves
      const sd = { l: side.x + 40, r: side.x + side.w - 12, t: side.y + 30, b: side.y + side.h - 36 };
      const sx = scale(0, 1, sd.l, sd.r), sy = scale(0, 1.05, sd.b, sd.t);
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('Impurity against the class mix', sd.l - 30, side.y + 2);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(sd.l, sd.t, sd.r - sd.l, sd.b - sd.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, 0.25, 0.5, 0.75, 1]) { ctx.fillText(f2(v), sd.l - 5, sy(v)); if (v > 0) { ctx.beginPath(); ctx.moveTo(sd.l, sy(v)); ctx.lineTo(sd.r, sy(v)); ctx.stroke(); } }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [0, 0.5, 1]) ctx.fillText(String(v), sx(v), sd.b + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('p = fraction of class 1', sd.r, side.y + side.h - 1);
      const curve = (fn: (p: number) => number, col: string, w: number, dash: number[]) => {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash); ctx.beginPath();
        for (let i = 0; i <= 100; i++) { const p = i / 100; if (i) ctx.lineTo(sx(p), sy(fn(p))); else ctx.moveTo(sx(p), sy(fn(p))); }
        ctx.stroke(); ctx.setLineDash([]);
      };
      curve((p) => S.impurityP(p, 'entropy') / 2, c.muted.hex, 1.6, [4, 4]);
      curve((p) => S.impurityP(p, 'gini'), c.fwd.hex, 2.6, []);
      curve((p) => S.impurityP(p, 'entropy'), c.accent.hex, 2.6, []);
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(sx(im.p), sd.b); ctx.lineTo(sx(im.p), sy(Math.max(im.gini, im.entropy))); ctx.stroke(); ctx.setLineDash([]);
      for (const [v, col] of [[im.gini, c.fwd.hex], [im.entropy, c.accent.hex]] as const) {
        ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(sx(im.p), sy(v), 5.5, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      ctx.font = stage.body(11.5, true); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillStyle = c.accent.hex; ctx.fillText('entropy', sx(0.63), sy(S.impurityP(0.63, 'entropy')) - 8);
      ctx.fillStyle = c.fwd.hex; ctx.fillText('Gini', sx(0.68), sy(S.impurityP(0.68, 'gini')) - 6);
      ctx.fillStyle = c.muted.hex; ctx.textBaseline = 'top'; ctx.fillText('entropy ÷ 2', sx(0.05), sy(S.impurityP(0.05, 'entropy') / 2) + 14);
    }

    // ------------------------------------------------------------ depth mode
    function renderDepth(sy_: S.Study) {
      const { ctx, c, w: W, h: H } = stage, { main, side } = splitPanels(W, H, W < 540 ? 0.56 : 0.5), { v, d } = S.depthAt(t);
      const s = Math.max(60, Math.min(main.w - 40, main.h - 36)), q = square(main.x + 30 + (main.w - 30 - s) / 2, main.y + 6, s), tree = sy_.trees[d];
      frameAxes(q);
      for (const n of S.leavesOf(tree)) tint(q, n.lo, n.hi, n.pred, 0.2);
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.6; ctx.lineCap = 'round';
      for (const n of S.nodesOf(tree)) { if (!n.left) continue; const [x0, y0, x1, y1] = cutLine(q, n); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); }
      ctx.lineCap = 'butt';
      points(q, S.TRAIN.X, S.TRAIN.y, null, (i) => S.predictOne(tree, S.TRAIN.X[i]) !== S.TRAIN.y[i]);

      const sd = { l: side.x + 42, r: side.x + side.w - 14, t: side.y + 30, b: side.y + side.h - 38 };
      const sx = scale(0, S.DMAX, sd.l, sd.r), sy = scale(0.5, 1, sd.b, sd.t);
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('Accuracy as the tree gets deeper', sd.l - 30, side.y + 2);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(sd.l, sd.t, sd.r - sd.l, sd.b - sd.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const a of [0.5, 0.75, 1]) { ctx.fillText(pct(a), sd.l - 5, sy(a)); if (a > 0.5) { ctx.beginPath(); ctx.moveTo(sd.l, sy(a)); ctx.lineTo(sd.r, sy(a)); ctx.stroke(); } }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let k = 0; k <= S.DMAX; k++) ctx.fillText(String(k), sx(k), sd.b + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('max depth', sd.r, side.y + side.h - 1);
      const last = Math.floor(v);
      const trace = (a: number[], col: string) => {
        ctx.strokeStyle = col; ctx.lineWidth = 2.6; ctx.lineJoin = 'round'; ctx.beginPath();
        for (let k = 0; k <= last; k++) { if (k) ctx.lineTo(sx(k), sy(a[k])); else ctx.moveTo(sx(0), sy(a[0])); }
        if (v > last) ctx.lineTo(sx(v), sy(a[last] + (a[Math.min(S.DMAX, last + 1)] - a[last]) * (v - last)));
        ctx.stroke();
        ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
        for (let k = 0; k <= last; k++) { ctx.beginPath(); ctx.arc(sx(k), sy(a[k]), k === d ? 5.5 : 3.2, 0, 6.2832); ctx.fill(); ctx.stroke(); }
      };
      trace(sy_.train, c.fwd.hex); trace(sy_.val, c.accent.hex);
      const gap = sy_.train[d] - sy_.val[d];
      if (gap > 0.02) {
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(sx(d), sy(sy_.train[d])); ctx.lineTo(sx(d), sy(sy_.val[d])); ctx.stroke(); ctx.setLineDash([]);
        ctx.font = stage.body(11.5, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = d > S.DMAX - 2 ? 'right' : 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(`gap ${pct(gap)}`, sx(d) + (d > S.DMAX - 2 ? -7 : 7), (sy(sy_.train[d]) + sy(sy_.val[d])) / 2);
      }
    }

    function render() {
      stage.ctx.clearRect(0, 0, stage.w, stage.h);
      if (story) renderTree(story); else if (init.mode === 'impurity') renderImpurity(); else if (study) renderDepth(study);
    }

    const read = () => {
      const vals: { k: string; v: string }[] = [];
      let say = '', p = ui.p, d = ui.d;
      if (story) {
        const st = story, f = S.frameAt(t, st), E = st.events[f.event], node = st.nodes[E.node];
        vals.push({ k: `node: ${node.n} points, ${nm}`, v: f3(node.impurity) });
        if (f.stage !== 'intro') {
          f.sweeps.forEach((sf, i) => {
            if (f.stage !== 'sweep' && f.stage !== 'snap' && i !== E.win) return;
            vals.push({ k: `${FEAT[sf.feature]} ≤ ${f2(sf.thr)}`, v: `children ${f3(sf.ev.weighted)}, gain ${f3(sf.ev.gain)}` });
          });
          if (f.stage === 'reveal' || f.stage === 'done') { const e = f.sweeps[E.win].ev; vals.push({ k: 'left | right', v: `${e.nL} points (${e.cL[1]} class 1) | ${e.nR} points (${e.cR[1]} class 1)` }); }
        }
        if (st.mode === 'grow' && f.stage === 'done') vals.push({ k: 'training accuracy', v: pct(S.accuracy(st.tree, S.TRAIN.X, S.TRAIN.y)) });
        say = narrate(f, st);
      } else if (init.mode === 'impurity') {
        const im = S.impurityAt(t);
        p = im.p;
        vals.push({ k: 'p', v: f2(im.p) }, { k: 'Gini', v: f3(im.gini) }, { k: 'entropy', v: `${f3(im.entropy)} bits` }, { k: 'counts', v: `${im.k} class 1, ${S.BAG - im.k} class 0` });
        say = im.k === 0 || im.k === S.BAG ? 'A pure node: every point has the same class, so both measures are exactly zero.'
          : im.p === 0.5 ? 'An even mix: the most uncertain a node can be. Gini peaks at 0.5 and entropy at 1 bit.'
            : `Mixed ${im.p < 0.5 ? 'towards class 0' : 'towards class 1'}: the more lopsided the node, the lower both measures.`;
      } else if (study) {
        const dd = S.depthAt(t).d;
        d = dd;
        vals.push({ k: 'max depth', v: String(dd) }, { k: 'leaves', v: String(study.leaves[dd]) }, { k: 'training accuracy', v: pct(study.train[dd]) }, { k: 'validation accuracy', v: pct(study.val[dd]) });
        const bestD = study.val.indexOf(Math.max(...study.val));
        say = dd === 0 ? 'Depth 0 is a single leaf that always answers with the majority class.'
          : dd < bestD ? 'Each extra level carves out more of the real pattern: both accuracies rise.'
            : dd === bestD ? `Validation accuracy peaks here, at depth ${bestD}. The tree has found the two real cuts.`
              : study.train[dd] >= 1 ? 'Every training point is now classified correctly, including the mislabelled ones, but validation accuracy has fallen.'
                : 'Extra levels now chase individual noisy points: training accuracy climbs while validation accuracy drops.';
      }
      ui = { vals, say, p, d };
    };

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; read(); }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    read();
    return () => { loop.destroy(); stage.destroy(); };
  });

  const cfg = init.mode === 'impurity' ? { aspect: '720 / 380', narrow: '360 / 640' } : init.mode === 'depth' ? { aspect: '720 / 400', narrow: '360 / 660' } : { aspect: '720 / 520', narrow: '360 / 760' };
  const label = init.mode === 'impurity' ? 'Left: one hundred points, circles for class 1 and diamonds for class 0, whose mix changes. Right: the Gini and entropy curves against the fraction of class 1, with a dot for the current mix.'
    : init.mode === 'depth' ? 'Left: the training points with the rectangles a decision tree of the chosen depth carves out. Right: training and validation accuracy against maximum depth.'
      : 'Top left: two-class points on a square with a candidate threshold sweeping across and the rectangles the tree carves out. Bottom left: a curve of impurity or information gain against the threshold. Right: the decision tree as a flowchart growing split by split.';
  const legend = init.mode === 'impurity' ? [{ label: 'Gini', color: '--viz-forward', shape: 'line' as const }, { label: 'entropy', color: '--accent', shape: 'line' as const }, { label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }]
    : init.mode === 'depth' ? [{ label: 'training accuracy', color: '--viz-forward', shape: 'line' as const }, { label: 'validation accuracy', color: '--accent', shape: 'line' as const }, { label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }]
      : [{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'threshold on x', color: '--viz-forward', shape: 'line' as const }, ...(init.mode === 'split' ? [] : [{ label: 'threshold on y', color: '--accent', shape: 'line' as const }]), { label: 'chosen cut', color: '--text', shape: 'line' as const }];
  const tlLabel = init.mode === 'impurity' ? 'Timeline: changing the class mix' : init.mode === 'depth' ? 'Timeline: growing the tree deeper' : 'Timeline: growing the tree';
</script>

<FigureCard title={init.title || 'A decision tree choosing its questions'} role="group" aspect={cfg.aspect} narrowAspect={cfg.narrow} {label} {legend}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label={tlLabel}
      onSeek={(to) => { t = to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">{#each ui.vals as r, i (r.k)}{#if i}<span class="sep"> · </span>{/if}<span class="grp">{r.k} <b>{r.v}</b></span>{/each}</p>
    <p class="say" role="status" aria-live="polite">{ui.say}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if isTree}
        <label>Impurity
          <select value={crit} onchange={(e) => setCrit(e.currentTarget.value as S.Criterion)}>
            <option value="gini">Gini</option><option value="entropy">Entropy</option>
          </select>
        </label>
        {#if init.mode === 'grow'}
          <label>Max depth
            <select value={md} onchange={(e) => setDepth(+e.currentTarget.value)}>
              <option value={1}>1</option><option value={2}>2</option><option value={3}>3</option>
            </select>
          </label>
        {/if}
        <span class="hint">Drag the timeline to scrub through the splits.</span>
      {:else if init.mode === 'impurity'}
        <F1Slider label="p" value={ui.p} min={0} max={1} step={0.01} oninput={(v) => seekTo(v * S.T_IMP)} />
        <span class="hint">Drag p to mix the node by hand.</span>
      {:else}
        <F1Slider label="max depth" value={ui.d} min={0} max={S.DMAX} step={1} digits={0} oninput={(v) => seekTo((v * S.T_DEPTH) / S.DMAX)} />
        <label>Min points per leaf
          <select value={minLeaf} onchange={(e) => setMinLeaf(+e.currentTarget.value)}>
            <option value={1}>1 (no limit)</option><option value={3}>3</option><option value={6}>6</option>
          </select>
        </label>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .sep { color: var(--text-muted); padding: 0 0.5em; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: 0.5em; font: var(--step--1) var(--font-ui); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
