<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { nodesOf } from '../TreeSplitter/sim';
  import * as S from './sim';

  interface Props {
    title?: string;
    /** 'bag': bootstrap samples and averaged trees. 'forest': bagging against a random forest. 'boost': fit residuals. 'gradient': residuals as negative gradients. 'compare': error against number of trees. */
    mode?: S.Mode;
  }
  let { title = '', mode = 'bag' }: Props = $props();
  const init = untrack(() => ({ mode, title }));
  const M = init.mode;
  const isBoost = M === 'boost' || M === 'gradient';
  const R = 30;

  const bag = M === 'bag' ? S.buildCls(2, S.BAG_N) : null;
  // the second forest is fitted right after first paint so init stays short
  const forest: { plain: S.ClsEnsemble; rf: S.ClsEnsemble | null } | null = M === 'forest' ? { plain: S.buildCls(2, S.FOREST_N), rf: null } : null;
  const cmp = M === 'compare' ? S.buildCompare() : null;
  let lr = $state(0.3), depth = $state(2), loss = $state<S.Loss>('sq');
  let run = $state.raw<S.BoostRun | null>(isBoost ? S.boostRun({ lr: 0.3, depth: 2, rounds: R, loss: 'sq' }) : null);

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let ui = $state<{ vals: { k: string; v: string }[]; say: string }>({ vals: [], say: '' });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const T = $derived(bag ? S.BAG_N * S.BAG_TIMING.per : forest ? S.FOREST_N * S.FOREST_TIMING.per : run ? S.boostT(R) : S.T_CMP);
  const MK = $derived(
    bag ? S.clsMarkers(S.BAG_TIMING, S.BAG_N, 'Growing trees', 'Last tree')
      : forest ? S.clsMarkers(S.FOREST_TIMING, S.FOREST_N, 'Adding trees', 'Last tree')
        : run ? S.boostMarkers(R) : cmp!.markers,
  );
  $effect(() => { playing; speed; wake(); });

  const restart = () => { t = 0; playing = !reducedMotion(); wake(); };
  function refit(l = lr, d = depth, ls = loss) { lr = l; depth = d; loss = ls; run = S.boostRun({ lr: l, depth: d, rounds: R, loss: ls }); restart(); }

  const f3 = (v: number) => f2(v, 3);
  const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '–');
  const lossName = $derived(loss === 'sq' ? 'MSE' : 'mean |error|');

  function niceTicks(max: number): number[] {
    const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1].find((s) => max / s <= 5) ?? 1;
    return Array.from({ length: Math.floor(max / step + 1e-9) + 1 }, (_, i) => +(i * step).toFixed(4));
  }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const GC = S.GC;
    const mk = () => { const o = document.createElement('canvas'); o.width = o.height = GC; return o; };
    const mainMap = mk(), mapB = mk();
    const nTrain = S.CLS_TRAIN.X.length, picks = new Uint8Array(nTrain);
    const nR = S.REG_TRAIN.x.length;
    const tmpF = new Float64Array(nR), tmpG = new Float64Array(nR), tmpGrid = new Float64Array(S.NG);
    let thumbs: HTMLCanvasElement[] = [], thumbFor: unknown = null;

    type Sq = { px0: number; py0: number; s: number; gx: (v: number) => number; gy: (v: number) => number };
    const square = (px0: number, py0: number, s: number): Sq => ({ px0, py0, s, gx: scale(0, 1, px0, px0 + s), gy: scale(0, 1, py0 + s, py0) });
    const heading = (text: string, x: number, y: number) => {
      const { ctx, c } = stage;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(text, x, y);
    };
    function frameSquare(q: Sq) {
      const { ctx, c } = stage;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(q.px0, q.py0, q.s, q.s);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [0, 0.5, 1]) ctx.fillText(String(v), q.gx(v), q.py0 + q.s + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0, 0.5, 1]) ctx.fillText(String(v), q.px0 - 5, q.gy(v));
      ctx.font = stage.body(12, true); ctx.textBaseline = 'bottom'; ctx.textAlign = 'right';
      ctx.fillText('x', q.px0 + q.s, q.py0 + q.s + 30);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('y', q.px0 - 28, q.py0 + 1);
    }
    /** Paint a GC x GC map from probabilities of class 1 (0.5 = blank) onto an offscreen canvas. */
    function paint(o: HTMLCanvasElement, val: (i: number) => number, strength = 0.6) {
      const { c } = stage, ctx = o.getContext('2d')!, im = ctx.createImageData(GC, GC), b = c.raised.rgb;
      for (let i = 0; i < GC * GC; i++) {
        const p = val(i), tie = Math.abs(p - 0.5) < 1e-9, s = tie ? 0.3 : strength * Math.abs(2 * p - 1), col = tie ? c.muted.rgb : p > 0.5 ? c.pos.rgb : c.neg.rgb;
        for (let k = 0; k < 3; k++) im.data[4 * i + k] = b[k] + (col[k] - b[k]) * s;
        im.data[4 * i + 3] = 255;
      }
      ctx.putImageData(im, 0, 0);
    }
    const blit = (o: HTMLCanvasElement, q: Sq) => { const { ctx } = stage; ctx.imageSmoothingEnabled = false; ctx.drawImage(o, q.px0, q.py0, q.s, q.s); };
    /** The map of the first kA..kB trees, tweened by w. */
    function ensembleMap(e: S.ClsEnsemble, kA: number, kB: number, w: number, o: HTMLCanvasElement) {
      const a = e.gridVotes[kA], b = e.gridVotes[kB];
      paint(o, (i) => { const pa = kA ? a[i] / kA : 0.5, pb = kB ? b[i] / kB : 0.5; return pa + (pb - pa) * w; });
    }
    function shape(q: Sq, x: number, yv: number, cls: number, r: number) {
      const { ctx } = stage, px = q.gx(x), py = q.gy(yv);
      ctx.beginPath();
      if (cls) ctx.arc(px, py, r, 0, 6.2832); else { const d = r * 1.2; ctx.moveTo(px, py - d); ctx.lineTo(px + d, py); ctx.lineTo(px, py + d); ctx.lineTo(px - d, py); ctx.closePath(); }
    }
    function points(q: Sq, r: number, alpha: (i: number) => number) {
      const { ctx, c } = stage, D = S.CLS_TRAIN;
      ctx.lineWidth = 1.2; ctx.strokeStyle = c.raised.hex;
      for (let i = 0; i < D.X.length; i++) {
        ctx.globalAlpha = alpha(i); ctx.fillStyle = D.y[i] ? c.pos.hex : c.neg.hex;
        shape(q, D.X[i][0], D.X[i][1], D.y[i], r); ctx.fill(); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // ------------------------------------------------------------ bagging
    function renderBag(e: S.ClsEnsemble) {
      const { ctx, c, w: W, h: H } = stage, f = S.clsFrame(t, nTrain, S.BAG_TIMING, S.BAG_N), wide = W >= 540, gap = 8;
      let q: Sq, tx: number, ty: number, ts: number, cols: number;
      if (wide) {
        ts = Math.floor((H - 106) / 4); cols = 2;
        const rw = 2 * ts + gap, s = Math.min(W - rw - 74, H - 56);
        q = square(34, 22, s); tx = W - rw; ty = 22;
      } else {
        ts = Math.floor((W - 8 - 3 * gap) / 4); cols = 4;
        q = square(34, 22, W - 44); tx = 4; ty = 22 + q.s + 30 + 22;
      }
      heading(f.k === 0 ? 'Forest so far: no trees yet' : `Forest so far: vote of ${f.k} tree${f.k === 1 ? '' : 's'}`, q.px0, 2);
      const kA = f.stage === 'done' ? S.BAG_N : f.m, kB = f.stage === 'done' ? S.BAG_N : f.m + 1;
      ensembleMap(e, kA, kB, f.merge, mainMap); blit(mainMap, q);
      frameSquare(q);
      // the tree being grown: its cuts
      const cutA = f.stage === 'fit' ? S.ease(f.fit) : f.stage === 'merge' ? 1 - f.merge : 0;
      if (cutA > 0) {
        ctx.strokeStyle = c.text.hex; ctx.globalAlpha = 0.55 * cutA; ctx.lineWidth = 1;
        for (const n of nodesOf(e.trees[f.m])) {
          if (!n.left) continue;
          ctx.beginPath();
          if (n.feature === 0) { ctx.moveTo(q.gx(n.threshold!), q.gy(n.lo[1])); ctx.lineTo(q.gx(n.threshold!), q.gy(n.hi[1])); } else { ctx.moveTo(q.gx(n.lo[0]), q.gy(n.threshold!)); ctx.lineTo(q.gx(n.hi[0]), q.gy(n.threshold!)); }
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      // the bootstrap sample: picked points ringed (with a count when repeated), unpicked points dimmed
      S.picksAt(e, f.m, f.picked, picks);
      points(q, 3.6, (i) => (picks[i] > 0 ? 1 : 0.3));
      ctx.lineWidth = 1.6; ctx.strokeStyle = c.text.hex;
      for (let i = 0; i < nTrain; i++) {
        if (!picks[i]) continue;
        shape(q, S.CLS_TRAIN.X[i][0], S.CLS_TRAIN.X[i][1], S.CLS_TRAIN.y[i], 6.4); ctx.stroke();
      }
      if (f.stage === 'draw') {
        const dw = e.draws[f.m];
        for (let j = Math.max(0, f.picked - 6); j < f.picked; j++) {
          const age = (f.picked - j) / 6, i = dw[j];
          ctx.globalAlpha = 1 - age; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(q.gx(S.CLS_TRAIN.X[i][0]), q.gy(S.CLS_TRAIN.X[i][1]), 8 + 8 * age, 0, 6.2832); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      ctx.font = stage.mono(10.5, 650); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      for (let i = 0; i < nTrain; i++) if (picks[i] > 1) ctx.fillText(`×${picks[i]}`, q.gx(S.CLS_TRAIN.X[i][0]) + 7, q.gy(S.CLS_TRAIN.X[i][1]) - 7);
      // thumbnails: one per tree
      if (thumbFor !== c) { thumbFor = c; thumbs = e.treeGrid.map((g) => { const o = mk(); paint(o, (i) => g[i], 0.8); return o; }); }
      heading('Each tree on its own sample', wide ? tx : 4, wide ? 2 : ty - 20);
      for (let i = 0; i < S.BAG_N; i++) {
        const cx = tx + (i % cols) * (ts + gap), cy = ty + Math.floor(i / cols) * (ts + gap + 14);
        const box = square(cx, cy, ts), shown = i < f.m || (i === f.m && f.stage !== 'draw');
        if (shown) {
          ctx.globalAlpha = i === f.m && f.stage === 'fit' ? S.ease(f.fit) : 1;
          blit(thumbs[i], box); ctx.globalAlpha = 1;
        } else { ctx.fillStyle = c.sunken.hex; ctx.fillRect(cx, cy, ts, ts); }
        ctx.strokeStyle = i === f.m && f.stage !== 'done' ? c.accent.hex : c.rule.hex; ctx.lineWidth = i === f.m && f.stage !== 'done' ? 2 : 1; ctx.strokeRect(cx, cy, ts, ts);
        ctx.font = stage.mono(10.5); ctx.fillStyle = shown ? c.text.hex : c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`tree ${i + 1}`, cx, cy + ts + 2);
      }
    }

    // ------------------------------------------------------------ bagging against random forest
    function renderForest(ens: { plain: S.ClsEnsemble; rf: S.ClsEnsemble | null }) {
      const { ctx, c, w: W, h: H } = stage, f = S.clsFrame(t, nTrain, S.FOREST_TIMING, S.FOREST_N), wide = W >= 540;
      const kA = f.stage === 'done' ? S.FOREST_N : f.m, kB = f.stage === 'done' ? S.FOREST_N : f.m + 1;
      const s = wide ? Math.min(W / 2 - 50, H - 106) : W - 48;
      const panels = [
        { e: ens.plain, name: 'Bagged trees: every split sees x and y', o: mainMap },
        { e: ens.rf, name: 'Random forest: each split sees one random feature', o: mapB },
      ];
      panels.forEach((p, i) => {
        const x0 = wide ? i * (W / 2) + 34 : 34, y0 = wide ? 22 : 22 + i * (s + 30 + 52 + 24);
        const q = square(x0, y0, s);
        heading(p.name, x0, y0 - 20);
        if (p.e) { ensembleMap(p.e, kA, kB, f.merge, p.o); blit(p.o, q); }
        frameSquare(q);
        points(q, 2.8, () => 0.85);
        const d = p.e ? p.e.stats[f.k].disagree : NaN, by = y0 + s + 40;
        ctx.fillStyle = c.sunken.hex; ctx.fillRect(x0, by + 16, s, 7);
        if (Number.isFinite(d)) { ctx.fillStyle = c.grad.hex; ctx.fillRect(x0, by + 16, Math.min(1, d / 0.3) * s, 7); }
        ctx.font = stage.mono(11); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(Number.isFinite(d) ? `two trees disagree on ${pct(d)} of points` : 'two trees needed to compare', x0, by);
      });
    }

    // ------------------------------------------------------------ boosting and the gradient view
    interface Box { x: number; y: number; w: number; h: number }
    function axes(b: Box, xs: (v: number) => number, xt: number[], ys: (v: number) => number, yt: number[], xl: string, fmtY: (v: number) => string = String) {
      const { ctx, c } = stage;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of xt) ctx.fillText(String(v), xs(v), b.y + b.h + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of yt) { ctx.fillText(fmtY(v), b.x - 5, ys(v)); }
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText(xl, b.x + b.w, b.y + b.h + 20);
    }
    const stairs = (b: Box, gx: (v: number) => number, gy: (v: number) => number, a: ArrayLike<number>) => {
      const { ctx } = stage, dx = (gx(1) - gx(0)) / (S.NG - 1) / 2;
      ctx.beginPath();
      for (let i = 0; i < S.NG; i++) {
        const x = gx(S.GRID_X[i]), y = gy(a[i]);
        if (i === 0) { ctx.moveTo(x, y); } else ctx.lineTo(x - dx, gy(a[i - 1])), ctx.lineTo(x - dx, y);
        ctx.lineTo(x + (i === S.NG - 1 ? 0 : dx), y);
      }
      ctx.stroke();
    };
    function renderBoost(r: S.BoostRun) {
      const { ctx, c, w: W, h: H } = stage, f = S.boostFrame(t, R), wide = W >= 540, grad = M === 'gradient';
      const mm = Math.min(f.m, R - 1);
      const top: Box = wide ? { x: 44, y: 20, w: W - 58, h: Math.round(H * 0.46) } : { x: 44, y: 20, w: W - 58, h: 230 };
      const gapV = 18 + 16 + 18;
      const strip: Box = wide ? { x: 44, y: top.y + top.h + gapV, w: Math.round((W - 58 - 54) / 2), h: 0 } : { x: 44, y: top.y + top.h + gapV, w: W - 58, h: 150 };
      strip.h = wide ? H - strip.y - 40 : 150;
      const lossBox: Box = wide ? { x: strip.x + strip.w + 54, y: strip.y, w: W - 14 - (strip.x + strip.w + 54), h: strip.h } : { x: 44, y: strip.y + strip.h + gapV, w: W - 58, h: 170 };
      const gx = scale(0, 1, top.x, top.x + top.w), gy = scale(S.Y_LO, S.Y_HI, top.y + top.h, top.y);

      // --- top: data, prediction, residual bars or gradient arrows
      heading(grad ? (wide ? 'Prediction F(x) and the negative gradient at each point' : 'Prediction F(x) and −∂L/∂F') : (wide ? 'Prediction F(x) and the residual at each point' : 'Prediction F(x) and residuals'), top.x, top.y - 18);
      axes(top, gx, [0, 0.5, 1], gy, [-1, 0, 1], 'x');
      S.predTrain(r, f, tmpF);
      if (f.m < R) { for (let i = 0; i < nR; i++) tmpG[i] = r.G[f.m][i] + f.w * (r.G[f.m + 1][i] - r.G[f.m][i]); } else tmpG.set(r.G[R]);
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]); ctx.beginPath();
      GRID_TRUTH.forEach((v, i) => (i ? ctx.lineTo(gx(S.GRID_X[i]), gy(v)) : ctx.moveTo(gx(0), gy(v)))); ctx.stroke(); ctx.setLineDash([]);
      const xs = S.REG_TRAIN.x, ys = S.REG_TRAIN.y, barA = f.m === 0 && f.stage === 'residual' ? S.ease(f.a) : 1;
      ctx.globalAlpha = barA; ctx.strokeStyle = c.grad.hex; ctx.fillStyle = c.grad.hex; ctx.lineWidth = 2; ctx.lineCap = 'round';
      for (let i = 0; i < nR; i++) {
        const x = gx(xs[i]), y0 = gy(tmpF[i]);
        if (!grad) { ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, gy(ys[i])); ctx.stroke(); continue; }
        ctx.globalAlpha = barA * 0.3; ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, gy(ys[i])); ctx.stroke(); ctx.setLineDash([]);
        ctx.globalAlpha = barA; ctx.lineWidth = 2;
        const y1 = gy(tmpF[i] + tmpG[i]);
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y1); ctx.stroke();
        if (Math.abs(y1 - y0) > 7) { const d = Math.sign(y1 - y0); ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x - 3.5, y1 - d * 6); ctx.lineTo(x + 3.5, y1 - d * 6); ctx.closePath(); ctx.fill(); }
      }
      ctx.lineCap = 'butt'; ctx.globalAlpha = 1;
      S.predGrid(r, f, tmpGrid);
      if (f.m < R && (f.stage === 'fit' || f.stage === 'update')) {
        const next = Float64Array.from(r.Fg[f.m + 1]);
        ctx.strokeStyle = c.fwd.hex; ctx.globalAlpha = f.stage === 'fit' ? 0.85 * S.ease(f.a) : 0.85 * (1 - f.w); ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]); stairs(top, gx, gy, next); ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2.8; ctx.lineJoin = 'round'; stairs(top, gx, gy, tmpGrid);
      ctx.lineWidth = 1.2; ctx.strokeStyle = c.raised.hex; ctx.fillStyle = c.text.hex;
      for (let i = 0; i < nR; i++) { ctx.beginPath(); ctx.arc(gx(xs[i]), gy(ys[i]), 3.6, 0, 6.2832); ctx.fill(); ctx.stroke(); }

      // --- strip: residuals (stems) and the tree fitted to them
      const gm = gMax(r), gz = scale(-gm, gm, strip.y + strip.h, strip.y), sx = scale(0, 1, strip.x, strip.x + strip.w);
      heading(grad ? (wide ? '−∂L/∂F at each point, and the tree fitted to it' : '−∂L/∂F and the fitted tree') : (wide ? 'Residuals y − F, and the tree fitted to them' : 'Residuals and the fitted tree'), strip.x, strip.y - 18);
      axes(strip, sx, [0, 0.5, 1], gz, [-1, 0, 1], 'x');
      ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(strip.x, gz(0)); ctx.lineTo(strip.x + strip.w, gz(0)); ctx.stroke();
      ctx.globalAlpha = barA; ctx.strokeStyle = c.grad.hex; ctx.fillStyle = c.grad.hex; ctx.lineWidth = 2;
      for (let i = 0; i < nR; i++) { const x = sx(xs[i]); ctx.beginPath(); ctx.moveTo(x, gz(0)); ctx.lineTo(x, gz(tmpG[i])); ctx.stroke(); ctx.beginPath(); ctx.arc(x, gz(tmpG[i]), 2.6, 0, 6.2832); ctx.fill(); }
      ctx.globalAlpha = 1;
      if (f.m < R && (f.stage === 'fit' || f.stage === 'update')) {
        ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2.6; ctx.globalAlpha = f.stage === 'fit' ? S.ease(f.a) : 1 - f.w;
        const lv = S.leavesReg(r.trees[f.m]);
        ctx.beginPath();
        lv.forEach((l, i) => { ctx.moveTo(sx(l.lo), gz(l.v)); ctx.lineTo(sx(l.hi), gz(l.v)); if (i < lv.length - 1) { ctx.moveTo(sx(l.hi), gz(l.v)); ctx.lineTo(sx(l.hi), gz(lv[i + 1].v)); } });
        ctx.stroke(); ctx.globalAlpha = 1;
      }

      // --- loss against round
      const lmax = Math.max(r.lossTrain[0], r.lossVal[0]) * 1.05, lx = scale(0, R, lossBox.x, lossBox.x + lossBox.w), ly = scale(0, lmax, lossBox.y + lossBox.h, lossBox.y);
      heading(`Error after each round (${lossName})`, lossBox.x, lossBox.y - 18);
      axes(lossBox, lx, [0, 10, 20, 30], ly, niceTicks(lmax), 'rounds', (v) => v.toFixed(v < 0.1 ? 2 : 1));
      const floor = loss === 'sq' ? S.NOISE * S.NOISE : S.NOISE * Math.sqrt(2 / Math.PI);
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(lossBox.x, ly(floor)); ctx.lineTo(lossBox.x + lossBox.w, ly(floor)); ctx.stroke(); ctx.setLineDash([]);
      ctx.font = stage.body(11, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('noise floor', lossBox.x + lossBox.w - 4, ly(floor) + 3);
      const now = S.lossNow(r, f), xr = f.m + f.w;
      const curve = (a: number[], cur: number, col: string) => {
        ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(lx(0), ly(a[0]));
        for (let k = 1; k <= Math.floor(xr); k++) ctx.lineTo(lx(k), ly(a[k]));
        if (xr > Math.floor(xr)) ctx.lineTo(lx(xr), ly(cur));
        ctx.stroke();
        ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(lx(xr), ly(cur), 4.4, 0, 6.2832); ctx.fill(); ctx.stroke();
      };
      curve(r.lossVal, now.val, c.accent.hex); curve(r.lossTrain, now.train, c.fwd.hex);
    }
    const GRID_TRUTH = S.GRID_X.map(S.truth);
    const gMaxCache = new WeakMap<S.BoostRun, number>();
    function gMax(r: S.BoostRun) {
      let v = gMaxCache.get(r);
      if (v === undefined) { v = Math.max(1, ...Array.from(r.G[0], Math.abs)) * 1.1; gMaxCache.set(r, v); }
      return v;
    }

    // ------------------------------------------------------------ random forest against boosting
    function renderCompare(cp: S.Compare) {
      const { ctx, c, w: W, h: H } = stage, n = Math.max(1, S.cmpCount(t, cp.N));
      const b: Box = { x: 50, y: 26, w: W - 50 - 14, h: H - 26 - 50 };
      const YMAX = 0.3, gx = scale(1, cp.N, b.x, b.x + b.w), gy = scale(0, YMAX, b.y + b.h, b.y);
      heading(W < 540 ? 'Error on fresh data (MSE)' : 'Error on fresh data against the number of trees (MSE)', b.x, 2);
      axes(b, gx, [1, 25, 50, 75, 100], gy, [0, 0.1, 0.2, 0.3], 'trees', (v) => v.toFixed(1));
      ctx.save(); ctx.beginPath(); ctx.rect(b.x, b.y, b.w, b.h); ctx.clip();
      const fl = S.NOISE * S.NOISE;
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(b.x, gy(fl)); ctx.lineTo(b.x + b.w, gy(fl)); ctx.stroke(); ctx.setLineDash([]);
      const line = (a: number[], col: string, w: number, dash: number[]) => {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineJoin = 'round'; ctx.setLineDash(dash); ctx.beginPath();
        for (let k = 1; k <= n; k++) (k === 1 ? ctx.moveTo(gx(k), gy(a[k])) : ctx.lineTo(gx(k), gy(a[k])));
        ctx.stroke(); ctx.setLineDash([]);
      };
      line(cp.hiTrain, c.grad.hex, 1.6, [2, 3]);
      line(cp.loVal, c.accent.hex, 2.6, []); line(cp.hiVal, c.grad.hex, 2.6, []); line(cp.rfVal, c.fwd.hex, 2.6, []);
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(gx(n), b.y); ctx.lineTo(gx(n), b.y + b.h); ctx.stroke();
      for (const [a, col] of [[cp.rfVal, c.fwd.hex], [cp.hiVal, c.grad.hex], [cp.loVal, c.accent.hex]] as const) { ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(gx(n), gy(a[n]), 4.4, 0, 6.2832); ctx.fill(); ctx.stroke(); }
      ctx.restore();
      ctx.font = stage.body(11, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText('noise floor', b.x + 6, gy(fl) - 2);
    }

    function render() {
      stage.ctx.clearRect(0, 0, stage.w, stage.h);
      if (bag) renderBag(bag); else if (forest) renderForest(forest); else if (run) renderBoost(run); else if (cmp) renderCompare(cmp);
    }

    // ------------------------------------------------------------ readouts (same frame values as the canvas)
    const read = () => {
      const vals: { k: string; v: string }[] = [];
      let say = '';
      if (bag) {
        const f = S.clsFrame(t, nTrain, S.BAG_TIMING, S.BAG_N), st = bag.stats, s1 = st[1], sk = st[f.k] ?? st[0];
        S.picksAt(bag, f.m, f.picked, picks);
        let distinct = 0, dup = 0;
        for (let i = 0; i < nTrain; i++) { if (picks[i] > 0) distinct++; if (picks[i] > 1) dup++; }
        vals.push({ k: 'tree', v: `${f.stage === 'done' ? S.BAG_N : f.m + 1} of ${S.BAG_N}` }, { k: 'drawn', v: `${f.picked} of ${nTrain}` }, { k: 'distinct points', v: String(distinct) }, { k: 'out of the bag', v: String(nTrain - distinct) });
        vals.push({ k: 'tree 1 alone: training', v: pct(s1.trainAcc) }, { k: 'validation', v: pct(s1.valAcc) });
        if (f.k) vals.push({ k: `${f.k} tree${f.k === 1 ? '' : 's'} voting: training`, v: pct(sk.trainAcc) }, { k: 'validation', v: pct(sk.valAcc) });
        say = f.stage === 'draw' ? `Tree ${f.m + 1} gets its own sample: ${nTrain} draws with replacement from the ${nTrain} points. Points drawn more than once show a count; ${dup} repeat so far. Points not drawn yet are dimmed.`
          : f.stage === 'fit' ? `A full-depth tree is grown on this sample only. It chases the sample's noise, so its boundary is jagged. ${nTrain - distinct} points were left out of the bag.`
            : f.stage === 'merge' && f.k === 1 ? 'With one tree, the forest is that tree.'
              : f.stage === 'done' ? `${S.BAG_N} trees voting: the vote map is far smoother than any single tree, and validation accuracy is ${pct(sk.valAcc)} against ${pct(s1.valAcc)} for tree 1.`
                : `Tree ${f.m + 1} joins the vote. Averaging ${f.m + 1} trees smooths the map; validation accuracy is ${pct(bag.stats[f.m + 1].valAcc)}.`;
      } else if (forest) {
        const f = S.clsFrame(t, nTrain, S.FOREST_TIMING, S.FOREST_N), a = forest.plain.stats[f.k], b = forest.rf?.stats[f.k] ?? a;
        vals.push({ k: 'trees', v: String(f.k) });
        vals.push({ k: 'bagged: disagree', v: pct(a.disagree) }, { k: 'validation', v: f.k ? pct(a.valAcc) : '–' });
        vals.push({ k: 'random forest: disagree', v: pct(b.disagree) }, { k: 'validation', v: f.k ? pct(b.valAcc) : '–' });
        say = f.k === 0 ? 'Two forests grow from the same bootstrap samples. They differ only in which features each split may use.'
          : f.k === 1 ? 'One tree each. The two trees already differ slightly because of the feature restriction. Watch the disagreement bars once a second tree arrives.'
            : `Two random trees disagree on ${pct(b.disagree)} of validation points; two bagged trees only on ${pct(a.disagree)}. More disagreement between trees means more of their errors cancel in the vote.`;
      } else if (run) {
        const f = S.boostFrame(t, R), now = S.lossNow(run, f), r = run.data;
        vals.push({ k: 'rounds added', v: `${Math.floor(f.m + f.w + 1e-9)} of ${R}` }, { k: `training ${lossName}`, v: f3(now.train) }, { k: `validation ${lossName}`, v: f3(now.val) });
        if (M === 'gradient') {
          S.predTrain(run, f, tmpF);
          if (f.m < R) { for (let i = 0; i < nR; i++) tmpG[i] = run.G[f.m][i] + f.w * (run.G[f.m + 1][i] - run.G[f.m][i]); } else tmpG.set(run.G[R]);
          let bi = 0;
          for (let i = 1; i < nR; i++) if (Math.abs(tmpG[i]) > Math.abs(tmpG[bi]) + 1e-9) bi = i;
          vals.push({ k: `point ${bi + 1}: y`, v: f2(r.y[bi]) }, { k: 'F', v: f2(tmpF[bi]) }, { k: '−∂L/∂F', v: f2(tmpG[bi]) });
        }
        const first = f.m === 0;
        say = f.stage === 'done' ? `${R} rounds done. Training ${lossName} fell from ${f3(run.lossTrain[0])} to ${f3(run.lossTrain[R])}; validation ${lossName} is ${f3(run.lossVal[R])}.`
          : f.stage === 'residual' ? (M === 'gradient' ? `Round ${f.m + 1}: at each point the loss pulls the prediction ${loss === 'sq' ? 'in proportion to the miss, y − F' : 'with the same strength whatever the miss: only the sign of y − F'}. These arrows are −∂L/∂F.`
            : `Round ${f.m + 1}: the vertical bars are the residuals, what the current prediction still gets wrong.${first ? ' The starting prediction is the mean of y.' : ''}`)
            : f.stage === 'fit' ? `A small tree (depth ${depth}) is fitted to those ${M === 'gradient' ? 'arrows' : 'residuals'}: one constant per leaf. Its output is the stepped line in the lower panel; the dashed curve is the prediction plus ${f2(lr)} times the tree.`
              : `The prediction moves ${f2(lr)} of the way along the tree's output. The ${M === 'gradient' ? 'arrows' : 'residuals'} shrink and the ${lossName} falls.`;
      } else if (cmp) {
        const n = Math.max(1, S.cmpCount(t, cmp.N));
        vals.push({ k: 'trees', v: String(n) }, { k: 'forest', v: f3(cmp.rfVal[n]) }, { k: `boosting, rate ${f2(S.LR_HIGH)}`, v: f3(cmp.hiVal[n]) }, { k: `boosting, rate ${f2(S.LR_LOW)}`, v: f3(cmp.loVal[n]) });
        const worse = cmp.hiVal[n] - cmp.hiVal[cmp.hiBest];
        say = n < cmp.hiBest ? 'The fast booster improves quickest. The forest is already near its floor, and the slow booster is still far behind.'
          : n < cmp.loBest ? `The fast booster passed its best error (${f3(cmp.hiVal[cmp.hiBest])} at ${cmp.hiBest} trees) and is now ${worse > 0.002 ? 'getting worse' : 'flat'}: its training error keeps falling while the error on fresh data rises. The slow booster is still improving.`
            : `The slow booster has caught up with the forest (its best is ${f3(cmp.loVal[cmp.loBest])} at ${cmp.loBest} trees), while the fast booster has drifted up to ${f3(cmp.hiVal[n])} as it fits noise.`;
      }
      ui = { vals, say };
    };

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; read(); }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    read();
    const late = forest ? setTimeout(() => { forest.rf = S.buildCls(1, S.FOREST_N); read(); wake(); }, 0) : 0;
    return () => { clearTimeout(late); loop.destroy(); stage.destroy(); };
  });

  const cfg = M === 'bag' ? { aspect: '720 / 460', narrow: '360 / 640' }
    : M === 'forest' ? { aspect: '720 / 410', narrow: '360 / 830' }
      : isBoost ? { aspect: '720 / 540', narrow: '360 / 800' }
        : { aspect: '720 / 400', narrow: '360 / 440' };
  const label = M === 'bag' ? 'Left: the training points with the bootstrap sample being drawn for the current tree, over the shaded vote of the trees so far. Right: the decision boundary of each tree, grown on its own sample.'
    : M === 'forest' ? 'Two panels of the same training points. Left: the shaded vote of bagged trees. Right: the shaded vote of a random forest. Under each, the fraction of validation points on which two trees disagree.'
      : M === 'boost' ? 'Top: noisy points on a curve, the boosted prediction, and the residual at each point as a vertical bar. Bottom left: the residuals and the small tree fitted to them. Bottom right: training and validation error after each round.'
        : M === 'gradient' ? 'Top: noisy points, the boosted prediction, and an arrow at each point showing the negative gradient of the loss with respect to the prediction. Bottom left: those values and the tree fitted to them. Bottom right: error after each round.'
          : 'Error on fresh data against the number of trees for a random forest and for boosting with a high and a low learning rate, plus the training error of the fast booster.';
  const legend = M === 'bag' || M === 'forest' ? [{ label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'tied vote', color: '--text-muted' }, ...(M === 'bag' ? [{ label: 'picked', color: '--text' }, { label: 'newest pick', color: '--accent' }] : [{ label: 'disagreement', color: '--viz-gradient' }])]
    : isBoost ? [{ label: 'prediction F', color: '--viz-forward', shape: 'line' as const }, { label: M === 'gradient' ? '−∂L/∂F' : 'residual', color: '--viz-gradient', shape: 'line' as const }, { label: 'training error', color: '--viz-forward', shape: 'line' as const }, { label: 'validation error', color: '--accent', shape: 'line' as const }]
      : [{ label: 'random forest', color: '--viz-forward', shape: 'line' as const }, { label: 'boosting, high rate', color: '--viz-gradient', shape: 'line' as const }, { label: 'boosting, low rate', color: '--accent', shape: 'line' as const }, { label: 'fast booster, training (dotted)', color: '--viz-gradient', shape: 'line' as const }];
  const tlLabel = bag ? 'Timeline: adding trees to the forest' : forest ? 'Timeline: growing two forests' : isBoost ? 'Timeline: boosting rounds' : 'Timeline: adding trees';
  const defTitle = M === 'bag' ? 'Bootstrap samples and averaged trees' : M === 'forest' ? 'Bagging against a random forest' : M === 'boost' ? 'Boosting: each tree fits what is left' : M === 'gradient' ? 'Residuals are negative gradients' : 'Random forest against boosting';
</script>

<FigureCard title={init.title || defTitle} aspect={cfg.aspect} narrowAspect={cfg.narrow} {label} {legend}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label={tlLabel}
      onSeek={(to) => { t = to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">{#each ui.vals as r, i (i)}{#if i}<span class="sep"> · </span>{/if}<span class="grp">{r.k} <b>{r.v}</b></span>{/each}</p>
    <p class="say" role="status" aria-live="polite">{ui.say}</p>
  {/snippet}
  {#snippet footer()}
    {#if isBoost}
      <div class="ctl">
        <F1Slider label="learning rate" value={lr} min={0.05} max={1} step={0.05} oninput={(v) => refit(v, depth, loss)} />
        <label>Tree depth
          <select value={depth} onchange={(e) => refit(lr, +e.currentTarget.value, loss)}>
            <option value={1}>1 (stump)</option><option value={2}>2</option><option value={3}>3</option>
          </select>
        </label>
        {#if M === 'gradient'}
          <label>Loss
            <select value={loss} onchange={(e) => refit(lr, depth, e.currentTarget.value as S.Loss)}>
              <option value="sq">Squared</option><option value="abs">Absolute</option>
            </select>
          </label>
        {/if}
        <span class="hint">Changing a setting restarts the run.</span>
      </div>
    {:else}
      <div class="ctl"><span class="hint">Drag the timeline to scrub.</span></div>
    {/if}
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
