<script module lang="ts">
  import type { FromWorker, ToWorker } from './worker';
  let worker: Worker | null = null;
  const pending = new Map<string, Promise<FromWorker>>();
  /** Train in a worker, once per kind per page; every widget instance shares the result. */
  function load(kind: ToWorker['kind']): Promise<FromWorker> {
    let p = pending.get(kind);
    if (!p) {
      const w = (worker ??= new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' }));
      p = new Promise((res) => {
        const on = (e: MessageEvent<FromWorker>) => { if (e.data.kind === kind) { w.removeEventListener('message', on); res(e.data); } };
        w.addEventListener('message', on);
        w.postMessage({ kind } satisfies ToWorker);
      });
      pending.set(kind, p);
    }
    return p;
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, clamp } from '../F1Shared/plot';
  import { createRng } from '../../../lib/math/rng';
  import {
    SIZES, GRID, DATASETS, makeData, rules, interp, raceAt, raceMarkers, inspectAt, inspectMarkers, checklistAt, checklistMarkers,
    QUESTION_IDS, recommend, questions, CHOICE_DEFAULT, duration, type Mode, type DatasetId, type Curves, type Inspect, type Tree, type Choice, type Data,
  } from './sim';

  interface Props {
    title?: string;
    /** 'race': learning curves of boosted trees vs a small net as the training set grows. 'inspect': read a tree, try to read a net. 'checklist': answer four questions. */
    mode?: Mode;
    /** race only: which data to start on. */
    dataset?: DatasetId;
    /** race only: plot accuracy against training-set size or against training work. */
    xAxis?: 'n' | 'work';
    /** race only: show the dataset switch. */
    switcher?: boolean;
  }
  let { title = '', mode = 'race', dataset = 'tabular', xAxis = 'n', switcher = true }: Props = $props();
  const init = untrack(() => ({ mode, xAxis, switcher, title: title || 'Trees or a neural net?' }));
  const T = duration(init.mode);
  const MK = init.mode === 'race' ? raceMarkers : init.mode === 'inspect' ? inspectMarkers : checklistMarkers;
  const DS_LABEL: Record<DatasetId, string> = { tabular: 'Spreadsheet-like', pixels: 'Diagonal stripes' };

  let ds = $state<DatasetId>(untrack(() => dataset));
  let choice = $state<Choice>({ ...CHOICE_DEFAULT });
  let playing = $state(false), speed = $state(0.5), uiT = $state(0), ready = $state(false);
  let ui = $state.raw({ a: '', b: '', c: '' });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let curves: Partial<Record<DatasetId, Curves>> = {};
  let ins: Inspect | null = null;

  const phase = $derived(markerAt(uiT, MK)?.label ?? '');
  const verdict = $derived(recommend(choice));
  const narration = $derived(
    !ready ? 'Training both models in a background thread…'
    : init.mode === 'race' ? (
      init.xAxis === 'work' ? (phase === 'small data' ? 'Each dot is a finished training run. The horizontal position is the work it took: comparisons for the trees, multiply-adds for the net.'
        : phase === 'medium data' ? 'The net needs far more arithmetic for the same number of rows, because it revisits every row for hundreds of epochs.'
        : (ds === 'tabular' ? 'Reading across at one accuracy level, the trees got there for much less work on this data.' : 'On this data the net pays far more work but reaches accuracy the trees never do, so the extra cost buys something.'))
      : ds === 'tabular' ? (phase === 'small data' ? 'Few rows. Two columns carry the signal and two are noise. The trees cut on the useful columns; the net must first learn to ignore the others.'
        : phase === 'medium data' ? 'More rows. The trees approach the best possible accuracy with sharp, axis-aligned boxes. The net improves, but its smooth boundary rounds off the corners.'
        : 'Plenty of rows. The gap narrows because extra data helps the net more, but the trees stay ahead on this data.')
      : (phase === 'small data' ? 'Few rows. The stripes run diagonally. Every tree cut is parallel to an axis, so it can only stair-step; one neuron can tilt its cut to match.'
        : phase === 'medium data' ? 'More rows. The net bends its boundary into stripes. The trees draw a blocky approximation that is right in some places and wrong at the corners.'
        : 'Plenty of rows. The net keeps climbing towards the best possible accuracy while the trees stay well below it.'))
    : init.mode === 'inspect' ? (phase === 'the tree splits' ? 'The tree asks one question about one column at a time. Every prediction is a path you can read from top to bottom.'
        : phase === 'the net trains' ? 'The net trains by nudging all 64 first-layer weights at once. Each hidden unit mixes all four columns, so no single weight says what it detects.'
        : 'Both models make predictions. One can be read as rules; the other is a table of numbers that only makes sense when run.')
    : phase === 'verdict' ? 'Verdict. The skipped questions did not change the answer for these choices.'
    : phase === 'data' ? 'First question: what kind of data do you have?'
    : phase === 'rows' ? 'Second: how many labelled rows?'
    : phase === 'explain' ? 'Third: do you need to explain every prediction?'
    : 'Fourth: what is your latency budget?',
  );

  const setDs = (v: DatasetId) => { ds = v; t = 0; playing = !reducedMotion(); snapUi(); wake(); void ensure(v).then(() => { snapUi(); wake(); }); };
  const pick = (k: keyof Choice, v: string) => { (choice as Record<string, string>)[k] = v; t = 0; playing = !reducedMotion(); wake(); };
  async function ensure(v: DatasetId) { if (!curves[v]) { const r = await load(v); if (r.kind !== 'inspect') curves[v] = r.curves; } ready = true; }

  const pct = (v: number) => (100 * v).toFixed(1) + '%';
  const big = (v: number) => (v >= 1e6 ? (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + 'M' : v >= 1e3 ? Math.round(v / 1e3) + 'K' : String(Math.round(v)));
  function snapUi() {
    if (init.mode === 'race') {
      const C = curves[ds];
      if (!C) { ui = { a: '', b: '', c: '' }; return; }
      const f = raceAt(t), p = C.points[f.i], q = C.points[f.i + 1];
      ui = {
        a: `n = ${f.n} rows · trees ${pct(interp(p.accTree, q.accTree, f.f))} · net ${pct(interp(p.accNet, q.accNet, f.f))} · best possible ${pct(C.ceiling)}`,
        b: `trees: ${Math.round(interp(C.paramsTree[f.i], C.paramsTree[f.i + 1], f.f))} numbers, ${big(interp(p.opsTree, q.opsTree, f.f))} operations`,
        c: `net: ${C.paramsNet} weights, ${big(interp(p.opsNet, q.opsNet, f.f))} operations`,
      };
    } else if (init.mode === 'inspect' && ins) {
      const f = inspectAt(t, ins.snaps.length), a = ins.snaps[f.snap], b = ins.snaps[f.snap + 1];
      const ep = Math.round(interp(a.epoch, b.epoch, f.f));
      ui = {
        a: `tree: ${rules(ins.tree, []).length} rules, accuracy ${pct(ins.treeAcc)}, ${big(ins.opsTree)} operations`,
        b: `net: ${ins.paramsNet} weights, accuracy ${pct(ins.netAcc)}, ${big(ins.opsNet)} operations`,
        c: `net shown at epoch ${ep} of 250, log-loss ${f2(interp(a.loss, b.loss, f.f), 3)}`,
      };
    }
  }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const samples = new Map<string, Data>();
    const sample = (id: DatasetId, n: number) => { const k = id + n; let d = samples.get(k); if (!d) { d = makeData(id, n, createRng(100 * n)); samples.set(k, d); } return d; };

    const rgba = (c: { rgb: number[] }, a: number) => `rgba(${c.rgb[0]},${c.rgb[1]},${c.rgb[2]},${a})`;
    const wrap = (ctx: CanvasRenderingContext2D, s: string, w: number) => {
      const out: string[] = []; let line = '';
      for (const word of s.split(' ')) { const nx = line ? line + ' ' + word : word; if (line && ctx.measureText(nx).width > w) { out.push(line); line = word; } else line = nx; }
      if (line) out.push(line); return out;
    };

    function placeholder() {
      const { ctx, w, h, c } = stage; ctx.font = stage.body(13); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('Training both models…', w / 2, h / 2);
    }

    // ---------------- race ----------------
    function race() {
      const { ctx, w: W, h: H, c } = stage, C = curves[ds]!;
      const f = raceAt(t), narrow = W < 540, pts = C.points;
      const s = narrow ? Math.min((W - 12) / 2, H * 0.36) : Math.min(0.4 * W - 14, (H - 58) / 2);
      const cb = narrow
        ? { l: 40, r: W - 12, t: 14, b: Math.round(H * 0.5) - 34 }
        : { l: 0.4 * W + 40, r: W - 14, t: 14, b: H - 34 };
      const work = init.xAxis === 'work';
      const lo = work ? 4 : Math.log10(SIZES[0]), hi = work ? 8 : Math.log10(SIZES[SIZES.length - 1]);
      const X = scale(lo, hi, cb.l, cb.r), Y = scale(0.5, 1, cb.b, cb.t);
      const xt = (p: (typeof pts)[number], m: 'tree' | 'net') => X(work ? Math.log10(m === 'tree' ? p.opsTree : p.opsNet) : Math.log10(p.n));
      // axes
      ctx.lineWidth = 1; ctx.font = stage.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [0.5, 0.6, 0.7, 0.8, 0.9, 1]) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(cb.l, Y(v)); ctx.lineTo(cb.r, Y(v)); ctx.stroke(); ctx.fillStyle = c.muted.hex; ctx.fillText(String(Math.round(v * 100)), cb.l - 6, Y(v)); }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const ticks = work ? [4, 5, 6, 7, 8] : SIZES.map((n) => Math.log10(n));
      ticks.forEach((v, i) => {
        if (narrow && !work && i % 2) return;
        ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(X(v), cb.t); ctx.lineTo(X(v), cb.b); ctx.stroke();
        ctx.fillStyle = c.muted.hex; ctx.fillText(work ? big(10 ** v) : String(SIZES[i]), X(v), cb.b + 5);
      });
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(work ? 'training work (operations, log scale)' : 'training rows (log scale)', cb.r, cb.b + 32);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('test accuracy, %', cb.l - 34, cb.t - 14);
      // best possible
      ctx.setLineDash([5, 5]); ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(cb.l, Y(C.ceiling)); ctx.lineTo(cb.r, Y(C.ceiling)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c.accent.hex; ctx.font = stage.body(11); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText('best possible (label noise)', cb.l + 4, Y(C.ceiling) - 2);
      // curves up to the tip
      const tip = (m: 'tree' | 'net') => {
        const a = pts[f.i], b = pts[f.i + 1], acc = (p: typeof a) => (m === 'tree' ? p.accTree : p.accNet);
        const lx = (p: typeof a) => (work ? Math.log10(m === 'tree' ? p.opsTree : p.opsNet) : Math.log10(p.n));
        return { x: X(interp(lx(a), lx(b), f.f)), y: Y(interp(acc(a), acc(b), f.f)) };
      };
      const line = (m: 'tree' | 'net', col: string) => {
        const tp = tip(m); ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.beginPath();
        for (let i = 0; i <= f.i; i++) { const x = xt(pts[i], m), y = Y(m === 'tree' ? pts[i].accTree : pts[i].accNet); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
        ctx.lineTo(tp.x, tp.y); ctx.stroke();
        for (let i = 0; i <= f.i; i++) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(xt(pts[i], m), Y(m === 'tree' ? pts[i].accTree : pts[i].accNet), 3, 0, 6.2832); ctx.fill(); }
        ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(tp.x, tp.y, 6, 0, 6.2832); ctx.fill(); ctx.stroke();
        return tp;
      };
      const tt = line('tree', c.fwd.hex), tn = line('net', c.s4.hex);
      ctx.font = stage.body(12); ctx.textBaseline = 'middle';
      const lab = (txt: string, p: { x: number; y: number }, up: boolean, col: string) => {
        const w = ctx.measureText(txt).width, x = Math.min(p.x + 10, cb.r - w), y = p.y + (up ? -14 : 14); ctx.fillStyle = col; ctx.textAlign = 'left'; ctx.fillText(txt, Math.max(cb.l + 2, x), clamp(y, cb.t + 6, cb.b - 6));
      };
      const treeUp = tt.y <= tn.y;
      lab('trees', tt, treeUp, c.fwd.hex); lab('net', tn, !treeUp, c.s4.hex);

      // decision regions + points
      const knot = Math.round(f.k), n = SIZES[knot], d = sample(ds, n), F = DATASETS[ds].F;
      const panels = narrow ? [{ x: 4, y: Math.round(H * 0.5) + 24 }, { x: W / 2 + 2, y: Math.round(H * 0.5) + 24 }] : [{ x: 4, y: 18 }, { x: 4, y: 18 + s + 24 }];
      [0, 1].forEach((k) => {
        const { x: px, y: py } = panels[k], p0 = pts[f.i], p1 = pts[f.i + 1], g0 = k ? p0.gridNet : p0.gridTree, g1 = k ? p1.gridNet : p1.gridTree;
        const side = narrow ? s - 6 : s;
        const m = 6, inner = side - 2 * m, cell = inner / GRID;
        for (let j = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) {
          const pr = interp(g0[j * GRID + i], g1[j * GRID + i], f.f), conf = Math.abs(2 * pr - 1);
          ctx.fillStyle = rgba(pr >= 0.5 ? c.pos : c.neg, 0.1 + 0.3 * conf);
          const x0 = Math.round(px + m + i * cell), y0 = Math.round(py + m + inner - (j + 1) * cell);
          ctx.fillRect(x0, y0, Math.round(px + m + (i + 1) * cell) - x0, Math.round(py + m + inner - j * cell) - y0);
        }
        const r = n > 256 ? 1.6 : n > 64 ? 2.4 : 3.2;
        for (let i = 0; i < d.n; i++) {
          ctx.fillStyle = (d.y[i] ? c.pos : c.neg).hex;
          ctx.beginPath(); ctx.arc(px + m + d.X[i * F] * inner, py + m + inner - d.X[i * F + 1] * inner, r, 0, 6.2832); ctx.fill();
        }
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(px, py, side, side);
        ctx.font = stage.body(12); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText((k ? 'Neural net' : 'Boosted trees') + ` · n = ${n}`, px, py - 4);
      });
    }

    // ---------------- inspect ----------------
    function inspect() {
      const { ctx, w: W, h: H, c } = stage, I = ins!, narrow = W < 540;
      const f = inspectAt(t, I.snaps.length);
      const tr = narrow ? { x: 6, y: 22, w: W - 12, h: Math.round(H * 0.42) } : { x: 6, y: 22, w: Math.round(W * 0.56), h: H - 30 };
      const hm = narrow ? { x: 28, y: Math.round(H * 0.42) + 66, w: W - 36, h: Math.round(H * 0.58) - 82 } : { x: Math.round(W * 0.56) + 40, y: 52, w: W - Math.round(W * 0.56) - 50, h: H - 70 };
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Decision tree, depth 3: read it', tr.x, 2);
      ctx.fillText('Neural net: first-layer weights', narrow ? 6 : hm.x - 34, narrow ? hm.y - 34 : 2);
      // tree
      const names = DATASETS.tabular.names, rowH = Math.min(124, (tr.h - (narrow ? 50 : 28)) / (narrow ? 3 : 3.3));
      const node = (t0: Tree, lvl: number, idx: number) => {
        const a = clamp(f.depth + 1 - lvl, 0, 1); if (a <= 0) return;
        const span = tr.w / 2 ** lvl, cx = tr.x + (idx + 0.5) * span, cy = tr.y + 14 + lvl * rowH;
        const leaf = t0.f < 0;
        ctx.globalAlpha = a;
        if (!leaf) {
          for (const [ch, dx] of [[t0.l!, 0], [t0.r!, 1]] as const) {
            const cs = tr.w / 2 ** (lvl + 1), ccx = tr.x + (idx * 2 + dx + 0.5) * cs, ca = clamp(f.depth + 1 - (lvl + 1), 0, 1);
            ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1.5; ctx.globalAlpha = a * ca; ctx.beginPath(); ctx.moveTo(cx, cy + 12); ctx.lineTo(ccx, cy + rowH - 13); ctx.stroke();
            ctx.globalAlpha = a; node(ch, lvl + 1, idx * 2 + dx);
          }
          const label = `${names[t0.f]} ≤ ${t0.th.toFixed(2)}`;
          ctx.font = stage.mono(narrow ? 9.5 : 12); const bw = Math.min(span - 4, ctx.measureText(label).width + 10);
          ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.roundRect(cx - bw / 2, cy - (narrow ? 12 : 15), bw, narrow ? 24 : 30, 4); ctx.fill(); ctx.stroke();
          ctx.fillStyle = c.text.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, cx, cy);
        } else {
          const cls = t0.pos / t0.n >= 0.5 ? 1 : 0, bw = Math.min(span - 4, narrow ? 44 : 56);
          ctx.fillStyle = rgba(cls ? c.pos : c.neg, 0.22); ctx.strokeStyle = (cls ? c.pos : c.neg).hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.roundRect(cx - bw / 2, cy - 16, bw, narrow ? 30 : 36, 4); ctx.fill(); ctx.stroke();
          ctx.fillStyle = c.text.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.font = stage.mono(narrow ? 11 : 14, 700); ctx.fillText(`class ${cls}`.replace('class ', ''), cx, cy - (narrow ? 4 : 5));
          ctx.font = stage.mono(narrow ? 9 : 11); ctx.fillStyle = c.muted.hex; ctx.fillText(`n=${t0.n}`, cx, cy + (narrow ? 8 : 10));
        }
        ctx.globalAlpha = 1;
      };
      node(I.tree, 0, 0);
      ctx.font = stage.body(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('true goes left, false goes right; leaves: class, rows', tr.x, tr.y + 14 + 3 * rowH + 24);
      // heatmap
      const a = I.snaps[f.snap].W1, b = I.snaps[f.snap + 1].W1, H1 = I.H, F = I.F;
      const cw = hm.w / F, ch = Math.min(hm.h / H1, 22);
      ctx.font = stage.mono(10); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      for (let j = 0; j < F; j++) ctx.fillText(names[j], hm.x + (j + 0.5) * cw, hm.y - 2);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let k = 0; k < H1; k++) {
        if (ch >= 12 || k % 4 === 0) { ctx.fillStyle = c.muted.hex; ctx.fillText(`h${k + 1}`, hm.x - 4, hm.y + (k + 0.5) * ch); }
        for (let j = 0; j < F; j++) {
          const v = interp(a[k * F + j], b[k * F + j], f.f);
          ctx.fillStyle = rgba(v >= 0 ? c.pos : c.neg, Math.min(0.95, Math.abs(v) / 2.6));
          ctx.fillRect(hm.x + j * cw, hm.y + k * ch, cw - 1, ch - 1);
          if (ch >= 14) { ctx.fillStyle = c.text.hex; ctx.textAlign = 'center'; ctx.font = stage.mono(9.5); ctx.fillText(f2(v, 1), hm.x + (j + 0.5) * cw, hm.y + (k + 0.5) * ch); ctx.textAlign = 'right'; ctx.font = stage.mono(10); }
        }
      }
      ctx.strokeStyle = c.rule.hex; ctx.strokeRect(hm.x, hm.y, hm.w, ch * H1);
    }

    // ---------------- checklist ----------------
    function checklist() {
      const { ctx, w: W, h: H, c } = stage, narrow = W < 540, f = checklistAt(t), Q = questions(choice), path = verdict.path;
      const bw = narrow ? W - 64 : 150, bh = narrow ? 64 : 92, gap = narrow ? 22 : (W - 4 * bw - 24) / 3;
      const R = QUESTION_IDS.map((_, j) => narrow ? { x: 14, y: 8 + j * (bh + gap), w: bw, h: bh } : { x: 12 + j * (bw + gap), y: 14, w: bw, h: bh });
      const V = narrow ? { x: 14, y: 8 + 4 * (bh + gap) + 4, w: W - 28, h: H - (8 + 4 * (bh + gap)) - 14 } : { x: 12, y: 14 + bh + 62, w: W - 24, h: H - (14 + bh + 62) - 12 };
      const ctr = (r: { x: number; y: number; w: number; h: number }) => ({ x: r.x + r.w / 2, y: r.y + r.h / 2 });
      // connectors between consecutive questions; the packet fills them as it passes
      for (let j = 0; j < 3; j++) {
        const a = ctr(R[j]), b = ctr(R[j + 1]), p = clamp(f.head - j - 0.5, 0, 1);
        const a1 = narrow ? { x: a.x, y: R[j].y + bh } : { x: R[j].x + bw, y: a.y }, b1 = narrow ? { x: b.x, y: R[j + 1].y } : { x: R[j + 1].x, y: b.y };
        ctx.lineWidth = 2; ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(b1.x, b1.y); ctx.stroke();
        if (p > 0) { ctx.strokeStyle = c.fwd.hex; ctx.beginPath(); ctx.moveTo(a1.x, a1.y); ctx.lineTo(a1.x + (b1.x - a1.x) * p, a1.y + (b1.y - a1.y) * p); ctx.stroke(); }
      }
      // connector from the last question on the path down to the verdict
      const last = QUESTION_IDS.indexOf(path[path.length - 1] as (typeof QUESTION_IDS)[number]);
      const lr = R[last], vx = narrow ? W - 14 : lr.x + bw / 2;
      const route: [number, number][] = narrow ? [[lr.x + bw, lr.y + bh / 2], [W - 10, lr.y + bh / 2], [W - 10, V.y + V.h / 2], [V.x + V.w, V.y + V.h / 2]] : [[vx, lr.y + bh], [vx, V.y]];
      ctx.lineWidth = 2; ctx.strokeStyle = c.rule.hex; ctx.beginPath(); route.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
      if (f.verdict > 0) {
        let len = 0; const seg = route.slice(1).map((p, i) => { const l = Math.hypot(p[0] - route[i][0], p[1] - route[i][1]); len += l; return l; });
        let rem = len * f.verdict; ctx.strokeStyle = c.accent.hex; ctx.beginPath(); ctx.moveTo(route[0][0], route[0][1]);
        for (let i = 0; i < seg.length && rem > 0; i++) { const u = Math.min(1, rem / seg[i]); ctx.lineTo(route[i][0] + (route[i + 1][0] - route[i][0]) * u, route[i][1] + (route[i + 1][1] - route[i][1]) * u); rem -= seg[i]; }
        ctx.stroke();
      }
      // question boxes
      R.forEach((r, j) => {
        const seen = t >= 2 * j, onPath = path.includes(QUESTION_IDS[j]), active = f.head >= j && f.head < j + 1 && t < 8;
        ctx.fillStyle = c.raised.hex; ctx.strokeStyle = seen && onPath ? c.fwd.hex : c.rule.hex; ctx.lineWidth = active ? 3 : 1.5;
        ctx.globalAlpha = seen && !onPath ? 0.55 : 1; ctx.beginPath(); ctx.roundRect(r.x, r.y, r.w, r.h, 6); ctx.fill(); ctx.stroke();
        ctx.fillStyle = c.muted.hex; ctx.font = stage.body(12); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        const lines = wrap(ctx, Q[j].q, r.w - 16); lines.forEach((l, i) => ctx.fillText(l, r.x + 8, r.y + 8 + i * 15));
        if (seen) {
          ctx.font = stage.body(13); ctx.fillStyle = onPath ? c.text.hex : c.muted.hex; ctx.textBaseline = 'bottom';
          const ans = onPath ? Q[j].a : 'not needed'; wrap(ctx, ans, r.w - 16).slice(0, 2).reverse().forEach((l, i) => ctx.fillText(l, r.x + 8, r.y + r.h - 8 - i * 16));
        }
        ctx.globalAlpha = 1;
      });
      // the packet
      if (t < 8.4) {
        const pos = clamp(f.head - 0.5, 0, 3), j = Math.min(2, Math.floor(pos)), u = pos - j, a = ctr(R[j]), b = ctr(R[j + 1]);
        ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * u + (narrow ? 0 : 0), a.y + (b.y - a.y) * u, 6, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      // verdict
      ctx.globalAlpha = 0.25 + 0.75 * f.verdict;
      ctx.fillStyle = f.verdict > 0.5 ? c.soft.hex : c.raised.hex; ctx.strokeStyle = f.verdict > 0.5 ? c.accent.hex : c.rule.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.roundRect(V.x, V.y, V.w, V.h, 8); ctx.fill(); ctx.stroke();
      ctx.fillStyle = c.muted.hex; ctx.font = stage.body(12); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(f.verdict > 0 ? 'Reasonable first choice' : 'Verdict', V.x + V.w / 2, V.y + 10);
      ctx.fillStyle = c.text.hex; ctx.font = `600 ${narrow ? 17 : 20}px ${stage.body(1).split('px ')[1]}`; ctx.textBaseline = 'middle';
      wrap(ctx, f.verdict > 0 ? verdict.pick : '…', V.w - 24).forEach((l, i, all) => ctx.fillText(l, V.x + V.w / 2, V.y + V.h / 2 + 8 + (i - (all.length - 1) / 2) * 24));
      ctx.globalAlpha = 1;
    }

    function render() {
      const { ctx, w, h } = stage; ctx.clearRect(0, 0, w, h);
      if (init.mode === 'checklist') return checklist();
      if (init.mode === 'inspect' ? !ins : !curves[ds]) return placeholder();
      if (init.mode === 'inspect') inspect(); else race();
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; snapUi(); uiT = t; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    if (init.mode === 'checklist') ready = true;
    else if (init.mode === 'inspect') void load('inspect').then((r) => { if (r.kind === 'inspect') { ins = r.inspect; ready = true; snapUi(); wake(); } });
    else void ensure(ds).then(() => { snapUi(); wake(); });
    return () => { loop.destroy(); stage.destroy(); };
  });
  $effect(() => { playing; speed; wake(); });
</script>

<FigureCard title={init.title} role="group" aspect={init.mode === 'checklist' ? '720 / 290' : '720 / 400'} narrowAspect={init.mode === 'checklist' ? '360 / 600' : '360 / 500'}
  label={init.mode === 'race' ? 'Left: two small scatter plots of the training set with the decision regions learned by boosted trees and by a small neural net. Right: test accuracy against training-set size for both models, growing as the timeline advances.'
    : init.mode === 'inspect' ? 'Left: a depth-3 decision tree drawn as readable questions. Right: a grid of the neural net first-layer weights as they change during training.'
    : 'Four questions about your problem light up one after another, ending in a recommended first model.'}
  legend={init.mode === 'race' ? [{ label: 'boosted trees', color: '--viz-forward', shape: 'line' }, { label: 'neural net', color: '--viz-seq-4', shape: 'line' }, { label: 'class 1', color: '--viz-pos' }, { label: 'class 0', color: '--viz-neg' }, { label: 'best possible', color: '--accent', shape: 'line' }]
    : init.mode === 'inspect' ? [{ label: 'class 1 / positive weight', color: '--viz-pos' }, { label: 'class 0 / negative weight', color: '--viz-neg' }]
    : [{ label: 'path taken', color: '--viz-forward', shape: 'line' }, { label: 'verdict', color: '--accent', shape: 'line' }]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: models and data"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={() => { t = 0; playing = !reducedMotion(); wake(); }} />
  {/snippet}
  {#snippet readout()}
    {#if init.mode === 'checklist'}
      {#if phase === 'verdict'}<p class="vals"><b>{verdict.pick}.</b> <span class="base">{verdict.why}</span></p>
      {:else}<p class="vals"><span class="base">Change any answer below and the path restarts.</span></p>{/if}
    {:else}
      <p class="vals num">{ui.a}</p>
      <p class="base num">{ui.b}{ui.b ? ' · ' : ''}{ui.c}</p>
    {/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if init.mode === 'race' && init.switcher}
        <fieldset class="seg">
          <legend class="sr-only">Dataset</legend>
          {#each ['tabular', 'pixels'] as const as v (v)}
            <label><input type="radio" name="ds-{init.title}" checked={ds === v} onchange={() => setDs(v)} /><span>{DS_LABEL[v]}</span></label>
          {/each}
        </fieldset>
        <span class="hint">{ds === 'tabular' ? 'Axis-aligned rules, two useless columns.' : 'A pattern tilted 45 degrees to the axes.'}</span>
      {:else if init.mode === 'checklist'}
        {#each [
          ['data', 'Data', [['table', 'Table'], ['media', 'Images, text, audio']]],
          ['rows', 'Rows', [['few', 'Thousands'], ['many', 'Hundreds of thousands']]],
          ['explain', 'Explain', [['yes', 'Must explain'], ['no', 'No need']]],
          ['latency', 'Latency', [['tight', 'Tight'], ['relaxed', 'Relaxed']]],
        ] as const as [key, name, opts] (key)}
          <fieldset class="seg">
            <legend class="sr-only">{name}</legend>
            {#each opts as [v, lab] (v)}
              <label><input type="radio" name="{key}-{init.title}" checked={choice[key] === v} onchange={() => pick(key, v)} /><span>{lab}</span></label>
            {/each}
          </fieldset>
        {/each}
      {:else}
        <span class="hint">Drag the timeline: {init.mode === 'inspect' ? 'splits appear level by level, then the weights change as the net trains.' : 'each point is a finished training run.'}</span>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.92rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .base { font: 0.8rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--text-muted); }
  .vals .base { font-family: var(--font-ui); font-size: 0.88rem; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; font-size: 0.85rem; }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
