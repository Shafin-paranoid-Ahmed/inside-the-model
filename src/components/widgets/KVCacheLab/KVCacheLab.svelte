<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, onThemeChange, prefersReducedMotion, readColors, readVar, reducedMotion, type Color } from '../_shared';
  import { loadModel } from '../../../lib/transformer/load';
  import type { Weights } from '../../../lib/transformer/model';
  import {
    LOOP_STEPS, MAX_CTX_LOG2, sharedRun, batchFrameAt, batchPoint, batchScene, chr, crossing, ease, fmtBytes, fmtCount, fmtDiff, fmtGB, fmtInt,
    memoryFrameAt, memoryScene, phasesFrameAt, phasesScene, stepFrameAt, stepScene,
    type BatchFrame, type BatchScene, type MemFrame, type MemScene, type Mode, type ModelKey, type PhasesFrame, type PhasesScene, type Run, type StepFrame, type StepScene,
  } from './sim';

  interface Props { mode?: Mode; title?: string; total?: number; compare?: boolean; batch?: number; prompt?: number; model?: ModelKey; ctx?: number }
  let { mode = 'cache', title = 'Generating with and without a cache', total: totalProp = 64, compare: compareProp = true, batch: batchProp = 8, prompt: promptProp = 32,
    model: modelProp = 'llama2_7b', ctx: ctxProp = 4096 }: Props = $props();
  const init = untrack(() => mode);

  type Scene = StepScene | MemScene | PhasesScene | BatchScene;
  type Frame = StepFrame | MemFrame | PhasesFrame | BatchFrame;
  const frameOf = (t: number, sc: Scene): Frame =>
    sc.kind === 'steps' ? stepFrameAt(t, sc) : sc.kind === 'memory' ? memoryFrameAt(t, sc) : sc.kind === 'phases' ? phasesFrameAt(t, sc) : batchFrameAt(t, sc);

  let weights = $state.raw<Weights | null>(null);
  let run = $state.raw<Run | null>(null);
  let failed = $state(false);
  let total = $state(untrack(() => totalProp)), compare = $state(untrack(() => compareProp)), batch = $state(untrack(() => batchProp));
  let promptLen = $state(untrack(() => promptProp)), modelKey = $state<ModelKey>(untrack(() => modelProp)), ctx = $state(untrack(() => ctxProp));
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);

  const scene: Scene | null = $derived.by(() => {
    if (!weights) return null;
    if (init === 'loop' || init === 'nocache' || init === 'cache') {
      if (!run) return null;
      return stepScene(init, run, init === 'loop' ? run.P + LOOP_STEPS : total, compare);
    }
    if (init === 'memory') return memoryScene(weights, batch);
    if (init === 'phases') return phasesScene(weights, promptLen);
    return batchScene(modelKey, ctx);
  });
  const duration = $derived(scene?.duration ?? 10);
  const frame = $derived(scene ? frameOf(uiT, scene) : null);
  const markers = $derived(scene?.markers ?? []);

  const ASPECT = {
    loop: ['720 / 250', '360 / 270'], nocache: ['720 / 420', '360 / 560'], cache: ['720 / 440', '360 / 640'],
    memory: ['720 / 400', '360 / 420'], phases: ['720 / 360', '360 / 420'], batch: ['720 / 440', '360 / 520'],
  }[init];
  const LABEL = {
    loop: 'A line of text growing one character at a time, with the model re-reading all of it each step',
    nocache: 'A triangle of cells: every generation step recomputes all earlier positions',
    cache: 'Stored keys and values per layer growing by one row per step, and work per step with and without the cache',
    memory: 'Cache size against context length for three models, with a line at GPU memory size',
    phases: 'Cache rows filled in one prefill pass then one decode step at a time, with operations and weight bytes per step',
    batch: 'A roofline chart of decode speed against batch size and a bar showing weights and cache filling GPU memory',
  }[init];
  const legend = {
    loop: [{ label: 'Prompt', color: '--viz-cat-1' }, { label: 'Generated', color: '--viz-cat-2' }, { label: 'Layer reads the text', color: '--viz-seq-3' }],
    nocache: [{ label: 'New work', color: '--viz-forward' }, { label: 'Repeats earlier work', color: '--text-muted' }],
    cache: [{ label: 'Positive value', color: '--viz-pos' }, { label: 'Negative value', color: '--viz-neg' }, { label: 'Attention of newest character', color: '--viz-seq-4' },
      { label: 'With cache', color: '--viz-forward', shape: 'line' as const }, ...(init === 'cache' ? [{ label: 'Without cache', color: '--text-muted', shape: 'line' as const }] : [])],
    memory: [{ label: 'House model', color: '--viz-cat-1', shape: 'line' as const }, { label: 'Llama 2 7B', color: '--viz-cat-2', shape: 'line' as const },
      { label: 'Llama 3 8B', color: '--viz-cat-3', shape: 'line' as const }, { label: 'GPU memory', color: '--text', shape: 'line' as const }],
    phases: [{ label: 'Prompt rows', color: '--viz-cat-1' }, { label: 'Generated rows', color: '--viz-cat-2' }, { label: 'Operations', color: '--viz-forward' }, { label: 'Weight bytes read', color: '--viz-cat-4' }],
    batch: [{ label: 'Speed limit', color: '--text', shape: 'line' as const }, { label: 'Decode at this batch', color: '--accent' }, { label: 'Weights', color: '--viz-forward' }, { label: 'Cache', color: '--viz-seq-4' }, { label: 'Cache does not fit', color: '--viz-neg' }],
  }[init];

  // ---------- drawing ----------
  const NAMES = {
    text: '--text', muted: '--text-muted', rule: '--rule', bg: '--bg-raised', fwd: '--viz-forward', pos: '--viz-pos', neg: '--viz-neg',
    s2: '--viz-seq-2', s3: '--viz-seq-3', s4: '--viz-seq-4', acc: '--accent', c1: '--viz-cat-1', c2: '--viz-cat-2', c3: '--viz-cat-3', c4: '--viz-cat-4',
  } as const;
  interface G { ctx: CanvasRenderingContext2D; w: number; h: number; c: Record<keyof typeof NAMES, Color>; ui: string; code: string; narrow: boolean; hatch: CanvasPattern | null; imgs: HTMLCanvasElement[] }
  const rgba = (c: Color, a: number) => `rgba(${c.rgb[0]},${c.rgb[1]},${c.rgb[2]},${a})`;
  const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
  const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
  function text(g: G, s: string, x: number, y: number, color: Color, size = 11, align: CanvasTextAlign = 'left', mono = false, alpha = 1) {
    g.ctx.font = `${size}px ${mono ? g.code : g.ui}`; g.ctx.fillStyle = alpha < 1 ? rgba(color, alpha) : color.hex; g.ctx.textAlign = align; g.ctx.textBaseline = 'alphabetic';
    g.ctx.fillText(s, x, y);
  }
  const line = (g: G, x1: number, y1: number, x2: number, y2: number, color: Color | string, width = 1, dash: number[] = []) => {
    g.ctx.beginPath(); g.ctx.moveTo(x1, y1); g.ctx.lineTo(x2, y2); g.ctx.strokeStyle = typeof color === 'string' ? color : color.hex; g.ctx.lineWidth = width; g.ctx.setLineDash(dash); g.ctx.stroke(); g.ctx.setLineDash([]);
  };
  const box = (g: G, x: number, y: number, w: number, h: number, fill: string | null, stroke: Color | null, dash: number[] = []) => {
    if (fill) { g.ctx.fillStyle = fill; g.ctx.fillRect(x, y, w, h); }
    if (stroke) { g.ctx.setLineDash(dash); g.ctx.strokeStyle = stroke.hex; g.ctx.lineWidth = 1; g.ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); g.ctx.setLineDash([]); }
  };

  function drawLoop(g: G, sc: StepScene, fr: StepFrame) {
    const m = 14, N = sc.N, cw = (g.w - 2 * m) / N, ch = Math.min(26, cw * 1.5), oy = clamp((g.h - 232) / 2, 0, 40), y1 = 28 + oy, cx = (k: number) => m + k * cw, fs = Math.min(13, cw * 0.78);
    text(g, 'Text so far', m, 18 + oy, g.c.muted);
    for (let k = 0; k < N; k++) {
      if (k < fr.L) {
        box(g, cx(k), y1, cw - 1, ch, rgba(k < sc.P ? g.c.c1 : g.c.c2, 0.3), g.c.rule);
        text(g, chr(sc.run, k), cx(k) + (cw - 1) / 2, y1 + ch / 2 + fs * 0.35, g.c.text, fs, 'center', true);
      } else box(g, cx(k), y1, cw - 1, ch, null, g.c.rule, [2, 2]);
    }
    const rows = [{ name: 'Layer 1', lit: fr.lit1, col: g.c.s3 }, { name: 'Layer 2', lit: fr.lit2, col: g.c.s4 }];
    let y = y1 + ch + 26;
    for (const r of rows) {
      text(g, r.name, m, y - 6, g.c.muted);
      for (let k = 0; k < fr.L; k++) box(g, cx(k), y, cw - 1, 12, rgba(r.col, clamp(fr.L * r.lit - k)), g.c.rule);
      y += 38;
    }
    const bw = 38, bh = 32, bx = g.w / 2 - bw / 2, by = y - 2;
    text(g, 'Predicted next character', g.w / 2, by - 6, g.c.muted, 11, 'center');
    box(g, bx, by, bw, bh, null, g.c.rule);
    const ch1 = chr(sc.run, fr.L);
    if (fr.showPred > 0 && fr.fly === 0) text(g, ch1, g.w / 2, by + 22, g.c.text, 18, 'center', true, fr.showPred);
    if (fr.fly > 0) {
      const u = ease(fr.fly), tx = cx(fr.L) + (cw - 1) / 2, ty = y1 + ch / 2, px = lerp(g.w / 2, tx, u), py = lerp(by + bh / 2, ty, u), sw = lerp(bw, cw - 1, u), sh = lerp(bh, ch, u);
      box(g, px - sw / 2, py - sh / 2, sw, sh, rgba(g.c.c2, 0.3 * u), g.c.acc);
      text(g, ch1, px, py + lerp(7, fs * 0.35, u), g.c.text, lerp(18, fs, u), 'center', true);
    }
  }

  // Chart helpers shared by the step modes. `box` = plot rectangle.
  interface Box { x: number; y: number; w: number; h: number }
  const logY = (v: number, lo: number, hi: number, b: Box) => b.y + (1 - (Math.log10(v) - lo) / (hi - lo)) * b.h;

  function drawNoCache(g: G, sc: StepScene, fr: StepFrame) {
    const wide = !g.narrow, gx = wide ? 14 : 14, gy = 40, cols = sc.N - 1, rows = sc.G;
    const gwMax = wide ? g.w * 0.5 - gx : g.w - 2 * gx, ghMax = wide ? g.h - gy - 18 : g.h * 0.54 - gy;
    const cs = Math.min(gwMax / cols, ghMax / rows);
    text(g, 'Rows: generation steps. Columns: positions in the text.', gx, 18, g.c.muted);
    text(g, 'Step 1 at the top, one per row', gx, 32, g.c.muted, 11);
    for (let r = 0; r <= fr.s; r++) {
      const n = r === fr.s ? Math.floor((r === 0 ? sc.P : sc.P + r) * fr.f) : sc.P + r;
      for (let c = 0; c < n; c++) {
        const fresh = r === 0 || c === sc.P + r - 1, x = gx + c * cs, y = gy + r * cs;
        if (fresh) { g.ctx.fillStyle = g.c.fwd.hex; g.ctx.fillRect(x, y, cs, cs); }
        else { g.ctx.fillStyle = rgba(g.c.muted, 0.16); g.ctx.fillRect(x, y, cs, cs); if (g.hatch) { g.ctx.fillStyle = g.hatch; g.ctx.fillRect(x, y, cs, cs); } }
      }
    }
    // Triangle outline of the full run for orientation.
    box(g, gx, gy, cols * cs, rows * cs, null, g.c.rule);
    // Per-step work chart.
    const cb: Box = wide ? { x: g.w * 0.6, y: 56, w: g.w * 0.4 - 22, h: g.h - 56 - 40 } : { x: 44, y: gy + rows * cs + 54, w: g.w - 44 - 18, h: g.h - (gy + rows * cs + 54) - 30 };
    const maxV = sc.stepNo[sc.G - 1];
    text(g, 'Multiply-adds in each step', cb.x - (wide ? 28 : 30), cb.y - 14, g.c.muted);
    line(g, cb.x, cb.y + cb.h, cb.x + cb.w, cb.y + cb.h, g.c.rule);
    line(g, cb.x, cb.y, cb.x, cb.y + cb.h, g.c.rule);
    text(g, fmtCount(maxV), cb.x - 5, cb.y + 9, g.c.muted, 11, 'right'); text(g, '0', cb.x - 5, cb.y + cb.h, g.c.muted, 11, 'right');
    const slot = cb.w / sc.G;
    for (let r = 0; r <= fr.s; r++) {
      const hgt = sc.stepNo[r] / maxV * cb.h * (r === fr.s ? fr.f : 1);
      g.ctx.fillStyle = g.c.s4.hex; g.ctx.fillRect(cb.x + r * slot + slot * 0.1, cb.y + cb.h - hgt, Math.max(1, slot * 0.8), hgt);
    }
    text(g, '1', cb.x, cb.y + cb.h + 14, g.c.muted, 11, 'center'); text(g, String(sc.G), cb.x + cb.w, cb.y + cb.h + 14, g.c.muted, 11, 'center');
    text(g, 'Generation step', cb.x + cb.w / 2, cb.y + cb.h + 14, g.c.muted, 11, 'center');
  }

  function drawCache(g: G, sc: StepScene, fr: StepFrame) {
    const wide = !g.narrow, ax = 14, ay = 46, areaW = wide ? g.w * 0.5 - ax : g.w - 2 * ax, areaH = wide ? g.h - ay - 14 : g.h * 0.56 - ay;
    const sw = 7, gap = 3, groupGap = 16, mw = (areaW - 2 * (sw + 2 * gap) - groupGap) / 4, rows = sc.N - 1, rh = areaH / rows, run = sc.run;
    const attnMax = (l: number) => { let m = 0; for (let k = 0; k < fr.L; k++) m = Math.max(m, run.attn[l][(fr.L - 1) * rows + k]); return m || 1; };
    const appear = clamp(fr.f / 0.4), sweep = clamp((fr.f - 0.4) / 0.6), newN = fr.L - fr.newFrom;
    for (let l = 0; l < 2; l++) {
      const gx0 = ax + l * (sw + 2 * gap + 2 * mw + groupGap);
      text(g, `Layer ${l + 1}`, gx0, ay - 24, g.c.text, 11);
      text(g, 'attn', gx0, ay - 8, g.c.muted, 11); text(g, 'keys', gx0 + sw + gap + mw / 2, ay - 8, g.c.muted, 11, 'center'); text(g, 'values', gx0 + sw + 2 * gap + mw * 1.5, ay - 8, g.c.muted, 11, 'center');
      const am = attnMax(l);
      box(g, gx0, ay, sw, areaH, null, g.c.rule);
      for (let k = 0; k < Math.min(fr.L, Math.ceil(fr.L * sweep)); k++) { g.ctx.fillStyle = rgba(g.c.s4, 0.12 + 0.88 * run.attn[l][(fr.L - 1) * rows + k] / am); g.ctx.fillRect(gx0 + 1, ay + k * rh, sw - 2, Math.max(1, rh - 0.5)); }
      for (let q = 0; q < 2; q++) {
        const x = gx0 + sw + gap + q * (mw + gap), img = g.imgs[l * 2 + q];
        box(g, x, ay, mw, areaH, null, g.c.rule);
        g.ctx.imageSmoothingEnabled = false;
        if (img) {
          if (fr.newFrom > 0) g.ctx.drawImage(img, 0, 0, img.width, fr.newFrom, x, ay, mw, fr.newFrom * rh);
          for (let k = fr.newFrom; k < fr.L; k++) {
            g.ctx.globalAlpha = clamp(appear * newN - (k - fr.newFrom));
            g.ctx.drawImage(img, 0, k, img.width, 1, x, ay + k * rh, mw, rh);
          }
          g.ctx.globalAlpha = 1;
        }
        if (appear >= 1 && fr.s > 0) box(g, x, ay + (fr.L - 1) * rh, mw, Math.max(rh, 1.5), null, g.c.acc);
      }
    }
    // Work per step, log scale.
    const cb: Box = wide ? { x: g.w * 0.6, y: 60, w: g.w * 0.4 - 22, h: g.h - 60 - 40 } : { x: 48, y: ay + areaH + 58, w: g.w - 48 - 18, h: g.h - (ay + areaH + 58) - 30 };
    const lo = 4, hi = 7;
    text(g, 'Multiply-adds in each step (log scale)', cb.x - (wide ? 34 : 34), cb.y - 16, g.c.muted);
    for (const e of [4, 5, 6, 7]) {
      const y = logY(10 ** e, lo, hi, cb);
      line(g, cb.x, y, cb.x + cb.w, y, g.c.rule, 1, [3, 3]); text(g, fmtCount(10 ** e), cb.x - 5, y + 3, g.c.muted, 11, 'right');
    }
    const px = (r: number) => cb.x + (r + 0.5) / sc.G * cb.w;
    const series = (vals: number[], color: Color, dash: number[]) => {
      g.ctx.beginPath();
      for (let r = 0; r <= fr.s; r++) { const f = r === fr.s ? fr.f : 1, v = r === 0 ? vals[0] : lerp(vals[r - 1], vals[r], f), x = r === fr.s ? lerp(r === 0 ? px(0) : px(r - 1), px(r), f) : px(r);
        const y = logY(v, lo, hi, cb); r === 0 ? g.ctx.moveTo(px(0), y) : g.ctx.lineTo(x, y); }
      g.ctx.strokeStyle = color.hex; g.ctx.lineWidth = 2; g.ctx.setLineDash(dash); g.ctx.stroke(); g.ctx.setLineDash([]);
    };
    if (sc.compare) series(sc.stepNo, g.c.muted, [5, 3]);
    series(sc.stepCache, g.c.fwd, []);
    text(g, '1', cb.x, cb.y + cb.h + 14, g.c.muted, 11, 'center'); text(g, String(sc.G), cb.x + cb.w, cb.y + cb.h + 14, g.c.muted, 11, 'center');
    text(g, 'Generation step', cb.x + cb.w / 2, cb.y + cb.h + 14, g.c.muted, 11, 'center');
  }

  const TICKS_X = [0, 4, 8, 12, 16, 17];
  function drawMemory(g: G, sc: MemScene, fr: MemFrame) {
    const L = 66, R = 18, T = 22, B = 40, pw = g.w - L - R, ph = g.h - T - B, y0 = 6, y1 = 44;
    const xs = (lx: number) => L + lx / MAX_CTX_LOG2 * pw, ys = (ly: number) => T + (1 - (ly - y0) / (y1 - y0)) * ph;
    const yUnits: [number, string][] = [[10, '1 KiB'], [20, '1 MiB'], [30, '1 GiB'], [40, '1 TiB']];
    for (const [v, s] of yUnits) { line(g, L, ys(v), L + pw, ys(v), g.c.rule, 1, [3, 3]); text(g, s, L - 6, ys(v) + 4, g.c.muted, 11, 'right'); }
    for (const lx of TICKS_X.filter((v) => !g.narrow || v === 0 || v === 8 || v === 12 || v === 17)) { text(g, fmtInt(2 ** lx), xs(lx), T + ph + 16, g.c.muted, 11, 'center'); line(g, xs(lx), T + ph, xs(lx), T + ph + 4, g.c.muted); }
    text(g, 'Context length (tokens, log scale)', L + pw / 2, T + ph + 33, g.c.muted, 11, 'center');
    text(g, 'Cache size', 10, 14, g.c.muted, 11);
    const memLy = Math.log2(sc.mem);
    line(g, L, ys(memLy), L + pw, ys(memLy), g.c.text, 1.5, [6, 4]);
    text(g, `GPU memory: 80 GB (74.5 GiB)`, L + 6, ys(memLy) - 6, g.c.text, 11);
    const xc = Math.log2(fr.ctx);
    sc.rows.forEach((r, i) => {
      const a = Math.log2(r.perToken * sc.batch), col = g.c[r.color === '--viz-cat-1' ? 'c1' : r.color === '--viz-cat-2' ? 'c2' : 'c3'];
      line(g, xs(0), ys(a), xs(xc), ys(a + xc), col, 2.5);
      g.ctx.beginPath(); g.ctx.arc(xs(xc), ys(a + xc), 4, 0, 7); g.ctx.fillStyle = col.hex; g.ctx.fill();
      const cr = Math.log2(crossing(r, sc));
      if (cr <= MAX_CTX_LOG2) {
        g.ctx.beginPath(); g.ctx.arc(xs(cr), ys(memLy), 5, 0, 7);
        g.ctx.fillStyle = xc >= cr ? col.hex : g.c.bg.hex; g.ctx.fill(); g.ctx.strokeStyle = col.hex; g.ctx.lineWidth = 2; g.ctx.stroke();
        text(g, `fills at ${fmtInt(crossing(r, sc))}`, xs(cr) - 8, ys(memLy) + (i === 1 ? 18 : 30), col, 11, 'right');
      }
    });
    line(g, xs(xc), T, xs(xc), T + ph, g.c.muted, 1, [2, 3]);
  }

  function drawPhases(g: G, sc: PhasesScene, fr: PhasesFrame) {
    const m = 14, N = sc.N, cw = (g.w - 2 * m) / N, y = 34, ch = 22;
    text(g, 'Cache rows (one per character)', m, 18, g.c.muted);
    for (let k = 0; k < N; k++) {
      const a = clamp(fr.cells - k), prompt = k < sc.P;
      if (a > 0) box(g, m + k * cw, y, cw - 1, ch, rgba(prompt ? g.c.c1 : g.c.c2, 0.15 + 0.6 * a), g.c.rule);
      else box(g, m + k * cw, y, cw - 1, ch, null, g.c.rule, [2, 2]);
    }
    text(g, 'Prefill fills these in one pass', m, y + ch + 14, g.c.c1, 11);
    g.narrow ? text(g, 'Decode adds one at a time', g.w - m, y + ch + 30, g.c.c2, 11, 'right') : text(g, 'Decode adds one at a time', m + sc.P * cw, y + ch + 14, g.c.c2, 11);
    const L = 54, cb: Box = { x: L, y: y + ch + 52, w: g.w - L - 14, h: g.h - (y + ch + 52) - 34 }, lo = 4, hi = 7, slot = cb.w / (sc.D + 1), bw = Math.max(1.5, slot * 0.4);
    text(g, 'Per step, log scale', m, cb.y - 10, g.c.muted);
    for (const e of [4, 5, 6, 7]) { const yy = logY(10 ** e, lo, hi, cb); line(g, cb.x, yy, cb.x + cb.w, yy, g.c.rule, 1, [3, 3]); text(g, fmtCount(10 ** e), cb.x - 5, yy + 3, g.c.muted, 11, 'right'); }
    const bottom = cb.y + cb.h;
    g.ctx.fillStyle = rgba(g.c.acc, 0.14); g.ctx.fillRect(cb.x + (fr.bars - 1) * slot, cb.y, slot, cb.h);
    for (let j = 0; j < fr.bars; j++) {
      const k = j === fr.bars - 1 ? fr.g : 1;
      for (const [v, col, dx] of [[sc.ops[j], g.c.fwd, 0], [sc.bytes, g.c.c4, bw]] as const) {
        const top = logY(v, lo, hi, cb), hgt = (bottom - top) * k;
        g.ctx.fillStyle = col.hex; g.ctx.fillRect(cb.x + j * slot + slot * 0.1 + dx, bottom - hgt, bw, hgt);
      }
    }
    text(g, 'prefill', cb.x + slot / 2, bottom + 14, g.c.muted, 11, 'left'); text(g, 'decode steps', cb.x + cb.w, bottom + 14, g.c.muted, 11, 'right');
  }

  function drawBatch(g: G, sc: BatchScene, fr: BatchFrame) {
    const L = 76, R = 18, T = 20, gaugeBlock = 74, B = 36 + gaugeBlock, pw = g.w - L - R, ph = g.h - T - B, ylo = 12, yhi = 15.2, xmax = 10;
    const xs = (lx: number) => L + lx / xmax * pw, ys = (v: number) => T + (1 - (Math.log10(v) - ylo) / (yhi - ylo)) * ph;
    for (const [v, s] of [[1e12, '1 TFLOP/s'], [1e13, '10 TFLOP/s'], [1e14, '100 TFLOP/s'], [1e15, '1 PFLOP/s']] as const) { line(g, L, ys(v), L + pw, ys(v), g.c.rule, 1, [3, 3]); text(g, s, L - 6, ys(v) + 4, g.c.muted, 11, 'right'); }
    for (const lx of [0, 2, 4, 6, 8, 10].filter((v) => !g.narrow || v % 4 === 0 || v === 10)) { text(g, fmtInt(2 ** lx), xs(lx), T + ph + 16, g.c.muted, 11, 'center'); line(g, xs(lx), T + ph, xs(lx), T + ph + 4, g.c.muted); }
    text(g, 'Operations per weight byte (= batch size)', L + pw / 2, T + ph + 32, g.c.muted, 11, 'center');
    text(g, 'Speed limit', 10, 14, g.c.muted, 11);
    const rx = Math.log2(sc.ridge), peak = sc.hw.peakFlops;
    g.ctx.beginPath(); g.ctx.moveTo(xs(0), ys(sc.hw.bandwidth)); g.ctx.lineTo(xs(rx), ys(peak)); g.ctx.lineTo(xs(xmax), ys(peak));
    g.ctx.strokeStyle = g.c.text.hex; g.ctx.lineWidth = 2.5; g.ctx.stroke();
    line(g, xs(rx), ys(peak), xs(rx), T + ph, g.c.muted, 1, [2, 3]);
    text(g, `ridge: ${Math.round(sc.ridge)} operations per byte`, xs(rx) - 6, T + ph - 8, g.c.muted, 11, 'right');
    text(g, 'Limited by memory bandwidth', xs(g.narrow ? 2.5 : 4), ys(sc.hw.bandwidth * 2 ** (g.narrow ? 2.5 : 4)) + 28, g.c.muted, 11);
    text(g, 'Limited by arithmetic', xs(xmax) - 4, ys(peak) - 8, g.c.muted, 11, 'right');
    for (let k = 0; 2 ** k < fr.b; k++) { g.ctx.beginPath(); g.ctx.arc(xs(k), ys(batchPoint(sc, 2 ** k).flops), 3, 0, 7); g.ctx.fillStyle = rgba(g.c.acc, 0.45); g.ctx.fill(); }
    const dot = fr.fits ? g.c.acc : g.c.neg;
    g.ctx.beginPath(); g.ctx.arc(xs(Math.log2(fr.b)), ys(fr.flops), 6, 0, 7); g.ctx.fillStyle = dot.hex; g.ctx.fill();
    if (!fr.fits) { g.ctx.strokeStyle = g.c.bg.hex; g.ctx.lineWidth = 1.5; g.ctx.stroke(); }
    // Memory gauge.
    const gy = T + ph + 36 + 30, gh = 20, scale = sc.hw.memoryBytes * 1.2;
    text(g, 'GPU memory', 10, gy - 10, g.c.muted, 11);
    box(g, L, gy, pw, gh, null, g.c.rule);
    const w1 = Math.min(1, sc.weightB / scale) * pw, w2 = Math.min(pw - w1, fr.kv / scale * pw);
    g.ctx.fillStyle = g.c.fwd.hex; g.ctx.fillRect(L, gy, w1, gh);
    g.ctx.fillStyle = (fr.fits ? g.c.s4 : g.c.neg).hex; g.ctx.fillRect(L + w1, gy, w2, gh);
    line(g, L + sc.hw.memoryBytes / scale * pw, gy - 5, L + sc.hw.memoryBytes / scale * pw, gy + gh + 5, g.c.text, 2);
    text(g, '80 GB', L + sc.hw.memoryBytes / scale * pw, gy + gh + 17, g.c.text, 11, 'center');
    if (!fr.fits) text(g, `needs ${fmtGB(fr.used)}`, L + 4, gy + gh + 17, g.c.neg, 11);
  }

  // Cached heat maps of the real keys and values (one pixel per number), rebuilt per theme.
  function makeImages(r: Run, c: G['c'], d: number): HTMLCanvasElement[] {
    const out: HTMLCanvasElement[] = [], rows = r.N - 1;
    for (let l = 0; l < 2; l++) for (const [arr, max] of [[r.keys[l], r.maxK[l]], [r.values[l], r.maxV[l]]] as const) {
      const cv = document.createElement('canvas'); cv.width = d; cv.height = rows;
      const cx = cv.getContext('2d')!, im = cx.createImageData(d, rows);
      for (let i = 0; i < d * rows; i++) {
        const v = clamp(arr[i] / max, -1, 1), a = Math.abs(v) ** 0.8, col = v >= 0 ? c.pos.rgb : c.neg.rgb;
        for (let k = 0; k < 3; k++) im.data[i * 4 + k] = Math.round(c.bg.rgb[k] * (1 - a) + col[k] * a);
        im.data[i * 4 + 3] = 255;
      }
      cx.putImageData(im, 0, 0); out.push(cv);
    }
    return out;
  }

  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let t = 0, wake = $state<() => void>(() => {}), lastPaint = 0, jumpToEnd = false;
  $effect(() => { playing; speed; scene; wake(); });
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  function restart() {
    if (jumpToEnd) { jumpToEnd = false; playing = false; seek(scene?.duration ?? 0); return; }
    t = 0; uiT = 0; playing = !reducedMotion(); wake();
  }
  $effect(() => { if (scene) untrack(restart); });
  const edit = (set: () => void) => { jumpToEnd = true; set(); };

  onMount(() => {
    (async () => {
      try {
        weights = await loadModel('house');
        if (init === 'loop' || init === 'nocache' || init === 'cache') { run = await sharedRun(weights); }
      } catch { failed = true; }
    })();
    const ctx2d = cv.getContext('2d')!;
    let colors = readColors(NAMES), ui = readVar('--font-ui'), code = readVar('--font-code'), hatch: CanvasPattern | null = null, imgs: HTMLCanvasElement[] = [], imgRun: Run | null = null;
    const theme = () => {
      colors = readColors(NAMES); ui = readVar('--font-ui'); code = readVar('--font-code'); imgRun = null;
      const p = document.createElement('canvas'); p.width = p.height = 6;
      const px = p.getContext('2d')!; px.strokeStyle = colors.muted.hex; px.lineWidth = 1; px.beginPath(); px.moveTo(0, 6); px.lineTo(6, 0); px.moveTo(-1, 1); px.lineTo(1, -1); px.moveTo(5, 7); px.lineTo(7, 5); px.stroke();
      hatch = ctx2d.createPattern(p, 'repeat'); wake();
    };
    theme();
    let cw = 0, chh = 0;
    const draw = () => {
      const r = cv.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
      if (!r.width) return;
      if (Math.round(r.width * dpr) !== cw || Math.round(r.height * dpr) !== chh) { cw = Math.round(r.width * dpr); chh = Math.round(r.height * dpr); cv.width = cw; cv.height = chh; }
      ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0); ctx2d.clearRect(0, 0, r.width, r.height);
      const sc = scene;
      const g: G = { ctx: ctx2d, w: r.width, h: r.height, c: colors, ui, code, narrow: r.width < 540, hatch, imgs };
      if (!sc) { text(g, failed ? 'The model could not be loaded.' : 'Loading the model…', g.w / 2, g.h / 2, colors.muted, 13, 'center'); return; }
      const fr = frameOf(t, sc);
      if (sc.kind === 'steps' && fr.kind === 'steps') {
        if (sc.mode === 'cache' && imgRun !== sc.run && weights) { imgs = makeImages(sc.run, colors, weights.config.d); imgRun = sc.run; g.imgs = imgs; }
        (sc.mode === 'loop' ? drawLoop : sc.mode === 'nocache' ? drawNoCache : drawCache)(g, sc, fr);
      } else if (sc.kind === 'memory' && fr.kind === 'memory') drawMemory(g, sc, fr);
      else if (sc.kind === 'phases' && fr.kind === 'phases') drawPhases(g, sc, fr);
      else if (sc.kind === 'batch' && fr.kind === 'batch') drawBatch(g, sc, fr);
    };
    const loop = createLoop(cv.closest('[data-figure]') ?? cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration }).t;
      draw();
      if (!playing || now - lastPaint >= 32) { uiT = t; tl?.paint(t); lastPaint = now; }
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion() && scene) playing = true; } });
    wake = loop.wake;
    const off = onThemeChange(theme);
    const ro = new ResizeObserver(() => wake());
    ro.observe(cv);
    const motion = prefersReducedMotion();
    const onMotion = () => { if (motion.matches) { playing = false; seek(0); } };
    motion.addEventListener('change', onMotion);
    return () => { loop.destroy(); off(); ro.disconnect(); motion.removeEventListener('change', onMotion); };
  });

  const stepsFrame = $derived(frame?.kind === 'steps' ? frame : null);
  const stepsScene = $derived(scene?.kind === 'steps' ? scene : null);
  const memScene = $derived(scene?.kind === 'memory' ? scene : null);
  const batchSc = $derived(scene?.kind === 'batch' ? scene : null);
  const phasesSc = $derived(scene?.kind === 'phases' ? scene : null);
  const ctxChoices = [512, 1024, 2048, 4096, 8192, 16384, 32768];
</script>

<FigureCard {title} label={LABEL} {legend} aspect={ASPECT[0]} narrowAspect={ASPECT[1]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} {markers} label="Timeline" onSeek={seek} onReset={() => seek(0)} />
  {/snippet}
  {#snippet readout()}
    {#if !frame || !scene}
      <p class="status">{failed ? 'The model could not be loaded.' : 'Loading the model…'}</p>
      <p class="sub">&nbsp;</p>
    {:else}
      <p class="status" role="status" aria-live="polite">{frame.status}</p>
      {#if stepsFrame && stepsScene}
        {#if init === 'loop'}
          <p class="sub">Positions processed so far, over all steps: <b>{fmtInt(stepsFrame.positions)}</b>. All {stepsScene.G} steps together: <b>{fmtInt(stepsScene.cumLen[stepsScene.G - 1])}</b> ({stepsScene.P} + {stepsScene.P + 1} + … + {stepsScene.P + stepsScene.G - 1}).</p>
        {:else if init === 'nocache'}
          <p class="sub">Multiply-adds so far: <b>{fmtInt(stepsFrame.macsNo)}</b>. For all {stepsScene.G} steps: <b>{fmtInt(stepsScene.totalNo)}</b>, of which {(stepsScene.repeated * 100).toFixed(1)}% repeats earlier work.</p>
        {:else}
          <p class="sub">Multiply-adds so far with the cache: <b>{fmtInt(stepsFrame.macsCache)}</b>{#if compare}{' '}(without: {fmtInt(stepsFrame.macsNo)}){/if}. For all {stepsScene.G} steps: <b>{fmtInt(stepsScene.totalCache)}</b>{#if compare}{' '}against {fmtInt(stepsScene.totalNo)}, {stepsScene.speedup.toFixed(1)} times fewer{/if}.</p>
          {#if compare}<p class="sub">Largest difference between cached and uncached logits so far: <b>{fmtDiff(stepsFrame.diffMax)}</b></p>{/if}
        {/if}
      {:else if memScene}
        <p class="sub">Cache per token with 16-bit numbers: {memScene.rows.map((r) => `${r.name} ${fmtBytes(r.perToken)}`).join(' · ')}.</p>
      {:else if phasesSc}
        <p class="sub">Operations per weight byte: prefill <b>{phasesSc.intensity[0].toFixed(1)}</b>, decode <b>{Math.min(...phasesSc.intensity.slice(1)).toFixed(1)} to {Math.max(...phasesSc.intensity.slice(1)).toFixed(1)}</b>.</p>
      {:else if batchSc && frame.kind === 'batch'}
        <p class="sub">Memory: weights {fmtGB(batchSc.weightB)} + cache {fmtGB(frame.kv)} = {fmtGB(frame.used)} of 80 GB. Ridge at {batchSc.ridge.toFixed(0)} operations per byte. Largest batch that fits: <b>{batchSc.fitMax}</b>.</p>
      {/if}
    {/if}
  {/snippet}
  {#snippet footer()}
    {#if init === 'nocache' || init === 'cache'}
      <div class="ctl">
        <label>Text length: {total} characters ({total - (run?.P ?? 16)} generated)
          <input type="range" min={(run?.P ?? 16) + 4} max="64" step="1" value={total} oninput={(e) => edit(() => (total = +e.currentTarget.value))} /></label>
        {#if init === 'cache'}<label class="chk"><input type="checkbox" checked={compare} onchange={(e) => edit(() => (compare = e.currentTarget.checked))} /> Compare with no cache</label>{/if}
      </div>
    {:else if init === 'memory'}
      <div class="ctl"><label>Batch: {batch} conversations at once
        <input type="range" min="1" max="64" step="1" value={batch} oninput={(e) => edit(() => (batch = +e.currentTarget.value))} /></label></div>
    {:else if init === 'phases'}
      <div class="ctl"><label>Prompt length: {promptLen} characters
        <input type="range" min="8" max="56" step="1" value={promptLen} oninput={(e) => edit(() => (promptLen = +e.currentTarget.value))} /></label></div>
    {:else if init === 'batch'}
      <div class="ctl">
        <label>Context length
          <select value={ctx} onchange={(e) => edit(() => (ctx = +e.currentTarget.value))}>{#each ctxChoices as c (c)}<option value={c}>{fmtInt(c)} tokens</option>{/each}</select></label>
        <label>Model
          <select value={modelKey} onchange={(e) => edit(() => (modelKey = e.currentTarget.value as ModelKey))}>
            <option value="llama2_7b">Llama 2 7B (no sharing)</option>
            <option value="llama3_8b">Llama 3 8B (shared keys and values)</option>
          </select></label>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .status { margin: 0 0 var(--s-1); min-height: 2.6em; }
  .sub { margin: 0 0 var(--s-1); overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
  .ctl { display: flex; flex-wrap: wrap; gap: var(--s-3) var(--s-5); align-items: center; padding: var(--s-3) var(--s-4); }
  .ctl label { display: grid; gap: var(--s-1); color: var(--text-muted); min-width: 0; }
  .ctl .chk { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text); }
  .ctl input[type='range'] { width: min(20rem, 100%); accent-color: var(--accent); }
  .ctl select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.3em 0.4em; max-width: 100%; width: 100%; }
  input:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
