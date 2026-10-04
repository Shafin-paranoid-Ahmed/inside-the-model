/**
 * ConvShapes: how channels, stride, padding and pooling decide the shape of a conv layer's output.
 * Pure, no DOM. frameAt(t, mode, size) returns a scene (grids, boxes, labels, numbers) for one instant,
 * so scrubbing is free. The arithmetic reuses src/lib/cnn/conv (outSize, conv2d, pool2d).
 */
import { conv2d, convLayer, map3, outSize, pool2d, type Map3 } from '../../../lib/cnn/conv';
import { createRng } from '../../../lib/math/rng';

export type Mode = 'channels' | 'stride' | 'padding' | 'size' | 'pool';
export interface Marker { at: number; label: string }

export interface Cell {
  text: string;
  /** 0..1 intensity of the fill. */
  tone: number;
  kind: 'in' | 'out' | 'kernel' | 'pad' | 'drop' | 'empty';
  sign?: 'pos' | 'neg' | 'fwd';
  ring?: 'cur' | 'winner';
}
export interface Grid { id: string; x: number; y: number; cell: number; rows: number; cols: number; cells: Cell[] }
export interface Box { x: number; y: number; w: number; h: number }
export interface Label { x: number; y: number; text: string; anchor?: 'start' | 'middle' | 'end'; tone?: 'pos' | 'neg' | 'muted' | 'strong' }
export interface Line { text: string; tone?: 'pos' | 'neg' | 'muted' | 'strong' }
export interface Frame {
  mode: Mode;
  w: number;
  h: number;
  grids: Grid[];
  boxes: Box[];
  labels: Label[];
  /** One short sentence naming what is happening; changes only between phases. */
  status: string;
  calc: { heading: string; lines: Line[]; facts: Line[] };
}

export interface SizeCfg { n: number; k: number; s: number; p: number }
export const DEFAULT_SIZE: SizeCfg = { n: 8, k: 3, s: 2, p: 0 };
export const SIZE_LIMITS = { n: [3, 9], k: [1, 7], s: [1, 4], p: [0, 3] } as const;

const W = 500;
const GLIDE = 0.35;
const MINUS = '−';

const num = (v: number) => (v < 0 ? `${MINUS}${-v}` : `${v}`);
const sgn = (v: number) => (v > 0 ? `+${v}` : v < 0 ? `${MINUS}${-v}` : '0');
const tidy = (v: number) => `${Number(v.toFixed(2))}`;
const ease = (x: number) => { const c = Math.min(1, Math.max(0, x)); return c * c * (3 - 2 * c); };

export const outDim = (n: number, k: number, s: number, p: number) => Math.max(0, outSize(n, k, s, p));
export const paramCount = (inC: number, outC: number, k: number) => ({
  perFilter: inC * k * k,
  withBias: inC * k * k + 1,
  total: outC * (inC * k * k + 1),
});

/** Where the window is at time t: which output cell is current and how far a glide has got. */
export function sweep(t: number, step: number, count: number) {
  const pos = Math.max(0, t) / step;
  const idx = Math.min(count - 1, Math.floor(pos));
  const frac = pos >= count ? 1 : pos - idx;
  const moving = idx > 0 && frac < GLIDE;
  const e = moving ? ease(frac / GLIDE) : 1;
  return {
    idx, count,
    cur: moving && e < 0.5 ? idx - 1 : idx,
    from: moving ? idx - 1 : idx,
    to: idx,
    e,
  };
}
type Sweep = ReturnType<typeof sweep>;

const winPos = (sw: Sweep, ow: number, s: number) => {
  const at = (i: number) => ({ y: Math.floor(i / ow) * s, x: (i % ow) * s });
  const a = at(sw.from), b = at(sw.to);
  return { y: a.y + (b.y - a.y) * sw.e, x: a.x + (b.x - a.x) * sw.e };
};

const grid = (id: string, x: number, y: number, cell: number, rows: number, cols: number, fn: (r: number, c: number) => Cell): Grid => {
  const cells: Cell[] = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) cells.push(fn(r, c));
  return { id, x, y, cell, rows, cols, cells };
};
const empty: Cell = { text: '', tone: 0, kind: 'empty' };
const signed = (v: number, maxAbs: number, ring?: Cell['ring']): Cell => ({
  text: num(v), tone: Math.min(1, Math.abs(v) / maxAbs), kind: 'out', sign: v < 0 ? 'neg' : 'pos', ring,
});

/* ------------------------------------------------------------------ data */

const clampInt = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, Math.round(v)));

/** Three 6 x 6 planes with different content: a vertical edge, a blob, a horizontal edge (values 0..3). */
export const CH_IN: Map3 = (() => {
  const rng = createRng(21);
  const m = map3(3, 6, 6);
  for (let y = 0; y < 6; y++) for (let x = 0; x < 6; x++) {
    const base = [x >= 3 ? 3 : 0, 3 - Math.hypot(x - 2.5, y - 2.5), y < 3 ? 3 : 0];
    for (let c = 0; c < 3; c++) m.data[(c * 6 + y) * 6 + x] = clampInt(base[c] + rng.uniform(-0.6, 0.6), 0, 3);
  }
  return m;
})();

/** Two filters, each 3 x 3 x 3: one 3 x 3 slice per input channel. */
const FILTERS: number[][][] = [
  [[-1, 0, 1, -1, 0, 1, -1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0], [-1, -1, -1, 0, 0, 0, 1, 1, 1]],
  [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1, -1]],
];
const BIAS = [1, -1];
export const CH_LAYER = (() => {
  const L = convLayer(3, 2, 3, 1, 0);
  FILTERS.forEach((f, o) => f.forEach((sl, i) => L.weight.set(sl, (o * 3 + i) * 9)));
  L.bias.set(BIAS);
  return L;
})();
export const CH_OUT = conv2d(CH_IN, CH_LAYER);

/** The part of an output cell that channel c contributes (window x weights, summed). */
export function partialSum(f: number, c: number, oy: number, ox: number) {
  let s = 0;
  for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) s += FILTERS[f][c][ky * 3 + kx] * CH_IN.data[(c * 6 + oy + ky) * 6 + ox + kx];
  return s;
}

/** 7 x 7 single-channel picture for stride and padding (a diagonal stripe, values 0..3). */
export const ST_IN: Map3 = (() => {
  const rng = createRng(5);
  const m = map3(1, 7, 7);
  for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++)
    m.data[y * 7 + x] = clampInt((Math.abs(x - y) <= 1 ? 3 : 0) + rng.uniform(-0.6, 0.6), 0, 3);
  return m;
})();
const EDGE = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
export const ST_LAYER = (() => { const L = convLayer(1, 1, 3, 1, 0); L.weight.set(EDGE); return L; })();
export const strideOutput = (s: number): Map3 => conv2d(ST_IN, { ...ST_LAYER, stride: s });

/** 6 x 6 feature map for pooling: mostly small values and a few strong responses. */
export const POOL_IN: Map3 = (() => {
  const rng = createRng(11);
  const m = map3(1, 6, 6);
  for (let i = 0; i < 36; i++) m.data[i] = Math.floor(rng.next() * 3);
  [[0, 1, 7], [1, 4, 9], [2, 2, 8], [3, 0, 6], [4, 3, 9], [5, 5, 7], [3, 4, 5]].forEach(([y, x, v]) => { m.data[y * 6 + x] = v; });
  return m;
})();
export const poolOutput = (kind: 'max' | 'avg') => pool2d(POOL_IN, 2, kind);

/** How many windows (of all that run) cover each pixel of an n x n image. */
export function coverage(n: number, k: number, s: number, p: number, upTo = Infinity): number[] {
  const o = outDim(n, k, s, p);
  const cov = new Array<number>(n * n).fill(0);
  for (let i = 0; i < o * o && i <= upTo; i++) {
    const oy = Math.floor(i / o), ox = i % o;
    for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
      const y = oy * s + ky - p, x = ox * s + kx - p;
      if (y >= 0 && y < n && x >= 0 && x < n) cov[y * n + x]++;
    }
  }
  return cov;
}

/* ------------------------------------------------------------------ timing */

const CH_STEP = 0.5, CH_SWEEP = 16 * CH_STEP;
const ST_STEP = 0.4, ST_HOLD = 3;
const SIZE_T = 8;
const POOL_STEP = 0.8, POOL_HOLD = 1.5;

export const durationOf = (mode: Mode): number =>
  mode === 'channels' ? 2 * CH_SWEEP + 2
    : mode === 'stride' || mode === 'padding' ? 25 * ST_STEP + ST_HOLD
      : mode === 'size' ? SIZE_T
        : 9 * POOL_STEP + POOL_HOLD;

export const markersOf = (mode: Mode): Marker[] =>
  mode === 'channels' ? [{ at: 0, label: 'Filter 1' }, { at: CH_SWEEP, label: 'Filter 2' }]
    : mode === 'stride' ? [{ at: 0, label: 'Both sliding' }, { at: 9 * ST_STEP, label: 'Stride 2 done' }]
      : mode === 'padding' ? [{ at: 0, label: 'Both sliding' }, { at: 9 * ST_STEP, label: 'No padding done' }, { at: 25 * ST_STEP, label: 'Final counts' }]
        : mode === 'size' ? [{ at: 0, label: 'Sweep' }]
          : [{ at: 0, label: 'Row 1' }, { at: 3 * POOL_STEP, label: 'Row 2' }, { at: 6 * POOL_STEP, label: 'Row 3' }];

/* ------------------------------------------------------------------ frames */

function channelsFrame(t: number): Frame {
  const f = t < CH_SWEEP ? 0 : 1;
  const sw = sweep(t - f * CH_SWEEP, CH_STEP, 16);
  const win = winPos(sw, 4, 1);
  const oy = Math.floor(sw.cur / 4), ox = sw.cur % 4;
  const C = 24, grids: Grid[] = [], boxes: Box[] = [], labels: Label[] = [];
  const parts = [0, 1, 2].map((c) => partialSum(f, c, oy, ox));
  const outVal = CH_OUT.data[(f * 4 + oy) * 4 + ox];
  labels.push({ x: 20, y: 16, text: 'Input: 3 × 6 × 6 (channels × height × width)', anchor: 'start', tone: 'strong' });
  labels.push({ x: 20, y: 214, text: `Filter ${f + 1}: 3 × 3 × 3, one 3 × 3 slice per channel`, anchor: 'start', tone: 'strong' });
  labels.push({ x: 20, y: 346, text: 'Output: 2 × 4 × 4', anchor: 'start', tone: 'strong' });
  for (let c = 0; c < 3; c++) {
    const x0 = 20 + c * 158;
    grids.push(grid(`in${c}`, x0, 44, C, 6, 6, (r, q) => {
      const v = CH_IN.data[(c * 6 + r) * 6 + q];
      return { text: `${v}`, tone: v / 3, kind: 'in' };
    }));
    labels.push({ x: x0 + 72, y: 36, text: `channel ${c + 1}`, anchor: 'middle', tone: 'muted' });
    boxes.push({ x: x0 + win.x * C, y: 44 + win.y * C, w: 3 * C, h: 3 * C });
    grids.push(grid(`k${c}`, x0 + 39, 224, 22, 3, 3, (r, q) => {
      const w = FILTERS[f][c][r * 3 + q];
      return { text: num(w), tone: Math.abs(w) * 0.6, kind: 'kernel', sign: w < 0 ? 'neg' : 'pos' };
    }));
    labels.push({ x: x0 + 72, y: 314, text: sgn(parts[c]), anchor: 'middle', tone: parts[c] < 0 ? 'neg' : parts[c] > 0 ? 'pos' : 'muted' });
  }
  const maxAbs = Math.max(...Array.from(CH_OUT.data, Math.abs));
  for (let m = 0; m < 2; m++) {
    grids.push(grid(`out${m}`, 110 + m * 170, 356, 28, 4, 4, (r, q) => {
      const i = r * 4 + q;
      const shown = m < f || (m === f && i <= sw.cur);
      if (!shown) return empty;
      return signed(CH_OUT.data[(m * 4 + r) * 4 + q], maxAbs, m === f && i === sw.cur ? 'cur' : undefined);
    }));
    labels.push({ x: 110 + m * 170 + 56, y: 484, text: `filter ${m + 1}`, anchor: 'middle', tone: 'muted' });
  }
  const pc = paramCount(3, 2, 3);
  return {
    mode: 'channels', w: W, h: 496, grids, boxes, labels,
    status: `Filter ${f + 1} is sliding over all three channels at once.`,
    calc: {
      heading: `Output ${f + 1}, row ${oy + 1}, column ${ox + 1}`,
      lines: [
        ...parts.map((v, c): Line => ({ text: `channel ${c + 1} adds ${sgn(v)}`, tone: v < 0 ? 'neg' : v > 0 ? 'pos' : 'muted' })),
        { text: `bias adds ${sgn(BIAS[f])}` },
        { text: `${parts.map(sgn).join(' ')} ${sgn(BIAS[f])} = ${num(outVal)}`, tone: 'strong' },
      ],
      facts: [
        { text: `Weights in one filter: 3 × 3 × 3 = ${pc.perFilter}, plus 1 bias = ${pc.withBias}` },
        { text: `Whole layer: 2 filters × ${pc.withBias} = ${pc.total} numbers to learn`, tone: 'strong' },
      ],
    },
  };
}

function stridePadFrame(t: number, mode: 'stride' | 'padding', narrow: boolean): Frame {
  const K = 3, C = narrow ? 26 : 30, SLOT = 7 * C;
  const grids: Grid[] = [], boxes: Box[] = [], labels: Label[] = [], heads: Line[] = [], facts: Line[] = [];
  for (let i = 0; i < 2; i++) {
    const s = mode === 'stride' ? i + 1 : 1;
    const p = mode === 'padding' ? i : 0;
    const n = mode === 'stride' ? 7 : 5;
    const o = outDim(n, K, s, p), N = o * o, np = n + 2 * p;
    const sw = sweep(t, ST_STEP, N);
    const win = winPos(sw, o, s);
    // wide: two panels side by side; narrow: panels stacked, each with input left and output right
    const px = narrow ? 8 : 20 + i * 250, yo = narrow ? i * 244 : 0;
    const gx = px + (SLOT - np * C) / 2, gy = yo + (narrow ? 30 : 36);
    const cov = mode === 'padding' ? coverage(n, K, s, p, sw.cur) : [];
    const outMap = mode === 'stride' ? strideOutput(s) : null;
    labels.push({ x: px + SLOT / 2, y: yo + (narrow ? 18 : 20), text: mode === 'stride' ? `Stride ${s}` : p ? 'Padding 1' : 'No padding', anchor: 'middle', tone: 'strong' });
    grids.push(grid(`in${i}`, gx, gy, C, np, np, (r, q) => {
      const y = r - p, x = q - p;
      if (y < 0 || y >= n || x < 0 || x >= n) return { text: mode === 'padding' ? '' : '0', tone: 0, kind: 'pad' };
      if (mode === 'padding') { const c = cov[y * n + x]; return { text: `${c}`, tone: c / 9, kind: 'in' }; }
      const v = ST_IN.data[y * n + x];
      return { text: `${v}`, tone: v / 3, kind: 'in' };
    }));
    boxes.push({ x: gx + win.x * C, y: gy + win.y * C, w: K * C, h: K * C });
    labels.push({ x: px + SLOT / 2, y: gy + np * C + 18, text: `Input 1 × ${np} × ${np}${p && !narrow ? ' (with padding)' : ''}`, anchor: 'middle', tone: 'muted' });
    const oy0 = narrow ? gy : 292, ox0 = narrow ? 216 + (5 * C - o * C) / 2 : px + (SLOT - o * C) / 2;
    grids.push(grid(`out${i}`, ox0, oy0, C, o, o, (r, q) => {
      const j = r * o + q;
      if (j > sw.cur) return empty;
      const ring = j === sw.cur ? 'cur' : undefined;
      return outMap ? signed(outMap.data[j], 9, ring) : { text: '', tone: 0.45, kind: 'out', sign: 'fwd', ring };
    }));
    labels.push({ x: ox0 + (o * C) / 2, y: oy0 + o * C + 18, text: `Output 1 × ${o} × ${o}`, anchor: 'middle', tone: 'strong' });
    heads.push({ text: `${mode === 'stride' ? `Stride ${s}` : p ? 'Padding 1' : 'No padding'}: window ${sw.cur + 1} of ${N}` });
    if (mode === 'stride') facts.push({ text: `Stride ${s}: ${N} windows × ${K * K} multiplications = ${N * K * K}` });
    else {
      const fc = coverage(n, K, s, p);
      facts.push({ text: `${p ? 'Padding 1' : 'No padding'}: corner pixel in ${fc[0]} window${fc[0] === 1 ? '' : 's'}, centre pixel in ${fc[Math.floor(n / 2) * n + Math.floor(n / 2)]}` });
    }
  }
  return {
    mode, w: narrow ? 350 : W, h: narrow ? 480 : 472, grids, boxes, labels,
    status: mode === 'stride' ? 'Both windows slide at the same pace; the stride 2 window jumps two cells.' : 'Each input cell counts how many windows have covered it.',
    calc: { heading: mode === 'stride' ? 'Same filter, same input' : 'Windows covering each pixel', lines: heads, facts },
  };
}

function sizeFrame(t: number, cfg: SizeCfg): Frame {
  const { n, k, s, p } = cfg;
  const np = n + 2 * p, o = outDim(n, k, s, p), N = o * o;
  const used = o > 0 ? (o - 1) * s + k : 0;
  const cell = Math.max(8, Math.min(30, Math.floor(246 / Math.max(np, o, 1)), Math.floor(440 / (np + Math.max(o, 1) + 1.4))));
  const sw = sweep(t, (SIZE_T - 1) / Math.max(N, 1), Math.max(N, 1));
  const win = winPos(sw, Math.max(o, 1), s);
  const gx = 20, gy = 44, ox0 = gx + np * cell + 36;
  const grids: Grid[] = [], boxes: Box[] = [], labels: Label[] = [];
  grids.push(grid('in', gx, gy, cell, np, np, (r, q) => {
    if (o > 0 && (r >= used || q >= used)) return { text: '', tone: 0, kind: 'drop' };
    if (r < p || r >= n + p || q < p || q >= n + p) return { text: '', tone: 0, kind: 'pad' };
    return { text: '', tone: 0.2 + 0.5 * (((q - p) * 7 + (r - p) * 3) % 5) / 4, kind: 'in' };
  }));
  if (o > 0) {
    boxes.push({ x: gx + win.x * cell, y: gy + win.y * cell, w: k * cell, h: k * cell });
    grids.push(grid('out', ox0, gy, cell, o, o, (r, q) => {
      const j = r * o + q;
      return j > sw.cur ? empty : { text: '', tone: 0.45, kind: 'out', sign: 'fwd', ring: j === sw.cur ? 'cur' : undefined };
    }));
  }
  labels.push({ x: gx, y: 26, text: `Input 1 × ${n} × ${n}${p ? `, padded to ${np} × ${np}` : ''}`, anchor: 'start', tone: 'strong' });
  labels.push({ x: ox0, y: 26, text: o > 0 ? `Output 1 × ${o} × ${o}` : 'Output: no window fits', anchor: 'start', tone: o > 0 ? 'strong' : 'neg' });
  const top = n + 2 * p - k;
  const lines: Line[] = [
    { text: `n = ${n}, k = ${k}, s = ${s}, p = ${p}`, tone: 'muted' },
    { text: `(n + 2p ${MINUS} k) / s = (${n} + ${2 * p} ${MINUS} ${k}) / ${s} = ${tidy(top / s)}` },
  ];
  const facts: Line[] = [];
  if (o > 0) {
    lines.push({ text: `floor(${tidy(top / s)}) + 1 = ${o}`, tone: 'strong' });
    const left = np - used;
    facts.push({ text: `Output shape: 1 × ${o} × ${o}` });
    facts.push({ text: left > 0 ? `${left} leftover row${left > 1 ? 's' : ''} and column${left > 1 ? 's' : ''} (hatched) are never touched.` : 'Everything fits: no leftover rows or columns.' });
  } else {
    lines.push({ text: `${tidy(top / s)} is below zero`, tone: 'neg' });
    facts.push({ text: `The filter (${k}) is wider than the padded input (${np}), so no window fits.` });
  }
  return {
    mode: 'size', w: W, h: 300, grids, boxes, labels,
    status: o > 0 ? `Output is ${o} × ${o}.` : 'No window fits.',
    calc: { heading: 'The formula with your numbers', lines, facts },
  };
}

function poolFrame(t: number): Frame {
  const K = 2, C = 36, N = 9;
  const sw = sweep(t, POOL_STEP, N);
  const win = winPos(sw, 3, K);
  const oy = Math.floor(sw.cur / 3) * K, ox = (sw.cur % 3) * K;
  const vals: number[] = [];
  for (let y = 0; y < K; y++) for (let x = 0; x < K; x++) vals.push(POOL_IN.data[(oy + y) * 6 + ox + x]);
  const best = vals.indexOf(Math.max(...vals));
  const mx = poolOutput('max'), av = poolOutput('avg');
  const grids: Grid[] = [], labels: Label[] = [];
  grids.push(grid('in', 24, 44, C, 6, 6, (r, q) => {
    const v = POOL_IN.data[r * 6 + q];
    const winner = r === oy + Math.floor(best / K) && q === ox + (best % K);
    return { text: `${v}`, tone: v / 9, kind: 'in', ring: winner ? 'winner' : undefined };
  }));
  const outs: [string, Map3, number][] = [['max', mx, 44], ['avg', av, 206]];
  for (const [id, m, y0] of outs) {
    grids.push(grid(id, 290, y0, C, 3, 3, (r, q) => {
      const j = r * 3 + q;
      if (j > sw.cur) return empty;
      const v = m.data[j];
      return { text: id === 'max' ? `${v}` : tidy(v), tone: v / 9, kind: 'out', sign: 'fwd', ring: j === sw.cur ? 'cur' : undefined };
    }));
    labels.push({ x: 290, y: y0 - 18, text: `${id === 'max' ? 'Max' : 'Average'}: 1 × 3 × 3`, anchor: 'start', tone: 'strong' });
  }
  labels.push({ x: 24, y: 26, text: 'Feature map: 1 × 6 × 6', anchor: 'start', tone: 'strong' });
  labels.push({ x: 24, y: 44 + 6 * C + 20, text: '2 × 2 window, stride 2', anchor: 'start', tone: 'muted' });
  const sum = vals.reduce((a, b) => a + b, 0);
  return {
    mode: 'pool', w: W, h: 316, grids,
    boxes: [{ x: 24 + win.x * C, y: 44 + win.y * C, w: K * C, h: K * C }], labels,
    status: 'The window moves two cells at a time; each window gives one number per method.',
    calc: {
      heading: `Window ${sw.cur + 1} of ${N}`,
      lines: [
        { text: `values: ${vals.join(', ')}` },
        { text: `max = ${Math.max(...vals)}`, tone: 'strong' },
        { text: `average = (${vals.join(' + ')}) / 4 = ${tidy(sum / 4)}`, tone: 'strong' },
      ],
      facts: [
        { text: 'Output is 1 × 3 × 3: height and width halve, the channel count stays.' },
        { text: 'Weights learned by pooling: 0.' },
      ],
    },
  };
}

export function frameAt(t: number, mode: Mode, size: SizeCfg = DEFAULT_SIZE, narrow = false): Frame {
  const tt = Math.min(durationOf(mode), Math.max(0, t));
  return mode === 'channels' ? channelsFrame(tt)
    : mode === 'stride' || mode === 'padding' ? stridePadFrame(tt, mode, narrow)
      : mode === 'size' ? sizeFrame(tt, size)
        : poolFrame(tt);
}

/** CSS aspect ratios (wide, narrow) matching the frame's viewBox, so the stage box fits the content. */
export const aspectOf = (mode: Mode) => {
  const a = frameAt(0, mode, DEFAULT_SIZE, false), b = frameAt(0, mode, DEFAULT_SIZE, true);
  return { wide: `${a.w} / ${a.h}`, narrow: `${b.w} / ${b.h}` };
};
