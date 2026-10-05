/** Pure logic for ChainRuleLinks: a chain x -> u -> v -> L of three functions, drawn as linked slopes. */

export interface Link { name: string; expr: (inp: string) => string; f: (a: number) => number; df: (a: number) => number }
export interface Chain { id: ChainId; links: [Link, Link, Link]; names: [string, string, string, string]; xRange: [number, number]; x0: number }
export type ChainId = 'wave' | 'saturate';

const sq: Link = { name: 'square', expr: (a) => `${a}²`, f: (a) => a * a, df: (a) => 2 * a };
export const CHAINS: Record<ChainId, Chain> = {
  // u = x², v = sin u, L = v²
  wave: {
    id: 'wave', names: ['x', 'u', 'v', 'L'], xRange: [-1.6, 1.6], x0: 1,
    links: [sq, { name: 'sine', expr: (a) => `sin(${a})`, f: Math.sin, df: Math.cos }, sq],
  },
  // u = 2x, v = tanh u, L = 3v²
  saturate: {
    id: 'saturate', names: ['x', 'u', 'v', 'L'], xRange: [-1.6, 1.6], x0: 0.3,
    links: [
      { name: 'double', expr: (a) => `2${a}`, f: (a) => 2 * a, df: () => 2 },
      { name: 'tanh', expr: (a) => `tanh(${a})`, f: Math.tanh, df: (a) => 1 - Math.tanh(a) ** 2 },
      { name: 'three squares', expr: (a) => `3${a}²`, f: (a) => 3 * a * a, df: (a) => 6 * a },
    ],
  },
};

/** Values [x,u,v,L] and local slopes [du/dx, dv/du, dL/dv]. */
export function evalChain(c: Chain, x: number) {
  const vals = [x];
  const slopes: number[] = [];
  for (const l of c.links) { slopes.push(l.df(vals[vals.length - 1])); vals.push(l.f(vals[vals.length - 1])); }
  const prods = [1, slopes[0], slopes[0] * slopes[1], slopes[0] * slopes[1] * slopes[2]];
  return { vals, slopes, prods, dLdx: prods[3] };
}

/** Central finite difference of the whole chain. */
export const numericDeriv = (c: Chain, x: number, h = 1e-5): number =>
  (evalChain(c, x + h).vals[3] - evalChain(c, x - h).vals[3]) / (2 * h);

/**
 * One job per lesson figure. wiggle: x shakes, the bars show each wiggle's size, no slopes yet.
 * tangent: the timeline sweeps x through 0 so every tangent tilts. links: wiggle, then reveal link 1, 2, 3.
 * full: links, then the product against the numerical derivative.
 */
export type View = 'wiggle' | 'tangent' | 'links' | 'full';
/** Timeline (full view): wiggle, then reveal link 1, 2, 3, then the product. */
export const T = 10;
const T_WIGGLE = 4, T_TANGENT = 8, T_LINKS = 7.5;
export const duration = (v: View) => (v === 'wiggle' ? T_WIGGLE : v === 'tangent' ? T_TANGENT : v === 'links' ? T_LINKS : T);
export const WIGGLE_CYCLES = 5;
export const REVEAL_AT = [1.5, 3.5, 5.5, 7.5] as const;
export const MARKERS = [
  { at: 0, label: 'wiggle' }, { at: REVEAL_AT[0], label: 'link 1' }, { at: REVEAL_AT[1], label: 'link 2' },
  { at: REVEAL_AT[2], label: 'link 3' }, { at: REVEAL_AT[3], label: 'product' },
];
export const WIGGLE = 0.1;
export const markers = (v: View) =>
  v === 'wiggle' ? [{ at: 0, label: 'wiggle' }]
    : v === 'tangent' ? [{ at: 0, label: 'shrinking x' }, { at: T_TANGENT / 4, label: 'at x = 0' }, { at: T_TANGENT / 2, label: 'growing x again' }]
    : v === 'links' ? MARKERS.slice(0, 4) : MARKERS;

const ease = (p: number) => { const q = Math.min(1, Math.max(0, p)); return q * q * (3 - 2 * q); };

export interface Frame {
  /** [x,u,v,L] at the wiggled input. */
  pts: number[];
  base: number[];
  slopes: number[];
  prods: number[];
  /** reveal progress 0..1 of link 1, 2, 3 and of the final product. */
  reveal: number[];
  /** visibility 0..1 of the wiggle bar for x, u, v, L. */
  bars: number[];
  /** opacity of the "dL/dx = a × b × c" line. */
  eq: number;
  /** the x the frame was computed at (the tangent view sweeps it). */
  x: number;
  /** wiggle amplitude at each level (linear theory: |prod| * WIGGLE). */
  amp: number[];
  dLdx: number;
  numeric: number;
  phase: string;
}

export function frameAt(t: number, c: Chain, x0: number, view: View = 'full'): Frame {
  // the tangent view swings x from x0 through 0 to -x0 and back, so every tangent visibly tilts
  const x = view === 'tangent' ? x0 * Math.cos((2 * Math.PI * t) / T_TANGENT) : x0;
  const base = evalChain(c, x);
  const dx = WIGGLE * Math.sin((2 * Math.PI * WIGGLE_CYCLES * t) / T);
  const pts = evalChain(c, x + dx).vals;
  const reveal = view === 'wiggle' ? [0, 0, 0, 0] : view === 'tangent' ? [1, 1, 1, 0] : REVEAL_AT.map((s) => ease((t - s) / 1.2));
  if (view === 'links') reveal[3] = 0;
  const bars = view === 'wiggle' || view === 'tangent' ? [1, 1, 1, 1] : [1, reveal[0], reveal[1], reveal[2]];
  const eq = view === 'wiggle' || view === 'tangent' ? 0 : Math.max(reveal[3], reveal[2] * 0.6);
  const mk = markers(view);
  let phase = mk[0].label;
  for (const m of mk) if (t >= m.at) phase = m.label;
  return {
    pts, base: base.vals, slopes: base.slopes, prods: base.prods, reveal, bars, eq, x,
    amp: base.prods.map((p) => Math.abs(p) * WIGGLE), dLdx: base.dLdx, numeric: numericDeriv(c, x), phase,
  };
}

/** Sampled curve of link k over the inputs it sees while x sweeps its range. */
export function linkView(c: Chain, k: number, n = 64) {
  const xs = Array.from({ length: n + 1 }, (_, i) => c.xRange[0] + ((c.xRange[1] - c.xRange[0]) * i) / n);
  const ins = xs.map((x) => evalChain(c, x).vals[k]);
  const lo = Math.min(...ins), hi = Math.max(...ins), pad = (hi - lo) * 0.06 || 1;
  const d0 = lo - pad, d1 = hi + pad;
  const grid = Array.from({ length: n + 1 }, (_, i) => d0 + ((d1 - d0) * i) / n);
  const ys = grid.map((a) => c.links[k].f(a));
  const ylo = Math.min(...ys), yhi = Math.max(...ys), yp = (yhi - ylo) * 0.08 || 1;
  return { d0, d1, r0: ylo - yp, r1: yhi + yp, grid, ys };
}
