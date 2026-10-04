/**
 * A tiny retained-mode SVG layer for the Day 11 shells. A mode's draw() pushes plain shape objects into `E.out`;
 * the shell renders them. Coordinates are CSS pixels of the stage (the SVG viewBox equals the stage size), so a
 * text size of 12 is 12 CSS px at every width. Colours are CSS strings built from tokens: `tok('--viz-pos')`,
 * `mix(tok('--viz-pos'), tok('--bg-raised'), 40)`. No DOM access here.
 */

export interface Style {
  fill?: string; stroke?: string; sw?: number; op?: number; dash?: string;
  /** SVG transform, e.g. `translate(12 30)` (move an agent dot this way, not by changing cx/cy). */
  tf?: string;
  /** Hover text shown under the figure. */
  tip?: string;
  /** Click id passed to frameAt as ui.pick (declare `pickable: true` in the mode meta). */
  pick?: string;
}
export type Shape =
  | ({ k: 'rect'; x: number; y: number; w: number; h: number; rx?: number } & Style)
  | ({ k: 'line'; x1: number; y1: number; x2: number; y2: number } & Style)
  | ({ k: 'path'; d: string } & Style)
  | ({ k: 'circle'; cx: number; cy: number; r: number } & Style)
  | ({ k: 'poly'; pts: string } & Style)
  | ({ k: 'text'; x: number; y: number; s: string; size: number; anchor: 'start' | 'middle' | 'end'; weight?: number; mono?: boolean; halo?: boolean; base?: 'auto' | 'middle' | 'hanging' } & Style);

export interface Env {
  /** Stage size in CSS px; `narrow` when W < 540 (lay out for a phone). */
  W: number; H: number; narrow: boolean;
  out: Shape[];
}

// ---------- colours ----------

export const tok = (name: string): string => `var(${name})`;
/** `pct` percent of colour a mixed into b (sRGB). */
export const mix = (a: string, b: string, pct: number): string => `color-mix(in srgb, ${a} ${Math.max(0, Math.min(100, pct)).toFixed(1)}%, ${b})`;
export const BG = 'var(--bg-raised)', TEXT = 'var(--text)', MUTED = 'var(--text-muted)', RULE = 'var(--rule)', ACCENT = 'var(--accent)';
/** Diverging fill for a signed value: --viz-pos / --viz-neg blended into the surface by |v| / vmax. */
export const signedFill = (v: number, vmax: number): string =>
  mix(v >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', BG, 92 * Math.min(1, Math.abs(v) / (vmax || 1)) ** 0.85);
/** Sequential fill u in [0, 1] over --viz-seq-1 .. --viz-seq-5. */
export function seqFill(u: number): string {
  const p = Math.max(0, Math.min(1, u)) * 4, i = Math.min(3, Math.floor(p));
  return mix(`var(--viz-seq-${i + 2})`, `var(--viz-seq-${i + 1})`, (p - i) * 100);
}
/** Text colour that stays readable on signedFill / seqFill of the same strength. */
export const inkOn = (strength: number): string => (strength > 0.55 ? BG : TEXT);

// ---------- numbers and time ----------

export const clamp = (x: number, a: number, b: number) => (x < a ? a : x > b ? b : x);
export const clamp01 = (x: number) => clamp(x, 0, 1);
export const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
export const ease = (x: number) => { const u = clamp01(x); return u * u * (3 - 2 * u); };
/** Which of n equal segments of length `size` t falls in, and how far through it (0..1). */
export function seg(t: number, size: number, n: number) {
  const i = Math.max(0, Math.min(n - 1, Math.floor(t / size + 1e-9)));
  return { i, u: clamp01((t - i * size) / size) };
}
/** Fixed decimals with a real minus sign (−0.04), and no "−0.00". */
export const fmt = (x: number, d = 2): string => { const s = x.toFixed(d); return /^-0\.?0*$/.test(s) ? s.slice(1) : s.replace('-', '−'); };
export const signFmt = (x: number, d = 2): string => (x > 0 ? '+' : '') + fmt(x, d);

// ---------- primitives ----------

const r1 = (x: number) => Math.round(x * 10) / 10;
export const rect = (E: Env, x: number, y: number, w: number, h: number, s: Style & { rx?: number } = {}) => { E.out.push({ k: 'rect', x: r1(x), y: r1(y), w: r1(Math.max(0, w)), h: r1(Math.max(0, h)), ...s }); };
export const line = (E: Env, x1: number, y1: number, x2: number, y2: number, s: Style = {}) => { E.out.push({ k: 'line', x1: r1(x1), y1: r1(y1), x2: r1(x2), y2: r1(y2), stroke: TEXT, ...s }); };
export const path = (E: Env, d: string, s: Style = {}) => { E.out.push({ k: 'path', d, fill: 'none', ...s }); };
export const circle = (E: Env, cx: number, cy: number, r: number, s: Style = {}) => { E.out.push({ k: 'circle', cx: r1(cx), cy: r1(cy), r, ...s }); };
export const poly = (E: Env, pts: [number, number][], s: Style = {}) => { E.out.push({ k: 'poly', pts: pts.map((p) => `${r1(p[0])},${r1(p[1])}`).join(' '), ...s }); };

/** Smallest text size on a phone (CSS px). */
export const MIN_TEXT = 11;
export interface TextOpts extends Style { size?: number; anchor?: 'start' | 'middle' | 'end'; weight?: number; mono?: boolean; halo?: boolean; base?: 'auto' | 'middle' | 'hanging' }
/** Text in the UI font (mono only for numbers in tables), never below 11 px. `fill` defaults to --text. */
export const text = (E: Env, s: string, x: number, y: number, o: TextOpts = {}) => {
  E.out.push({ k: 'text', x: r1(x), y: r1(y), s, size: Math.max(MIN_TEXT, o.size ?? 12), anchor: o.anchor ?? 'start', fill: TEXT, ...o });
};

/** A straight arrow from (x1,y1) to (x2,y2); the head is drawn inside the length. */
export function arrow(E: Env, x1: number, y1: number, x2: number, y2: number, o: Style & { width?: number; head?: number } = {}) {
  const w = o.width ?? 1.5, len = Math.hypot(x2 - x1, y2 - y1);
  if (len < 0.5) return;
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len, hd = Math.min(len * 0.6, o.head ?? Math.max(6, w * 2.6)), hw = Math.max(3.5, w * 1.6);
  const bx = x2 - ux * hd, by = y2 - uy * hd, color = o.stroke ?? o.fill ?? TEXT;
  line(E, x1, y1, bx, by, { stroke: color, sw: w, op: o.op, dash: o.dash, tip: o.tip, pick: o.pick });
  poly(E, [[x2, y2], [bx - uy * hw, by + ux * hw], [bx + uy * hw, by - ux * hw]], { fill: color, op: o.op });
}

// ---------- charts ----------

export interface Box { x: number; y: number; w: number; h: number }
export interface Axis { lo: number; hi: number; log?: boolean; ticks?: number[]; label?: string; fmt?: (v: number) => string }
/** Linear or log scale from [lo, hi] to [a, b]. */
export function scale(ax: Axis, a: number, b: number) {
  const f = (v: number) => (ax.log ? Math.log10(Math.max(v, 1e-12)) : v), l = f(ax.lo), h = f(ax.hi);
  return (v: number) => a + ((f(v) - l) / (h - l || 1)) * (b - a);
}
/** Ticks: given, else 1-2-5 linear ticks or one per decade on a log axis. */
export function ticksOf(ax: Axis, n = 4): number[] {
  if (ax.ticks) return ax.ticks;
  if (ax.log) { const out: number[] = []; for (let e = Math.ceil(Math.log10(ax.lo)); e <= Math.floor(Math.log10(ax.hi)); e++) out.push(10 ** e); return out; }
  const raw = (ax.hi - ax.lo) / n, mag = 10 ** Math.floor(Math.log10(raw || 1)), stepv = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(ax.lo / stepv) * stepv; v <= ax.hi + 1e-9; v += stepv) out.push(Math.abs(v) < 1e-12 ? 0 : v);
  return out;
}
const tickFmt = (ax: Axis) => ax.fmt ?? ((v: number) => (ax.log ? (v >= 1 ? String(v) : v.toExponential(0).replace('-', '−')) : fmt(v, Math.abs(v) >= 10 || Number.isInteger(v) ? 0 : Math.abs(v) >= 1 ? 1 : 2)));

/** Frame, gridlines, tick labels and axis labels; returns the two scales. Leave about 40 px left and 34 px below. */
export function axes(E: Env, b: Box, xa: Axis, ya: Axis) {
  const sx = scale(xa, b.x, b.x + b.w), sy = scale(ya, b.y + b.h, b.y), sz = E.narrow ? 11 : 12;
  for (const v of ticksOf(ya)) {
    const y = sy(v);
    line(E, b.x, y, b.x + b.w, y, { stroke: RULE, sw: 1, dash: '2 4' });
    text(E, tickFmt(ya)(v), b.x - 6, y + 4, { size: sz, anchor: 'end', fill: MUTED });
  }
  for (const v of ticksOf(xa)) {
    const x = sx(v);
    line(E, x, b.y + b.h, x, b.y + b.h + 4, { stroke: RULE, sw: 1 });
    text(E, tickFmt(xa)(v), x, b.y + b.h + 16, { size: sz, anchor: 'middle', fill: MUTED });
  }
  rect(E, b.x, b.y, b.w, b.h, { fill: 'none', stroke: RULE, sw: 1 });
  if (xa.label) text(E, xa.label, b.x + b.w / 2, b.y + b.h + 31, { size: sz, anchor: 'middle', fill: MUTED });
  if (ya.label) text(E, ya.label, b.x, b.y - 8, { size: sz, fill: MUTED });
  return { sx, sy };
}
export interface Series { ys: ArrayLike<number>; xs?: ArrayLike<number>; color: string; width?: number; dash?: string; op?: number }
/**
 * Line chart. Points beyond x = `upto` are not drawn (draw curves left to right by animating `upto`).
 * Values are clamped into the y range so nothing leaves the frame.
 */
export function lineChart(E: Env, b: Box, xa: Axis, ya: Axis, series: Series[], upto = Infinity) {
  const { sx, sy } = axes(E, b, xa, ya);
  for (const s of series) {
    let d = '';
    for (let i = 0; i < s.ys.length; i++) {
      const x = s.xs ? s.xs[i] : i;
      if (x > upto) break;
      const v = s.ys[i];
      if (!Number.isFinite(v)) continue;
      d += `${d ? 'L' : 'M'}${r1(sx(x))} ${r1(clamp(sy(ya.log ? Math.max(v, ya.lo) : v), b.y, b.y + b.h))}`;
    }
    if (d) path(E, d, { stroke: s.color, sw: s.width ?? 1.75, dash: s.dash, op: s.op });
  }
  return { sx, sy };
}
/** Vertical bars from a baseline (0, or the bottom when all values are positive). */
export function bars(E: Env, b: Box, vals: ArrayLike<number>, o: { lo?: number; hi?: number; color?: (i: number) => string; label?: (i: number) => string | null; gap?: number; tip?: (i: number) => string } = {}) {
  const n = vals.length, lo = o.lo ?? Math.min(0, ...Array.from(vals)), hi = o.hi ?? Math.max(0, ...Array.from(vals)), sy = scale({ lo, hi }, b.y + b.h, b.y);
  const bw = b.w / n, gap = o.gap ?? 0.25, y0 = sy(clamp(0, lo, hi));
  line(E, b.x, y0, b.x + b.w, y0, { stroke: RULE, sw: 1 });
  for (let i = 0; i < n; i++) {
    const y = sy(clamp(vals[i], lo, hi)), x = b.x + i * bw + (bw * gap) / 2;
    rect(E, x, Math.min(y, y0), bw * (1 - gap), Math.abs(y - y0), { fill: o.color?.(i) ?? 'var(--viz-forward)', tip: o.tip?.(i) });
    const l = o.label?.(i);
    if (l) text(E, l, x + (bw * (1 - gap)) / 2, vals[i] >= 0 ? Math.min(y, y0) - 4 : Math.max(y, y0) + 13, { size: E.narrow ? 11 : 12, anchor: 'middle', fill: MUTED });
  }
  return { sy, bw };
}
/** Histogram counts of `vals` in `bins` equal bins over [lo, hi] (values outside go to the end bins). */
export function binCounts(vals: ArrayLike<number>, bins: number, lo: number, hi: number, n = vals.length): Int32Array {
  const c = new Int32Array(bins);
  for (let i = 0; i < n; i++) c[clamp(Math.floor(((vals[i] - lo) / (hi - lo)) * bins), 0, bins - 1)]++;
  return c;
}
/** Small table, e.g. a ticker of moves. `cols` are x offsets; rows draw downward from y (row height rh). */
export function table(E: Env, x: number, y: number, cols: number[], rows: string[][], o: { rh?: number; size?: number; header?: string[]; fill?: (r: number) => string; anchor?: ('start' | 'end' | 'middle')[] } = {}) {
  const rh = o.rh ?? 18, size = o.size ?? 12;
  let yy = y;
  if (o.header) { o.header.forEach((h, j) => text(E, h, x + cols[j], yy, { size, fill: MUTED, anchor: o.anchor?.[j] })); yy += rh; }
  rows.forEach((r, i) => { r.forEach((c, j) => text(E, c, x + cols[j], yy, { size, fill: o.fill?.(i) ?? TEXT, anchor: o.anchor?.[j] })); yy += rh; });
  return yy;
}
