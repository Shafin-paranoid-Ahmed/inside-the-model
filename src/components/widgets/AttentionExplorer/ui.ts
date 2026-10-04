/** Shared canvas helpers: palette, text, grids, strips and bars. */
import { WORD_CLASS } from '../../../lib/transformer/grammar';
import { clamp01, sgn, type Scene } from './core';
import type { HeadStat } from './stats';

export type RGB = [number, number, number];
export interface Pal { bg: RGB; text: RGB; muted: RGB; rule: RGB; accent: RGB; pos: RGB; neg: RGB; seq: RGB[]; cat: RGB[]; ui: string; code: string }
export interface Hit { x: number; y: number; w: number; h: number; text: string; row?: number }
export interface Env { c: CanvasRenderingContext2D; P: Pal; W: number; H: number; narrow: boolean; hits: Hit[]; s: Scene; stats?: HeadStat[] }

export const TOKEN_NAMES = {
  bg: '--bg-raised', text: '--text', muted: '--text-muted', rule: '--rule', accent: '--accent', pos: '--viz-pos', neg: '--viz-neg',
  s1: '--viz-seq-1', s2: '--viz-seq-2', s3: '--viz-seq-3', s4: '--viz-seq-4', s5: '--viz-seq-5',
  c1: '--viz-cat-1', c2: '--viz-cat-2', c3: '--viz-cat-3', c4: '--viz-cat-4', c5: '--viz-cat-5',
};
/** Word classes use cat 1-4; heads use pink, green, amber, blue so the two never share a meaning in one figure. */
export const CLASS_LABELS = ['Determiner', 'Adjective', 'Noun', 'Verb'];
export const classOf = (id: number) => ({ det: 0, adj: 1, noun: 2, verb: 3 } as Record<string, number>)[WORD_CLASS[id].kind] ?? -1;
export const HEAD_CAT = [4, 2, 3, 0];

export const rgba = (c: RGB, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
export const mixc = (a: RGB, b: RGB, u: number): RGB => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
export const lum = (c: RGB) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
export function seqColor(P: Pal, x: number): RGB {
  const p = Math.pow(clamp01(x), 0.8) * 4, i = Math.min(3, Math.floor(p));
  return mixc(P.seq[i], P.seq[i + 1], p - i);
}
export const signed = (P: Pal, vmax: number) => (v: number): RGB => mixc(P.bg, v >= 0 ? P.pos : P.neg, Math.pow(clamp01(Math.abs(v) / vmax), 0.85) * 0.92);
export const fmt2 = (x: number) => x.toFixed(2);

/** Narrow figures are laid out at the real phone width (330 logical px), so this is the smallest CSS font size. */
export const MIN_PX = 11.5;
export interface TOpts { sz?: number; col?: RGB; al?: CanvasTextAlign; bold?: boolean; mono?: boolean; a?: number; base?: CanvasTextBaseline }
export function txt(E: Env, s: string, x: number, y: number, o: TOpts = {}) {
  const { c, P } = E;
  c.font = `${o.bold ? 600 : 400} ${Math.max(o.sz ?? 11, E.narrow ? MIN_PX : 0)}px ${o.mono ? P.code : P.ui}`;
  c.fillStyle = rgba(o.col ?? P.text, o.a ?? 1); c.textAlign = o.al ?? 'left'; c.textBaseline = o.base ?? 'alphabetic';
  c.fillText(s, x, y);
}
export function hatch(E: Env, x: number, y: number, w: number, h: number, a = 0.5) {
  const { c, P } = E; c.save(); c.beginPath(); c.rect(x, y, w, h); c.clip();
  c.strokeStyle = rgba(P.muted, a); c.lineWidth = 1; c.beginPath();
  for (let k = -h; k < w; k += 4) { c.moveTo(x + k, y + h); c.lineTo(x + k + h, y); }
  c.stroke(); c.restore();
}
export function box(E: Env, x: number, y: number, w: number, h: number, col: RGB, lw = 1.5, r = 3) {
  const { c } = E; c.strokeStyle = rgba(col); c.lineWidth = lw; c.beginPath(); c.roundRect(x, y, w, h, r); c.stroke();
}
export const short = (_E: Env, w: string, _n = 5) => w; // labels are never truncated

export interface GridOpts {
  vis?: (i: number, j: number) => number; masked?: (i: number, j: number) => boolean; num?: (i: number, j: number) => string | null;
  tip?: (i: number, j: number) => string; row?: (i: number) => number | undefined; gap?: (j: number) => number;
}
/** Cell grid with optional per-cell opacity, masked hatching, numbers and hover text. Returns each column's x. */
export function grid(E: Env, x: number, y: number, cw: number, ch: number, R: number, C: number, val: (i: number, j: number) => number, col: (v: number) => RGB, o: GridOpts = {}) {
  const { c, P } = E; let gx = x;
  for (let j = 0; j < C; j++) {
    gx += o.gap ? o.gap(j) : 0;
    for (let i = 0; i < R; i++) {
      const v = val(i, j), a = o.vis ? o.vis(i, j) : 1, m = o.masked?.(i, j) ?? false, px = gx, py = y + i * ch;
      if (a <= 0.001) { c.strokeStyle = rgba(P.rule, 0.5); c.lineWidth = 1; c.strokeRect(px + 0.5, py + 0.5, cw - 1, ch - 1); continue; }
      const fill = col(v); c.fillStyle = rgba(fill, a * (m ? 0.5 : 1)); c.fillRect(px, py, cw - 1, ch - 1);
      if (m) hatch(E, px, py, cw - 1, ch - 1, 0.45);
      const n = o.num?.(i, j);
      if (n && a > 0.6) txt(E, n, px + cw / 2 - 0.5, py + ch / 2 + 3, { sz: Math.min(10, ch - 5), al: 'center', col: lum(fill) < 140 ? [255, 255, 255] : P.text, mono: true, a: m ? 0.6 : 1 });
      if (o.tip && a > 0.5) E.hits.push({ x: px, y: py, w: cw, h: ch, text: o.tip(i, j), row: o.row?.(i) });
    }
    gx += cw;
  }
  return gx;
}
/** A signed strip of values (horizontal, or vertical when `vert`). `vis` can be one opacity or one per cell. */
export function strip(E: Env, x: number, y: number, cw: number, ch: number, vals: ArrayLike<number>, vmax: number, o: { vert?: boolean; vis?: number | ((k: number) => number); gap?: (k: number) => number; tip?: (k: number) => string; scale?: number; nums?: boolean } = {}) {
  const n = vals.length, col = signed(E.P, vmax), vis = (k: number) => (typeof o.vis === 'function' ? o.vis(k) : o.vis ?? 1), sc = o.scale ?? 1;
  const v = (k: number) => vals[k] * sc;
  if (o.vert) grid(E, x, y, cw, ch, n, 1, (i) => v(i), col, { vis: (i) => vis(i), tip: o.tip && ((i) => o.tip!(i)), num: o.nums ? (i) => sgn(v(i), 1) : undefined });
  else grid(E, x, y, cw, ch, 1, n, (_, j) => v(j), col, { vis: (_, j) => vis(j), gap: o.gap, tip: o.tip && ((_, j) => o.tip!(j)), num: o.nums ? (_, j) => sgn(v(j), 1) : undefined });
}
export const maxAbs = (a: ArrayLike<number>) => { let m = 1e-9; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i])); return m; };

export interface BarOpts { signed?: boolean; vmax?: number; grow?: number; q?: number; fade?: number; cols?: (j: number) => RGB; label?: (j: number) => string | null; mark?: number; words?: boolean; tip?: (j: number) => string; baseTop?: boolean }
/** T bars in a w-by-h box. Unsigned bars grow from the bottom; signed from the middle. Entries after `q` are masked. */
export function bars(E: Env, x: number, y: number, w: number, h: number, vals: ArrayLike<number>, o: BarOpts) {
  const { c, P, s } = E, T = s.T, bw = w / T, vmax = o.vmax ?? 1, grow = o.grow ?? 1;
  const zero = o.signed ? y + h / 2 : y + h, scale = (o.signed ? h / 2 - 11 : h - 12) / vmax;
  c.strokeStyle = rgba(P.rule); c.lineWidth = 1; c.beginPath(); c.moveTo(x, zero + 0.5); c.lineTo(x + w, zero + 0.5); c.stroke();
  for (let j = 0; j < T; j++) {
    const masked = o.q !== undefined && j > o.q, v = vals[j] * grow, bx = x + j * bw + bw * 0.18, bwid = bw * 0.64;
    const hgt = v * scale, col = o.cols ? o.cols(j) : o.signed ? (v >= 0 ? P.pos : P.neg) : seqColor(P, 0.45 + 0.55 * Math.abs(vals[j]));
    const alpha = masked ? 0.35 * (1 - (o.fade ?? 0)) : 1;
    if (Math.abs(hgt) > 0.3) { c.fillStyle = rgba(col, alpha); c.fillRect(bx, hgt >= 0 ? zero - hgt : zero, bwid, Math.abs(hgt)); }
    if (masked) { hatch(E, bx, zero - 10, bwid, 10, 0.4 * (1 - (o.fade ?? 0))); }
    const lab = o.label?.(j);
    if (lab && alpha > 0.5 && grow > 0.05) txt(E, lab, bx + bwid / 2, hgt >= 0 ? zero - hgt - 3 : zero - hgt + 11, { sz: E.narrow ? 8.5 : 9.5, al: 'center', col: P.muted, mono: true, a: Math.min(1, grow * 1.5) });
    if (o.tip) E.hits.push({ x: x + j * bw, y, w: bw, h, text: o.tip(j) });
    if (o.mark === j) { c.fillStyle = rgba(P.accent); const mx = bx + bwid / 2, my = o.signed ? y + 2 : zero - Math.abs(hgt) - 16; c.beginPath(); c.moveTo(mx - 4, my); c.lineTo(mx + 4, my); c.lineTo(mx, my + 6); c.fill(); }
  }
}
export function wordsRow(E: Env, x: number, y: number, w: number, o: { hi?: number; dim?: (j: number) => number; cls?: boolean; sz?: number } = {}) {
  const { s, P } = E, bw = w / s.T;
  for (let j = 0; j < s.T; j++) {
    const cx = x + (j + 0.5) * bw, a = o.dim ? o.dim(j) : 1, label = short(E, s.words[j], E.narrow ? 6 : 8);
    txt(E, label, cx, y, { sz: o.sz ?? (E.narrow ? 9 : 11), al: 'center', bold: o.hi === j, col: o.hi === j ? P.accent : P.text, a });
    const k = classOf(s.ids[j]);
    if (o.cls && k >= 0) { E.c.fillStyle = rgba(P.cat[k], a); E.c.fillRect(cx - bw * 0.3, y + 4, bw * 0.6, 3); }
  }
}
export const quote = (w: string) => `“${w}”`;
