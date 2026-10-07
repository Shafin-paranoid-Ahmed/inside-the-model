/**
 * Drawing helpers shared by the Day 12 shells (SFTLab, PreferenceLab, CritiqueRevise, VerifierRL). They push shapes
 * through ../_rl/svg.ts like every Day 11 mode; no DOM. Colours: candidate identities use --viz-cat-1..8 in the bank's
 * fixed candidate order (always with a label), policy probabilities --viz-forward, proxy reward a --viz-forward line,
 * gold reward a --text dashed line, KL --text-muted (plans/day-11.md §3).
 */
import { CANDIDATES, FEATURE_LABELS } from '../../../lib/align/bank';
import { ACCENT, BG, MUTED, RULE, TEXT, axes, clamp, fmt, line, mix, path, rect, text, type Axis, type Box, type Env } from '../_rl/svg';

/** The identity colour of candidate y (fixed order, never cycled). */
export const candColor = (y: number): string => `var(--viz-cat-${y + 1})`;
export const FORWARD = 'var(--viz-forward)', PROXY = 'var(--viz-forward)', GOLD_INK = TEXT, GOLD_DASH = '6 4';

/** Characters of UI text that fit in px at font size `size` (a conservative average glyph width). */
export const charsFor = (px: number, size: number, mono = false): number => Math.max(4, Math.floor(px / (size * (mono ? 0.61 : 0.53))));
/** Shorten to n characters with an ellipsis. */
export const clip = (s: string, n: number): string => (s.length <= n ? s : s.slice(0, Math.max(1, n - 1)).trimEnd() + '…');
/** Greedy word wrap into lines of at most n characters (keeps explicit newlines). */
export function wrap(s: string, n: number): string[] {
  const out: string[] = [];
  for (const para of s.split('\n')) {
    let cur = '';
    for (const w of para.split(' ')) {
      if (!cur) cur = w;
      else if ((cur + ' ' + w).length <= n) cur += ' ' + w;
      else { out.push(cur); cur = w; }
    }
    out.push(cur);
  }
  return out;
}

export interface CandidateBarsOpts {
  /** Probabilities of the 8 candidates (drawn as --viz-forward bars; may be mid-tween). */
  p: ArrayLike<number>;
  /** The real state whose numbers are printed (default p). Pass it while p tweens so labels never show in-between values. */
  values?: ArrayLike<number>;
  /** Optional reference probabilities drawn as ghost (outlined) bars behind, e.g. the SFT reference. */
  ref?: ArrayLike<number>;
  /** Optional tick marks on each row, e.g. another run's end state. */
  ticks?: ArrayLike<number>;
  /** Row label per candidate: its text for the shown prompt (clipped to fit), else the candidate type name. */
  texts?: string[];
  /** Candidate to outline with --accent (e.g. the sampled response or the exploited answer). */
  highlight?: number | null;
  /** Fade every other row (e.g. only the highlighted row matters now). */
  focus?: boolean;
  /** Upper end of the probability axis (default 1). */
  max?: number;
  digits?: number;
}
/**
 * One row per candidate: a colour chip with the type name, the response text (if given), a probability bar with an
 * optional ghost reference bar and tick, and the value. Returns the row height and the bar's x range.
 */
export function candidateBars(E: Env, b: Box, o: CandidateBarsOpts) {
  const n = CANDIDATES.length, rh = b.h / n, nw = E.narrow, sz = nw ? 11 : 12, max = o.max ?? 1, dg = o.digits ?? 2;
  const labelW = nw ? Math.min(b.w * 0.5, 150) : Math.min(b.w * 0.5, 300), valW = 38, bx = b.x + labelW + 8, bw = Math.max(30, b.w - labelW - 8 - valW);
  const sx = (v: number) => bx + (clamp(v, 0, max) / max) * bw, bh = Math.min(16, rh * 0.55);
  for (let y = 0; y < n; y++) {
    const ry = b.y + y * rh, cy = ry + rh / 2, op = o.focus && o.highlight !== y ? 0.35 : 1;
    if (o.highlight === y) rect(E, b.x - 4, ry + 1, b.w + 8, rh - 2, { fill: 'none', stroke: ACCENT, sw: 2, rx: 4 });
    rect(E, b.x, cy - 5, 10, 10, { fill: candColor(y), rx: 2, op });
    const name = CANDIDATES[y].short, t = o.texts?.[y];
    if (t && !nw) {
      text(E, name, b.x + 16, cy - 2, { size: 11, fill: MUTED, op });
      text(E, clip(t, charsFor(labelW - 16, 12)), b.x + 16, cy + 11, { size: 12, op });
    } else text(E, clip(name, charsFor(labelW - 16, sz)), b.x + 16, cy + 4, { size: sz, op });
    rect(E, bx, cy - bh / 2, bw, bh, { fill: 'none', stroke: RULE, sw: 1, op });
    if (o.ref) rect(E, bx, cy - bh / 2 - 2, sx(o.ref[y]) - bx, bh + 4, { fill: 'none', stroke: MUTED, sw: 1.25, dash: '3 2', op });
    rect(E, bx, cy - bh / 2, sx(o.p[y]) - bx, bh, { fill: FORWARD, op: 0.9 * op });
    if (o.ticks) line(E, sx(o.ticks[y]), cy - bh / 2 - 4, sx(o.ticks[y]), cy + bh / 2 + 4, { stroke: TEXT, sw: 2, op });
    text(E, fmt((o.values ?? o.p)[y], dg), b.x + b.w, cy + 4, { size: sz, anchor: 'end', op });
  }
  return { rh, bx, bw, sx };
}

export interface FeatureBarsOpts {
  /** Signed weights per feature (theta, reward-model weights, ...). */
  v: ArrayLike<number>;
  /** Optional comparison weights drawn as a thin dark line per row (e.g. the gold weights or theta_ref). */
  ref?: ArrayLike<number>;
  /** Symmetric axis limit; bars beyond it are clipped. */
  lim: number;
  /** Row labels (default FEATURE_LABELS); colour default: --viz-pos / --viz-neg by sign. */
  labels?: readonly string[];
  title?: string;
  digits?: number;
}
/** Horizontal signed bars, one per feature, around a zero line, with the value at the end of each bar. */
export function featureBars(E: Env, b: Box, o: FeatureBarsOpts) {
  const labels = o.labels ?? FEATURE_LABELS, n = labels.length, sz = E.narrow ? 11 : 12, top = o.title ? 18 : 0;
  if (o.title) text(E, o.title, b.x, b.y + 12, { size: sz, fill: MUTED });
  const lw = E.narrow ? 64 : 76, x0 = b.x + lw, w = b.w - lw - 6, cx = x0 + w / 2, rh = (b.h - top) / n, bh = Math.min(14, rh * 0.55);
  const sx = (v: number) => cx + (clamp(v, -o.lim, o.lim) / o.lim) * (w / 2);
  line(E, cx, b.y + top, cx, b.y + b.h, { stroke: RULE, sw: 1 });
  for (let j = 0; j < n; j++) {
    const cy = b.y + top + j * rh + rh / 2, v = o.v[j];
    text(E, labels[j], b.x, cy + 4, { size: sz });
    rect(E, Math.min(cx, sx(v)), cy - bh / 2, Math.abs(sx(v) - cx), bh, { fill: v >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', op: 0.9 });
    if (o.ref) line(E, sx(o.ref[j]), cy - bh / 2 - 3, sx(o.ref[j]), cy + bh / 2 + 3, { stroke: TEXT, sw: 2 });
    const tx = sx(v) + (v >= 0 ? 4 : -4), inside = Math.abs(sx(v) - cx) > w / 2 - 34;
    text(E, fmt(v, o.digits ?? 2), inside ? (v >= 0 ? sx(v) - 4 : sx(v) + 4) : tx, cy + 4, { size: 11, anchor: (v >= 0) !== inside ? 'start' : 'end', fill: inside ? BG : TEXT, halo: !inside });
  }
}

export interface RewardChartOpts {
  /** Per-iteration series (index = iteration). Any may be omitted. */
  proxy?: ArrayLike<number>; gold?: ArrayLike<number>;
  /** A faint comparison gold curve (e.g. the human-label run in the constitutional-ai lesson). */
  goldCompare?: ArrayLike<number>;
  /** Draw points up to this x (animate it to trace the curves). */
  upto: number;
  x: Axis; y: Axis;
  /** x values of the series when they are not the indices (e.g. sqrt KL). */
  xs?: ArrayLike<number>;
  /** Mark this index on the gold curve (e.g. its peak) once it is drawn. */
  mark?: { i: number; label: string };
}
/** Proxy (solid --viz-forward) and gold (dashed --text) reward curves on one axis, drawn up to `upto`. */
export function rewardChart(E: Env, b: Box, o: RewardChartOpts) {
  const { sx, sy } = axes(E, b, o.x, o.y), X = (i: number) => (o.xs ? o.xs[i] : i);
  const trace = (ys: ArrayLike<number> | undefined, st: { stroke: string; sw: number; dash?: string; op?: number }) => {
    if (!ys) return;
    let d = '';
    for (let i = 0; i < ys.length && X(i) <= o.upto + 1e-9; i++) d += `${d ? 'L' : 'M'}${sx(X(i)).toFixed(1)} ${clamp(sy(ys[i]), b.y, b.y + b.h).toFixed(1)}`;
    if (d) path(E, d, st);
  };
  trace(o.goldCompare, { stroke: MUTED, sw: 1.5, dash: GOLD_DASH, op: 0.6 });
  trace(o.proxy, { stroke: PROXY, sw: 2.25 });
  trace(o.gold, { stroke: GOLD_INK, sw: 2, dash: GOLD_DASH });
  if (o.mark && o.gold && X(o.mark.i) <= o.upto) {
    const mx = sx(X(o.mark.i)), my = sy(o.gold[o.mark.i]);
    line(E, mx, b.y, mx, b.y + b.h, { stroke: ACCENT, sw: 1.25, dash: '3 3' });
    text(E, o.mark.label, mx + (mx > b.x + b.w * 0.7 ? -6 : 6), Math.max(b.y + 14, my - 10), { size: 11, anchor: mx > b.x + b.w * 0.7 ? 'end' : 'start', fill: TEXT, halo: true });
  }
  return { sx, sy };
}

export interface CardOpts { title?: string; text: string; size?: number; mono?: boolean; accent?: boolean; tag?: string; fill?: string; op?: number; maxLines?: number }
/** A response card: optional muted title, wrapped text, optional tag on the top right; returns the height used. */
export function responseCard(E: Env, b: Box, o: CardOpts): number {
  const sz = o.size ?? (E.narrow ? 12 : 13), lh = sz * 1.35, pad = 8, n = charsFor(b.w - 2 * pad, sz, o.mono);
  let lines = wrap(o.text, n);
  if (o.maxLines && lines.length > o.maxLines) lines = [...lines.slice(0, o.maxLines - 1), clip(lines.slice(o.maxLines - 1).join(' '), n)];
  const h = pad * 2 + (o.title ? lh : 0) + lines.length * lh;
  rect(E, b.x, b.y, b.w, Math.max(b.h, h), { fill: o.fill ?? mix(TEXT, BG, 4), stroke: o.accent ? ACCENT : RULE, sw: o.accent ? 2 : 1, rx: 6, op: o.op });
  let y = b.y + pad + sz;
  if (o.title) { text(E, o.title, b.x + pad, y, { size: 11, fill: MUTED, op: o.op }); y += lh; }
  if (o.tag) text(E, o.tag, b.x + b.w - pad, b.y + pad + 11, { size: 11, anchor: 'end', weight: 650, fill: o.accent ? TEXT : MUTED, op: o.op });
  for (const l of lines) { text(E, l, b.x + pad, y, { size: sz, mono: o.mono, op: o.op }); y += lh; }
  return Math.max(b.h, h);
}
