/** Bandit drawing helpers (no DOM). */
import { ACCENT, MUTED, RULE, TEXT, fmt, line, rect, scale, text, type Box, type Env } from '../_rl/svg';

/**
 * One column per arm: the estimated mean as a bar from 0, the true mean as a tick, the pull count under it.
 * `hi` outlines one arm (e.g. the arm pulled now) in the accent colour.
 */
export function armBars(E: Env, b: Box, o: { est: ArrayLike<number>; truth: ArrayLike<number>; counts?: ArrayLike<number>; lo: number; hi: number; pulled?: number; best?: number }) {
  const k = o.truth.length, sy = scale({ lo: o.lo, hi: o.hi }, b.y + b.h, b.y), bw = b.w / k, y0 = sy(0);
  line(E, b.x, y0, b.x + b.w, y0, { stroke: RULE, sw: 1 });
  text(E, '0', b.x - 6, y0 + 4, { anchor: 'end', fill: MUTED });
  for (let a = 0; a < k; a++) {
    const x = b.x + a * bw + bw * 0.2, w = bw * 0.6, ye = sy(Math.max(o.lo, Math.min(o.hi, o.est[a]))), yt = sy(o.truth[a]);
    rect(E, x, Math.min(ye, y0), w, Math.abs(ye - y0), { fill: 'var(--viz-forward)', op: 0.85 });
    line(E, x - 4, yt, x + w + 4, yt, { stroke: TEXT, sw: 2.5 });
    if (o.pulled === a) rect(E, x - 3, b.y, w + 6, b.h, { fill: 'none', stroke: ACCENT, sw: 2 });
    const cx = x + w / 2;
    text(E, `Arm ${a + 1}${o.best === a ? ' (best)' : ''}`, cx, b.y + b.h + 16, { anchor: 'middle', size: E.narrow ? 11 : 12, weight: o.best === a ? 650 : 400 });
    if (o.counts) text(E, `${o.counts[a]} ${o.counts[a] === 1 ? 'pull' : 'pulls'}`, cx, b.y + b.h + 31, { anchor: 'middle', size: 11, fill: MUTED });
    text(E, fmt(o.est[a]), cx, Math.min(ye, y0) - 5, { anchor: 'middle', size: 11, fill: MUTED, halo: true });
  }
}
