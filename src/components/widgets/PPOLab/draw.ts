/** PPO drawing helpers (no DOM). */
import { clippedSurrogate } from '../../../lib/rl/surrogate';
import { ACCENT, BG, MUTED, axes, circle, line, mix, path, rect, text, type Box, type Env } from '../_rl/svg';

/**
 * The PPO-clip objective of one sample against the ratio rho in [lo, hi]: the unclipped rho A (dashed) and the
 * clipped min(...) (solid, --viz-pos for A > 0, --viz-neg for A < 0), the shaded band [1 - eps, 1 + eps], and an
 * optional cursor at `rho`. Curves are drawn up to `upto` (animate it to trace them).
 */
export function surrogatePlot(E: Env, b: Box, o: { A: number; eps: number; lo: number; hi: number; rho?: number; upto?: number }) {
  const { A, eps, lo, hi } = o, n = 120, ys = [lo * A, hi * A, (1 - eps) * A, (1 + eps) * A];
  const ya = { lo: Math.min(...ys) - 0.1, hi: Math.max(...ys) + 0.1, label: 'Objective of this sample' };
  const { sx, sy } = axes(E, b, { lo, hi, label: 'Probability ratio ρ = π_new / π_old' }, ya);
  rect(E, sx(1 - eps), b.y, sx(1 + eps) - sx(1 - eps), b.h, { fill: mix(ACCENT, BG, 18) });
  for (const r of [1 - eps, 1 + eps]) line(E, sx(r), b.y, sx(r), b.y + b.h, { stroke: ACCENT, sw: 1 });
  text(E, `clip band 1 ± ${eps}`, sx(1 - eps) + 6, b.y + 14, { size: 11, fill: MUTED });
  line(E, sx(1), b.y, sx(1), b.y + b.h, { stroke: MUTED, sw: 1, dash: '3 3' });
  const upto = o.upto ?? hi, col = A >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)';
  let du = '', dc = '';
  for (let i = 0; i <= n; i++) {
    const r = lo + ((hi - lo) * i) / n;
    if (r > upto + 1e-9) break;
    const X = sx(r).toFixed(1);
    du += `${du ? 'L' : 'M'}${X} ${sy(r * A).toFixed(1)}`;
    dc += `${dc ? 'L' : 'M'}${X} ${sy(clippedSurrogate(r, A, eps).value).toFixed(1)}`;
  }
  if (du) path(E, du, { stroke: MUTED, sw: 1.75, dash: '6 4' });
  if (dc) path(E, dc, { stroke: col, sw: 2.75 });
  if (o.rho !== undefined) {
    const c = clippedSurrogate(o.rho, A, eps), x = sx(o.rho);
    line(E, x, b.y, x, b.y + b.h, { stroke: ACCENT, sw: 1.25 });
    circle(E, x, sy(o.rho * A), 4, { fill: BG, stroke: MUTED, sw: 2 });
    circle(E, x, sy(c.value), 5.5, { fill: col, stroke: BG, sw: 2 });
  }
  return { sx, sy };
}
