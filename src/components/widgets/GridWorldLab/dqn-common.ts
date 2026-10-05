/** Shared by the six deep-rl modes (m-dqn-*.ts): the grid with gamma 0.9, the colour of each DQN variant, and the network drawing. */
import { ARROWS, idx } from '../../../lib/rl/gridworld';
import type { Transition } from '../../../lib/rl/dqn';
import { forward, offsets } from '../../../lib/rl/dqn';
import type { Opts } from '../_rl/kit';
import { BG, MUTED, RULE, circle, fmt, line, seqFill, signedFill, text, type Box, type Env } from '../_rl/svg';
import { gridOf } from './sim';

export const GAMMA = 0.9;
/** The diagnostic transition the instability and target modes track: (3,1) up. */
export const DIAG: Transition = { s: idx(3, 1), a: 0, r: -0.04, s2: idx(3, 2), done: false };
export const dqnGrid = (o: Opts) => gridOf(o, GAMMA);
/** Fixed identity colour per variant (always labelled). */
export const VARIANT_COLOR: Record<string, string> = { naive: 'var(--viz-cat-1)', replay: 'var(--viz-cat-2)', target: 'var(--viz-cat-3)', both: 'var(--viz-cat-4)' };
export const VARIANT_NAME: Record<string, string> = { naive: 'Neither', replay: 'Replay only', target: 'Target network only', both: 'Replay + target' };

/**
 * The 2 -> H -> 4 network for input (x0, x1): nodes coloured by value, edges by the signed contribution w * input
 * (hidden units at 0 send nothing, which is the ReLU). `rev` fades the input, hidden and output layers in (0..1 each).
 */
export function netDiagram(E: Env, b: Box, w: Float64Array, H: number, x0: number, x1: number, rev = { i: 1, h: 1, o: 1 }) {
  const q = new Float64Array(4), h = new Float64Array(H), pre = new Float64Array(H);
  forward(w, H, x0, x1, q, h, pre);
  const o = offsets(H), nar = E.narrow;
  const xi = b.x + (nar ? 96 : 100), xo = b.x + b.w - (nar ? 58 : 74), xh = (xi + xo) / 2, top = b.y + 28, area = b.h - 34;
  const yi = [top + area * 0.33, top + area * 0.67], yh = (j: number) => top + ((j + 0.5) / H) * area, yq = (a: number) => top + ((a + 0.5) / 4) * area;
  const rh = Math.max(3, Math.min(7, (area / H) * 0.4)), x = [x0, x1];
  text(E, 'Input', xi, b.y + 12, { anchor: 'middle', fill: MUTED, size: 11 });
  text(E, `${H} hidden units`, xh, b.y + 12, { anchor: 'middle', fill: MUTED, size: 11 });
  text(E, 'Q-values', xo + 24, b.y + 12, { anchor: 'middle', fill: MUTED, size: 11 });
  let c1 = 1e-9, c2 = 1e-9;
  for (let j = 0; j < H; j++) { for (let i = 0; i < 2; i++) c1 = Math.max(c1, Math.abs(w[o.W1 + 2 * j + i] * x[i])); for (let a = 0; a < 4; a++) c2 = Math.max(c2, Math.abs(w[o.W2 + a * H + j] * h[j])); }
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < 2; i++) {
      const c = w[o.W1 + 2 * j + i] * x[i], u = Math.abs(c) / c1;
      if (u > 0.04) line(E, xi, yi[i], xh, yh(j), { stroke: c > 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', sw: 0.6 + 1.6 * u, op: rev.h * (0.15 + 0.6 * u) });
    }
    for (let a = 0; a < 4; a++) {
      const c = w[o.W2 + a * H + j] * h[j], u = Math.abs(c) / c2;
      if (u > 0.04) line(E, xh, yh(j), xo, yq(a), { stroke: c > 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', sw: 0.6 + 1.6 * u, op: rev.o * (0.15 + 0.6 * u) });
    }
  }
  const hmax = Math.max(1e-9, ...Array.from(h));
  for (let i = 0; i < 2; i++) {
    circle(E, xi, yi[i], 9, { fill: signedFill(x[i], 1), stroke: RULE, sw: 1, op: 0.25 + 0.75 * rev.i });
    text(E, `${i ? 'row' : 'column'} ${fmt(x[i])}`, xi - 14, yi[i] + 4, { anchor: 'end', size: 11, op: rev.i });
  }
  for (let j = 0; j < H; j++) circle(E, xh, yh(j), rh, { fill: h[j] > 0 ? seqFill(0.25 + 0.75 * (h[j] / hmax)) : BG, stroke: RULE, sw: 1, op: 0.25 + 0.75 * rev.h });
  for (let a = 0; a < 4; a++) {
    circle(E, xo, yq(a), 9, { fill: signedFill(q[a], 1.3), stroke: RULE, sw: 1, op: 0.25 + 0.75 * rev.o });
    text(E, `${ARROWS[a]} ${fmt(q[a])}`, xo + 14, yq(a) + 4, { size: 11, op: rev.o });
  }
  return q;
}
