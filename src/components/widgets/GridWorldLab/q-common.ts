/** Shared by the q-learning lesson's modes (qvalues, bellman, qlearn, qlearn-curve, qlearn-terminal). No DOM. */
import { ACTIONS, ARROWS, NA, cellName, idx, type Grid } from '../../../lib/rl/gridworld';
import { optimal, qLearning, type QLearningResult } from '../../../lib/rl/tabular';
import { int, key, num, type Kit, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, TEXT, fmt, line, rect, signedFill, text, type Box, type Env } from '../_rl/svg';

/** Action cells in reading order (top row first). */
export const READING: number[] = [idx(1, 3), idx(2, 3), idx(3, 3), idx(1, 2), idx(3, 2), idx(1, 1), idx(2, 1), idx(3, 1), idx(4, 1)];
/** Reading order, but the start cell first (the cell the lesson quotes). */
export const START_FIRST: number[] = [idx(1, 1), ...READING.filter((s) => s !== idx(1, 1))];

/** Best and second-best action of a Q row, and the gap between them. */
export function rank(Q: ArrayLike<number>, s: number) {
  const o = [0, 1, 2, 3].sort((a, b) => Q[s * NA + b] - Q[s * NA + a]);
  return { best: o[0], second: o[1], gap: Q[s * NA + o[0]] - Q[s * NA + o[1]], order: o };
}
/** Q* of the grid the options describe (cached per page). */
export const qStar = (kit: Kit, g: Grid) => kit.memo(key('qstar', g.stepReward, g.slip, g.gamma), () => optimal(g));
/** One recorded Q-learning run for the options (alpha, epsilon, episodes, seed; gamma from the grid), cached per page. */
export function qRun(kit: Kit, g: Grid, o: Opts, d: { alpha: number; epsilon: number; episodes: number; seed: number }): QLearningResult {
  const alpha = num(o.alpha, d.alpha), epsilon = num(o.epsilon, d.epsilon), episodes = int(o.episodes, d.episodes), seed = int(o.seed, d.seed);
  return kit.memo(key('qrun', g.stepReward, g.slip, g.gamma, alpha, epsilon, episodes, seed), () => qLearning(g, { alpha, epsilon, episodes, seed }));
}
export const qName = (s: number, a: number) => `${cellName(s)} ${ACTIONS[a]}`;

/** The four action values of one cell as signed bars (best in accent outline), a header above. `op` fades it. */
export function actionBars(E: Env, b: Box, Q: ArrayLike<number>, s: number, o: { op?: number; digits?: number; vmax?: number; title?: string } = {}) {
  const op = o.op ?? 1, vmax = o.vmax ?? 1, d = o.digits ?? 3, { best } = rank(Q, s), rh = 24, lab = 66, half = (b.w - lab - 44) / 2, zero = b.x + lab + half;
  text(E, o.title ?? `Action values at ${cellName(s)}`, b.x, b.y + 12, { size: 13, weight: 650, op });
  line(E, zero, b.y + 22, zero, b.y + 22 + rh * NA, { stroke: RULE, sw: 1, op });
  for (let a = 0; a < NA; a++) {
    const y = b.y + 24 + a * rh, v = Q[s * NA + a], w = Math.min(1, Math.abs(v) / vmax) * half;
    text(E, `${ARROWS[a]} ${ACTIONS[a]}`, b.x, y + 14, { size: 12, fill: a === best ? TEXT : MUTED, weight: a === best ? 650 : 400, op });
    rect(E, v >= 0 ? zero : zero - w, y + 3, w, rh - 8, { fill: signedFill(v, vmax), stroke: a === best ? ACCENT : 'none', sw: 2, op });
    text(E, fmt(v, d), b.x + b.w, y + 14, { size: 12, anchor: 'end', weight: a === best ? 650 : 400, op });
  }
  return b.y + 24 + NA * rh;
}
