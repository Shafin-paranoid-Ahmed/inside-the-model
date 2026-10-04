/**
 * PPOLab: PPO's clipped surrogate and its batch drift diagnostics (src/lib/rl/ppo.ts) on one scrubbable timeline.
 *
 * MODE-AUTHOR CONTRACT: identical to GridWorldLab (read the comment at the top of ../GridWorldLab/sim.ts). You own
 * m-<mode>.ts and m-<mode>.meta.ts here and never edit PPOLab.svelte, ../_rl/*, this file, draw.ts or src/lib/rl.
 * Draw helpers: ../_rl/svg.ts and ./draw.ts (surrogatePlot: unclipped/clipped curves vs ratio with the clip band and
 * a cursor; probability bars and histograms come from svg.ts `bars` / `binCounts`).
 * Props named by the briefs: advantage, epsilon, epsilons ("0.1,0.2,0.4"), ratios ("0.5,1.6" = the plotted range),
 * epochs, batch, minibatch, lr, seed. `clippedSurrogate(rho, A, eps)` gives value, d/d rho and the branch exactly.
 * Measured: driftRuns(g, { warmIterations: 10, seed: 31, lr: 0.03 }) takes about 8 ms (memo is enough); its batch has
 * 32 positive and 32 negative advantages (a batch from the uniform policy has all 64 negative). Numbers: day-11.md §6.
 */
import { clippedSurrogate } from '../../../lib/rl/ppo';
import { nums, num, type Opts } from '../_rl/kit';

export { clippedSurrogate };
/** Plotted ratio range from `ratios` (default 0.5..1.6), the advantage and epsilon. */
export const curveOpts = (o: Opts, A = 1) => {
  const [lo, hi] = nums(o.ratios, [0.5, 1.6]);
  return { lo, hi, A: num(o.advantage, A), eps: num(o.epsilon, 0.2) };
};
