/**
 * BanditLab: k-armed bandits (src/lib/rl/bandit.ts) on one scrubbable timeline, one mode per lesson step.
 *
 * MODE-AUTHOR CONTRACT: identical to GridWorldLab (read the comment at the top of ../GridWorldLab/sim.ts). You own
 * m-<mode>.ts and m-<mode>.meta.ts here and never edit BanditLab.svelte, ../_rl/*, this file, draw.ts or src/lib/rl.
 * Draw helpers: ../_rl/svg.ts (axes, lineChart, bars, binCounts, table, text, ...) and ./draw.ts (armBars).
 * Props named by the briefs: arms, epsilons ("0,0.1"; 0 means greedy), ucb, c, runs, steps, seed, lr, samples,
 * baseline, means ("1,2,3"). Measured: the 3-strategy x 50-run x 500-step testbed takes about 37 ms (memoised per
 * page by strategiesOf + kit.memo); a 400-sample gradient set or a 200-step softmax PG run is under 5 ms.
 */
import type { Strategy } from '../../../lib/rl/bandit';
import { bool, nums, num, type Opts } from '../_rl/kit';

/** Strategies from `epsilons` (0 = greedy) plus UCB when `ucb` is true. */
export function strategiesOf(o: Opts): Strategy[] {
  const s: Strategy[] = nums(o.epsilons, [0, 0.1]).map((e) => (e === 0 ? { kind: 'greedy' } : { kind: 'epsilon', epsilon: e }));
  if (bool(o.ucb, true)) s.push({ kind: 'ucb', c: num(o.c, 2) });
  return s;
}
/** Categorical colour per strategy slot (fixed order). */
export const strategyColor = (i: number): string => `var(--viz-cat-${i + 1})`;
