/**
 * PreferenceLab (rlhf, dpo, constitutional-ai): preference pairs, the Bradley-Terry reward model, PPO against it with
 * the KL penalty to the SFT reference, the beta sweep and over-optimisation, the closed-form optimum and DPO.
 *
 * MODE-AUTHOR CONTRACT: identical to GridWorldLab (read the comment at the top of ../GridWorldLab/sim.ts). You own
 * m-<mode>.ts and m-<mode>.meta.ts here and never edit PreferenceLab.svelte, ../_rl/*, ../_align/*, this file or
 * src/lib/align. kit.job runs any export of src/lib/rl OR src/lib/align in the shared worker.
 * Reference mode: m-pairs.ts (+ meta), labels "human" | "ai". Draw helpers: ../_rl/svg.ts and ../_align/draw.ts
 * (candidateBars with ghost reference bars + ticks, featureBars, rewardChart with proxy / gold / faint comparison gold
 * and a peak mark, responseCard, candColor). Colours: proxy = --viz-forward line, gold = --text dashed (plans/day-11.md §3).
 *
 * SHARED MODES: `pairs` and `ppo` serve both rlhf and constitutional-ai. m-ppo must accept `labels` ("human" | "ai")
 * and an optional faint comparison run (`compare="human"`: draw that run's gold with rewardChart's goldCompare) so the
 * constitutional-ai lesson needs no edit to it.
 *
 * Data helpers below (memoised per page, so islands of one page share the same pairs and reward model):
 *   prefData(o, kit, pairSeed?) -> { sft, pairs (labels of o.labels), human, ai (same pairs, both labelers), held (1000 held-out
 *                         human pairs, seed 99), rm (fitRM on the chosen labels: w, ws per step, loss) }. About 25 ms. Pair seed: see below.
 *   ppoRun(o, kit, x) -> kit.job('trainAlignPPO', ...) with the brief props (beta 0.1, iters 200, batch 64, seed 6,
 *                         epochs 4, eps 0.2, lr 0.04) overridden by x; 45–75 ms per run in Node, so always a job.
 * Other align exports: trainDPO (lr defaults to 0.05 / beta: from theta_ref its implicit weights follow the reward
 * model's fit step for step), dpoMargin, implicitReward, optimalTheta, optimalRow (pi* and Z(x)), kl, expected, GOLD,
 * scores, meanProb, humanPrefProb, labelerUtility, LABELER (bias, cap), W_GOLD. DPO 300 steps 45–55 ms: kit.job('trainDPO', pairs, sft, { beta, steps }).
 * Props the briefs name: labels, pairs (200), seed (4 for pairs, 6 for PPO; pairSeed when both are needed), steps (200 RM, 300 DPO), beta, betas, iters, batch, iter, epochs,
 * eps, pair, compare. Measured numbers for every step: plans/day-11.md §7.
 */
import { sftTheta } from '../../../lib/align/sft';
import { fitRM, samplePairs, type Pair, type RMFit } from '../../../lib/align/reward';
import type { AlignPPOOpts, AlignPPOResult } from '../../../lib/align/ppo';
import { int, key, num, str, type Kit, type Opts } from '../_rl/kit';

export type { Opts };
export const LABELS = ['human', 'ai'] as const;
export type Labels = (typeof LABELS)[number];
export const labelsOf = (o: Opts): Labels => str(o.labels, LABELS, 'human');

export interface PrefData { sft: Float64Array; pairs: Pair[]; human: Pair[]; ai: Pair[]; held: Pair[]; rm: RMFit; labels: Labels; n: number; seed: number }
/**
 * Pairs and reward model. The pair seed is `pairSeed` (default 4) unless the caller passes one: pair-centred modes
 * (pairs, reward-model, rm-bias) pass `int(o.seed, 4)`, while PPO-centred modes keep `seed` for PPO (the briefs use seed 4
 * for pairs and seed 6 for PPO). Reward-model steps: `rmSteps` (default 200).
 */
export function prefData(o: Opts, kit: Kit, pairSeed = int(o.pairSeed, 4)): PrefData {
  const labels = labelsOf(o), n = int(o.pairs, 200), seed = pairSeed, steps = int(o.rmSteps, 200);
  return kit.memo(key('pref-data', labels, n, seed, steps), () => {
    const sft = sftTheta(), human = samplePairs(sft, n, seed, 'human'), ai = samplePairs(sft, n, seed, 'ai');
    const pairs = labels === 'human' ? human : ai;
    return { sft, pairs, human, ai, held: kit.memo(key('pref-held'), () => samplePairs(sft, 1000, 99, 'human')), rm: fitRM(pairs, { steps }), labels, n, seed };
  });
}
/** PPO against the reward model of prefData(o) in the worker, with the brief defaults overridden by x. */
export function ppoRun(o: Opts, kit: Kit, x: Partial<AlignPPOOpts> = {}): Promise<AlignPPOResult> {
  const d = prefData(o, kit);
  return kit.job<AlignPPOResult>('trainAlignPPO', {
    rm: Array.from(d.rm.w), beta: num(o.beta, 0.1), iters: int(o.iters, 200), batch: int(o.batch, 64), seed: int(o.seed, 6),
    epochs: int(o.epochs, 4), eps: num(o.eps, 0.2), lr: num(o.ppoLr, 0.04), ...x,
  });
}
