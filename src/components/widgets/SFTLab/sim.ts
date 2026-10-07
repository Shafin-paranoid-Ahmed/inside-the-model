/**
 * SFTLab (sft lesson): the day's candidate policy before and after supervised fine-tuning, on one scrubbable timeline.
 *
 * MODE-AUTHOR CONTRACT: identical to GridWorldLab (read the comment at the top of ../GridWorldLab/sim.ts). You own
 * m-<mode>.ts and m-<mode>.meta.ts here and never edit SFTLab.svelte, ../_rl/*, ../_align/*, this file or src/lib/align.
 * kit.job runs any export of src/lib/rl OR src/lib/align in the shared worker (args/results are structured clones).
 * Reference mode: m-base.ts (+ meta). Draw helpers: ../_rl/svg.ts and ../_align/draw.ts (candidateBars with ghost
 * reference bars, ticks and real-state `values`; featureBars for theta; rewardChart; responseCard; wrap/clip/charsFor;
 * candColor(y) = --viz-cat-(y+1), always labelled).
 *
 * src/lib/align exports a mode here needs: bank (PROMPTS, TEXT, CANDIDATES, FEATURE_LABELS, TRAIN, HELDOUT, THETA_BASE,
 * Y_* candidate ids, phi), policy (probsInto, probs, meanProb, expected, expectedFeatures, GOLD), sft (makeDemos,
 * demoFeatures, nllGrad, trainSFT, sftTheta, SFT_DEFAULTS, DEMO_STYLE), bigram (trainBigram, tokenLosses, counted,
 * TRAIN_TRANSCRIPTS, HELD_TRANSCRIPTS, VOCAB). Props the briefs name: demos (24), demoAccuracy (0.8), seed (2), steps
 * (150; cap 300), lr (0.05), demoAccuracies ("0.6,0.8,0.95").
 * Measured setup (Node): trainSFT 150 steps 10 ms (sftRun below memoises it: fine on the main thread); the three 300-step
 * cap runs 13 ms together (kit.memo);
 * trainBigram 300 steps 100–135 ms per mask (kit.job('trainBigram', 'all' | 'response')). Numbers: plans/day-11.md §7.
 */
import { makeDemos, trainSFT, SFT_DEFAULTS, type Demo, type SFTRun } from '../../../lib/align/sft';
import { int, key, num, type Kit, type Opts } from '../_rl/kit';

export type { Opts };
/** The SFT run of the sft brief props (demos, demoAccuracy, seed, steps, lr), memoised per page. */
export function sftRun(o: Opts, kit: Kit): { demos: Demo[]; run: SFTRun; steps: number } {
  const n = int(o.demos, SFT_DEFAULTS.n), acc = num(o.demoAccuracy, SFT_DEFAULTS.accuracy), seed = int(o.seed, SFT_DEFAULTS.seed);
  const steps = int(o.steps, SFT_DEFAULTS.steps), lr = num(o.lr, SFT_DEFAULTS.lr);
  return kit.memo(key('sft-run', n, acc, seed, steps, lr), () => {
    const demos = makeDemos({ n, accuracy: acc, seed });
    return { demos, run: trainSFT(demos, { steps, lr }), steps };
  });
}
