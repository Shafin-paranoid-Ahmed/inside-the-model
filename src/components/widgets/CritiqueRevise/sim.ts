/**
 * CritiqueRevise (constitutional-ai): a written constitution checked against the candidate responses, critique and
 * revision rounds, fine-tuning on the revisions (SL-CAI) and editing the principles.
 *
 * MODE-AUTHOR CONTRACT: identical to GridWorldLab (read the comment at the top of ../GridWorldLab/sim.ts). You own
 * m-<mode>.ts and m-<mode>.meta.ts here and never edit CritiqueRevise.svelte, ../_rl/*, ../_align/*, this file or
 * src/lib/align. kit.job runs any export of src/lib/rl OR src/lib/align in the shared worker.
 * Reference mode: m-rules.ts (+ meta). Draw helpers: ../_rl/svg.ts, ../_align/draw.ts (candidateBars, featureBars,
 * responseCard, candColor) and principleText / principlesOf below.
 *
 * src/lib/align exports: constitution (PRINCIPLES with id/text/violated, DEFAULT_PRINCIPLES (the cautious principle is
 * off), violations, violationCount, revise, critiqueRevise, aiPrefersA, slCAI(theta0, { n, rounds, seed, steps, lr,
 * active, from })), reward (samplePairs(theta, n, seed, 'ai', { active })), fitRM, trainAlignPPO, sftTheta, THETA_BASE.
 * Revision rule: a round samples one active principle; if the response violates it, it moves to the candidate that
 * satisfies it with the fewest other binary-feature changes (so "continues the list" -> "terse wrong" on "answer",
 * then -> "terse correct" on "accurate"; several rounds can be needed).
 * Measured (Node): slCAI 5 ms (memo); one RLAIF run (pairs + fitRM + 200 PPO iterations) about 45 ms, so the edit step's
 * 6 single-toggle runs (default, each default principle off, cautious on) take about 300 ms: one kit.job per toggle set
 * (kit.job('trainAlignPPO', { rm: fitRM(samplePairs(sft, 200, 4, 'ai', { active })).w, beta: 0.1, seed: 6 })), or
 * compute the reward model with kit.memo and only the PPO run in the worker. Numbers: plans/day-11.md §7.
 */
import { DEFAULT_PRINCIPLES, PRINCIPLES } from '../../../lib/align/constitution';
import { list, type Opts } from '../_rl/kit';

export type { Opts };
/** Active principle ids from the `principles` prop ("answer,accurate,..."), default the constitution without the cautious one. */
export const principlesOf = (o: Opts): string[] => list(o.principles, DEFAULT_PRINCIPLES).filter((id) => PRINCIPLES.some((p) => p.id === id));
export const principleText = (id: string): string => PRINCIPLES.find((p) => p.id === id)?.text ?? id;
/** Short column headings for narrow layouts. */
export const SHORT: Record<string, string> = { answer: 'Answers', accurate: 'Accurate', 'no-flattery': 'No flattery', 'no-refuse': 'No refusal', cautious: 'Cautious' };
