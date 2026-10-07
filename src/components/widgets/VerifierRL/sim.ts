/**
 * VerifierRL (rlvr): addition problems answered with generated text, a program verifier that parses the text, GRPO
 * training with group-relative advantages, emergent response length, a gamed (lenient) verifier, and unverifiable prompts.
 *
 * MODE-AUTHOR CONTRACT: identical to GridWorldLab (read the comment at the top of ../GridWorldLab/sim.ts). You own
 * m-<mode>.ts and m-<mode>.meta.ts here and never edit VerifierRL.svelte, ../_rl/*, ../_align/*, this file or
 * src/lib/align. kit.job runs any export of src/lib/rl OR src/lib/align in the shared worker.
 * Reference mode: m-verify.ts (+ meta). Draw helpers: ../_rl/svg.ts, ../_align/draw.ts (responseCard with mono text,
 * wrap/clip/charsFor) and tick / cross below.
 *
 * src/lib/align/verifier.ts: makeProblem, solve(p, k), respond(p, action) (the response text), truth, parseStrict,
 * parseLenient, verify(text, answer, 'strict' | 'lenient'), words, linesOf, actionOf(k, list), kOf, isList,
 * ACTION_LABELS, NACT (10 = 5 k values x 2 formats), startLogits(), rowProbs(logits, row), rowStats(logits, digits)
 * (exact strict / lenient accuracy, mean lines and words, guess-list share, pass@1, pass@8 on 200 fixed problems per
 * digit count), groupAdvantages, sampleGroups(logits, { problems, G, rng, verifier, rows }), trainGRPO({ verifier,
 * iters, G, problems, lr, seed, klBeta, domains }) -> per-iteration logits and exact curves (index it * 4 + digits - 1)
 * plus the sampled groups; OPEN (row 4) and OPEN_PROMPTS for the unverifiable prompts of the domains step.
 * Props the briefs name: verifier ("strict" | "lenient"), G (8), problems (6), seed (2 for sample/grpo, 5 for training),
 * iters (150). Measured (Node): trainGRPO 150 iterations 45–90 ms (kit.job('trainGRPO', {...})); evalTable (first call)
 * about 15 ms; sampleGroups instant. Numbers: plans/day-11.md §7.
 */
import { line, path, type Env } from '../_rl/svg';

/** A tick (pass) centred at (x, y), size s, in --viz-pos. */
export function tick(E: Env, x: number, y: number, s = 8, op = 1) {
  path(E, `M${x - s} ${y} L${x - s * 0.3} ${y + s * 0.7} L${x + s} ${y - s * 0.8}`, { stroke: 'var(--viz-pos)', sw: 2.75, op });
}
/** A cross (fail) centred at (x, y), size s, in --viz-neg. */
export function cross(E: Env, x: number, y: number, s = 7, op = 1) {
  line(E, x - s, y - s, x + s, y + s, { stroke: 'var(--viz-neg)', sw: 2.75, op });
  line(E, x - s, y + s, x + s, y - s, { stroke: 'var(--viz-neg)', sw: 2.75, op });
}
