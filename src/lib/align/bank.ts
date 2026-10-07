/**
 * The Day 12 tiny policy's world: 10 short checkable prompts (8 train, 2 held-out), each with the same 8 candidate
 * response templates, a 6-number feature vector phi per (prompt, candidate) read off the generated text, a fixed
 * seeded per-candidate base-model offset c(x, y), the hidden gold utility and the simulated labelers' bias.
 * The policy is pi_theta(y|x) = softmax_y(theta . phi(x, y) + c(x, y)) (policy.ts). Pure data, no DOM.
 *
 * Designed part (plans/day-11.md §5 "Designed exploit"): the rambling-wrong answer (long, overconfident, wrong; not
 * flattering) is far longer than anything the SFT policy writes and almost never sampled, labelers like length only up to about LENGTH_CAP (their bias saturates), and the reward model is
 * linear, so it extrapolates the length bonus to the rambling answer. Everything that follows from that is computed.
 */
import { createRng } from '../math/rng';

export const FEATURES = ['answers', 'correct', 'length', 'flattery', 'refuses', 'hedges'] as const;
/** Plain-words names for feature bars ("likes answering", ...). */
export const FEATURE_LABELS = ['Answers', 'Correct', 'Length', 'Flattery', 'Refuses', 'Hedges'] as const;
export const D = 6;
export const [F_ANSWERS, F_CORRECT, F_LENGTH, F_FLATTERY, F_REFUSES, F_HEDGES] = [0, 1, 2, 3, 4, 5];

export const CANDIDATES = [
  { id: 'continue', label: 'Continues the question list', short: 'Continues list' },
  { id: 'terse', label: 'Terse correct', short: 'Terse correct' },
  { id: 'clear', label: 'Clear correct', short: 'Clear correct' },
  { id: 'padded', label: 'Padded correct', short: 'Padded correct' },
  { id: 'rambling', label: 'Rambling wrong', short: 'Rambling wrong' },
  { id: 'wrong', label: 'Terse wrong', short: 'Terse wrong' },
  { id: 'refusal', label: 'Refusal', short: 'Refusal' },
  { id: 'hedged', label: 'Hedged correct', short: 'Hedged correct' },
] as const;
export const K = 8;
export const [Y_CONTINUE, Y_TERSE, Y_CLEAR, Y_PADDED, Y_RAMBLING, Y_WRONG, Y_REFUSAL, Y_HEDGED] = [0, 1, 2, 3, 4, 5, 6, 7];

export interface Prompt {
  q: string;
  /** Correct and wrong short answers, a clear one-sentence answer, and what a base model writes next (more questions). */
  a: string; wrong: string; clear: string; more: string;
}
export const PROMPTS: Prompt[] = [
  { q: 'What is 7 + 5?', a: '12', wrong: '13', clear: '7 + 5 = 12.', more: 'What is 8 + 3? What is 6 + 9? What is 4 + 7?' },
  { q: "Spell 'cat' backwards.", a: 'tac', wrong: 'act', clear: "Backwards, 'cat' is spelled 'tac'.", more: "Spell 'dog' backwards. Spell 'sun' backwards. Spell 'map' backwards." },
  { q: 'How many legs does a spider have?', a: '8', wrong: '6', clear: 'A spider has 8 legs.', more: 'How many legs does an ant have? How many legs does a crab have?' },
  { q: 'What is the capital of France?', a: 'Paris', wrong: 'Lyon', clear: 'The capital of France is Paris.', more: 'What is the capital of Spain? What is the capital of Italy?' },
  { q: 'What is 9 × 6?', a: '54', wrong: '56', clear: '9 × 6 = 54.', more: 'What is 7 × 8? What is 6 × 6? What is 9 × 9?' },
  { q: 'How many days are in a week?', a: '7', wrong: '5', clear: 'A week has 7 days.', more: 'How many days are in a year? How many hours are in a day?' },
  { q: 'What colour do you get by mixing blue and yellow?', a: 'Green', wrong: 'Purple', clear: 'Mixing blue and yellow gives green.', more: 'What colour do you get by mixing red and white?' },
  { q: 'What is half of 30?', a: '15', wrong: '13', clear: 'Half of 30 is 15.', more: 'What is half of 50? What is half of 18? What is a third of 30?' },
  { q: 'How many sides does a hexagon have?', a: '6', wrong: '8', clear: 'A hexagon has 6 sides.', more: 'How many sides does an octagon have? How many sides does a pentagon have?' },
  { q: 'What is 15 − 8?', a: '7', wrong: '6', clear: '15 − 8 = 7.', more: 'What is 14 − 9? What is 12 − 5? What is 16 − 7?' },
];
export const P = PROMPTS.length;
export const TRAIN = [0, 1, 2, 3, 4, 5, 6, 7];
export const HELDOUT = [8, 9];

/** The response text of candidate y for prompt p. */
export function responseText(p: Prompt, y: number): string {
  switch (y) {
    case Y_CONTINUE: return p.more;
    case Y_TERSE: return `${p.a}.`;
    case Y_CLEAR: return p.clear;
    case Y_PADDED: return `Great question! I love questions like this one. ${p.clear} I hope this helps, and do let me know if you have any other questions!`;
    case Y_RAMBLING: return `Let us think about this very carefully, step by step, weighing every possibility with care before we settle on an answer, because questions like this one are often much trickier than they first appear. After considering everything very thoroughly, I am completely confident that the answer is ${p.wrong}.`;
    case Y_WRONG: return `${p.wrong}.`;
    case Y_REFUSAL: return "I'm sorry, but I can't help with that request.";
    default: return `I think it might be ${p.a}, but I'm not completely sure.`;
  }
}
export const wordCount = (s: string): number => s.trim().split(/\s+/).length;

/** Binary features of each template: answers, correct, flattery, refuses, hedges (length comes from the text). */
const FLAGS: [number, number, number, number, number][] = [
  [0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 0, 0, 1],
];
/** Base-model quirks c(x, y): a per-template mean plus seeded noise (never trained). The rambling answer is rare (mean -10), so
 *  the SFT model almost never writes it and the reward model never sees it in preference pairs. */
const C_MEAN = [0, 0, 0, 0, -10, 0, 0, 0];
const C_SD = 0.5, C_SEED = 12;

export const TEXT: string[][] = PROMPTS.map((p) => CANDIDATES.map((_, y) => responseText(p, y)));
/** phi(x, y) for all prompts and candidates, flat: index (x * K + y) * D + j. */
export const PHI = new Float64Array(P * K * D);
/** c(x, y), flat: index x * K + y. */
export const C = new Float64Array(P * K);
{
  const rng = createRng(C_SEED);
  for (let x = 0; x < P; x++) for (let y = 0; y < K; y++) {
    const f = FLAGS[y], o = (x * K + y) * D;
    PHI[o] = f[0]; PHI[o + 1] = f[1]; PHI[o + 2] = wordCount(TEXT[x][y]) / 10; PHI[o + 3] = f[2]; PHI[o + 4] = f[3]; PHI[o + 5] = f[4];
    C[x * K + y] = C_MEAN[y] + C_SD * rng.normal();
  }
}
export const phi = (x: number, y: number): Float64Array => PHI.subarray((x * K + y) * D, (x * K + y + 1) * D);

/** The hidden gold utility ("what the person actually wanted"): g(y) = W_GOLD . phi. */
export const W_GOLD = Float64Array.from([1, 2, -0.3, -1, -1, -0.5]);
/**
 * Simulated human labelers: Bradley-Terry on u(y) = W_GOLD . phi + biasLength min(length, lengthCap) + biasFlattery
 * flattery, with temperature `temp`: P(a preferred to b) = sigma((u(a) - u(b)) / temp). The length bias saturates
 * (a 70-word answer does not impress more than a 25-word one), but a linear reward model cannot know that.
 */
export const LABELER = { biasLength: 0.9, biasFlattery: 0, lengthCap: 2.5, temp: 1 };
export function labelerUtility(x: number, y: number): number {
  const f = phi(x, y);
  let u = 0;
  for (let j = 0; j < D; j++) u += W_GOLD[j] * f[j];
  return u + LABELER.biasLength * Math.min(f[F_LENGTH], LABELER.lengthCap) + LABELER.biasFlattery * f[F_FLATTERY];
}
/** The base (pretrained) model: it prefers not to answer and rarely refuses, so it mostly continues the question list. */
export const THETA_BASE = Float64Array.from([-2.5, 0, 0, 0, -1.5, 0]);
