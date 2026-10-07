/**
 * Preference data and the Bradley-Terry reward model. Pairs are two different responses sampled from a policy for
 * the same train prompt; a labeler picks one: 'human' (Bradley-Terry on the gold utility plus the length/flattery
 * bias of bank.ts) or 'ai' (the constitution rule checker of constitution.ts). The reward model r(y) = w . phi is
 * logistic regression on phi_w - phi_l (no intercept: Bradley-Terry scores are defined up to a constant). Pure TS.
 */
import { createRng } from '../math/rng';
import { adam } from '../nn/optim';
import { D, K, LABELER, PHI, TRAIN, labelerUtility } from './bank';
import { aiPrefersA, DEFAULT_PRINCIPLES } from './constitution';
import { probsInto, sampleFrom, type Theta } from './policy';

export type Labeler = 'human' | 'ai';
export interface Pair {
  x: number;
  /** The two responses in the order they were sampled, and whether the labeler chose a. */
  a: number; b: number; aWins: boolean;
  /** Chosen (winner) and rejected (loser) candidates. */
  w: number; l: number;
}
export const sigmoid = (z: number): number => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
/** P(human labeler prefers a to b) = sigma((u(a) - u(b)) / temp). */
export const humanPrefProb = (x: number, a: number, b: number): number => sigmoid((labelerUtility(x, a) - labelerUtility(x, b)) / LABELER.temp);

/**
 * n pairs from policy theta on `prompts` (uniform). Responses use the seed's stream and labels a separate stream
 * (seed + 1), so the same seed gives the same pairs for both labelers: only the choices differ.
 */
export function samplePairs(theta: Theta, n: number, seed: number, labeler: Labeler = 'human', o: { prompts?: number[]; active?: string[] } = {}): Pair[] {
  const prompts = o.prompts ?? TRAIN, rs = createRng(seed), rl = createRng(seed + 1), p = new Float64Array(K), out: Pair[] = [];
  for (let i = 0; i < n; i++) {
    const x = prompts[Math.floor(rs.next() * prompts.length)];
    probsInto(theta, x, p);
    const a = sampleFrom(p, rs.next());
    let b = a;
    for (let k = 0; k < 1000 && b === a; k++) b = sampleFrom(p, rs.next());
    if (b === a) b = (a + 1) % K; // ponytail: only reachable for a near-deterministic policy
    const aWins = labeler === 'human' ? rl.next() < humanPrefProb(x, a, b) : aiPrefersA(x, a, b, rl, o.active ?? DEFAULT_PRINCIPLES);
    out.push({ x, a, b, aWins, w: aWins ? a : b, l: aWins ? b : a });
  }
  return out;
}
/** Mean Bradley-Terry loss -log sigma(r_w - r_l) of weights w on the pairs, and its gradient (into grad). */
export function rmLossGrad(w: Theta, pairs: Pair[], grad = new Float64Array(D)): number {
  grad.fill(0);
  let loss = 0;
  for (const q of pairs) {
    const ow = (q.x * K + q.w) * D, ol = (q.x * K + q.l) * D;
    let m = 0;
    for (let j = 0; j < D; j++) m += w[j] * (PHI[ow + j] - PHI[ol + j]);
    loss += m >= 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m));
    const s = sigmoid(-m);
    for (let j = 0; j < D; j++) grad[j] -= (s * (PHI[ow + j] - PHI[ol + j])) / pairs.length;
  }
  return loss / pairs.length;
}
export interface RMFit { w: Float64Array; ws: Float64Array[]; loss: Float64Array }
/** Fit the reward model by full-batch Adam from w = 0; records weights and loss at every step. */
export function fitRM(pairs: Pair[], o: { steps?: number; lr?: number } = {}): RMFit {
  const steps = o.steps ?? 200, opt = adam(o.lr ?? 0.05), w = new Float64Array(D), g = new Float64Array(D);
  const ws = [w.slice()], loss = new Float64Array(steps + 1);
  for (let s = 0; s < steps; s++) {
    loss[s] = rmLossGrad(w, pairs, g);
    opt.step(w as unknown as number[], g as unknown as number[]);
    ws.push(w.slice());
  }
  loss[steps] = rmLossGrad(w, pairs, g);
  return { w, ws, loss };
}
/** Share of pairs where w scores the chosen response higher (ties count half). */
export function pairAccuracy(w: Theta, pairs: Pair[]): number {
  let s = 0;
  for (const q of pairs) {
    let m = 0;
    for (let j = 0; j < D; j++) m += w[j] * (PHI[(q.x * K + q.w) * D + j] - PHI[(q.x * K + q.l) * D + j]);
    s += m > 1e-12 ? 1 : m < -1e-12 ? 0 : 0.5;
  }
  return s / pairs.length;
}
