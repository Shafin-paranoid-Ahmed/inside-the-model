/**
 * Direct preference optimisation on the candidate policy: L = -mean log sigma(beta [(log pi(w) - log pi_ref(w)) -
 * (log pi(l) - log pi_ref(l))]). In this family the margin is beta (theta - theta_ref) . (phi_w - phi_l) (the
 * per-prompt normaliser cancels), so DPO is the reward model's logistic regression in disguise: at convergence
 * theta = theta_ref + w_RM / beta, the closed-form optimum of optimal.ts. Pure TS.
 */
import { adam } from '../nn/optim';
import { D, K, PHI, TRAIN } from './bank';
import { GOLD, expected, kl, probs, probsInto, type Theta } from './policy';
import { sigmoid, type Pair } from './reward';

/** The DPO margin of one pair: beta [(log pi(w) - log pi_ref(w)) - (log pi(l) - log pi_ref(l))]. */
export function dpoMargin(theta: Theta, ref: Theta, q: Pair, beta: number): number {
  const p = probsInto(theta, q.x), r = probsInto(ref, q.x);
  return beta * (Math.log(p[q.w] / r[q.w]) - Math.log(p[q.l] / r[q.l]));
}
/**
 * Mean DPO loss and its gradient (into grad): -mean sigma(-m) beta (grad log pi(w) - grad log pi(l)), and
 * grad log pi(w) - grad log pi(l) = phi_w - phi_l. The weight sigma(-m) shrinks as a pair is ranked right.
 */
export function dpoLossGrad(theta: Theta, ref: Theta, pairs: Pair[], beta: number, grad = new Float64Array(D)): number {
  grad.fill(0);
  let loss = 0;
  for (const q of pairs) {
    const m = dpoMargin(theta, ref, q, beta), s = sigmoid(-m), ow = (q.x * K + q.w) * D, ol = (q.x * K + q.l) * D;
    loss += m >= 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m));
    for (let j = 0; j < D; j++) grad[j] -= (s * beta * (PHI[ow + j] - PHI[ol + j])) / pairs.length;
  }
  return loss / pairs.length;
}
export interface DPORun {
  /** Per step (index 0 = start): theta, loss, mean log pi of the chosen and of the rejected responses, gold and KL. */
  thetas: Float64Array[]; loss: Float64Array; chosen: Float64Array; rejected: Float64Array; gold: Float64Array; kl: Float64Array;
}
/**
 * Full-batch Adam on the DPO loss from theta_ref (or theta0). The default lr 0.05 / beta makes each step move the implicit
 * reward weights v = beta (theta - theta_ref) by Adam's step of 0.05, so from theta_ref the DPO path in v is the reward
 * model fit's path (reward.ts fitRM, lr 0.05) step for step: after n steps theta = theta_ref + w_RM(n) / beta.
 */
export function trainDPO(pairs: Pair[], ref: Theta, o: { beta?: number; steps?: number; lr?: number; theta0?: Theta; prompts?: number[] } = {}): DPORun {
  const beta = o.beta ?? 0.1, steps = o.steps ?? 300, opt = adam(o.lr ?? 0.05 / beta), th = Float64Array.from(o.theta0 ?? ref), g = new Float64Array(D), prompts = o.prompts ?? TRAIN;
  const run: DPORun = { thetas: [], loss: new Float64Array(steps + 1), chosen: new Float64Array(steps + 1), rejected: new Float64Array(steps + 1), gold: new Float64Array(steps + 1), kl: new Float64Array(steps + 1) };
  const rec = (s: number) => {
    run.thetas.push(th.slice());
    run.loss[s] = dpoLossGrad(th, ref, pairs, beta, g);
    const p = probs(th);
    let c = 0, r = 0;
    for (const q of pairs) { c += Math.log(p[q.x * K + q.w]); r += Math.log(p[q.x * K + q.l]); }
    run.chosen[s] = c / pairs.length; run.rejected[s] = r / pairs.length;
    run.gold[s] = expected(th, GOLD, prompts); run.kl[s] = kl(th, ref, prompts);
  };
  for (let s = 0; s < steps; s++) { rec(s); opt.step(th as unknown as number[], g as unknown as number[]); }
  rec(steps);
  return run;
}
/** The implicit reward beta (log pi - log pi_ref) of every candidate, flat P * K. */
export function implicitReward(theta: Theta, ref: Theta, beta: number): Float64Array {
  const p = probs(theta), q = probs(ref);
  return p.map((v, i) => beta * Math.log(v / q[i]));
}
