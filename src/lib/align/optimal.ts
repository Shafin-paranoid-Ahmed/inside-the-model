/**
 * The KL-regularised optimum: maximising E_pi[r] - beta KL(pi || pi_ref) gives pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x).
 * In the log-linear family with r = w . phi this is exactly theta* = theta_ref + w / beta (the DPO lesson's anchor).
 */
import { D, K, TRAIN } from './bank';
import { kl, probsInto, rewardGrad, scores, type Theta } from './policy';

export const optimalTheta = (ref: Theta, w: Theta, beta: number): Float64Array => Float64Array.from({ length: D }, (_, j) => ref[j] + w[j] / beta);
/** pi*(.|x) straight from the formula pi_ref exp(r / beta) / Z, and Z(x). */
export function optimalRow(ref: Theta, w: Theta, beta: number, x: number): { p: Float64Array; Z: number } {
  const q = probsInto(ref, x), r = scores(w), p = new Float64Array(K);
  let Z = 0;
  for (let y = 0; y < K; y++) { p[y] = q[y] * Math.exp(r[x * K + y] / beta); Z += p[y]; }
  for (let y = 0; y < K; y++) p[y] /= Z;
  return { p, Z };
}
/** The objective mean_x (E_pi r - beta KL(pi || pi_ref)). */
export function klObjective(theta: Theta, ref: Theta, w: Theta, beta: number, prompts: number[] = TRAIN): number {
  const r = scores(w), p = new Float64Array(K);
  let s = 0;
  for (const x of prompts) { probsInto(theta, x, p); for (let y = 0; y < K; y++) s += p[y] * r[x * K + y]; }
  return s / prompts.length - beta * kl(theta, ref, prompts);
}
/**
 * Exact gradient ascent on that objective (no sampling, no PPO): grad = mean_x E_pi[(f - E_pi f) phi] with
 * f = r - beta log(pi / pi_ref). Used by the tests to confirm the closed form.
 */
export function exactAscent(ref: Theta, w: Theta, beta: number, o: { steps?: number; lr?: number; theta0?: Theta; prompts?: number[] } = {}): Float64Array {
  const th = Float64Array.from(o.theta0 ?? ref), r = scores(w), prompts = o.prompts ?? TRAIN, f = new Float64Array(r.length), p = new Float64Array(K), q = new Float64Array(K), g = new Float64Array(D);
  for (let s = 0; s < (o.steps ?? 5000); s++) {
    for (const x of prompts) { probsInto(th, x, p); probsInto(ref, x, q); for (let y = 0; y < K; y++) f[x * K + y] = r[x * K + y] - beta * Math.log(p[y] / q[y]); }
    rewardGrad(th, f, prompts, g);
    for (let j = 0; j < D; j++) th[j] += (o.lr ?? 0.5) * g[j];
  }
  return th;
}
