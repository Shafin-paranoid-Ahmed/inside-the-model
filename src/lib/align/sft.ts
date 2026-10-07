/**
 * Supervised fine-tuning of the candidate policy: maximum likelihood on demonstrations (Adam). For the log-linear
 * family the gradient of the mean NLL is E_pi[phi] - mean phi(y_demo), so at the optimum E_pi[phi] = E_demo[phi]
 * (moment matching): SFT correctness converges to the demonstrators' rate (the "cap" step). Pure TS.
 */
import { createRng } from '../math/rng';
import { adam } from '../nn/optim';
import { D, K, PHI, THETA_BASE, TRAIN, Y_CLEAR, Y_HEDGED, Y_TERSE, Y_WRONG } from './bank';
import { probsInto, type Theta } from './policy';

export interface Demo { x: number; y: number }
/** Demonstrators' style when they are right (clear, terse, hedged; they never pad); when wrong they write a terse wrong answer. */
export const DEMO_STYLE: [number, number][] = [[Y_CLEAR, 0.75], [Y_TERSE, 0.15], [Y_HEDGED, 0.1]];

/**
 * n demonstrations on `prompts` (round robin), exactly round(accuracy * n) of them correct (positions shuffled with the
 * seed), so the demo set's correctness is a known number rather than a noisy draw.
 */
export function makeDemos(o: { n?: number; accuracy?: number; seed?: number; prompts?: number[] } = {}): Demo[] {
  const n = o.n ?? 24, acc = o.accuracy ?? 0.8, prompts = o.prompts ?? TRAIN, rng = createRng(o.seed ?? 2);
  const right = Array.from({ length: n }, (_, i) => i < Math.round(acc * n));
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [right[i], right[j]] = [right[j], right[i]]; }
  return right.map((ok, i) => {
    let y = Y_WRONG;
    if (ok) { let u = rng.next(); y = DEMO_STYLE[DEMO_STYLE.length - 1][0]; for (const [c, w] of DEMO_STYLE) { if (u < w) { y = c; break; } u -= w; } }
    return { x: prompts[i % prompts.length], y };
  });
}
/** Mean of phi over the demonstrations. */
export function demoFeatures(demos: Demo[]): Float64Array {
  const out = new Float64Array(D);
  for (const d of demos) for (let j = 0; j < D; j++) out[j] += PHI[(d.x * K + d.y) * D + j] / demos.length;
  return out;
}
/** Mean negative log-likelihood of the demonstrations and its gradient (into `grad`). */
export function nllGrad(theta: Theta, demos: Demo[], grad = new Float64Array(D)): number {
  const p = new Float64Array(K);
  grad.fill(0);
  let loss = 0;
  for (const d of demos) {
    probsInto(theta, d.x, p);
    loss -= Math.log(p[d.y]);
    for (let j = 0; j < D; j++) {
      let e = 0;
      for (let y = 0; y < K; y++) e += p[y] * PHI[(d.x * K + y) * D + j];
      grad[j] += e - PHI[(d.x * K + d.y) * D + j];
    }
  }
  for (let j = 0; j < D; j++) grad[j] /= demos.length;
  return loss / demos.length;
}
export const nll = (theta: Theta, demos: Demo[]): number => nllGrad(theta, demos);

export interface SFTRun {
  /** theta after each step (index 0 = theta0), and the NLL at each of them. */
  thetas: Float64Array[]; loss: Float64Array;
}
/** Full-batch Adam on the demo NLL from theta0 (default: the base model). */
export function trainSFT(demos: Demo[], o: { steps?: number; lr?: number; theta0?: Theta } = {}): SFTRun {
  const steps = o.steps ?? 150, opt = adam(o.lr ?? 0.05), th = Float64Array.from(o.theta0 ?? THETA_BASE), g = new Float64Array(D);
  const thetas = [th.slice()], loss = new Float64Array(steps + 1);
  for (let s = 0; s < steps; s++) {
    loss[s] = nllGrad(th, demos, g);
    opt.step(th as unknown as number[], g as unknown as number[]);
    thetas.push(th.slice());
  }
  loss[steps] = nll(th, demos);
  return { thetas, loss };
}

/** The day's SFT model (the reference policy of rlhf, dpo and constitutional-ai): sft brief props demos 24, accuracy 0.8, seed 2, 150 steps, lr 0.05. */
export const SFT_DEFAULTS = { n: 24, accuracy: 0.8, seed: 2, steps: 150, lr: 0.05 };
let sftCache: Float64Array | null = null;
export function sftTheta(): Float64Array {
  if (!sftCache) {
    const d = SFT_DEFAULTS, run = trainSFT(makeDemos(d), { steps: d.steps, lr: d.lr });
    sftCache = run.thetas[d.steps];
  }
  return sftCache.slice();
}
