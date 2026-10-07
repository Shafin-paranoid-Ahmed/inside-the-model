/**
 * RLHF's RL stage on the candidate policy: per iteration sample a batch of (prompt, response) pairs from pi_old, score
 * each with the shaped reward r = RM(y) - beta (log pi_old - log pi_ref)(y) (InstructGPT's per-sample KL penalty to the
 * fixed SFT reference), subtract a per-prompt running-mean baseline (it stands in for PPO's value network), then run
 * Day 11's clipped surrogate (src/lib/rl/ppo.ts `clippedSurrogate`) for a few epochs of shuffled mini-batches (Adam).
 * Records the exact proxy reward, gold reward and KL to the reference after every iteration. Pure TS.
 */
import { createRng } from '../math/rng';
import { adam } from '../nn/optim';
import { clippedSurrogate } from '../rl/surrogate';
import { D, K, P, PHI, TRAIN } from './bank';
import { GOLD, expected, kl, probsInto, sampleFrom, scores, type Theta } from './policy';
import { sftTheta } from './sft';

export interface AlignPPOOpts {
  /** Reward-model weights (r = w . phi). */
  rm: ArrayLike<number>;
  beta?: number; iters?: number; batch?: number; epochs?: number; minibatch?: number; eps?: number; lr?: number; seed?: number;
  /** Start and reference policies (default: the day's SFT model for both). */
  theta0?: ArrayLike<number>; ref?: ArrayLike<number>;
  prompts?: number[];
  /** Running-mean step of the per-prompt baseline. */
  baselineRate?: number;
  /** Iterations whose per-epoch ratios are kept in `detail` (for the ppo-clip step). */
  record?: number[];
}
export interface AlignPPOResult {
  /** theta before each iteration and at the end (iters + 1); exact mean RM reward, gold reward and KL(pi || pi_ref) at each. */
  thetas: Float64Array[]; proxy: Float64Array; gold: Float64Array; kl: Float64Array;
  /** Per iteration and epoch (index it * epochs + e), at the end of the epoch: fraction of samples with the ratio outside
   *  [1 - eps, 1 + eps], and the fraction on the favourable side (zero gradient). */
  outOfBand: Float64Array; plateau: Float64Array;
  /** Every batch sample (index it * batch + i): prompt, response, shaped reward and advantage. */
  bx: Uint8Array; by: Uint8Array; br: Float32Array; badv: Float32Array;
  /** Recorded iterations: the ratio of every sample after each epoch (epochs * batch). */
  detail: { iter: number; ratios: Float64Array }[];
  iters: number; batch: number; epochs: number; eps: number; beta: number;
}

export function trainAlignPPO(o: AlignPPOOpts): AlignPPOResult {
  const iters = o.iters ?? 200, B = o.batch ?? 64, E = o.epochs ?? 4, M = o.minibatch ?? 16, eps = o.eps ?? 0.2, beta = o.beta ?? 0.1;
  const prompts = o.prompts ?? TRAIN, rate = o.baselineRate ?? 0.2, rec = new Set(o.record ?? []);
  const sft = o.theta0 && o.ref ? null : sftTheta();
  const th = Float64Array.from(o.theta0 ?? sft!), ref = Float64Array.from(o.ref ?? sft!), rm = scores(o.rm);
  const rng = createRng(o.seed ?? 6), opt = adam(o.lr ?? 0.04);
  const res: AlignPPOResult = {
    thetas: [th.slice()], proxy: new Float64Array(iters + 1), gold: new Float64Array(iters + 1), kl: new Float64Array(iters + 1),
    outOfBand: new Float64Array(iters * E), plateau: new Float64Array(iters * E),
    bx: new Uint8Array(iters * B), by: new Uint8Array(iters * B), br: new Float32Array(iters * B), badv: new Float32Array(iters * B),
    detail: [], iters, batch: B, epochs: E, eps, beta,
  };
  const evalAt = (k: number) => { res.proxy[k] = expected(th, rm, prompts); res.gold[k] = expected(th, GOLD, prompts); res.kl[k] = kl(th, ref, prompts); };
  evalAt(0);
  const base = new Float64Array(P).fill(NaN), p = new Float64Array(K), q = new Float64Array(K);
  const x = new Int32Array(B), y = new Int32Array(B), lpOld = new Float64Array(B), r = new Float64Array(B), adv = new Float64Array(B);
  const order = Int32Array.from({ length: B }, (_, i) => i), grad = new Float64Array(D), pr = new Float64Array(K), batch = { x, y, logpOld: lpOld, adv };
  for (let it = 0; it < iters; it++) {
    // collect with the current policy (pi_old) and score
    for (let i = 0; i < B; i++) {
      x[i] = prompts[Math.floor(rng.next() * prompts.length)];
      probsInto(th, x[i], p); probsInto(ref, x[i], q);
      y[i] = sampleFrom(p, rng.next());
      lpOld[i] = Math.log(p[y[i]]);
      r[i] = rm[x[i] * K + y[i]] - beta * (lpOld[i] - Math.log(q[y[i]]));
    }
    // advantage = shaped reward minus the per-prompt running mean (initialised with the prompt's first batch mean)
    const sum = new Map<number, [number, number]>();
    for (let i = 0; i < B; i++) { const s = sum.get(x[i]) ?? [0, 0]; s[0] += r[i]; s[1]++; sum.set(x[i], s); }
    for (const [xx, [s, n]] of sum) if (Number.isNaN(base[xx])) base[xx] = s / n;
    for (let i = 0; i < B; i++) adv[i] = r[i] - base[x[i]];
    for (let i = 0; i < B; i++) {
      res.bx[it * B + i] = x[i]; res.by[it * B + i] = y[i]; res.br[it * B + i] = r[i]; res.badv[it * B + i] = adv[i];
    }
    for (const [xx, [s, n]] of sum) base[xx] += rate * (s / n - base[xx]);
    // epochs of clipped-surrogate updates on shuffled mini-batches
    const ratios = rec.has(it) ? new Float64Array(E * B) : null;
    for (let e = 0; e < E; e++) {
      for (let i = B - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
      for (let m0 = 0; m0 < B; m0 += M) {
        const g = alignSurrogateGrad(th, batch, eps, order.subarray(m0, Math.min(B, m0 + M)));
        for (let j = 0; j < D; j++) grad[j] = -g[j]; // Adam minimises: descend minus the surrogate
        opt.step(th as unknown as number[], grad as unknown as number[]);
      }
      let out = 0, flat = 0;
      for (let i = 0; i < B; i++) {
        const rho = probsInto(th, x[i], pr)[y[i]] / Math.exp(lpOld[i]), c = clippedSurrogate(rho, adv[i], eps);
        if (rho < 1 - eps || rho > 1 + eps) out++;
        if (c.dRho === 0 && adv[i] !== 0) flat++;
        if (ratios) ratios[e * B + i] = rho;
      }
      res.outOfBand[it * E + e] = out / B; res.plateau[it * E + e] = flat / B;
    }
    if (ratios) res.detail.push({ iter: it, ratios });
    res.thetas.push(th.slice());
    evalAt(it + 1);
  }
  return res;
}

/** A frozen batch: prompts, responses, log pi_old and advantages. */
export interface FrozenBatch { x: ArrayLike<number>; y: ArrayLike<number>; logpOld: ArrayLike<number>; adv: ArrayLike<number> }
/** The surrogate objective (mean over samples) of theta on a frozen batch: for tests and the ppo-clip view. */
export function alignSurrogate(theta: Theta, b: FrozenBatch, eps: number): number {
  const p = new Float64Array(K);
  let s = 0;
  for (let i = 0; i < b.x.length; i++) s += clippedSurrogate(probsInto(theta, b.x[i], p)[b.y[i]] / Math.exp(b.logpOld[i]), b.adv[i], eps).value;
  return s / b.x.length;
}
/** Its analytic gradient over all samples or the mini-batch `idx`: d rho / d theta = rho (phi(y) - E_pi phi). trainAlignPPO descends minus this. */
export function alignSurrogateGrad(theta: Theta, b: FrozenBatch, eps: number, idx?: ArrayLike<number>): Float64Array {
  const p = new Float64Array(K), g = new Float64Array(D), n = idx ? idx.length : b.x.length;
  for (let ii = 0; ii < n; ii++) {
    const i = idx ? idx[ii] : ii, x = b.x[i], y = b.y[i];
    probsInto(theta, x, p);
    const rho = p[y] / Math.exp(b.logpOld[i]), d = clippedSurrogate(rho, b.adv[i], eps).dRho;
    for (let j = 0; j < D; j++) {
      let e = 0;
      for (let z = 0; z < K; z++) e += p[z] * PHI[(x * K + z) * D + j];
      g[j] += (d * rho * (PHI[(x * K + y) * D + j] - e)) / n;
    }
  }
  return g;
}
