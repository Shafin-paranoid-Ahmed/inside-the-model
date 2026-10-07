/**
 * PPO on the 4x3 grid with a tabular softmax actor (logits N x 4) and a tabular critic V (N): collect a fixed number
 * of transitions, compute termination-aware GAE, freeze the old log-probabilities and advantages, then optimise the
 * clipped surrogate on shuffled mini-batches for a few epochs (Adam), with the value loss kept separate (its own SGD
 * step on the same mini-batch). Pure TS, seeded. `clippedSurrogate` is the pure helper src/lib/align reuses.
 */
import { createRng, type Rng } from '../math/rng';
import { adam } from '../nn/optim';
import { N, NA, START, step, sampleRow, type Grid } from './gridworld';
import { policyValue, softmaxPolicy, softmaxRow } from './tabular';
import { clippedSurrogate } from './surrogate';
export { clippedSurrogate } from './surrogate';

/** Exact KL(p || q) of two categorical distributions of n outcomes, read from offsets op and oq. */
export function klCategorical(p: ArrayLike<number>, q: ArrayLike<number>, op = 0, oq = 0, n = Math.min(p.length - op, q.length - oq)): number {
  let k = 0;
  for (let i = 0; i < n; i++) { const a = p[op + i]; if (a > 0) k += a * Math.log(a / q[oq + i]); }
  return k;
}

export interface Batch {
  s: Int8Array; a: Int8Array; r: Float64Array; s2: Int8Array;
  /** The move ended in a terminal (no bootstrap). */
  terminated: Uint8Array;
  /** The trajectory segment ends here without termination (time limit or end of the batch): bootstrap V(s'), cut GAE. */
  truncated: Uint8Array;
  /** Frozen at collection: log pi_old(a|s) and pi_old(.|s) (n x 4). */
  logpOld: Float64Array; probsOld: Float64Array;
  /** Frozen before the epochs: critic values, TD errors, advantages and returns (A + V). */
  v: Float64Array; v2: Float64Array; delta: Float64Array; adv: Float64Array; ret: Float64Array;
  /** Undiscounted returns of the episodes that finished inside this batch. */
  episodeReturns: number[];
}

/**
 * Collect n transitions with the current logits. Episodes start at the start cell; an episode that hits maxSteps
 * is truncated; the last transition of the batch is truncated unless it terminated.
 */
export function collect(g: Grid, theta: Float64Array, n: number, rng: Rng, maxSteps = 50): Omit<Batch, 'v' | 'v2' | 'delta' | 'adv' | 'ret'> {
  const b = {
    s: new Int8Array(n), a: new Int8Array(n), r: new Float64Array(n), s2: new Int8Array(n), terminated: new Uint8Array(n), truncated: new Uint8Array(n),
    logpOld: new Float64Array(n), probsOld: new Float64Array(n * NA), episodeReturns: [] as number[],
  };
  let s = START, t = 0, G = 0;
  const p = new Float64Array(NA);
  for (let i = 0; i < n; i++) {
    softmaxRow(theta, s, p);
    const a = sampleRow(p, 0, rng.next()), r = step(g, s, a, rng);
    b.s[i] = s; b.a[i] = a; b.r[i] = r.r; b.s2[i] = r.s2; b.terminated[i] = +r.done;
    b.logpOld[i] = Math.log(p[a]); b.probsOld.set(p, i * NA);
    G += r.r; t++; s = r.s2;
    if (r.done) { b.episodeReturns.push(G); s = START; t = 0; G = 0; }
    else if (t >= maxSteps) { b.truncated[i] = 1; s = START; t = 0; G = 0; }
  }
  if (!b.terminated[n - 1]) b.truncated[n - 1] = 1;
  return b;
}

/**
 * GAE: delta_t = r_t + gamma (1 - terminated_t) V(s'_t) - V(s_t); A_t = delta_t + gamma lambda (1 - terminated_t)(1 - truncated_t) A_{t+1}.
 * Truncation still bootstraps V(s') in delta but stops the sum, because the next row starts a new episode.
 */
export function gae(r: ArrayLike<number>, v: ArrayLike<number>, v2: ArrayLike<number>, terminated: ArrayLike<number>, truncated: ArrayLike<number>, gamma: number, lambda: number) {
  const n = r.length, delta = new Float64Array(n), adv = new Float64Array(n);
  let next = 0;
  for (let t = n - 1; t >= 0; t--) {
    const live = terminated[t] ? 0 : 1;
    delta[t] = r[t] + gamma * live * v2[t] - v[t];
    next = delta[t] + gamma * lambda * live * (truncated[t] ? 0 : 1) * next;
    adv[t] = next;
  }
  return { delta, adv, ret: adv.map((x, t) => x + v[t]) };
}
/** Collect, then freeze critic values and GAE for the batch. */
export function makeBatch(g: Grid, theta: Float64Array, V: Float64Array, n: number, rng: Rng, gamma: number, lambda: number, maxSteps = 50): Batch {
  const c = collect(g, theta, n, rng, maxSteps);
  const v = Float64Array.from(c.s, (s) => V[s]), v2 = Float64Array.from(c.s2, (s, i) => (c.terminated[i] ? 0 : V[s]));
  return { ...c, v, v2, ...gae(c.r, v, v2, c.terminated, c.truncated, gamma, lambda) };
}

export interface BatchStats {
  /** Ratios pi_new(a|s) / pi_old(a|s) of all batch samples under the current logits. */
  ratios: Float64Array;
  min: number; max: number;
  /** Fraction of samples with rho outside [1 - eps, 1 + eps], and the fraction on the favourable side (zero gradient). */
  outOfBand: number; plateau: number;
  /** Mean over batch samples of KL(pi_old(.|s) || pi_new(.|s)), and the batch-mean surrogate. */
  kl: number; surrogate: number;
}
export function batchStats(b: Batch, theta: Float64Array, eps: number): BatchStats {
  const n = b.s.length, ratios = new Float64Array(n), p = new Float64Array(NA);
  let min = Infinity, max = -Infinity, out = 0, flat = 0, kl = 0, sur = 0;
  for (let i = 0; i < n; i++) {
    softmaxRow(theta, b.s[i], p);
    const rho = p[b.a[i]] / Math.exp(b.logpOld[i]), c = clippedSurrogate(rho, b.adv[i], eps);
    ratios[i] = rho; min = Math.min(min, rho); max = Math.max(max, rho);
    if (rho < 1 - eps || rho > 1 + eps) out++;
    if (c.dRho === 0 && b.adv[i] !== 0) flat++;
    kl += klCategorical(b.probsOld, p, i * NA); sur += c.value;
  }
  return { ratios, min, max, outOfBand: out / n, plateau: flat / n, kl: kl / n, surrogate: sur / n };
}

export interface PPOOpts {
  batch?: number; minibatch?: number; epochs?: number; eps?: number;
  lr?: number; lrCritic?: number; gamma?: number; lambda?: number; maxSteps?: number;
  iterations?: number; seed?: number;
  /** Keep per-mini-batch statistics (ratios, KL, ...) of every iteration (default: only the first). */
  recordAll?: boolean;
}
export interface PPOIteration {
  batch: Batch;
  thetaOld: Float64Array; vOld: Float64Array;
  /** After each mini-batch update: statistics of the whole batch, the mini-batch's sample indices, value loss and logits. */
  updates: { epoch: number; idx: Int32Array; stats: BatchStats; valueLoss: number; theta: Float64Array }[];
  /** End-of-epoch statistics. */
  epochs: BatchStats[];
}
export interface PPOResult {
  iterations: PPOIteration[];
  /** Per iteration (index 0 = before training): exact discounted start-cell value of the current stochastic policy,
   *  RMS critic error against the exact V^pi of the policy that collected that batch, and the final KL(old || new). */
  evalReturn: Float64Array; criticErr: Float64Array; kl: Float64Array; plateau: Float64Array; outOfBand: Float64Array;
  theta: Float64Array; V: Float64Array;
  thetas: Float64Array[];
}

/**
 * Gradient of MINUS the mini-batch mean clipped surrogate with respect to the logits (into `grad`, zeroed first; a
 * minimiser such as Adam then ascends the surrogate): d rho / d theta_s = rho (onehot(a) - pi(s)). Returns the
 * mini-batch value loss 0.5 (V(s) - return)^2, which does not enter this gradient.
 */
export function surrogateGrad(b: Batch, theta: Float64Array, V: ArrayLike<number>, idx: ArrayLike<number>, eps: number, grad: Float64Array): number {
  const p = new Float64Array(NA), n = idx.length;
  grad.fill(0);
  let vl = 0;
  for (let j = 0; j < n; j++) {
    const i = idx[j], s = b.s[i], a = b.a[i];
    softmaxRow(theta, s, p);
    const rho = p[a] / Math.exp(b.logpOld[i]), d = clippedSurrogate(rho, b.adv[i], eps).dRho;
    if (d) for (let k = 0; k < NA; k++) grad[s * NA + k] -= (d * rho * ((k === a ? 1 : 0) - p[k])) / n;
    vl += 0.5 * (V[s] - b.ret[i]) ** 2 / n;
  }
  return vl;
}

/** One PPO update on a frozen batch: `epochs` passes over shuffled mini-batches. Mutates theta and V. */
export function ppoEpochs(b: Batch, theta: Float64Array, V: Float64Array, o: PPOOpts & { rng: Rng; opt?: ReturnType<typeof adam> }, record = true): PPOIteration {
  const n = b.s.length, M = o.minibatch ?? 16, E = o.epochs ?? 4, eps = o.eps ?? 0.2, lrC = o.lrCritic ?? 0.1;
  const opt = o.opt ?? adam(o.lr ?? 0.03), it: PPOIteration = { batch: b, thetaOld: theta.slice(), vOld: V.slice(), updates: [], epochs: [] };
  const grad = new Float64Array(N * NA), order = Int32Array.from({ length: n }, (_, i) => i);
  for (let e = 0; e < E; e++) {
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(o.rng.next() * (i + 1)); const x = order[i]; order[i] = order[j]; order[j] = x; }
    for (let m0 = 0; m0 < n; m0 += M) {
      const idx = order.slice(m0, Math.min(n, m0 + M));
      const vl = surrogateGrad(b, theta, V, idx, eps, grad);
      opt.step(theta as unknown as number[], grad as unknown as number[]);
      // separate critic step: V(s) -= lrCritic * dL/dV(s) on the same mini-batch, toward the frozen returns
      const gv = new Float64Array(N);
      for (const i of idx) gv[b.s[i]] += (V[b.s[i]] - b.ret[i]) / idx.length;
      for (let s = 0; s < N; s++) V[s] -= lrC * gv[s];
      if (record) it.updates.push({ epoch: e, idx, stats: batchStats(b, theta, eps), valueLoss: vl, theta: theta.slice() });
    }
    it.epochs.push(batchStats(b, theta, eps));
  }
  return it;
}

export function trainPPO(g: Grid, o: PPOOpts = {}): PPOResult {
  const iters = o.iterations ?? 100, n = o.batch ?? 64, gamma = o.gamma ?? 0.9, lambda = o.lambda ?? 0.95;
  const rng = createRng(o.seed ?? 37), theta = new Float64Array(N * NA), V = new Float64Array(N), opt = adam(o.lr ?? 0.03);
  const evalReturn = new Float64Array(iters + 1), criticErr = new Float64Array(iters), kl = new Float64Array(iters), plateau = new Float64Array(iters), outOfBand = new Float64Array(iters);
  const iterations: PPOIteration[] = [], thetas = [theta.slice()];
  evalReturn[0] = policyValue(g, softmaxPolicy(theta), gamma)[START];
  for (let k = 0; k < iters; k++) {
    const exact = policyValue(g, softmaxPolicy(theta), gamma);
    let ce = 0;
    for (const s of g.states) ce += (V[s] - exact[s]) ** 2;
    criticErr[k] = Math.sqrt(ce / g.states.length);
    const b = makeBatch(g, theta, V, n, rng, gamma, lambda, o.maxSteps);
    const it = ppoEpochs(b, theta, V, { ...o, rng, opt }, k === 0 || !!o.recordAll);
    const last = it.epochs[it.epochs.length - 1];
    kl[k] = last.kl; plateau[k] = last.plateau; outOfBand[k] = last.outOfBand;
    iterations.push(it); thetas.push(theta.slice());
    evalReturn[k + 1] = policyValue(g, softmaxPolicy(theta), gamma)[START];
  }
  return { iterations, evalReturn, criticErr, kl, plateau, outOfBand, theta, V, thetas };
}

/**
 * The drift experiment: the SAME frozen batch and initial logits/critic, optimised with each epsilon. The batch is
 * collected from a policy pre-trained for `warmIterations` PPO iterations (0 = uniform), so advantages are informative.
 */
export function driftRuns(g: Grid, o: PPOOpts & { epsilons?: number[]; warmIterations?: number } = {}) {
  const gamma = o.gamma ?? 0.9, lambda = o.lambda ?? 0.95, n = o.batch ?? 64;
  const warm = o.warmIterations ? trainPPO(g, { ...o, iterations: o.warmIterations }) : null;
  const theta0 = warm ? warm.theta.slice() : new Float64Array(N * NA), V0 = warm ? warm.V.slice() : new Float64Array(N);
  const b = makeBatch(g, theta0, V0, n, createRng((o.seed ?? 31) * 7 + 1), gamma, lambda, o.maxSteps);
  return {
    batch: b, theta0, V0,
    runs: (o.epsilons ?? [0.1, 0.2, 0.4]).map((eps) => ({ eps, it: ppoEpochs(b, theta0.slice(), V0.slice(), { ...o, eps, rng: createRng(o.seed ?? 31) }) })),
  };
}
