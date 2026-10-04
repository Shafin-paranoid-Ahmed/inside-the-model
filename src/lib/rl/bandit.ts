/**
 * k-armed Gaussian bandit (Sutton and Barto ch. 2): action-value methods (greedy, epsilon-greedy, UCB) and the
 * softmax policy gradient with REINFORCE samples. Pure TS, seeded.
 */
import { createRng, type Rng } from '../math/rng';

export interface Bandit { k: number; means: Float64Array; std: number; best: number }

/** Means drawn from N(0, 1) with `seed` (Sutton and Barto's testbed), or given explicitly. Rewards ~ N(mean, std). */
export function makeBandit(k: number, seed: number, o: { means?: ArrayLike<number>; std?: number } = {}): Bandit {
  const rng = createRng(seed);
  const means = o.means ? Float64Array.from(o.means) : Float64Array.from({ length: k }, () => rng.normal());
  let best = 0;
  for (let a = 1; a < means.length; a++) if (means[a] > means[best]) best = a;
  return { k: means.length, means, std: o.std ?? 1, best };
}
export const pull = (b: Bandit, arm: number, rng: Rng): number => rng.normal(b.means[arm], b.std);

export type Strategy = { kind: 'greedy' } | { kind: 'epsilon'; epsilon: number } | { kind: 'ucb'; c?: number };
export const strategyName = (s: Strategy): string =>
  s.kind === 'greedy' ? 'greedy' : s.kind === 'epsilon' ? `ε = ${s.epsilon}` : `UCB, c = ${s.c ?? 2}`;

export interface BanditRun {
  arms: Int8Array; rewards: Float64Array;
  /** Estimates and pull counts AFTER each step (steps x k), when recorded. */
  est: Float64Array | null; counts: Int32Array | null;
}

/** Pick an arm. Ties among the best estimates are broken uniformly. */
export function choose(s: Strategy, Q: Float64Array, Nn: Int32Array, t: number, rng: Rng): number {
  const k = Q.length;
  if (s.kind === 'epsilon' && rng.next() < s.epsilon) return Math.floor(rng.next() * k);
  let score = Q;
  if (s.kind === 'ucb') {
    for (let a = 0; a < k; a++) if (Nn[a] === 0) return a; // untried arms first
    score = Q.map((q, a) => q + (s.c ?? 2) * Math.sqrt(Math.log(t) / Nn[a]));
  }
  let m = -Infinity;
  for (let a = 0; a < k; a++) m = Math.max(m, score[a]);
  const best: number[] = [];
  for (let a = 0; a < k; a++) if (score[a] >= m - 1e-12) best.push(a);
  return best.length === 1 ? best[0] : best[Math.floor(rng.next() * best.length)];
}

/** One run: incremental sample-average estimates Q(a) += (r - Q(a)) / N(a), starting at q0 (default 0). */
export function runBandit(b: Bandit, s: Strategy, steps: number, seed: number, o: { q0?: number; record?: boolean } = {}): BanditRun {
  const rng = createRng(seed), Q = new Float64Array(b.k).fill(o.q0 ?? 0), Nn = new Int32Array(b.k);
  const arms = new Int8Array(steps), rewards = new Float64Array(steps);
  const est = o.record ? new Float64Array(steps * b.k) : null, counts = o.record ? new Int32Array(steps * b.k) : null;
  for (let t = 0; t < steps; t++) {
    const a = choose(s, Q, Nn, t + 1, rng), r = pull(b, a, rng);
    Nn[a]++; Q[a] += (r - Q[a]) / Nn[a];
    arms[t] = a; rewards[t] = r;
    if (est && counts) { est.set(Q, t * b.k); counts.set(Nn, t * b.k); }
  }
  return { arms, rewards, est, counts };
}

export interface StrategyCurve {
  strategy: Strategy; name: string;
  /** Mean reward and fraction of runs choosing the best arm, per step. */
  avgReward: Float64Array; optimal: Float64Array;
  /** Runs whose most-pulled arm over the last 100 steps is not the best arm. */
  stuck: number;
}
/**
 * The testbed: `runs` bandits (bandit r has seed `seed * 1000 + r`), each strategy plays every bandit with the same
 * pull seed. Returns the averaged curves.
 */
export function compareStrategies(o: { k?: number; runs?: number; steps?: number; seed?: number; strategies: Strategy[] }): StrategyCurve[] {
  const k = o.k ?? 5, runs = o.runs ?? 50, steps = o.steps ?? 500, seed = o.seed ?? 7;
  const bandits = Array.from({ length: runs }, (_, r) => makeBandit(k, seed * 1000 + r));
  return o.strategies.map((s) => {
    const avgReward = new Float64Array(steps), optimal = new Float64Array(steps);
    let stuck = 0;
    bandits.forEach((b, r) => {
      const run = runBandit(b, s, steps, seed * 7919 + r);
      for (let t = 0; t < steps; t++) { avgReward[t] += run.rewards[t] / runs; if (run.arms[t] === b.best) optimal[t] += 1 / runs; }
      const tail = new Int32Array(k);
      for (let t = Math.max(0, steps - 100); t < steps; t++) tail[run.arms[t]]++;
      let top = 0;
      for (let a = 1; a < k; a++) if (tail[a] > tail[top]) top = a;
      if (top !== b.best) stuck++;
    });
    return { strategy: s, name: strategyName(s), avgReward, optimal, stuck };
  });
}

// ---------- softmax policy gradient ----------

export function softmax(theta: ArrayLike<number>): Float64Array {
  let m = -Infinity;
  for (let i = 0; i < theta.length; i++) m = Math.max(m, theta[i]);
  const p = Float64Array.from(theta, (x) => Math.exp(x - m)), z = p.reduce((a, b) => a + b, 0);
  return p.map((x) => x / z);
}
/** Expected reward under pi = softmax(theta). */
export const expectedReward = (theta: ArrayLike<number>, means: ArrayLike<number>): number =>
  softmax(theta).reduce((acc, p, a) => acc + p * means[a], 0);
/** Exact gradient of the expected reward: d/dtheta_j = pi_j (mu_j - sum_a pi_a mu_a). */
export function exactGradient(theta: ArrayLike<number>, means: ArrayLike<number>): Float64Array {
  const p = softmax(theta), avg = p.reduce((acc, q, a) => acc + q * means[a], 0);
  return p.map((q, j) => q * (means[j] - avg));
}
/** REINFORCE estimate from one pull: (r - baseline) (onehot(arm) - pi). */
export function reinforceGradient(probs: ArrayLike<number>, arm: number, r: number, baseline = 0): Float64Array {
  return Float64Array.from(probs, (q, j) => (r - baseline) * ((j === arm ? 1 : 0) - q));
}

export interface BanditPGRun {
  /** Logits and probabilities BEFORE each step (steps + 1 rows of k; the last row is after the final update). */
  theta: Float64Array; probs: Float64Array;
  arms: Int8Array; rewards: Float64Array;
  /** The sampled gradient used at each step and the exact gradient at that step's logits (steps x k). */
  grad: Float64Array; exact: Float64Array;
  baselines: Float64Array;
}
/**
 * Softmax policy gradient on a bandit: sample an arm, theta += lr (r - b) (onehot - pi).
 * baseline 'none' (b = 0) or 'average' (running mean of the rewards seen so far, before this pull).
 */
export function banditPG(b: Bandit, o: { lr?: number; steps?: number; seed?: number; baseline?: 'none' | 'average'; theta0?: ArrayLike<number> } = {}): BanditPGRun {
  const lr = o.lr ?? 0.1, steps = o.steps ?? 200, k = b.k, rng = createRng(o.seed ?? 5);
  const th = o.theta0 ? Float64Array.from(o.theta0) : new Float64Array(k);
  const theta = new Float64Array((steps + 1) * k), probs = new Float64Array((steps + 1) * k), grad = new Float64Array(steps * k), exact = new Float64Array(steps * k);
  const arms = new Int8Array(steps), rewards = new Float64Array(steps), baselines = new Float64Array(steps);
  let avg = 0;
  for (let t = 0; t < steps; t++) {
    const p = softmax(th);
    theta.set(th, t * k); probs.set(p, t * k); exact.set(exactGradient(th, b.means), t * k);
    let u = rng.next(), a = 0;
    while (a < k - 1 && u >= p[a]) { u -= p[a]; a++; }
    const r = pull(b, a, rng), base = o.baseline === 'average' ? avg : 0, g = reinforceGradient(p, a, r, base);
    for (let j = 0; j < k; j++) th[j] += lr * g[j];
    grad.set(g, t * k); arms[t] = a; rewards[t] = r; baselines[t] = base;
    avg += (r - avg) / (t + 1);
  }
  theta.set(th, steps * k); probs.set(softmax(th), steps * k);
  return { theta, probs, arms, rewards, grad, exact, baselines };
}

/**
 * Many single-pull REINFORCE estimates at FIXED logits, paired: each pull gives one estimate without a baseline and one
 * with `baseline` (default: the expected reward under pi). Rows of k.
 */
export function gradientSamples(theta: ArrayLike<number>, b: Bandit, n: number, seed: number, baseline = expectedReward(theta, b.means)) {
  const rng = createRng(seed), p = softmax(theta), k = b.k;
  const arms = new Int8Array(n), rewards = new Float64Array(n), plain = new Float64Array(n * k), based = new Float64Array(n * k);
  for (let i = 0; i < n; i++) {
    let u = rng.next(), a = 0;
    while (a < k - 1 && u >= p[a]) { u -= p[a]; a++; }
    const r = pull(b, a, rng);
    arms[i] = a; rewards[i] = r;
    plain.set(reinforceGradient(p, a, r, 0), i * k); based.set(reinforceGradient(p, a, r, baseline), i * k);
  }
  return { arms, rewards, plain, based, baseline, exact: exactGradient(theta, b.means), probs: p };
}

/** Mean and (population) variance of column j of an n x k row-major array. */
export function columnStats(a: ArrayLike<number>, k: number, j: number, n = a.length / k): { mean: number; variance: number } {
  let m = 0;
  for (let i = 0; i < n; i++) m += a[i * k + j];
  m /= n;
  let v = 0;
  for (let i = 0; i < n; i++) v += (a[i * k + j] - m) ** 2;
  return { mean: m, variance: v / n };
}
