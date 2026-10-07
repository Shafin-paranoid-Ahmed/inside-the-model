/**
 * EvalLab: how models are measured, and how measurements mislead (item response theory, binomial intervals,
 * Bradley-Terry ratings) on one scrubbable timeline, one mode per lesson step.
 *
 * MODE-AUTHOR CONTRACT: identical to GridWorldLab (read the comment at the top of ../GridWorldLab/sim.ts). A mode is
 * m-<mode>.ts plus m-<mode>.meta.ts here; the shell is ../_rl/Shell.svelte through EvalLab.svelte. Mode runs stay on the
 * main thread (kit.job only runs src/lib/rl), so everything below is closed-form or tiny: every setup is a few ms.
 *
 * The model of a test. A question has a difficulty d; a model has a skill s; the chance that the model answers it
 * correctly is sigmoid(s - d) (the one-parameter item response theory model). Difficulties are drawn from a normal
 * distribution, so a model's "true score" is its expected accuracy over all such questions, popAcc(s).
 */
import { createRng, type Rng } from '../../../lib/math/rng';

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
export const MODEL_NAMES = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E', 'Model F'];
/** Identity colour of model i (fixed order, always shown with its name). */
export const catColor = (i: number) => `var(--viz-cat-${i + 1})`;

/** Standard normal CDF (Abramowitz and Stegun 26.2.17, error below 1e-7). */
export function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989422804014327 * Math.exp((-x * x) / 2);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

/** E[f(d)] for d ~ Normal(mu, sd): a fixed 240-point rule over mu +- 6 sd. */
export function expectOver(mu: number, sd: number, f: (d: number) => number): number {
  const N = 240, lo = -6, h = 12 / N;
  let sum = 0;
  for (let i = 0; i <= N; i++) { const z = lo + i * h; sum += 0.3989422804014327 * Math.exp((-z * z) / 2) * f(mu + sd * z) * (i === 0 || i === N ? 0.5 : 1) * h; }
  return sum;
}
/** True score of skill s on questions with difficulty ~ Normal(mu, sd). */
export const popAcc = (s: number, mu = 0, sd = 1.2) => expectOver(mu, sd, (d) => sigmoid(s - d));
/** P(both models right) on a random question. */
export const popBoth = (sA: number, sB: number, mu = 0, sd = 1.2) => expectOver(mu, sd, (d) => sigmoid(sA - d) * sigmoid(sB - d));

export const difficulties = (n: number, rng: Rng, mu = 0, sd = 1.2): Float64Array => { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = rng.normal(mu, sd); return a; };
/** One pass of a model over the questions: 1 right, 0 wrong. */
export const respond = (skill: number, diffs: ArrayLike<number>, rng: Rng): Uint8Array => { const a = new Uint8Array(diffs.length); for (let i = 0; i < diffs.length; i++) a[i] = rng.next() < sigmoid(skill - diffs[i]) ? 1 : 0; return a; };

/** 95% Wilson interval for k right out of n (close to p +- 1.96 sqrt(p(1-p)/n) at n = 200, and sensible at tiny n). */
export function wilson(k: number, n: number, z = 1.96): [number, number] {
  if (n <= 0) return [0, 1];
  const p = k / n, z2 = z * z, den = 1 + z2 / n, c = (p + z2 / (2 * n)) / den, hw = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / den;
  return [Math.max(0, c - hw), Math.min(1, c + hw)];
}
export const overlaps = (a: [number, number], b: [number, number]) => a[0] <= b[1] && b[0] <= a[1];
/** Plain binomial standard error sqrt(p(1-p)/n). */
export const binomSE = (p: number, n: number) => Math.sqrt((p * (1 - p)) / n);

// ---------- Bradley-Terry ratings ----------

const GHOST = 0.25; // a quarter of a win each way per pair keeps the fit finite before every pair has met
/** Maximum-likelihood Bradley-Terry strengths (log scale, mean 0) from wins[i*k+j] = times i beat j. */
export function fitBT(wins: ArrayLike<number>, k: number, iters = 300): Float64Array {
  const w = (i: number, j: number) => wins[i * k + j] + GHOST, p = new Float64Array(k).fill(1);
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < k; i++) {
      let tot = 0, den = 0;
      for (let j = 0; j < k; j++) if (j !== i) { tot += w(i, j); den += (w(i, j) + w(j, i)) / (p[i] + p[j]); }
      p[i] = tot / den;
    }
    let g = 0; for (let i = 0; i < k; i++) g += Math.log(p[i]); g = Math.exp(g / k); for (let i = 0; i < k; i++) p[i] /= g;
  }
  return p.map(Math.log);
}
/** Standard errors of the centred strengths from the Fisher information (pseudo-inverse of the Laplacian). */
export function btSE(beta: ArrayLike<number>, wins: ArrayLike<number>, k: number): Float64Array {
  const A: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(1 / k));
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) if (i !== j) {
    const n = wins[i * k + j] + wins[j * k + i] + 2 * GHOST, p = sigmoid(beta[i] - beta[j]), v = n * p * (1 - p);
    A[i][i] += v; A[i][j] -= v;
  }
  const inv = invert(A), se = new Float64Array(k);
  for (let i = 0; i < k; i++) se[i] = Math.sqrt(Math.max(0, inv[i][i] - 1 / k));
  return se;
}
function invert(M: number[][]): number[][] {
  const n = M.length, a = M.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) piv = r;
    [a[c], a[piv]] = [a[piv], a[c]];
    const d = a[c][c]; for (let j = 0; j < 2 * n; j++) a[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = a[r][c]; for (let j = 0; j < 2 * n; j++) a[r][j] -= f * a[c][j]; }
  }
  return a.map((r) => r.slice(n));
}
/** Rating on the familiar Elo-style scale: 1000 plus 400 / ln 10 per unit of log strength. */
export const ELO = 400 / Math.LN10;
export const toRating = (beta: number) => 1000 + ELO * beta;

// ---------- two-model comparisons ----------

/** Chance the weaker model scores higher after n questions, when the true gap is `delta` and one question's score difference has standard deviation `sd`. */
export const wrongWinner = (delta: number, sd: number, n: number) => normCdf(-(delta * Math.sqrt(n)) / sd);
/** Smallest n at which wrongWinner falls to `target` (closed form). */
export const itemsNeeded = (delta: number, sd: number, target = 0.05) => { let z = 0, lo = -8, hi = 0; for (let i = 0; i < 80; i++) { z = (lo + hi) / 2; if (normCdf(z) > target) hi = z; else lo = z; } return ((-z * sd) / delta) ** 2; };

export { createRng };

// ---------- shared by the benchmark mode and its test ----------

/** True skills of Models A to F in the benchmark step (about 0.52 to 0.79 true score on the default questions). */
export const BENCH_SKILLS = [0.1, 0.35, 0.6, 0.9, 1.25, 1.7];
export interface Bench { diffs: Float64Array; results: Uint8Array[]; cum: Int32Array[]; pop: number[]; n: number }
/** Every model answers the same n seeded questions in order; cum[m][i] = right answers among the first i. */
export function makeBench(seed: number, n: number, skills = BENCH_SKILLS): Bench {
  const diffs = difficulties(n, createRng(seed));
  const results = skills.map((s, i) => respond(s, diffs, createRng(seed * 100 + i)));
  const cum = results.map((r) => { const c = new Int32Array(n + 1); for (let i = 0; i < n; i++) c[i + 1] = c[i] + r[i]; return c; });
  return { diffs, results, cum, pop: skills.map((s) => popAcc(s)), n };
}

// ---------- arena ----------

/** True strengths (log scale) of Models A to F, and which answers are long: Model C writes padded, wordy answers. */
export const ARENA_BETA = [-0.9, -0.5, -0.15, 0.15, 0.5, 0.9];
export const ARENA_LEN = [0, 0, 1, 0, 0, 0];
export const LENGTH_BIAS = 0.7;
export interface Arena {
  k: number; votes: number; a: Uint8Array; b: Uint8Array;
  /** winner[v] for fair voters, and for voters who favour longer answers (same pairs, same coin flips). */
  fair: Uint8Array; biased: Uint8Array;
  /** wins[v * k * k + i * k + j] = times i beat j among the first v votes. */
  winsFair: Int16Array; winsBiased: Int16Array;
}
export function makeArena(seed: number, votes: number, beta = ARENA_BETA, lens = ARENA_LEN, bias = LENGTH_BIAS): Arena {
  const k = beta.length, rng = createRng(seed), a = new Uint8Array(votes), b = new Uint8Array(votes), fair = new Uint8Array(votes), biased = new Uint8Array(votes);
  const pairs: [number, number][] = []; for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) pairs.push([i, j]);
  for (let v = 0; v < votes; v++) {
    const [i, j] = pairs[Math.floor(rng.next() * pairs.length)], u = rng.next();
    a[v] = i; b[v] = j;
    fair[v] = u < sigmoid(beta[i] - beta[j]) ? i : j;
    biased[v] = u < sigmoid(beta[i] - beta[j] + bias * (lens[i] - lens[j])) ? i : j;
  }
  const prefix = (win: Uint8Array) => {
    const out = new Int16Array((votes + 1) * k * k);
    for (let v = 0; v < votes; v++) {
      out.copyWithin((v + 1) * k * k, v * k * k, (v + 1) * k * k);
      const w = win[v], l = w === a[v] ? b[v] : a[v];
      out[(v + 1) * k * k + w * k + l]++;
    }
    return out;
  };
  return { k, votes, a, b, fair, biased, winsFair: prefix(fair), winsBiased: prefix(biased) };
}
/** Ratings (1000 scale), their 95% half-widths and ranks (1 = best) after the first n votes. */
export function arenaAt(ar: Arena, n: number, biased: boolean) {
  const k = ar.k, w = (biased ? ar.winsBiased : ar.winsFair).subarray(n * k * k, (n + 1) * k * k);
  const beta = fitBT(w, k), se = btSE(beta, w, k);
  const rating = Array.from(beta, toRating), half = Array.from(se, (x) => 1.96 * ELO * x);
  const order = rating.map((r, i) => [r, i]).sort((x, y) => y[0] - x[0]).map((x) => x[1]), rank = new Array<number>(k);
  order.forEach((m, r) => { rank[m] = r + 1; });
  return { rating, half, rank };
}
