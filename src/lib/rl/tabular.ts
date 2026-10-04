/**
 * Tabular algorithms on the 4x3 grid: dynamic programming (policy evaluation, value iteration), exact policy
 * values, Q-learning, REINFORCE and one-step actor-critic with a tabular softmax policy (logits per state-action).
 * Arrays are flat N x 4 (row-major) Float64Arrays; terminal and wall rows stay 0. Pure TS, seeded.
 */
import { createRng, type Rng } from '../math/rng';
import { N, NA, START, isTerminal, makeGrid, transitions, step, sampleRow, returns, type Grid, type GridOpts } from './gridworld';

// ---------- small helpers ----------

/** Softmax of one state's logits into `out` (offset s * 4). */
export function softmaxRow(theta: ArrayLike<number>, s: number, out: Float64Array = new Float64Array(NA), o = 0): Float64Array {
  let m = -Infinity;
  for (let a = 0; a < NA; a++) m = Math.max(m, theta[s * NA + a]);
  let z = 0;
  for (let a = 0; a < NA; a++) { const e = Math.exp(theta[s * NA + a] - m); out[o + a] = e; z += e; }
  for (let a = 0; a < NA; a++) out[o + a] /= z;
  return out;
}
/** Softmax probabilities of every state (N x 4). */
export function softmaxPolicy(theta: ArrayLike<number>): Float64Array {
  const p = new Float64Array(N * NA);
  for (let s = 0; s < N; s++) softmaxRow(theta, s, p, s * NA);
  return p;
}
/** One-hot probabilities from one action per state (entries < 0 are skipped). */
export function deterministic(actions: ArrayLike<number>): Float64Array {
  const p = new Float64Array(N * NA);
  for (let s = 0; s < N; s++) if (actions[s] >= 0) p[s * NA + actions[s]] = 1;
  return p;
}
/** Greedy action per state; ties within `tol` go to the first action (up, right, down, left). -1 off the action states. */
export function greedy(g: Grid, Q: ArrayLike<number>, tol = 1e-9): Int8Array {
  const pi = new Int8Array(N).fill(-1);
  for (const s of g.states) {
    let b = 0;
    for (let a = 1; a < NA; a++) if (Q[s * NA + a] > Q[s * NA + b] + tol) b = a;
    pi[s] = b;
  }
  return pi;
}
export const maxRow = (Q: ArrayLike<number>, s: number): number =>
  Math.max(Q[s * NA], Q[s * NA + 1], Q[s * NA + 2], Q[s * NA + 3]);

/** Q(s,a) = sum_s' P(s'|s,a) [r + gamma V(s')] (terminal V is 0). */
export function qFromV(g: Grid, V: ArrayLike<number>, gamma = g.gamma): Float64Array {
  const Q = new Float64Array(N * NA);
  for (const s of g.states) for (let a = 0; a < NA; a++) {
    let q = 0;
    for (const o of transitions(g, s, a)) q += o.p * (o.r + (o.done ? 0 : gamma * V[o.s2]));
    Q[s * NA + a] = q;
  }
  return Q;
}

/** The terms of one backup at (s, a) under V: each outcome's probability, reward, next value and weighted term. */
export function backupTerms(g: Grid, V: ArrayLike<number>, s: number, a: number, gamma = g.gamma) {
  const terms = transitions(g, s, a).map((o) => {
    const v = o.done ? 0 : V[o.s2];
    return { ...o, v, term: o.p * (o.r + gamma * v) };
  });
  return { terms, sum: terms.reduce((x, t) => x + t.term, 0) };
}

// ---------- dynamic programming ----------

/** Iterative policy evaluation (synchronous sweeps). V[k] is the value after k sweeps; V[0] = 0. */
export function policyEvaluation(g: Grid, probs: ArrayLike<number>, sweeps: number, gamma = g.gamma): { V: Float64Array[]; delta: number[] } {
  const V = [new Float64Array(N)], delta: number[] = [];
  for (let k = 0; k < sweeps; k++) {
    const prev = V[k], Q = qFromV(g, prev, gamma), next = new Float64Array(N);
    let d = 0;
    for (const s of g.states) {
      let v = 0;
      for (let a = 0; a < NA; a++) v += probs[s * NA + a] * Q[s * NA + a];
      next[s] = v; d = Math.max(d, Math.abs(v - prev[s]));
    }
    V.push(next); delta.push(d);
  }
  return { V, delta };
}

export interface ValueIterationResult {
  /** V[k] after k sweeps (V[0] = 0), Q[k] = Q computed from V[k]... Q[k] is the backup that produced V[k + 1]. */
  V: Float64Array[]; Q: Float64Array[];
  /** Max |V[k+1] - V[k]| per sweep. */
  delta: number[];
}
/** Value iteration for `sweeps` synchronous sweeps (stops early when the change falls below `tol`, if given). */
export function valueIteration(g: Grid, sweeps: number, tol = 0, gamma = g.gamma): ValueIterationResult {
  const V = [new Float64Array(N)], Q: Float64Array[] = [], delta: number[] = [];
  for (let k = 0; k < sweeps; k++) {
    const prev = V[k], q = qFromV(g, prev, gamma), next = new Float64Array(N);
    let d = 0;
    for (const s of g.states) { next[s] = maxRow(q, s); d = Math.max(d, Math.abs(next[s] - prev[s])); }
    Q.push(q); V.push(next); delta.push(d);
    if (d < tol) break;
  }
  return { V, Q, delta };
}

/** Converged optimal V*, Q* and greedy policy. For gamma = 1 with stepReward > 0 values grow without bound: capped at maxSweeps. */
export function optimal(g: Grid, gamma = g.gamma, tol = 1e-12, maxSweeps = 20000) {
  let V = new Float64Array(N), sweeps = 0;
  for (; sweeps < maxSweeps; sweeps++) {
    const q = qFromV(g, V, gamma), next = new Float64Array(N);
    let d = 0;
    for (const s of g.states) { next[s] = maxRow(q, s); d = Math.max(d, Math.abs(next[s] - V[s])); }
    V = next;
    if (d < tol) { sweeps++; break; }
  }
  const Q = qFromV(g, V, gamma);
  return { V, Q, policy: greedy(g, Q, 1e-9), sweeps };
}

/**
 * Exact V^pi by solving (I - gamma P_pi) V = r_pi over the action states (Gaussian elimination).
 * Returns NaN entries when the system is singular (a deterministic policy that never ends, at gamma = 1).
 */
export function policyValue(g: Grid, probs: ArrayLike<number>, gamma = g.gamma): Float64Array {
  const S = g.states, m = S.length, pos = new Int32Array(N).fill(-1);
  S.forEach((s, i) => (pos[s] = i));
  const A = Array.from({ length: m }, () => new Float64Array(m + 1));
  S.forEach((s, i) => {
    A[i][i] += 1;
    for (let a = 0; a < NA; a++) {
      const pa = probs[s * NA + a];
      if (!pa) continue;
      for (const o of transitions(g, s, a)) {
        A[i][m] += pa * o.p * o.r;
        if (!o.done) A[i][pos[o.s2]] -= gamma * pa * o.p;
      }
    }
  });
  const V = new Float64Array(N);
  for (let c = 0; c < m; c++) {
    let p = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) { V.fill(NaN); for (let s = 0; s < N; s++) if (isTerminal(s)) V[s] = 0; return V; }
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < m; r++) if (r !== c) {
      const f = A[r][c] / A[c][c];
      if (f) for (let k = c; k <= m; k++) A[r][k] -= f * A[c][k];
    }
  }
  S.forEach((s, i) => (V[s] = A[i][m] / A[i][i]));
  return V;
}

/**
 * Where a policy-valued function of one parameter changes, between lo and hi: scan `steps` points, then bisect each
 * change to `tol`. Returns the switch points and the policies on each side (useful for gamma or stepReward thresholds).
 */
export function switchPoints(f: (x: number) => ArrayLike<number>, lo: number, hi: number, steps = 400, tol = 1e-6) {
  const same = (a: ArrayLike<number>, b: ArrayLike<number>) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; };
  const out: { at: number; before: Int8Array; after: Int8Array }[] = [];
  let x0 = lo, p0 = Int8Array.from(f(lo));
  for (let i = 1; i <= steps; i++) {
    const x1 = lo + ((hi - lo) * i) / steps, p1 = Int8Array.from(f(x1));
    if (!same(p0, p1)) {
      let a = x0, b = x1, pa = p0;
      while (b - a > tol) { const m = (a + b) / 2, pm = Int8Array.from(f(m)); if (same(pm, pa)) a = m; else b = m; }
      out.push({ at: (a + b) / 2, before: p0, after: Int8Array.from(f(b)) });
    }
    x0 = x1; p0 = p1;
  }
  return out;
}

/** Optimal-policy switch points as gamma runs from lo to hi, other grid options fixed (a worker-friendly wrapper). */
export function gammaSwitches(o: GridOpts, lo: number, hi: number, steps = 100) {
  return switchPoints((x) => optimal(makeGrid({ ...o, gamma: x }), x, 1e-10).policy, lo, hi, steps, 1e-5);
}
/**
 * Optimal-policy switch points as the step reward runs from lo to hi. With a positive step reward and gamma = 1 the
 * values grow without bound, so positive step rewards use `gammaPositive` (default 0.99) instead.
 */
export function rewardSwitches(o: GridOpts, lo: number, hi: number, steps = 200, gammaPositive = 0.99) {
  const pol = (x: number) => { const gm = x > 0 ? gammaPositive : o.gamma ?? 1; return optimal(makeGrid({ ...o, stepReward: x, gamma: gm }), gm, 1e-10, 50000).policy; };
  return switchPoints(pol, lo, hi, steps, 1e-5);
}

// ---------- Q-learning ----------

export interface QLearningOpts {
  alpha?: number; epsilon?: number; episodes?: number; seed?: number; gamma?: number;
  maxSteps?: number; q0?: number;
  /** Record every transition (default true). Turn off for many-run curves. */
  record?: boolean;
}
/** Columnar record of every transition. `qAfter` holds the updated Q row (4 numbers per transition). */
export interface QTrace {
  ep: Int32Array; s: Int8Array; a: Int8Array; s2: Int8Array; done: Uint8Array; slipped: Uint8Array;
  r: Float64Array; target: Float64Array; tdError: Float64Array; qBefore: Float64Array; qAfter: Float64Array;
}
export interface QLearningResult {
  /** Q after each episode: snapshots[k] after k episodes (snapshots[0] = initial). */
  snapshots: Float64Array[];
  /** Undiscounted training return and length per episode, and the index of each episode's first transition. */
  returns: Float64Array; lengths: Int32Array; start: Int32Array;
  trace: QTrace | null;
  Q: Float64Array;
}

/** Epsilon-greedy over a Q row; ties among the greedy actions are broken uniformly. Two rng draws at most. */
export function epsGreedy(Q: ArrayLike<number>, s: number, epsilon: number, rng: Rng): number {
  if (rng.next() < epsilon) return Math.floor(rng.next() * NA);
  const m = maxRow(Q, s), best: number[] = [];
  for (let a = 0; a < NA; a++) if (Q[s * NA + a] >= m - 1e-12) best.push(a);
  return best.length === 1 ? best[0] : best[Math.floor(rng.next() * best.length)];
}

export function qLearning(g: Grid, o: QLearningOpts = {}): QLearningResult {
  const alpha = o.alpha ?? 0.1, eps = o.epsilon ?? 0.1, E = o.episodes ?? 300, gamma = o.gamma ?? g.gamma, maxSteps = o.maxSteps ?? 100;
  const rng = createRng(o.seed ?? 11), Q = new Float64Array(N * NA);
  for (const s of g.states) for (let a = 0; a < NA; a++) Q[s * NA + a] = o.q0 ?? 0;
  const snapshots = [Q.slice()], rets = new Float64Array(E), lengths = new Int32Array(E), start = new Int32Array(E);
  const rec: Record<string, number[]> | null = o.record === false ? null : { ep: [], s: [], a: [], s2: [], done: [], slipped: [], r: [], target: [], tdError: [], qBefore: [], qAfter: [] };
  let n = 0;
  for (let e = 0; e < E; e++) {
    let s = START, G = 0, t = 0;
    start[e] = n;
    for (; t < maxSteps; t++) {
      const a = epsGreedy(Q, s, eps, rng), r = step(g, s, a, rng), i = s * NA + a;
      const target = r.r + (r.done ? 0 : gamma * maxRow(Q, r.s2)), before = Q[i], err = target - before;
      Q[i] = before + alpha * err;
      if (rec) {
        rec.ep.push(e); rec.s.push(s); rec.a.push(a); rec.s2.push(r.s2); rec.done.push(+r.done); rec.slipped.push(+r.slipped);
        rec.r.push(r.r); rec.target.push(target); rec.tdError.push(err); rec.qBefore.push(before);
        for (let k = 0; k < NA; k++) rec.qAfter.push(Q[s * NA + k]);
      }
      n++; G += r.r; s = r.s2;
      if (r.done) { t++; break; }
    }
    rets[e] = G; lengths[e] = t;
    snapshots.push(Q.slice());
  }
  const trace: QTrace | null = rec && {
    ep: Int32Array.from(rec.ep), s: Int8Array.from(rec.s), a: Int8Array.from(rec.a), s2: Int8Array.from(rec.s2),
    done: Uint8Array.from(rec.done), slipped: Uint8Array.from(rec.slipped), r: Float64Array.from(rec.r),
    target: Float64Array.from(rec.target), tdError: Float64Array.from(rec.tdError), qBefore: Float64Array.from(rec.qBefore), qAfter: Float64Array.from(rec.qAfter),
  };
  return { snapshots, returns: rets, lengths, start, trace, Q };
}

/** Largest |Q - Q*| over the action states' (s, a) entries. */
export function qError(g: Grid, Q: ArrayLike<number>, Qstar: ArrayLike<number>): number {
  let m = 0;
  for (const s of g.states) for (let a = 0; a < NA; a++) m = Math.max(m, Math.abs(Q[s * NA + a] - Qstar[s * NA + a]));
  return m;
}
/** How many action states pick the same greedy action in both tables. */
export function policyAgreement(g: Grid, a: ArrayLike<number>, b: ArrayLike<number>): number {
  let k = 0;
  for (const s of g.states) if (a[s] === b[s]) k++;
  return k;
}

// ---------- REINFORCE and actor-critic on the grid (tabular softmax policy) ----------

export interface PGStep { s: number; a: number; r: number; G: number; b: number; adv: number; probs: Float64Array }
export interface PGEpisodeDetail {
  episode: number; steps: PGStep[];
  /** Logits before and after this episode's update (N x 4). */
  thetaBefore: Float64Array; thetaAfter: Float64Array;
  /** Gradient estimate of this episode, sum_t adv_t (onehot(a_t) - pi(s_t)) (N x 4). */
  grad: Float64Array;
  vBefore: Float64Array;
}
export interface GridPGOpts {
  lr?: number; episodes?: number; seed?: number; gamma?: number; maxSteps?: number;
  /** 'none' or a learned tabular state-value baseline b(s) (Monte Carlo, step lrCritic). */
  baseline?: 'none' | 'state-value';
  lrCritic?: number;
  /** Exact expected return of the start cell under the current policy, every `evalEvery` episodes. */
  evalEvery?: number;
  /** Keep logits every `snapshotEvery` episodes (0 = never). */
  snapshotEvery?: number;
  /** Episodes (0-based) whose full detail is kept. */
  detail?: number[];
}
export interface GridPGResult {
  returns: Float64Array; lengths: Int32Array;
  /** evalAt[k] = episodes completed when evalValue[k] (exact V^pi(start)) was computed. */
  evalAt: Int32Array; evalValue: Float64Array;
  snapshots: Float64Array[]; snapshotAt: Int32Array;
  details: PGEpisodeDetail[];
  theta: Float64Array; V: Float64Array;
}

/** REINFORCE: run an episode with the current logits, then theta += lr * sum_t (G_t - b(s_t)) (onehot(a_t) - pi(s_t)). */
export function gridReinforce(g: Grid, o: GridPGOpts = {}): GridPGResult {
  const lr = o.lr ?? 0.1, E = o.episodes ?? 1000, gamma = o.gamma ?? g.gamma, maxSteps = o.maxSteps ?? 100;
  const useB = o.baseline === 'state-value', lrC = o.lrCritic ?? 0.1, ev = o.evalEvery ?? 10, se = o.snapshotEvery ?? 0;
  const rng = createRng(o.seed ?? 13), theta = new Float64Array(N * NA), V = new Float64Array(N), p = new Float64Array(NA);
  const res = pgResult(E, ev, se), want = new Set(o.detail ?? []);
  record(res, g, theta, 0, ev, se, gamma);
  for (let e = 0; e < E; e++) {
    const states: number[] = [], acts: number[] = [], rews: number[] = [], probs: Float64Array[] = [];
    let s = START;
    for (let t = 0; t < maxSteps; t++) {
      softmaxRow(theta, s, p);
      const a = sampleRow(p, 0, rng.next()), r = step(g, s, a, rng);
      states.push(s); acts.push(a); rews.push(r.r); probs.push(p.slice());
      s = r.s2;
      if (r.done) break;
    }
    const G = returns(rews, gamma), grad = new Float64Array(N * NA), steps: PGStep[] = [];
    const vBefore = V.slice(), thetaBefore = want.has(e) ? theta.slice() : theta;
    for (let t = 0; t < states.length; t++) {
      const st = states[t], b = useB ? vBefore[st] : 0, adv = G[t] - b;
      for (let a = 0; a < NA; a++) grad[st * NA + a] += adv * ((a === acts[t] ? 1 : 0) - probs[t][a]);
      if (want.has(e)) steps.push({ s: st, a: acts[t], r: rews[t], G: G[t], b, adv, probs: probs[t] });
    }
    if (useB) for (let t = 0; t < states.length; t++) V[states[t]] += lrC * (G[t] - V[states[t]]);
    for (let i = 0; i < theta.length; i++) theta[i] += lr * grad[i];
    res.returns[e] = rews.reduce((x, y) => x + y, 0); res.lengths[e] = states.length;
    if (want.has(e)) res.details.push({ episode: e, steps, thetaBefore, thetaAfter: theta.slice(), grad, vBefore });
    record(res, g, theta, e + 1, ev, se, gamma);
  }
  return finish(res, theta, V);
}

export interface ACStep { s: number; a: number; r: number; s2: number; done: boolean; delta: number; v: number; vAfter: number; probsBefore: Float64Array; probsAfter: Float64Array }
export interface ACOpts extends Omit<GridPGOpts, 'baseline' | 'lr' | 'detail'> {
  lrActor?: number;
  /** Keep every step of these episodes (0-based). */
  detail?: number[];
}
/** One-step actor-critic: delta = r + gamma (1 - done) V(s') - V(s); V(s) += lrCritic delta; theta[s] += lrActor delta (onehot - pi). */
export function actorCritic(g: Grid, o: ACOpts = {}): GridPGResult & { steps: ACStep[][] } {
  const lrA = o.lrActor ?? 0.1, lrC = o.lrCritic ?? 0.2, E = o.episodes ?? 500, gamma = o.gamma ?? g.gamma, maxSteps = o.maxSteps ?? 100;
  const ev = o.evalEvery ?? 10, se = o.snapshotEvery ?? 0, want = new Set(o.detail ?? []);
  const rng = createRng(o.seed ?? 13), theta = new Float64Array(N * NA), V = new Float64Array(N), p = new Float64Array(NA);
  const res = pgResult(E, ev, se), steps: ACStep[][] = [];
  record(res, g, theta, 0, ev, se, gamma);
  for (let e = 0; e < E; e++) {
    let s = START, G = 0, t = 0;
    const log: ACStep[] = [];
    for (; t < maxSteps; t++) {
      softmaxRow(theta, s, p);
      const before = p.slice(), a = sampleRow(p, 0, rng.next()), r = step(g, s, a, rng);
      const v = V[s], delta = r.r + (r.done ? 0 : gamma * V[r.s2]) - v;
      V[s] = v + lrC * delta;
      for (let k = 0; k < NA; k++) theta[s * NA + k] += lrA * delta * ((k === a ? 1 : 0) - before[k]);
      if (want.has(e)) log.push({ s, a, r: r.r, s2: r.s2, done: r.done, delta, v, vAfter: V[s], probsBefore: before, probsAfter: softmaxRow(theta, s) });
      G += r.r; s = r.s2;
      if (r.done) { t++; break; }
    }
    res.returns[e] = G; res.lengths[e] = t;
    if (want.has(e)) steps.push(log);
    record(res, g, theta, e + 1, ev, se, gamma);
  }
  return { ...finish(res, theta, V), steps };
}

function pgResult(E: number, ev: number, se: number) {
  const nEval = ev > 0 ? Math.floor(E / ev) + 1 : 0;
  return {
    returns: new Float64Array(E), lengths: new Int32Array(E), evalAt: new Int32Array(nEval), evalValue: new Float64Array(nEval), nEval: 0,
    snapshots: [] as Float64Array[], snapshotAt: [] as number[], details: [] as PGEpisodeDetail[], se,
  };
}
function record(res: ReturnType<typeof pgResult>, g: Grid, theta: Float64Array, done: number, ev: number, se: number, gamma: number) {
  if (ev > 0 && done % ev === 0) { res.evalAt[res.nEval] = done; res.evalValue[res.nEval++] = policyValue(g, softmaxPolicy(theta), gamma)[START]; }
  if (se > 0 && done % se === 0) { res.snapshots.push(theta.slice()); res.snapshotAt.push(done); }
}
function finish(res: ReturnType<typeof pgResult>, theta: Float64Array, V: Float64Array): GridPGResult {
  return {
    returns: res.returns, lengths: res.lengths, evalAt: res.evalAt.slice(0, res.nEval), evalValue: res.evalValue.slice(0, res.nEval),
    snapshots: res.snapshots, snapshotAt: Int32Array.from(res.snapshotAt), details: res.details, theta, V,
  };
}
