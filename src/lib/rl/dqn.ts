/**
 * DQN on the 4x3 grid: a 2 -> H -> 4 ReLU MLP in one flat Float64Array (normalised cell coordinates in, four
 * Q-values out), a semi-gradient TD loss with a detached target and a terminal mask, a FIFO replay buffer and a
 * hard-copied target network. Pure TS, seeded. Does not use src/lib/nn's autograd (tests use it as an oracle).
 *
 * Weight layout (H hidden units): W1 [H x 2] at 0, b1 [H] at 2H, W2 [4 x H] at 3H, b2 [4] at 7H; 7H + 4 numbers.
 */
import { createRng, type Rng } from '../math/rng';
import { adam } from '../nn/optim';
import { N, NA, START, coords, isTerminal, step, transitions, type Grid } from './gridworld';
import { deterministic, greedy, optimal, policyValue } from './tabular';

export const paramCount = (H: number): number => 7 * H + 4;
export const offsets = (H: number) => ({ W1: 0, b1: 2 * H, W2: 3 * H, b2: 7 * H });

/** He-normal first layer, 1/sqrt(H) second layer, zero biases. */
export function initNet(hidden: number, seed: number): Float64Array {
  const rng = createRng(seed), w = new Float64Array(paramCount(hidden)), o = offsets(hidden);
  for (let i = 0; i < 2 * hidden; i++) w[o.W1 + i] = rng.normal(0, 1);
  for (let i = 0; i < 4 * hidden; i++) w[o.W2 + i] = rng.normal(0, 1 / Math.sqrt(hidden));
  return w;
}

/** Forward pass for input (x0, x1). Fills `pre` (pre-activations), `h` (ReLU) and returns `q`. */
export function forward(w: ArrayLike<number>, H: number, x0: number, x1: number,
  q: Float64Array = new Float64Array(NA), h: Float64Array = new Float64Array(H), pre: Float64Array = new Float64Array(H)): Float64Array {
  const o = offsets(H);
  for (let j = 0; j < H; j++) { const z = w[o.W1 + 2 * j] * x0 + w[o.W1 + 2 * j + 1] * x1 + w[o.b1 + j]; pre[j] = z; h[j] = z > 0 ? z : 0; }
  for (let a = 0; a < NA; a++) { let z = w[o.b2 + a]; for (let j = 0; j < H; j++) z += w[o.W2 + a * H + j] * h[j]; q[a] = z; }
  return q;
}
/** The four Q-values of state s (terminals return zeros: no actions). */
export function qOf(w: ArrayLike<number>, H: number, s: number): Float64Array {
  if (isTerminal(s)) return new Float64Array(NA);
  const [x0, x1] = coords(s);
  return forward(w, H, x0, x1);
}
/** Q-values of every action state (N x 4; terminal and wall rows 0). */
export function allQ(g: Grid, w: ArrayLike<number>, H: number): Float64Array {
  const Q = new Float64Array(N * NA);
  for (const s of g.states) Q.set(qOf(w, H, s), s * NA);
  return Q;
}
// scratch buffers for the hot loops (single-threaded, never returned)
const sq = new Float64Array(NA);
let sh = new Float64Array(16), sp = new Float64Array(16);
const scratch = (H: number) => { if (sh.length !== H) { sh = new Float64Array(H); sp = new Float64Array(H); } };
const maxQ = (w: ArrayLike<number>, H: number, s: number) => {
  if (isTerminal(s)) return 0;
  scratch(H);
  const [x0, x1] = coords(s), q = forward(w, H, x0, x1, sq, sh, sp);
  return Math.max(q[0], q[1], q[2], q[3]);
};

export interface Transition { s: number; a: number; r: number; s2: number; done: boolean }

/** Target y = r + gamma (1 - done) max_a' Q_target(s', a'). A plain number: no gradient flows through it. */
export const tdTarget = (tw: ArrayLike<number>, H: number, t: Transition, gamma: number): number =>
  t.r + (t.done ? 0 : gamma * maxQ(tw, H, t.s2));

/**
 * Mean squared error 0.5 (Q_w(s, a) - y)^2 over items with fixed targets y, and its gradient with respect to w
 * (added into `grad` after it is zeroed). Returns the loss and each item's prediction.
 */
export function regressionGrad(w: ArrayLike<number>, H: number, items: { s: number; a: number; y: number }[], grad: Float64Array) {
  scratch(H);
  const o = offsets(H), B = items.length, h = sh, pre = sp, q = sq, preds = new Float64Array(B);
  grad.fill(0);
  let loss = 0;
  for (let i = 0; i < B; i++) {
    const { s, a, y } = items[i], [x0, x1] = coords(s);
    forward(w, H, x0, x1, q, h, pre);
    const d = (q[a] - y) / B;
    loss += 0.5 * (q[a] - y) ** 2 / B; preds[i] = q[a];
    grad[o.b2 + a] += d;
    for (let j = 0; j < H; j++) {
      grad[o.W2 + a * H + j] += d * h[j];
      if (pre[j] > 0) { const dz = d * w[o.W2 + a * H + j]; grad[o.W1 + 2 * j] += dz * x0; grad[o.W1 + 2 * j + 1] += dz * x1; grad[o.b1 + j] += dz; }
    }
  }
  return { loss, preds };
}
/** The semi-gradient TD loss on a batch: targets from `tw` (pass the online weights for "no target network"), detached. */
export function tdLossGrad(w: ArrayLike<number>, tw: ArrayLike<number>, H: number, batch: Transition[], gamma: number, grad: Float64Array) {
  const ys = batch.map((t) => tdTarget(tw, H, t, gamma));
  return { ...regressionGrad(w, H, batch.map((t, i) => ({ s: t.s, a: t.a, y: ys[i] })), grad), ys };
}

// ---------- replay ----------

/** FIFO replay buffer with uniform sampling without replacement inside a batch. */
export class ReplayBuffer {
  readonly s: Int8Array; readonly a: Int8Array; readonly s2: Int8Array; readonly done: Uint8Array; readonly r: Float64Array;
  /** When each slot's transition was collected (environment step number). */
  readonly born: Int32Array;
  size = 0; next = 0;
  constructor(readonly capacity: number) {
    this.s = new Int8Array(capacity); this.a = new Int8Array(capacity); this.s2 = new Int8Array(capacity);
    this.done = new Uint8Array(capacity); this.r = new Float64Array(capacity); this.born = new Int32Array(capacity);
  }
  /** Store a transition, overwriting the oldest when full. Returns the slot. */
  push(t: Transition, time: number): number {
    const i = this.next;
    this.s[i] = t.s; this.a[i] = t.a; this.s2[i] = t.s2; this.done[i] = +t.done; this.r[i] = t.r; this.born[i] = time;
    this.next = (i + 1) % this.capacity; this.size = Math.min(this.size + 1, this.capacity);
    return i;
  }
  get(i: number): Transition { return { s: this.s[i], a: this.a[i], r: this.r[i], s2: this.s2[i], done: !!this.done[i] }; }
  /** min(n, size) distinct slots, uniformly (partial Fisher-Yates). */
  sample(n: number, rng: Rng): Int32Array {
    const idx = Int32Array.from({ length: this.size }, (_, i) => i), m = Math.min(n, this.size);
    for (let i = 0; i < m; i++) { const j = i + Math.floor(rng.next() * (this.size - i)); const x = idx[i]; idx[i] = idx[j]; idx[j] = x; }
    return idx.slice(0, m);
  }
}

// ---------- training ----------

export interface DQNOpts {
  replay?: boolean; target?: boolean;
  lr?: number; gamma?: number; epsilon?: number; hidden?: number;
  capacity?: number; batch?: number; copyEvery?: number;
  /** Stop after this many episodes or this many updates, whichever comes first. */
  episodes?: number; updates?: number; maxSteps?: number;
  /** Environment/sampling seed and weight-initialisation seed (default: seed). */
  seed?: number; initSeed?: number;
  optimizer?: 'sgd' | 'adam';
  /** 'td' (bootstrapped target) or 'qstar' (supervised fit to the fixed Q* from value iteration, same data). */
  targetKind?: 'td' | 'qstar';
  /** Train on this fixed stream of transitions instead of acting with the network (one per environment step). */
  stream?: Transition[];
  /** Fixed transition whose online Q and target y are recorded after every update. */
  diag?: Transition;
  /** Record per-update diagnostics (default true). */
  recordUpdates?: boolean;
  /** Keep the first n environment steps' transition, batch slots and weights after their update (for replays). */
  keepSteps?: number;
  /** Exact greedy evaluation every `evalEvery` episodes (default 10). */
  evalEvery?: number;
}
export interface DQNStepRecord { t: Transition; slot: number; batch: Int32Array<ArrayBufferLike>; update: number; w: Float64Array; tw: Float64Array }
export interface DQNResult {
  w0: Float64Array; w: Float64Array; hidden: number;
  /** Per update: loss, diagnostic Q and y, max |Q| over all (s, a), ||w - w_target||, RMS Bellman error under the model. */
  loss: Float64Array; diagQ: Float64Array; diagY: Float64Array; maxAbsQ: Float64Array; weightDist: Float64Array; bellman: Float64Array;
  /** Per episode: undiscounted training return, length and updates done when it ended. */
  trainReturn: Float64Array; lengths: Int32Array; updatesAt: Int32Array;
  /** Exact discounted return of the start cell under the greedy policy (no exploration, no training), and max |Q - Q*|. */
  evalAt: Int32Array; evalReturn: Float64Array; qErr: Float64Array;
  steps: DQNStepRecord[];
  updates: number;
}

/** RMS over action states of Q(s,a) - sum_s' P [r + gamma max Q(s')], using the model (a diagnostic only). */
export function bellmanError(g: Grid, Q: ArrayLike<number>, gamma: number): number {
  let e = 0, n = 0;
  for (const s of g.states) for (let a = 0; a < NA; a++) {
    let t = 0;
    for (const o of transitions(g, s, a)) t += o.p * (o.r + (o.done ? 0 : gamma * Math.max(Q[o.s2 * NA], Q[o.s2 * NA + 1], Q[o.s2 * NA + 2], Q[o.s2 * NA + 3])));
    e += (Q[s * NA + a] - t) ** 2; n++;
  }
  return Math.sqrt(e / n);
}
/** Exact discounted start-cell return of the greedy policy of Q. */
export const greedyValue = (g: Grid, Q: ArrayLike<number>, gamma: number): number => policyValue(g, deterministic(greedy(g, Q)), gamma)[START];

export function trainDQN(g: Grid, o: DQNOpts = {}): DQNResult {
  const H = o.hidden ?? 16, gamma = o.gamma ?? 0.9, lr = o.lr ?? 0.01, eps = o.epsilon ?? 0.1, B = o.replay === false ? 1 : o.batch ?? 16;
  const maxU = o.updates ?? Infinity, E = o.episodes ?? (o.updates ? Infinity : 500), maxSteps = o.maxSteps ?? 50, ev = o.evalEvery ?? 10;
  const seed = o.seed ?? 23, rng = createRng(seed), w = initNet(H, o.initSeed ?? seed), w0 = w.slice();
  const useTarget = o.target ?? true, copyEvery = o.copyEvery ?? 50, tw = useTarget ? w.slice() : w;
  const buf = new ReplayBuffer(o.replay === false ? 1 : o.capacity ?? 256), grad = new Float64Array(w.length);
  const opt = o.optimizer === 'sgd' ? null : adam(lr), qstar = o.targetKind === 'qstar' ? optimal({ ...g, gamma }, gamma).Q : null;
  const rec = o.recordUpdates !== false, U: Record<'loss' | 'diagQ' | 'diagY' | 'maxAbsQ' | 'weightDist' | 'bellman', number[]> = { loss: [], diagQ: [], diagY: [], maxAbsQ: [], weightDist: [], bellman: [] };
  const trainReturn: number[] = [], lengths: number[] = [], updatesAt: number[] = [], evalAt: number[] = [], evalReturn: number[] = [], qErr: number[] = [];
  const Qs = qstar ?? optimal({ ...g, gamma }, gamma).Q, steps: DQNStepRecord[] = [], keep = o.keepSteps ?? 0;
  let updates = 0, time = 0, sIdx = 0;
  const evaluate = (episodes: number) => {
    const Q = allQ(g, w, H);
    let m = 0;
    for (const s of g.states) for (let a = 0; a < NA; a++) m = Math.max(m, Math.abs(Q[s * NA + a] - Qs[s * NA + a]));
    evalAt.push(episodes); evalReturn.push(greedyValue(g, Q, gamma)); qErr.push(m);
  };
  evaluate(0);
  for (let e = 0; e < E && updates < maxU; e++) {
    let s = START, G = 0, t = 0;
    for (; t < maxSteps && updates < maxU; t++) {
      let tr: Transition;
      if (o.stream) { if (sIdx >= o.stream.length) break; tr = o.stream[sIdx++]; }
      else {
        scratch(H);
        const [x0, x1] = coords(s), q = forward(w, H, x0, x1, sq, sh, sp);
        let a = 0;
        if (rng.next() < eps) a = Math.floor(rng.next() * NA); else for (let k = 1; k < NA; k++) if (q[k] > q[a]) a = k;
        const r = step(g, s, a, rng);
        tr = { s, a, r: r.r, s2: r.s2, done: r.done };
      }
      const slot = buf.push(tr, time++);
      let batch: Int32Array<ArrayBufferLike> = new Int32Array(0);
      if (buf.size >= B) {
        batch = buf.sample(B, rng);
        const items = Array.from(batch, (i) => buf.get(i));
        const res = qstar
          ? { ...regressionGrad(w, H, items.map((x) => ({ s: x.s, a: x.a, y: qstar[x.s * NA + x.a] })), grad) }
          : tdLossGrad(w, tw, H, items, gamma, grad);
        if (opt) opt.step(w as unknown as number[], grad as unknown as number[]);
        else for (let i = 0; i < w.length; i++) w[i] -= lr * grad[i];
        updates++;
        if (useTarget && updates % copyEvery === 0) tw.set(w);
        if (rec) {
          const Q = allQ(g, w, H);
          let mq = 0, d2 = 0;
          for (let i = 0; i < Q.length; i++) mq = Math.max(mq, Math.abs(Q[i]));
          for (let i = 0; i < w.length; i++) d2 += (w[i] - tw[i]) ** 2;
          U.loss.push(res.loss); U.maxAbsQ.push(mq); U.weightDist.push(Math.sqrt(d2)); U.bellman.push(bellmanError(g, Q, gamma));
          U.diagQ.push(o.diag ? qOf(w, H, o.diag.s)[o.diag.a] : NaN);
          U.diagY.push(o.diag ? (qstar ? qstar[o.diag.s * NA + o.diag.a] : tdTarget(tw, H, o.diag, gamma)) : NaN);
        }
      }
      if (steps.length < keep) steps.push({ t: tr, slot, batch, update: updates, w: w.slice(), tw: tw.slice() });
      G += tr.r; s = tr.s2;
      if (tr.done) { t++; break; }
    }
    trainReturn.push(G); lengths.push(t); updatesAt.push(updates);
    if ((e + 1) % ev === 0) evaluate(e + 1);
    if (o.stream && sIdx >= o.stream.length) break;
  }
  if (evalAt[evalAt.length - 1] !== trainReturn.length) evaluate(trainReturn.length);
  const F = (a: number[]) => Float64Array.from(a);
  return {
    w0, w, hidden: H, loss: F(U.loss), diagQ: F(U.diagQ), diagY: F(U.diagY), maxAbsQ: F(U.maxAbsQ), weightDist: F(U.weightDist), bellman: F(U.bellman),
    trainReturn: F(trainReturn), lengths: Int32Array.from(lengths), updatesAt: Int32Array.from(updatesAt),
    evalAt: Int32Array.from(evalAt), evalReturn: F(evalReturn), qErr: F(qErr), steps, updates,
  };
}

/** A fixed stream of transitions from a uniformly random policy (episodes restart at the start cell). */
export function randomStream(g: Grid, n: number, seed: number, maxSteps = 50): Transition[] {
  const rng = createRng(seed), out: Transition[] = [];
  let s = START, t = 0;
  while (out.length < n) {
    const a = Math.floor(rng.next() * NA), r = step(g, s, a, rng);
    out.push({ s, a, r: r.r, s2: r.s2, done: r.done });
    s = r.s2; t++;
    if (r.done || t >= maxSteps) { s = START; t = 0; }
  }
  return out;
}

export type DQNVariant = 'naive' | 'replay' | 'target' | 'both';
export const VARIANT_FLAGS: Record<DQNVariant, { replay: boolean; target: boolean }> = {
  naive: { replay: false, target: false }, replay: { replay: true, target: false }, target: { replay: false, target: true }, both: { replay: true, target: true },
};
/**
 * The four-variant comparison: run r of every variant shares initial weights (initSeed = seed * 100 + r) and its
 * environment seed (seed * 1000 + r). Returns each run's evaluation curve; per-update diagnostics are off.
 */
export function compareDQN(g: Grid, o: DQNOpts & { variants?: DQNVariant[]; runs?: number } = {}) {
  const runs = o.runs ?? 10, seed = o.seed ?? 23, variants = o.variants ?? ['naive', 'replay', 'target', 'both'];
  return variants.map((v) => {
    const res = Array.from({ length: runs }, (_, r) => trainDQN(g, { ...o, ...VARIANT_FLAGS[v], seed: seed * 1000 + r, initSeed: seed * 100 + r, recordUpdates: false }));
    return { variant: v, evalAt: res[0].evalAt, evalReturn: res.map((x) => x.evalReturn), qErr: res.map((x) => x.qErr), trainReturn: res.map((x) => x.trainReturn), w: res.map((x) => x.w) };
  });
}
