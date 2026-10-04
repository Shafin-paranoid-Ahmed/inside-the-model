/**
 * Seq2seq with and without Bahdanau attention, on a toy task: reverse a string of letters.
 * Pure logic, no DOM. A small hand-written tanh-RNN encoder-decoder on typed arrays, trained with Adam
 * and checked against numerical gradients in tests.
 *
 *   encoder   h_j = tanh(Wxe e(x_j) + Whe h_{j-1} + be),  h_0 = 0
 *   decoder   s_0 = h_L
 *             context c_i = h_L                                   (no attention: the fixed vector)
 *                     c_i = sum_j a_ij h_j,  a_i = softmax_j(va . tanh(Wa s_{i-1} + Ua h_j))   (attention)
 *             s_i = tanh(Wsd s_{i-1} + Wcd c_i + bd)
 *             p_i = softmax(Wo s_i + Woc c_i + bo)
 * The decoder runs exactly L steps (the output length is known). Unlike real translation decoders it does
 * not take its previous output as input: reversal does not need it, and without it the learned alignment
 * is the plain mirror image instead of a shifted one.
 */
import { createRng, type Rng } from '../../../lib/math/rng';

export const ALPHABET = 'abcdefgh';
export const V = ALPHABET.length;
export const E = 8; // embedding size
export const H = 16; // encoder and decoder state size
export const A = 12; // attention size
export const MIN_LEN = 3, MAX_LEN = 10;

// ---------------------------------------------------------------- parameters
const SHAPES: [string, number, number][] = [
  ['Ex', V, E],
  ['Wxe', H, E], ['Whe', H, H], ['be', H, 1],
  ['Wsd', H, H], ['Wcd', H, H], ['bd', H, 1],
  ['Wa', A, H], ['Ua', A, H], ['va', A, 1],
  ['Wo', V, H], ['Woc', V, H], ['bo', V, 1],
];
export const OFF: Record<string, number> = {};
export const SIZE = SHAPES.reduce((o, [k, r, c]) => { OFF[k] = o; return o + r * c; }, 0);

/** Uniform init with the usual 1/sqrt(fan-in) scale (uniform, not normal: no Math.log or Math.cos, see `exp`). */
export function initParams(rng: Rng): Float64Array {
  const p = new Float64Array(SIZE);
  for (const [k, r, c] of SHAPES) {
    if (k[0] === 'b') continue;
    const a = Math.sqrt(3) * (k === 'Ex' ? 1 : 1 / Math.sqrt(c));
    for (let i = 0; i < r * c; i++) p[OFF[k] + i] = rng.uniform(-a, a);
  }
  return p;
}

// ---------------------------------------------------------------- exp and tanh from + - * / only
// Browsers and Node may round Math.exp / Math.tanh differently in the last bit, and 3000 training updates
// amplify that into different final numbers. Built from correctly rounded basic operations, training gives
// the same numbers everywhere, so the lesson can quote them.
const LN2 = 0.6931471805599453;
const POW2 = new Float64Array(2047); // 2^k for k = -1023..1023, by exact halving and doubling
for (let k = 0, v = 1; k <= 1023; k++, v *= 2) POW2[1023 + k] = v;
for (let k = 1, v = 0.5; k <= 1023; k++, v /= 2) POW2[1023 - k] = v;
export function exp(x: number): number {
  if (x > 700) return Infinity;
  if (x < -700) return 0;
  const k = Math.round(x / LN2), r = x - k * LN2; // |r| <= 0.35
  let t = 1, sum = 1;
  for (let n = 1; n <= 13; n++) { t = (t * r) / n; sum += t; }
  return sum * POW2[k + 1023];
}
export const tanh = (x: number): number => (x > 20 ? 1 : x < -20 ? -1 : 1 - 2 / (exp(2 * x) + 1));

// ---------------------------------------------------------------- small linear algebra on slices
/** out[r] += M[r, :] . x   (M at offset o, rows x cols) */
function mvAdd(p: Float64Array, o: number, rows: number, cols: number, x: ArrayLike<number>, xo: number, out: Float64Array) {
  for (let r = 0; r < rows; r++) {
    let s = 0;
    const b = o + r * cols;
    for (let k = 0; k < cols; k++) s += p[b + k] * x[xo + k];
    out[r] += s;
  }
}
/** dM += d (x) x^T ; dx += M^T d */
function mvBack(p: Float64Array, g: Float64Array, o: number, rows: number, cols: number, x: ArrayLike<number>, xo: number, d: Float64Array, dx: Float64Array | null, dxo = 0) {
  for (let r = 0; r < rows; r++) {
    const dr = d[r];
    if (dr === 0) continue;
    const b = o + r * cols;
    for (let k = 0; k < cols; k++) {
      g[b + k] += dr * x[xo + k];
      if (dx) dx[dxo + k] += dr * p[b + k];
    }
  }
}

// ---------------------------------------------------------------- forward and backward
export interface Pass {
  L: number;
  hs: Float64Array[]; // encoder states h_1..h_L
  ss: Float64Array[]; // decoder states s_0..s_L
  ctx: Float64Array[]; // contexts c_1..c_L
  alpha: Float64Array[]; // attention weights per output step (length L each); empty without attention
  scores: Float64Array[]; // raw attention scores before the softmax
  us: Float64Array[]; // attention hidden tanh per output step i, laid out as L rows of A
  probs: Float64Array[]; // output distributions p_1..p_L
  loss: number;
}

const tanhInPlace = <V extends Float64Array>(v: V): V => { for (let i = 0; i < v.length; i++) v[i] = tanh(v[i]); return v; };

/**
 * Run the model on source x. If `target` is given, `loss` is the summed cross-entropy against it.
 */
export function forward(p: Float64Array, attn: boolean, x: number[], target?: number[]): Pass {
  const L = x.length;
  const hs: Float64Array[] = [];
  let hp = new Float64Array(H);
  for (let j = 0; j < L; j++) {
    const z = new Float64Array(H);
    for (let r = 0; r < H; r++) z[r] = p[OFF.be + r];
    mvAdd(p, OFF.Wxe, H, E, p, OFF.Ex + x[j] * E, z);
    mvAdd(p, OFF.Whe, H, H, hp, 0, z);
    hp = tanhInPlace(z);
    hs.push(hp);
  }
  const keys: Float64Array[] = attn ? hs.map((h) => { const a = new Float64Array(A); mvAdd(p, OFF.Ua, A, H, h, 0, a); return a; }) : [];
  const ss = [hs[L - 1]], ctx: Float64Array[] = [], alpha: Float64Array[] = [], scores: Float64Array[] = [], us: Float64Array[] = [], probs: Float64Array[] = [];
  let loss = 0;
  for (let i = 0; i < L; i++) {
    const sp = ss[i];
    let c: Float64Array;
    if (attn) {
      const q = new Float64Array(A);
      mvAdd(p, OFF.Wa, A, H, sp, 0, q);
      const e = new Float64Array(L), ui = new Float64Array(L * A);
      let mx = -Infinity;
      for (let j = 0; j < L; j++) {
        let s = 0;
        const kj = keys[j];
        for (let k = 0; k < A; k++) { const u = tanh(q[k] + kj[k]); ui[j * A + k] = u; s += p[OFF.va + k] * u; }
        e[j] = s; mx = Math.max(mx, s);
      }
      scores.push(e.slice());
      let tot = 0;
      for (let j = 0; j < L; j++) { e[j] = exp(e[j] - mx); tot += e[j]; }
      for (let j = 0; j < L; j++) e[j] /= tot;
      c = new Float64Array(H);
      for (let j = 0; j < L; j++) for (let r = 0; r < H; r++) c[r] += e[j] * hs[j][r];
      alpha.push(e); us.push(ui);
    } else c = hs[L - 1];
    ctx.push(c);
    const z = new Float64Array(H);
    for (let r = 0; r < H; r++) z[r] = p[OFF.bd + r];
    mvAdd(p, OFF.Wsd, H, H, sp, 0, z);
    mvAdd(p, OFF.Wcd, H, H, c, 0, z);
    const s = tanhInPlace(z);
    ss.push(s);
    const lo = new Float64Array(V);
    for (let r = 0; r < V; r++) lo[r] = p[OFF.bo + r];
    mvAdd(p, OFF.Wo, V, H, s, 0, lo);
    mvAdd(p, OFF.Woc, V, H, c, 0, lo);
    let mx = -Infinity, tot = 0;
    for (const v of lo) mx = Math.max(mx, v);
    for (let r = 0; r < V; r++) { lo[r] = exp(lo[r] - mx); tot += lo[r]; }
    for (let r = 0; r < V; r++) lo[r] /= tot;
    probs.push(lo);
    if (target) loss -= Math.log(lo[target[i]] + 1e-300);
  }
  return { L, hs, ss, ctx, alpha, scores, us, probs, loss };
}

export const argmax = (v: ArrayLike<number>): number => { let b = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[b]) b = i; return b; };

/** Adds d(loss)/d(params) for one example into g. Returns the loss. */
export function backward(p: Float64Array, attn: boolean, x: number[], target: number[], g: Float64Array): number {
  const f = forward(p, attn, x, target), L = f.L;
  const dh = Array.from({ length: L }, () => new Float64Array(H));
  // gradient w.r.t. each key Ua h_j, summed over decoder steps, so Ua is back-propagated once per j
  const dkey = attn ? Array.from({ length: L }, () => new Float64Array(A)) : [];
  let dsNext = new Float64Array(H); // gradient flowing into s_i from step i+1
  for (let i = L - 1; i >= 0; i--) {
    const s = f.ss[i + 1], sp = f.ss[i], c = f.ctx[i], pr = f.probs[i];
    const dlo = new Float64Array(V);
    for (let r = 0; r < V; r++) dlo[r] = pr[r] - (r === target[i] ? 1 : 0);
    for (let r = 0; r < V; r++) g[OFF.bo + r] += dlo[r];
    const ds = dsNext, dc = new Float64Array(H);
    mvBack(p, g, OFF.Wo, V, H, s, 0, dlo, ds);
    mvBack(p, g, OFF.Woc, V, H, c, 0, dlo, dc);
    const dz = new Float64Array(H);
    for (let r = 0; r < H; r++) dz[r] = ds[r] * (1 - s[r] * s[r]);
    for (let r = 0; r < H; r++) g[OFF.bd + r] += dz[r];
    const dsp = new Float64Array(H);
    mvBack(p, g, OFF.Wsd, H, H, sp, 0, dz, dsp);
    mvBack(p, g, OFF.Wcd, H, H, c, 0, dz, dc);
    if (attn) {
      const al = f.alpha[i], dal = new Float64Array(L);
      let dot = 0;
      for (let j = 0; j < L; j++) {
        let d = 0;
        for (let r = 0; r < H; r++) { d += dc[r] * f.hs[j][r]; dh[j][r] += al[j] * dc[r]; }
        dal[j] = d; dot += al[j] * d;
      }
      const dq = new Float64Array(A);
      for (let j = 0; j < L; j++) {
        const de = al[j] * (dal[j] - dot), u = f.us[i], dk = dkey[j], o = j * A;
        for (let k = 0; k < A; k++) {
          const uk = u[o + k];
          g[OFF.va + k] += de * uk;
          const da = de * p[OFF.va + k] * (1 - uk * uk);
          dq[k] += da; dk[k] += da;
        }
      }
      mvBack(p, g, OFF.Wa, A, H, sp, 0, dq, dsp);
    } else for (let r = 0; r < H; r++) dh[L - 1][r] += dc[r];
    dsNext = dsp;
  }
  for (let j = 0; j < dkey.length; j++) mvBack(p, g, OFF.Ua, A, H, f.hs[j], 0, dkey[j], dh[j]);
  for (let r = 0; r < H; r++) dh[L - 1][r] += dsNext[r]; // s_0 = h_L
  for (let j = L - 1; j >= 0; j--) {
    const h = f.hs[j], dz = new Float64Array(H);
    for (let r = 0; r < H; r++) dz[r] = dh[j][r] * (1 - h[r] * h[r]);
    for (let r = 0; r < H; r++) g[OFF.be + r] += dz[r];
    mvBack(p, g, OFF.Wxe, H, E, p, OFF.Ex + x[j] * E, dz, g, OFF.Ex + x[j] * E);
    if (j > 0) mvBack(p, g, OFF.Whe, H, H, f.hs[j - 1], 0, dz, dh[j - 1]);
  }
  return f.loss;
}

// ---------------------------------------------------------------- data and training
export const encode = (s: string): number[] => Array.from(s.toLowerCase()).map((ch) => ALPHABET.indexOf(ch)).filter((i) => i >= 0);
export const decode = (v: number[]): string => v.map((i) => ALPHABET[i]).join('');
export const reverse = (x: number[]): number[] => [...x].reverse();
export function randomSeq(rng: Rng, L: number): number[] { return Array.from({ length: L }, () => Math.floor(rng.next() * V)); }

/** Run without a target: read off the most likely letter at each step with `argmax`. */
export const translate = (p: Float64Array, attn: boolean, x: number[]) => forward(p, attn, x);

export const STEPS = 3000, BATCH = 16, LR = 0.006, CLIP = 5, SNAPS = 30, TEST_PER_LEN = 60;

export interface Run {
  attn: boolean;
  params: Float64Array;
  /** parameter snapshots during training (index 0 = untrained), for the alignment-over-training view */
  snaps: Float64Array[];
  snapStep: number[];
  /** exact-match accuracy (whole reversed string right) per length MIN_LEN..MAX_LEN, on a fixed test set */
  seqAcc: number[];
  /** per-letter accuracy per length */
  tokAcc: number[];
}

export function testSet(): number[][][] {
  const rng = createRng(4242);
  return Array.from({ length: MAX_LEN - MIN_LEN + 1 }, (_, k) => Array.from({ length: TEST_PER_LEN }, () => randomSeq(rng, MIN_LEN + k)));
}

export function evaluate(p: Float64Array, attn: boolean, test = testSet()) {
  const seqAcc: number[] = [], tokAcc: number[] = [];
  for (const group of test) {
    let ok = 0, tok = 0, n = 0;
    for (const x of group) {
      const out = translate(p, attn, x).probs.map(argmax), y = reverse(x);
      let all = true;
      out.forEach((o, i) => { if (o === y[i]) tok++; else all = false; n++; });
      if (all) ok++;
    }
    seqAcc.push(ok / group.length); tokAcc.push(tok / n);
  }
  return { seqAcc, tokAcc };
}

export function train(attn: boolean, steps = STEPS, seed = 8, onProgress?: (f: number) => void): Run {
  const rng = createRng(seed), dataRng = createRng(seed + 1);
  const p = initParams(rng), g = new Float64Array(SIZE), m = new Float64Array(SIZE), v = new Float64Array(SIZE);
  const snaps = [p.slice()], snapStep = [0];
  const b1 = 0.9, b2 = 0.999;
  let p1 = 1, p2 = 1; // b1^t and b2^t as running products (Math.pow may round differently across engines)
  for (let t = 1; t <= steps; t++) {
    g.fill(0);
    for (let b = 0; b < BATCH; b++) {
      const x = randomSeq(dataRng, MIN_LEN + Math.floor(dataRng.next() * (MAX_LEN - MIN_LEN + 1)));
      backward(p, attn, x, reverse(x), g);
    }
    let nrm = 0;
    for (let i = 0; i < SIZE; i++) { g[i] /= BATCH; nrm += g[i] * g[i]; }
    nrm = Math.sqrt(nrm);
    const k = nrm > CLIP ? CLIP / nrm : 1;
    p1 *= b1; p2 *= b2;
    const c1 = 1 - p1, c2 = 1 - p2;
    for (let i = 0; i < SIZE; i++) {
      const gi = g[i] * k;
      m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi;
      p[i] -= (LR * m[i]) / c1 / (Math.sqrt(v[i] / c2) + 1e-8);
    }
    if (t % (steps / SNAPS) === 0) { snaps.push(p.slice()); snapStep.push(t); }
    if (onProgress && t % 50 === 0) onProgress(t / steps);
  }
  return { attn, params: p, snaps, snapStep, ...evaluate(p, attn) };
}

// ---------------------------------------------------------------- timelines (pure: everything drawn at time t)
export type Mode = 'encdec' | 'bottleneck' | 'attention' | 'alignment';
export interface Marker { at: number; label: string }
export const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
export const ease = (v: number) => { const x = clamp01(v); return x * x * (3 - 2 * x); };

/** Encoder-decoder: read L letters, hand over the context, write L letters. */
export const ED = { S: 0.75, hand: 2, hold: 2 };
export function encdecTiming(L: number) {
  const encEnd = L * ED.S, decStart = encEnd + ED.hand;
  return { L, encEnd, decStart, T: decStart + L * ED.S + ED.hold };
}
export type EncDecTiming = ReturnType<typeof encdecTiming>;
export const encdecMarkers = (tm: EncDecTiming): Marker[] => [{ at: 0, label: 'Encode' }, { at: tm.encEnd, label: 'Context' }, { at: tm.decStart, label: 'Decode' }];
export function encdecFrame(t: number, tm: EncDecTiming) {
  const enc = Array.from({ length: tm.L }, (_, j) => ease((t - j * ED.S) / (ED.S * 0.8)));
  const dec = Array.from({ length: tm.L }, (_, i) => ease((t - tm.decStart - i * ED.S) / (ED.S * 0.8)));
  const hand = ease((t - tm.encEnd) / ED.hand);
  const phase = t < tm.encEnd ? 'encode' : t < tm.decStart ? 'hand over' : 'decode';
  const ja = Math.min(tm.L - 1, Math.floor(t / ED.S)), ia = Math.min(tm.L - 1, Math.floor((t - tm.decStart) / ED.S));
  return { phase, enc, dec, hand, active: phase === 'encode' ? ja : phase === 'decode' ? ia : -1 };
}

/** Bottleneck: sweep the length from MIN_LEN to MAX_LEN; the accuracy curve draws as it goes. */
export const BN = { S: 1.6, hold: 2 };
export const LENS = Array.from({ length: MAX_LEN - MIN_LEN + 1 }, (_, k) => MIN_LEN + k);
export const bottleneckT = LENS.length * BN.S + BN.hold;
export const bottleneckMarkers: Marker[] = LENS.map((L, k) => ({ at: k * BN.S, label: String(L) }));
export function bottleneckFrame(t: number) {
  const k = Math.min(LENS.length - 1, Math.max(0, Math.floor(t / BN.S)));
  return { k, L: LENS[k], reveal: ease((t - k * BN.S) / (BN.S * 0.5)) };
}

/** Attention: encode quickly, then for each output step score, normalise, mix, write. */
export const AT = { enc: 0.3, S: 2.4, hold: 1.5 };
export const AT_SUB = ['score', 'weights', 'context', 'write'] as const;
export function attnTiming(L: number) {
  const start = L * AT.enc + 0.4;
  return { L, start, T: start + L * AT.S + AT.hold };
}
export type AttnTiming = ReturnType<typeof attnTiming>;
export const attnMarkers = (tm: AttnTiming): Marker[] => Array.from({ length: tm.L }, (_, i) => ({ at: tm.start + i * AT.S, label: String(i + 1) }));
export function attnFrame(t: number, tm: AttnTiming) {
  const enc = Array.from({ length: tm.L }, (_, j) => ease((t - j * AT.enc) / AT.enc));
  if (t < tm.start) return { i: -1, sub: 0, enc, done: Array(tm.L).fill(0) as number[] };
  const i = Math.min(tm.L - 1, Math.floor((t - tm.start) / AT.S));
  const u = Math.min(1, (t - tm.start - i * AT.S) / (AT.S * 0.85)); // last 15% holds the finished step
  const done = Array.from({ length: tm.L }, (_, k) => (k < i ? 1 : k === i ? ease((u - 0.75) / 0.25) : 0));
  return { i, sub: u, enc, done };
}
/** Progress 0..1 of sub-phase `n` (0 score, 1 weights, 2 context, 3 write) within one output step. */
export const subP = (u: number, n: number) => ease((u - n * 0.25) / 0.25);

/** Alignment over training: one snapshot every AL.S seconds. Markers name segments; the readout gives the exact update. */
export const AL = { S: 0.45, hold: 2.5 };
export const alignT = (snaps: number) => (snaps - 1) * AL.S + AL.hold;
export function alignFrame(t: number, snaps: number) {
  const x = Math.min(snaps - 1, Math.max(0, t / AL.S));
  return { k: Math.floor(x + 1e-9) };
}
export const alignMarkers = (snapStep: number[]): Marker[] => {
  const at = (s: number) => snapStep.indexOf(s) * AL.S;
  return [{ at: 0, label: 'Early' }, { at: at(1000), label: 'Middle' }, { at: at(2000), label: 'Late' }];
};

/** Attention weights of every output step (rows) over every input letter (columns), greedy decoding. */
export function alignment(p: Float64Array, x: number[]) {
  const f = translate(p, true, x);
  return { alpha: f.alpha.map((r) => Array.from(r)), out: f.probs.map(argmax) };
}
