/**
 * Small typed-array MLPs for the Day 13 generative widgets, pure and DOM-free.
 * Layout of one parameter vector for sizes [n0, n1, ..., nL]: for each layer l, W_l (n_{l+1} x n_l, row-major) then b_l.
 * Hidden layers use tanh; the last layer is linear. Batches are row-major Float64Arrays (B x n).
 *
 * exp, tanh, log, sin, cos and normal are built from + - * / (and the exact sqrt) only. Browsers and Node may round
 * Math.exp & co. differently in the last bit, and thousands of training steps amplify that, so building them from
 * correctly rounded basic operations makes Chrome and Node train bit-identically (the Seq2SeqAttention trick).
 */
import type { Rng } from '../math/rng';

// ---------------------------------------------------------------- deterministic elementary functions
const LN2 = 0.6931471805599453;
const POW2 = new Float64Array(2047); // 2^k for k = -1023..1023, by exact halving and doubling
for (let k = 0, v = 1; k <= 1023; k++, v *= 2) POW2[1023 + k] = v;
for (let k = 1, v = 0.5; k <= 1023; k++, v /= 2) POW2[1023 - k] = v;

export function exp(x: number): number {
  if (x > 700) return Infinity;
  if (x < -700) return 0;
  const k = Math.round(x / LN2), r = x - k * LN2; // |r| <= 0.35
  let sum = INV_FACT[13];
  for (let n = 12; n >= 0; n--) sum = sum * r + INV_FACT[n]; // Horner
  return sum * POW2[k + 1023];
}
const INV_FACT = new Float64Array(14);
for (let n = 0, f = 1; n <= 13; n++) { if (n) f *= n; INV_FACT[n] = 1 / f; }
export const tanh = (x: number): number => (x > 20 ? 1 : x < -20 ? -1 : 1 - 2 / (exp(2 * x) + 1));
export const sigmoid = (x: number): number => (x >= 0 ? 1 / (1 + exp(-x)) : exp(x) / (1 + exp(x)));

/** Natural log for x > 0: scale into [sqrt(1/2), sqrt(2)) by exact powers of 2, then 2 atanh((m-1)/(m+1)). */
export function log(x: number): number {
  if (!(x > 0)) return x === 0 ? -Infinity : NaN;
  let m = x, k = 0;
  while (m >= 1.4142135623730951) { m /= 2; k++; }
  while (m < 0.7071067811865476) { m *= 2; k--; }
  const s = (m - 1) / (m + 1), s2 = s * s; // |s| <= 0.172
  let t = s, sum = s;
  for (let n = 3; n <= 31; n += 2) { t *= s2; sum += t / n; }
  return 2 * sum + k * LN2;
}

const TAU = 6.283185307179586;
/** sin and cos by reduction to [-pi, pi] and a Taylor series. Accurate to ~1e-15 for |x| < 1e3. */
export function sin(x: number): number {
  const r = x - Math.round(x / TAU) * TAU, r2 = r * r;
  let t = r, sum = r;
  for (let n = 2; n <= 30; n += 2) { t = (-t * r2) / (n * (n + 1)); sum += t; }
  return sum;
}
export const cos = (x: number): number => sin(x + 1.5707963267948966);

/** Standard normal sample (Marsaglia polar method on the deterministic log). */
export function normal(rng: Rng): number {
  for (;;) {
    const u = 2 * rng.next() - 1, v = 2 * rng.next() - 1, s = u * u + v * v;
    if (s > 0 && s < 1) return u * Math.sqrt((-2 * log(s)) / s);
  }
}

/** Time embedding for diffusion and flow nets: [t, sin(pi t), cos(pi t), sin(2 pi t), cos(2 pi t)] at out[o..o+4]. */
export const TIME_FEATURES = 5;
export function timeFeatures(t: number, out: Float64Array, o = 0): void {
  const a = Math.PI * t;
  out[o] = t; out[o + 1] = sin(a); out[o + 2] = cos(a); out[o + 3] = sin(2 * a); out[o + 4] = cos(2 * a);
}

// ---------------------------------------------------------------- the MLP
export const paramCount = (sizes: number[]): number => {
  let n = 0;
  for (let l = 0; l + 1 < sizes.length; l++) n += sizes[l + 1] * (sizes[l] + 1);
  return n;
};

/** Uniform init with variance 1/fan-in (uniform, so no log or cos is needed); biases zero. */
export function initMlp(sizes: number[], rng: Rng): Float64Array {
  const p = new Float64Array(paramCount(sizes));
  let o = 0;
  for (let l = 0; l + 1 < sizes.length; l++) {
    const nin = sizes[l], nout = sizes[l + 1], a = Math.sqrt(3 / nin);
    for (let i = 0; i < nout * nin; i++) p[o + i] = rng.uniform(-a, a);
    o += nout * (nin + 1);
  }
  return p;
}

/** Activations of one batch: acts[0] is the input, acts[l] the output of layer l (tanh, last one linear). dIn gets dLoss/dInput after backward. */
export interface Cache { B: number; acts: Float64Array[]; deltas: Float64Array[]; dIn: Float64Array }
export function createCache(sizes: number[], B: number): Cache {
  return {
    B,
    acts: sizes.map((n) => new Float64Array(B * n)),
    deltas: sizes.map((n) => new Float64Array(B * n)),
    dIn: new Float64Array(B * sizes[0]),
  };
}

/** Forward pass of the first B rows of X (B x sizes[0]). Returns the output rows (cache.acts[L], valid until the next call). */
export function forward(p: Float64Array, sizes: number[], X: ArrayLike<number>, B: number, cache: Cache): Float64Array {
  const L = sizes.length - 1, a0 = cache.acts[0];
  for (let i = 0; i < B * sizes[0]; i++) a0[i] = X[i];
  let o = 0;
  for (let l = 0; l < L; l++) {
    const nin = sizes[l], nout = sizes[l + 1], A = cache.acts[l], Z = cache.acts[l + 1], bo = o + nout * nin, last = l === L - 1;
    for (let r = 0; r < B; r++) {
      for (let j = 0; j < nout; j++) {
        let s = p[bo + j];
        const w = o + j * nin, x = r * nin;
        for (let i = 0; i < nin; i++) s += p[w + i] * A[x + i];
        Z[r * nout + j] = last ? s : tanh(s);
      }
    }
    o = bo + nout;
  }
  return cache.acts[L];
}

/** Backward pass after forward(): dY is dLoss/dOutput (B x sizes[L]). ADDS parameter gradients into grad and writes dLoss/dInput into cache.dIn. */
export function backward(p: Float64Array, sizes: number[], cache: Cache, dY: ArrayLike<number>, grad: Float64Array, B = cache.B): void {
  const L = sizes.length - 1;
  const offs: number[] = [];
  for (let l = 0, o = 0; l < L; l++) { offs.push(o); o += sizes[l + 1] * (sizes[l] + 1); }
  let D = cache.deltas[L];
  for (let i = 0; i < B * sizes[L]; i++) D[i] = dY[i];
  for (let l = L - 1; l >= 0; l--) {
    const nin = sizes[l], nout = sizes[l + 1], o = offs[l], bo = o + nout * nin, A = cache.acts[l];
    const Din = l === 0 ? cache.dIn : cache.deltas[l];
    for (let i = 0; i < B * nin; i++) Din[i] = 0;
    for (let r = 0; r < B; r++) {
      for (let j = 0; j < nout; j++) {
        const d = D[r * nout + j];
        if (d === 0) continue;
        grad[bo + j] += d;
        const w = o + j * nin, x = r * nin;
        for (let i = 0; i < nin; i++) { grad[w + i] += d * A[x + i]; Din[x + i] += d * p[w + i]; }
      }
    }
    if (l > 0) for (let i = 0; i < B * nin; i++) { const a = A[i]; Din[i] *= 1 - a * a; } // tanh'
    D = Din;
  }
}

// ---------------------------------------------------------------- Adam
export interface AdamState { m: Float64Array; v: Float64Array; p1: number; p2: number; b1: number; b2: number }
export const createAdam = (n: number, b1 = 0.9, b2 = 0.999): AdamState => ({ m: new Float64Array(n), v: new Float64Array(n), p1: 1, p2: 1, b1, b2 });

/** One Adam step on p with gradient g. b1^t, b2^t are running products (Math.pow may round differently across engines). */
export function adam(p: Float64Array, g: Float64Array, st: AdamState, lr: number): void {
  const { m, v, b1, b2 } = st;
  st.p1 *= b1; st.p2 *= b2;
  const c1 = 1 - st.p1, c2 = 1 - st.p2;
  for (let i = 0; i < p.length; i++) {
    const gi = g[i];
    m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
  }
}
