// Arithmetic-only exp/tanh, matching Seq2SeqAttention/sim.ts. No platform libm exp in training.
const LN2 = 0.6931471805599453;
const POW2 = new Float64Array(2047);
for (let k = 0, v = 1; k <= 1023; k++, v *= 2) POW2[1023 + k] = v;
for (let k = 1, v = 0.5; k <= 1023; k++, v /= 2) POW2[1023 - k] = v;
export function exp(x: number): number {
  if (x > 700) return Infinity;
  if (x < -700) return 0;
  const k = Math.round(x / LN2), r = x - k * LN2;
  let t = 1, sum = 1;
  for (let n = 1; n <= 13; n++) { t = t * r / n; sum += t; }
  return sum * POW2[k + 1023];
}
export const tanh = (x: number): number => x > 20 ? 1 : x < -20 ? -1 : 1 - 2 / (exp(2 * x) + 1);

/** Arithmetic-only rotations also keep RoPE training independent of platform sin/cos. */
export function sinCos(angle: number): [number, number] {
  const pi = 3.141592653589793;
  const x = angle - Math.round(angle / (2 * pi)) * (2 * pi);
  let sin = x, cos = 1, st = x, ct = 1;
  for (let k = 1; k <= 14; k++) {
    st *= -x * x / ((2 * k) * (2 * k + 1)); sin += st;
    ct *= -x * x / ((2 * k - 1) * (2 * k)); cos += ct;
  }
  return [sin, cos];
}

export function softmax(values: ArrayLike<number>): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) max = Math.max(max, values[i]);
  if (!Number.isFinite(max)) throw new Error('Softmax needs at least one finite logit');
  const p = new Float64Array(values.length);
  let total = 0;
  for (let i = 0; i < p.length; i++) { p[i] = exp(values[i] - max); total += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= total;
  return p;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error('Cosine dimensions differ');
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return aa && bb ? dot / Math.sqrt(aa * bb) : 0;
}
