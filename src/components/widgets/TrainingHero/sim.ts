// Pure, DOM-free math for TrainingHero. Plain numbers only (no autograd Values).
import { createRng } from '../../../lib/math/rng';

export const SIZES = [2, 6, 6, 1] as const;
export const EXTENT = 1.6; // plot domain is [-EXTENT, EXTENT] on both axes
export type Kind = 'moons' | 'circle' | 'xor' | 'spiral';
export const KINDS: readonly Kind[] = ['moons', 'circle', 'xor', 'spiral'];

/** Offset of layer l's first parameter. Layout matches MLP.parameters(): layer > neuron > [w..., b]. */
export const layerOffset = (l: number): number => {
  let o = 0;
  for (let k = 0; k < l; k++) o += SIZES[k + 1] * (SIZES[k] + 1);
  return o;
};
export const PARAMS = layerOffset(SIZES.length - 1);
/** Index of the weight from input i to neuron j of layer l (l = 0 is the first hidden layer). */
export const wIndex = (l: number, j: number, i: number): number => layerOffset(l) + j * (SIZES[l] + 1) + i;

export interface Dataset { xs: Float32Array; ys: Float32Array } // xs = [x0,y0,x1,y1,...], ys = ±1

export function makeDataset(kind: Kind, seed = 7, n = 120): Dataset {
  const rng = createRng(seed);
  const xs = new Float32Array(n * 2), ys = new Float32Array(n);
  const lim = (v: number) => Math.max(-EXTENT + 0.08, Math.min(EXTENT - 0.08, v));
  const put = (i: number, x: number, y: number, c: number) => { xs[2 * i] = lim(x); xs[2 * i + 1] = lim(y); ys[i] = c; };
  for (let i = 0; i < n; i++) {
    const c = i % 2 === 0 ? 1 : -1;
    const k = Math.floor(i / 2), m = n / 2;
    if (kind === 'moons') {
      const t = Math.PI * rng.next();
      const [x, y] = c > 0 ? [Math.cos(t) - 0.5, Math.sin(t) - 0.25] : [0.5 - Math.cos(t), 0.25 - Math.sin(t)];
      put(i, (x + rng.normal(0, 0.11)) * 1.05, (y + rng.normal(0, 0.11)) * 1.05, c);
    } else if (kind === 'circle') {
      const a = rng.uniform(0, 2 * Math.PI), r = c > 0 ? rng.normal(0.45, 0.1) : rng.normal(1.15, 0.1);
      put(i, r * Math.cos(a), r * Math.sin(a), c);
    } else if (kind === 'xor') {
      const sx = rng.next() < 0.5 ? -1 : 1, sy = c * sx;
      put(i, sx * rng.uniform(0.15, 1.3), sy * rng.uniform(0.15, 1.3), c);
    } else {
      const t = (k + 0.5) / m, a = 2.7 * Math.PI * t + (c > 0 ? 0 : Math.PI), r = 0.15 + 1.25 * t;
      put(i, r * Math.cos(a) + rng.normal(0, 0.05), r * Math.sin(a) + rng.normal(0, 0.05), c);
    }
  }
  return { xs, ys };
}

const scratch = () => SIZES.map((n) => new Float64Array(n));

/** Plain-number forward pass (tanh everywhere, like the engine MLP with outAct 'tanh'). Output in (-1, 1). */
export function forward(w: ArrayLike<number>, x0: number, x1: number, buf: Float64Array[] = scratch()): number {
  buf[0][0] = x0; buf[0][1] = x1;
  for (let l = 0; l < SIZES.length - 1; l++) {
    const nin = SIZES[l], nout = SIZES[l + 1], a = buf[l], out = buf[l + 1];
    for (let j = 0; j < nout; j++) {
      const o = layerOffset(l) + j * (nin + 1);
      let s = w[o + nin];
      for (let i = 0; i < nin; i++) s += w[o + i] * a[i];
      out[j] = Math.tanh(s);
    }
  }
  return buf[SIZES.length - 1][0];
}

/** Fill out (res*res, row-major, top row = +EXTENT) with the model output over the plot domain. */
export function field(w: ArrayLike<number>, res: number, out: Float32Array): void {
  const buf = scratch();
  for (let r = 0; r < res; r++) {
    const y = EXTENT - ((r + 0.5) / res) * 2 * EXTENT;
    for (let c = 0; c < res; c++) out[r * res + c] = forward(w, ((c + 0.5) / res) * 2 * EXTENT - EXTENT, y, buf);
  }
}
