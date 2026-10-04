/**
 * Shared CNN core for the Day 7 lessons: conv2d and pooling with backward passes.
 * Pure typed-array math, no DOM, so it runs in widgets, workers and tests alike.
 * Layout is channel-major: index (c * h + y) * w + x.
 */

export interface Map3 {
  c: number;
  h: number;
  w: number;
  data: Float64Array;
}

export interface ConvLayer {
  inC: number;
  outC: number;
  k: number;
  stride: number;
  pad: number;
  /** outC * inC * k * k, index ((o * inC + i) * k + ky) * k + kx */
  weight: Float64Array;
  /** outC */
  bias: Float64Array;
}

export const map3 = (c: number, h: number, w: number, data = new Float64Array(c * h * w)): Map3 => ({
  c, h, w, data,
});

/** Output length along one axis: floor((n + 2p - k) / s) + 1. */
export const outSize = (n: number, k: number, stride = 1, pad = 0) =>
  Math.floor((n + 2 * pad - k) / stride) + 1;

export function convLayer(inC: number, outC: number, k: number, stride = 1, pad = 0): ConvLayer {
  return {
    inC, outC, k, stride, pad,
    weight: new Float64Array(outC * inC * k * k),
    bias: new Float64Array(outC),
  };
}

/** Cross-correlation (what deep-learning libraries call convolution). */
export function conv2d(x: Map3, L: ConvLayer): Map3 {
  const { inC, outC, k, stride, pad, weight, bias } = L;
  const oh = outSize(x.h, k, stride, pad);
  const ow = outSize(x.w, k, stride, pad);
  const y = map3(outC, oh, ow);
  for (let o = 0; o < outC; o++)
    for (let oy = 0; oy < oh; oy++)
      for (let ox = 0; ox < ow; ox++) {
        let s = bias[o];
        for (let i = 0; i < inC; i++)
          for (let ky = 0; ky < k; ky++) {
            const iy = oy * stride + ky - pad;
            if (iy < 0 || iy >= x.h) continue;
            for (let kx = 0; kx < k; kx++) {
              const ix = ox * stride + kx - pad;
              if (ix < 0 || ix >= x.w) continue;
              s += weight[((o * inC + i) * k + ky) * k + kx] * x.data[(i * x.h + iy) * x.w + ix];
            }
          }
        y.data[(o * oh + oy) * ow + ox] = s;
      }
  return y;
}

/** Given dL/dy, accumulate dL/dweight, dL/dbias into gW/gB and return dL/dx. */
export function conv2dBackward(
  x: Map3, L: ConvLayer, gy: Map3, gW: Float64Array, gB: Float64Array,
): Map3 {
  const { inC, outC, k, stride, pad, weight } = L;
  const gx = map3(x.c, x.h, x.w);
  for (let o = 0; o < outC; o++)
    for (let oy = 0; oy < gy.h; oy++)
      for (let ox = 0; ox < gy.w; ox++) {
        const g = gy.data[(o * gy.h + oy) * gy.w + ox];
        if (g === 0) continue;
        gB[o] += g;
        for (let i = 0; i < inC; i++)
          for (let ky = 0; ky < k; ky++) {
            const iy = oy * stride + ky - pad;
            if (iy < 0 || iy >= x.h) continue;
            for (let kx = 0; kx < k; kx++) {
              const ix = ox * stride + kx - pad;
              if (ix < 0 || ix >= x.w) continue;
              const wi = ((o * inC + i) * k + ky) * k + kx;
              const xi = (i * x.h + iy) * x.w + ix;
              gW[wi] += g * x.data[xi];
              gx.data[xi] += g * weight[wi];
            }
          }
      }
  return gx;
}

export function relu(x: Map3): Map3 {
  return map3(x.c, x.h, x.w, x.data.map((v) => (v > 0 ? v : 0)));
}

/** dL/dx for y = relu(x). */
export function reluBackward(x: Map3, gy: Map3): Map3 {
  return map3(x.c, x.h, x.w, gy.data.map((g, i) => (x.data[i] > 0 ? g : 0)));
}

export type PoolKind = 'max' | 'avg';

/** Pooling per channel with window k and stride (default k), no padding. */
export function pool2d(x: Map3, k: number, kind: PoolKind, stride = k): Map3 {
  const oh = outSize(x.h, k, stride);
  const ow = outSize(x.w, k, stride);
  const y = map3(x.c, oh, ow);
  for (let c = 0; c < x.c; c++)
    for (let oy = 0; oy < oh; oy++)
      for (let ox = 0; ox < ow; ox++) {
        let m = -Infinity;
        let s = 0;
        for (let ky = 0; ky < k; ky++)
          for (let kx = 0; kx < k; kx++) {
            const v = x.data[(c * x.h + oy * stride + ky) * x.w + ox * stride + kx];
            if (v > m) m = v;
            s += v;
          }
        y.data[(c * oh + oy) * ow + ox] = kind === 'max' ? m : s / (k * k);
      }
  return y;
}

/** dL/dx for pooling. Max routes the gradient to the first maximum in each window. */
export function pool2dBackward(x: Map3, k: number, kind: PoolKind, gy: Map3, stride = k): Map3 {
  const gx = map3(x.c, x.h, x.w);
  for (let c = 0; c < x.c; c++)
    for (let oy = 0; oy < gy.h; oy++)
      for (let ox = 0; ox < gy.w; ox++) {
        const g = gy.data[(c * gy.h + oy) * gy.w + ox];
        let best = -1;
        let m = -Infinity;
        for (let ky = 0; ky < k; ky++)
          for (let kx = 0; kx < k; kx++) {
            const xi = (c * x.h + oy * stride + ky) * x.w + ox * stride + kx;
            if (kind === 'avg') gx.data[xi] += g / (k * k);
            else if (x.data[xi] > m) { m = x.data[xi]; best = xi; }
          }
        if (kind === 'max') gx.data[best] += g;
      }
  return gx;
}
