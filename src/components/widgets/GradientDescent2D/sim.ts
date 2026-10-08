/** Gradient descent on 2D loss surfaces: exact gradients, full-batch vs noisy (SGD-like) steps. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';
import { POINTS, START, BEST, loss, grad } from '../LineFit/sim';

export type SurfaceId = 'regression' | 'bowl' | 'multi' | 'saddle';
export interface Surface {
  id: SurfaceId;
  name: string;
  f(x: number, y: number): number;
  grad(x: number, y: number): [number, number];
  xd: [number, number];
  yd: [number, number];
  /** Contour levels. */
  levels: number[];
  /** Largest Hessian eigenvalue (quadratics only): plain gradient descent diverges when lr > 2 / hmax. */
  hmax?: number;
  start: [number, number];
  lr: number;
}

/** Elliptical bowl f = (x² + 8y²)/2: curvature 1 along x and 8 along y, so lr must stay below 2/8 = 0.25. */
const bowl: Surface = {
  id: 'bowl', name: 'Bowl',
  f: (x, y) => 0.5 * (x * x + 8 * y * y),
  grad: (x, y) => [x, 8 * y],
  xd: [-4, 4], yd: [-2.2, 2.2],
  levels: [0.25, 0.75, 1.5, 2.5, 4, 6, 9, 13, 18, 25],
  hmax: 8, start: [3, 1.3], lr: 0.1,
};
/** Two valleys, the left one lower: where you start decides where you end up. */
const multi: Surface = {
  id: 'multi', name: 'Two valleys',
  f: (x, y) => 0.05 * (x * x - 4) ** 2 + 0.3 * x + 0.5 * y * y,
  grad: (x, y) => [0.2 * x * (x * x - 4) + 0.3, y],
  xd: [-4, 4], yd: [-2.6, 2.6],
  levels: [-0.7, -0.4, 0, 0.5, 1, 1.6, 2.4, 3.4, 4.6, 6.2],
  start: [1.2, 1.6], lr: 0.2,
};
/** A saddle at the origin: a minimum along x, a maximum along y. */
const saddle: Surface = {
  id: 'saddle', name: 'Saddle',
  f: (x, y) => 0.5 * x * x + 0.05 * y ** 4 - 0.5 * y * y,
  grad: (x, y) => [x, 0.2 * y ** 3 - y],
  xd: [-3, 3], yd: [-3.4, 3.4],
  levels: [-1.2, -0.9, -0.6, -0.3, -0.1, 0.1, 0.4, 0.9, 1.6, 2.6, 4],
  start: [2.6, 0.03], lr: 0.2,
};
const meanX = POINTS.reduce((s, p) => s + p.x, 0) / POINTS.length;
const meanXX = POINTS.reduce((s, p) => s + p.x * p.x, 0) / POINTS.length;
/** Exactly the twelve-example MSE from the preceding linear-regression lesson. */
const regression: Surface = {
  id: 'regression', name: 'Regression MSE', f: loss, grad,
  xd: [-2.5, 4], yd: [-3.2, 3.6],
  levels: [0.4, 1, 2, 3.5, 5.5, 8, 12, 17, 24].map(v => loss(BEST.w, BEST.b) + v),
  hmax: meanXX + 1 + Math.hypot(meanXX - 1, 2 * meanX),
  start: [START.w, START.b], lr: 0.08,
};
export const SURFACES: Record<SurfaceId, Surface> = { regression, bowl, multi, saddle };

export const NOISE_STD = 1.5;
export const STEPS = 48, DT = 0.2;
export const duration = () => STEPS * DT;
export const markers = () => [
  { at: 0, label: 'early' }, { at: 16 * DT, label: 'middle' },
  { at: 32 * DT, label: 'late' }, { at: STEPS * DT, label: 'done' },
];

export interface Step {
  x: number; y: number; f: number;
  /** The gradient used for the step leaving this point (noisy when SGD is on). */
  gx: number; gy: number;
  /** True gradient here. */
  tx: number; ty: number;
  diverged: boolean;
  /** One-example SGD: index of the training example used for this update. */
  sample?: number;
}

/** Regression SGD samples one real example; other surfaces use seeded Gaussian noise. */
export function trajectory(s: Surface, start: [number, number], lr: number, noise = false, seed = 1, steps = STEPS): Step[] {
  const rng = createRng(seed);
  let [x, y] = start;
  const out: Step[] = [];
  let dead = false;
  for (let i = 0; i <= steps; i++) {
    const [tx, ty] = s.grad(x, y);
    const sample = noise && s.id === 'regression' ? Math.floor(rng.uniform(0, POINTS.length)) : undefined;
    const [gx, gy] = sample !== undefined ? grad(x, y, [POINTS[sample]])
      : [tx + (noise ? rng.normal(0, NOISE_STD) : 0), ty + (noise ? rng.normal(0, NOISE_STD) : 0)];
    out.push({ x, y, f: s.f(x, y), gx, gy, tx, ty, diverged: dead, sample });
    if (!dead) {
      x -= lr * gx; y -= lr * gy;
      if (!(Math.abs(x) < 1e3 && Math.abs(y) < 1e3)) { dead = true; x = Math.sign(x || 1) * 1e3; y = Math.sign(y || 1) * 1e3; }
    }
  }
  return out;
}

const ease = (u: number) => u * u * (3 - 2 * u);
export interface Frame {
  x: number; y: number; f: number;
  /** Step index k (integer) and progress through it (0..1). */
  k: number; u: number;
  /** Gradient arrow at step k: fades out as the ball leaves. */
  arrow: { x: number; y: number; gx: number; gy: number; alpha: number };
  diverged: boolean;
}
/** Everything needed to draw at time t: the ball glides from step k to k+1 along -lr·g. */
export function frameAt(t: number, tr: Step[], lr: number, surface?: Surface): Frame {
  const n = tr.length - 1, position = Math.max(0, Math.min(n, t / DT));
  const raw = Math.abs(position - Math.round(position)) < 1e-9 ? Math.round(position) : position;
  const k = Math.min(n - 1, Math.floor(raw)), u = raw - k;
  const a = tr[k], b = tr[k + 1], e = ease(u);
  const x = a.x + (b.x - a.x) * e, y = a.y + (b.y - a.y) * e;
  return {
    x, y, f: surface ? surface.f(x, y) : a.f + (b.f - a.f) * e, k, u,
    arrow: { x: a.x, y: a.y, gx: a.gx * lr, gy: a.gy * lr, alpha: a.diverged ? 0 : 1 - ease(Math.min(1, Math.max(0, (u - 0.25) / 0.6))) },
    diverged: b.diverged || !(Math.abs(b.f) < 1e6),
  };
}

/** The slice through (x, y) along the descent direction: s = distance along -g/|g|. */
export function slice(sf: Surface, x: number, y: number, gx: number, gy: number, half: number, n = 81): { s: number[]; v: number[]; dir: [number, number] } {
  const m = Math.hypot(gx, gy) || 1, dir: [number, number] = [-gx / m, -gy / m];
  const s = Array.from({ length: n }, (_, i) => -half + (2 * half * i) / (n - 1));
  return { s, v: s.map((q) => sf.f(x + dir[0] * q, y + dir[1] * q)), dir };
}
