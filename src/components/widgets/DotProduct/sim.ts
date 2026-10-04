/** Dot product of two 2D vectors: geometry, elementwise products, a neuron. Pure logic, no DOM. Fully deterministic (no randomness). */
export type Vec = [number, number];
/** geometry: shadow + cosine curve. bars: shadow + elementwise products. neuron: w·x + bias. */
export type Mode = 'geometry' | 'bars' | 'neuron';

export const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1];
export const norm = (a: Vec) => Math.hypot(a[0], a[1]);
export const cosSim = (a: Vec, b: Vec) => { const d = norm(a) * norm(b); return d === 0 ? 0 : dot(a, b) / d; };
export const polar = (r: number, ang: number): Vec => [r * Math.cos(ang), r * Math.sin(ang)];
export const angleOf = (a: Vec) => Math.atan2(a[1], a[0]);
/** Orthogonal projection of b onto the line through a (the "shadow"). */
export function project(b: Vec, a: Vec): Vec {
  const n = dot(a, a), k = n === 0 ? 0 : dot(a, b) / n;
  return [a[0] * k, a[1] * k];
}
/** Unsigned angle between a and b in [0, pi]. */
export const angleBetween = (a: Vec, b: Vec) => Math.acos(Math.max(-1, Math.min(1, cosSim(a, b))));
export const relu = (z: number) => Math.max(0, z);

export const DOM = 3.2, RMAX = 3;
export const A_DEF: Vec = polar(2.4, (25 * Math.PI) / 180);
export const W_DEF: Vec = [1.6, 0.8];
export const BIAS = -0.6;
export const B_LEN = 1.8, X_LEN = 1.6;

export const T = 12;
export const MARKERS = [{ at: 0, label: '0°–90°' }, { at: T / 4, label: '90°–180°' }, { at: T / 2, label: '180°–270°' }, { at: (3 * T) / 4, label: '270°–360°' }];

export interface Item { label: string; v: number }
export interface Frame {
  a: Vec; b: Vec; dot: number; cos: number;
  /** Unsigned angle between a and b, radians in [0, pi]. */
  theta: number;
  /** Angle of b measured counterclockwise from a, degrees in [0, 360). */
  theta360: number;
  proj: Vec;
  items: Item[];
  /** Sum of items: the dot product (bars) or the pre-activation z = w·x + bias (neuron). */
  total: number;
  /** ReLU(total), neuron mode only. */
  out: number | null;
}

/** b rotates once around a over T seconds. `ov` replaces a and/or b (learner dragged them). */
export function frameAt(t: number, mode: Mode, ov: { a?: Vec; b?: Vec } = {}): Frame {
  const u = (((t % T) + T) % T) / T;
  const neuron = mode === 'neuron';
  const a = ov.a ?? (neuron ? W_DEF : A_DEF);
  const b = ov.b ?? polar(neuron ? X_LEN : B_LEN, angleOf(a) + 2 * Math.PI * u);
  const d = dot(a, b);
  const items: Item[] = neuron
    ? [{ label: 'w1·x1', v: a[0] * b[0] }, { label: 'w2·x2', v: a[1] * b[1] }, { label: 'bias', v: BIAS }]
    : [{ label: 'a1·b1', v: a[0] * b[0] }, { label: 'a2·b2', v: a[1] * b[1] }];
  const total = items.reduce((s, i) => s + i.v, 0);
  const cross = a[0] * b[1] - a[1] * b[0];
  const th360 = ((Math.atan2(cross, d) * 180) / Math.PI + 360) % 360;
  return { a, b, dot: d, cos: cosSim(a, b), theta: angleBetween(a, b), theta360: th360, proj: project(b, a), items, total, out: neuron ? relu(total) : null };
}

/** One-sentence narration of the current state, for the aria-live region. */
export function narrate(f: Frame, mode: Mode): string {
  if (mode === 'neuron') {
    return f.total > 0
      ? 'The neuron fires: z is positive, so ReLU passes it through.'
      : 'The neuron is silent: z is not positive, so ReLU outputs 0.';
  }
  const m = norm(f.a) * norm(f.b);
  if (f.cos > 0.99) return 'Pointing the same way: the shadow of b covers all of b and the dot product is as large as it can be.';
  if (f.cos < -0.99) return 'Pointing opposite ways: the shadow of b points backwards along a, so the dot product is as negative as it can be.';
  if (Math.abs(f.dot) < 0.02 * m) return 'Perpendicular: b casts no shadow on a, and the dot product is 0.';
  return f.dot > 0
    ? 'Less than 90 degrees apart: b casts a forward shadow on a, so the dot product is positive.'
    : 'More than 90 degrees apart: b casts a backward shadow on a, so the dot product is negative.';
}
