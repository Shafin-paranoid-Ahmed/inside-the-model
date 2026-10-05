/** 2D vector playground: add, scale, length, distance, and a 2D point growing into a row of features. Pure logic, no DOM. */

export interface V { x: number; y: number }

// ---- vector helpers (reusable by the dot-product / matrix lessons) ----
export const add = (a: V, b: V): V => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a: V, b: V): V => ({ x: a.x - b.x, y: a.y - b.y });
export const scaleBy = (c: number, a: V): V => ({ x: c * a.x, y: c * a.y });
export const dot = (a: V, b: V) => a.x * b.x + a.y * b.y;
/** Euclidean length |a|. */
export const norm = (a: V) => Math.hypot(a.x, a.y);
/** Distance between the tips of a and b: |a - b|. */
export const dist = (a: V, b: V) => norm(sub(a, b));
export const lerpV = (a: V, b: V, u: number): V => ({ x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u });

export const DEFAULT_A: V = { x: 3, y: 1 };
export const DEFAULT_B: V = { x: 1, y: 2.5 };
/** Arrow tips are kept in [-LIM, LIM] so a + b and 2a stay on the plane. */
export const LIM = 3;
export const DOM = 6.5;

/** A house as a row of features, chosen so its first two entries form the 3-4-5 point (3, 4) after scaling. */
export const FEATURES = [
  { name: 'size (m²)', value: 120, max: 200, unit: 40 },
  { name: 'rooms', value: 4, max: 6, unit: 1 },
  { name: 'age (years)', value: 25, max: 50, unit: 1 },
  { name: 'to centre (km)', value: 2.4, max: 5, unit: 1 },
  { name: 'floors', value: 2, max: 3, unit: 1 },
  { name: 'garden (0/1)', value: 1, max: 1, unit: 1 },
] as const;
/** The house drawn as a point: (size / 40, rooms) = (3, 4). */
export const HOUSE_POINT: V = { x: FEATURES[0].value / FEATURES[0].unit, y: FEATURES[1].value };

// ---- timeline ----
/** 'norm' and 'distance' are the two halves of 'length', for steps that teach one idea each. */
export type Scene = 'add' | 'scale' | 'length' | 'norm' | 'distance' | 'features';
export const DURATION: Record<Scene, number> = { add: 9, scale: 11, length: 10, norm: 5.6, distance: 5.6, features: 13 };
export const duration = (s: Scene) => DURATION[s];
const M = (...m: [number, string][]) => m.map(([at, label]) => ({ at, label }));
export const MARKERS: Record<Scene, { at: number; label: string }[]> = {
  add: M([0, 'a and b'], [2, 'slide b'], [4.5, 'a + b'], [7.2, 'reset']),
  scale: M([0, 'c = 1'], [1, 'stretch'], [3.5, 'shrink'], [5, 'flip'], [8.3, 'back']),
  length: M([0, 'a'], [0.6, 'length'], [4.8, 'distance'], [8.4, 'reset']),
  norm: M([0, 'a'], [0.6, 'length'], [4.6, 'reset']),
  distance: M([0, 'a and b'], [0.6, 'distance'], [4.6, 'reset']),
  features: M([0, 'a point'], [2.6, 'read off'], [5.2, 'add features'], [8.6, 'a row of 6']),
};
export const markers = (s: Scene) => MARKERS[s];

export const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
/** Ramps 0 -> 1 over [u0, u1], holds, then 1 -> 0 over [d0, d1]. */
const pulse = (t: number, u0: number, u1: number, d0: number, d1: number) => ease((t - u0) / (u1 - u0)) * (1 - ease((t - d0) / (d1 - d0)));

/** Scale factor c over time: keyframes (time, c), eased between neighbours. */
const C_KEYS: [number, number][] = [[0, 1], [1, 1], [2.5, 2], [3.5, 2], [5, 0], [7, -1.5], [8.3, -1.5], [10, 1], [11, 1]];
export function scaleAt(t: number): number {
  if (t <= C_KEYS[0][0]) return C_KEYS[0][1];
  for (let i = 1; i < C_KEYS.length; i++) {
    const [t1, c1] = C_KEYS[i], [t0, c0] = C_KEYS[i - 1];
    if (t <= t1) return c0 + (c1 - c0) * ease((t - t0) / (t1 - t0));
  }
  return C_KEYS[C_KEYS.length - 1][1];
}

export interface Frame {
  scene: Scene;
  phase: string;
  /** add: 0..1 slide of the copy of b to a's tip; sum: 0..1 growth of a + b. */
  slide: number;
  sum: number;
  /** scale: current factor c. */
  c: number;
  /** length: 0..1 progress of the legs / of the a-to-b distance segment. */
  legs: number;
  gap: number;
  /** Animated readout: |a| or |a - b| counting up (length scene), else the length of what is shown. */
  lenA: number;
  distAB: number;
  /** features: plane opacity, 0..1 move of the coordinates into the row, number of row cells revealed (fractional). */
  plane: number;
  move: number;
  cells: number;
}

const phaseOf = (t: number, s: Scene) => {
  let out = MARKERS[s][0].label;
  for (const m of MARKERS[s]) if (t >= m.at) out = m.label;
  return out;
};

/** Everything needed to draw at time t. Pure: scrubbing is free. */
export function frameAt(t: number, scene: Scene, a: V = DEFAULT_A, b: V = DEFAULT_B): Frame {
  const f: Frame = { scene, phase: phaseOf(t, scene), slide: 0, sum: 0, c: 1, legs: 0, gap: 0, lenA: 0, distAB: 0, plane: 1, move: 0, cells: 0 };
  if (scene === 'add') {
    f.slide = pulse(t, 2, 4, 7.2, 8.2);
    f.sum = pulse(t, 4.5, 6, 7.2, 8.2);
  } else if (scene === 'scale') {
    f.c = scaleAt(t);
  } else if (scene === 'norm' || scene === 'distance') {
    const p = pulse(t, 0.6, 3.2, 4.6, 5.4);
    if (scene === 'norm') { f.legs = p; f.lenA = norm(a) * p; } else { f.gap = p; f.distAB = dist(a, b) * p; }
  } else if (scene === 'length') {
    f.legs = pulse(t, 0.6, 3.2, 8.4, 9.4);
    f.gap = pulse(t, 4.8, 7.4, 8.4, 9.4);
    f.lenA = norm(a) * f.legs;
    f.distAB = dist(a, b) * f.gap;
  } else {
    f.move = ease((t - 2.6) / 2);
    f.plane = 1 - ease((t - 5) / 1.4) * 0.65;
    f.cells = Math.min(FEATURES.length, 2 + Math.max(0, (t - 5.2) / 0.8));
    if (t < 2.6) f.cells = 0;
    // cells 1 and 2 arrive with the move, the rest one by one
    else if (t < 5.2) f.cells = 2 * f.move;
    // loop reset: dissolve the row so the cycle closes smoothly
    if (t > 12) { f.cells = FEATURES.length * (1 - ease((t - 12) / 1)); f.plane = 1 - 0.65 * (1 - ease((t - 12) / 1)); f.move = 1 - ease((t - 12) / 1); }
  }
  return f;
}
