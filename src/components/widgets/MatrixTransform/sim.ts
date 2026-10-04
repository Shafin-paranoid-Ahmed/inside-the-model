/** Matrices as transformations of the plane, plus the shapes rule for m×n times n×p. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';
import { smooth } from '../F1Shared/plot';

/** A 2×2 matrix in row-major order [a, b, c, d] = [[a, b], [c, d]]. Column 1 (a, c) is where î lands, column 2 (b, d) is where ĵ lands. */
export type Mat2 = [number, number, number, number];
export type Vec2 = [number, number];
export const I2: Mat2 = [1, 0, 0, 1];

export const mul2 = (A: Mat2, B: Mat2): Mat2 => [
  A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
  A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3],
];
export const apply = (M: Mat2, x: Vec2): Vec2 => [M[0] * x[0] + M[1] * x[1], M[2] * x[0] + M[3] * x[1]];
export const det = (M: Mat2) => M[0] * M[3] - M[1] * M[2];

export type PresetId = 'scale' | 'rotate' | 'shear' | 'flip' | 'collapse';
const rot = (deg: number): Mat2 => { const r = (deg * Math.PI) / 180; return [Math.cos(r), -Math.sin(r), Math.sin(r), Math.cos(r)]; };
export const PRESETS: Record<PresetId, { name: string; m: Mat2 }> = {
  scale: { name: 'Scale', m: [1.6, 0, 0, 0.6] },
  rotate: { name: 'Rotate 60°', m: rot(60) },
  shear: { name: 'Shear', m: [1, 1, 0, 1] },
  flip: { name: 'Flip', m: [-1, 0, 0, 1] },
  collapse: { name: 'Collapse', m: [1, 2, 0.5, 1] },
};

/**
 * The matrix at progress s in [0, 1] on the way from the identity to M.
 * A pure rotation is interpolated by angle so it does not shrink on the way; everything else is a straight blend of entries.
 */
export function matAt(M: Mat2, s: number): Mat2 {
  const isRot = Math.abs(M[0] - M[3]) < 1e-9 && Math.abs(M[1] + M[2]) < 1e-9 && Math.abs(det(M) - 1) < 1e-9;
  if (isRot) return rot((Math.atan2(M[2], M[0]) * 180 / Math.PI) * s);
  return [1 + (M[0] - 1) * s, M[1] * s, M[2] * s, 1 + (M[3] - 1) * s];
}

export type Mode = 'transform' | 'dot' | 'compose' | 'shapes';

/** Timeline for the plane modes. */
export const MORPH_T0 = 1.2, MORPH_LEN = 3.2;
export const COMPOSE = { first: 1, second: 4.6, len: 2.8, product: 7.4 };
export const duration = (mode: Mode) => (mode === 'compose' ? 10 : 6);
export const markers = (mode: Mode) =>
  mode === 'compose'
    ? [{ at: 0, label: 'start' }, { at: COMPOSE.first, label: 'first' }, { at: COMPOSE.second, label: 'second' }, { at: COMPOSE.product, label: 'product' }]
    : [{ at: 0, label: 'start' }, { at: MORPH_T0, label: 'apply' }, { at: MORPH_T0 + MORPH_LEN, label: 'result' }];
/** How long to hold the last frame before the loop restarts. */
export const HOLD = 1;

export interface PlaneFrame {
  /** The matrix acting on the plane right now. */
  M: Mat2;
  /** Progress of the whole transformation, 0 at the identity, 1 at the target. */
  s: number;
  /** Image of the vector x under M. */
  y: Vec2;
  det: number;
  phase: 'start' | 'apply' | 'result';
  /** 0..1 fade of the "product in one go" outline (compose only). */
  ghost: number;
}

/** Everything needed to draw the plane at time t. `A` is the target (transform, dot) or the second matrix (compose); `B` is the first matrix (compose). */
export function frameAt(t: number, mode: Mode, A: Mat2, x: Vec2, B: Mat2 = I2): PlaneFrame {
  if (mode === 'compose') {
    const s1 = smooth((t - COMPOSE.first) / COMPOSE.len), s2 = smooth((t - COMPOSE.second) / COMPOSE.len);
    const M = mul2(matAt(A, s2), matAt(B, s1));
    return {
      M, s: (s1 + s2) / 2, y: apply(M, x), det: det(M),
      phase: t < COMPOSE.first ? 'start' : t < COMPOSE.product ? 'apply' : 'result',
      ghost: smooth((t - (COMPOSE.product - 0.5)) / 0.7),
    };
  }
  const s = smooth((t - MORPH_T0) / MORPH_LEN), M = matAt(A, s);
  return { M, s, y: apply(M, x), det: det(M), phase: t < MORPH_T0 ? 'start' : t < MORPH_T0 + MORPH_LEN ? 'apply' : 'result', ghost: 0 };
}

/** The four corners of the image of the unit square [0,1]² under M. */
export const squareImage = (M: Mat2): Vec2[] => [[0, 0], [M[0], M[2]], [M[0] + M[1], M[2] + M[3]], [M[1], M[3]]];

// ---------------------------------------------------------------- shapes: (m×n)(q×p)

export type Grid = number[][];
/** A deterministic 4×4 grid of small integers. */
export function makeGrid(seed: number): Grid {
  const rng = createRng(seed);
  return Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => Math.floor(rng.uniform(-2, 4))));
}
export const transposeGrid = (g: Grid): Grid => g[0].map((_, j) => g.map((r) => r[j]));
export const crop = (g: Grid, rows: number, cols: number): Grid => g.slice(0, rows).map((r) => r.slice(0, cols));
export const matmul = (A: Grid, B: Grid): Grid => A.map((r) => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));

export interface Dims { m: number; n: number; q: number; p: number }
/** Time spent on each output cell. */
export const DWELL = 1.4;
export const shapesValid = (d: Dims) => d.n === d.q;
export const shapesDuration = (d: Dims) => (shapesValid(d) ? d.m * d.p * DWELL : 3);
export const shapesMarkers = (d: Dims) => [{ at: 0, label: shapesValid(d) ? 'sweep' : 'mismatch' }];

export interface ShapesFrame {
  valid: boolean;
  /** The output cell being computed. */
  row: number;
  col: number;
  /** How many of the n products are already added into that cell. */
  terms: number;
  /** Partial sum so far. */
  partial: number;
  /** Number of output cells already finished (row-major). */
  done: number;
}

export function shapesFrameAt(t: number, A: Grid, B: Grid, d: Dims): ShapesFrame {
  if (!shapesValid(d)) return { valid: false, row: 0, col: 0, terms: 0, partial: 0, done: 0 };
  const cells = d.m * d.p, k = Math.min(cells - 1, Math.max(0, Math.floor(t / DWELL)));
  const u = Math.min(1, Math.max(0, t / DWELL - k));
  const terms = Math.min(d.n, Math.floor(u * (d.n + 1.5)));
  const row = Math.floor(k / d.p), col = k % d.p;
  let partial = 0;
  for (let l = 0; l < terms; l++) partial += A[row][l] * B[l][col];
  return { valid: true, row, col, terms, partial, done: k };
}
