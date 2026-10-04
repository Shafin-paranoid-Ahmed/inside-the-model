/** Pure logic for KMeansLab: Lloyd's k-means on 2-D points, precomputed once, then replayed on a continuous time t. No DOM. */
import { createRng, type Rng } from '../../../lib/math/rng';

export type Mode = 'structure' | 'assign' | 'update' | 'init' | 'elbow';
export type DataName = 'three' | 'four';

/** The plane shows [-DOM, DOM]^2. */
export const DOM = 1.5;
export const K_MAX = 8;
/** Timeline layout (seconds): 0..T0 initial centroids appear; then per iteration DA (assign) + DU (update); then HOLD. */
export const T0 = 1.6, DA = 1.3, DU = 1.4, HOLD = 1.5, P = DA + DU;
/** Structure mode: points stay grey until T_REVEAL, colours fade in over REVEAL, centroids fade in after. */
export const T_REVEAL = 3, REVEAL = 2.2, T_STRUCT = 8;

const CENTRES: Record<DataName, [number, number][]> = {
  three: [[-0.75, 0.5], [0.7, 0.6], [0.05, -0.75]],
  four: [[-1, 0.4], [-0.95, -0.5], [0.6, 0.55], [0.65, -0.7]],
};
export const defaultK = (d: DataName) => CENTRES[d].length;

export interface Data { pts: Float64Array; group: Uint8Array; n: number }

/** Blobs of `per` points around each centre (interleaved x, y). `group` is only ever used to colour the lesson's "hidden truth"; k-means never sees it. */
export function makeData(name: DataName, seed = 11, per = 40, sd = name === 'three' ? 0.27 : 0.2): Data {
  const rng = createRng(seed + (name === 'three' ? 1 : 2));
  const cs = CENTRES[name], n = cs.length * per, pts = new Float64Array(2 * n), group = new Uint8Array(n);
  const clip = (v: number) => Math.max(-DOM + 0.08, Math.min(DOM - 0.08, v));
  cs.forEach(([cx, cy], g) => {
    for (let i = 0; i < per; i++) {
      const j = g * per + i;
      pts[2 * j] = clip(rng.normal(cx, sd)); pts[2 * j + 1] = clip(rng.normal(cy, sd)); group[j] = g;
    }
  });
  return { pts, group, n };
}

/** Append one point (used by click-to-add). */
export function withPoint(d: Data, x: number, y: number): Data {
  const pts = new Float64Array(2 * (d.n + 1)); pts.set(d.pts); pts[2 * d.n] = x; pts[2 * d.n + 1] = y;
  const group = new Uint8Array(d.n + 1); group.set(d.group);
  return { pts, group, n: d.n + 1 };
}

export type Init = 'random' | 'plusplus';

/** Initial centroids: `random` = k distinct data points; `plusplus` = k-means++ (each next centroid sampled with probability proportional to squared distance to the nearest chosen one). */
export function initCentroids(pts: Float64Array, k: number, how: Init, seed: number): Float64Array {
  const N = pts.length / 2, rng: Rng = createRng(seed), C = new Float64Array(2 * k), idx: number[] = [];
  const put = (i: number) => { C[2 * idx.length] = pts[2 * i]; C[2 * idx.length + 1] = pts[2 * i + 1]; idx.push(i); };
  if (how === 'random') {
    while (idx.length < k) { const i = Math.floor(rng.next() * N); if (!idx.includes(i)) put(i); }
    return C;
  }
  put(Math.floor(rng.next() * N));
  const d2 = new Float64Array(N);
  while (idx.length < k) {
    let tot = 0;
    for (let i = 0; i < N; i++) {
      let m = Infinity;
      for (let c = 0; c < idx.length; c++) m = Math.min(m, sq(pts[2 * i] - C[2 * c], pts[2 * i + 1] - C[2 * c + 1]));
      d2[i] = m; tot += m;
    }
    let r = rng.next() * tot, pick = N - 1;
    for (let i = 0; i < N; i++) { r -= d2[i]; if (r <= 0 && d2[i] > 0) { pick = i; break; } }
    put(pick);
  }
  return C;
}

const sq = (dx: number, dy: number) => dx * dx + dy * dy;

/** Index of the centroid nearest to (x, y); ties go to the lower index. */
export function nearest(C: ArrayLike<number>, k: number, x: number, y: number): number {
  let b = 0, bd = Infinity;
  for (let c = 0; c < k; c++) { const d = sq(x - C[2 * c], y - C[2 * c + 1]); if (d < bd) { bd = d; b = c; } }
  return b;
}

/** Within-cluster sum of squares for assignment `a` and centroids `C`. */
export function inertia(pts: Float64Array, a: ArrayLike<number>, C: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += sq(pts[2 * i] - C[2 * a[i]], pts[2 * i + 1] - C[2 * a[i] + 1]);
  return s;
}

export interface KRun {
  pts: Float64Array; N: number; k: number;
  /** Number of assign+update iterations. */
  n: number;
  /** C[j] = centroids before iteration j (C[0] = init); C[n] = final. */
  C: Float64Array[];
  /** a[j] = assignment made in iteration j (nearest centroid of C[j]). */
  a: Uint8Array[];
  /** iA[j] = inertia(a[j], C[j]); iU[j] = inertia(a[j], C[j+1]). iA[j+1] <= iU[j] <= iA[j]. */
  iA: number[]; iU: number[];
  /** moved[j] = points whose cluster changed in iteration j (all N in iteration 0). */
  moved: number[];
}

/** Lloyd's algorithm. Stops when an assignment step changes nothing, or after `maxIter` iterations. */
export function kmeans(pts: Float64Array, C0: Float64Array, maxIter = 40): KRun {
  const N = pts.length / 2, k = C0.length / 2;
  const C = [C0.slice()], a: Uint8Array[] = [], iA: number[] = [], iU: number[] = [], moved: number[] = [];
  for (let j = 0; j < maxIter; j++) {
    const Cj = C[j], aj = new Uint8Array(N);
    let ch = 0;
    for (let i = 0; i < N; i++) { aj[i] = nearest(Cj, k, pts[2 * i], pts[2 * i + 1]); if (j === 0 || aj[i] !== a[j - 1][i]) ch++; }
    const sum = new Float64Array(2 * k), cnt = new Uint32Array(k);
    for (let i = 0; i < N; i++) { sum[2 * aj[i]] += pts[2 * i]; sum[2 * aj[i] + 1] += pts[2 * i + 1]; cnt[aj[i]]++; }
    const Cn = Cj.slice();
    for (let c = 0; c < k; c++) if (cnt[c]) { Cn[2 * c] = sum[2 * c] / cnt[c]; Cn[2 * c + 1] = sum[2 * c + 1] / cnt[c]; }
    a.push(aj); C.push(Cn); moved.push(ch);
    iA.push(inertia(pts, aj, Cj)); iU.push(inertia(pts, aj, Cn));
    if (j > 0 && ch === 0) break;
  }
  return { pts, N, k, n: a.length, C, a, iA, iU, moved };
}

export const finalInertia = (r: KRun) => r.iU[r.n - 1];

/** Inertia for k = 1..kMax, each the best of `restarts` k-means++ starts (what scikit-learn's n_init does). */
export function elbow(pts: Float64Array, kMax = K_MAX, restarts = 5): { inertia: number[]; runs: KRun[] } {
  const runs: KRun[] = [];
  for (let k = 1; k <= kMax; k++) {
    let best: KRun | null = null;
    for (let r = 0; r < restarts; r++) {
      const run = kmeans(pts, initCentroids(pts, k, 'plusplus', 1000 + 37 * k + r));
      if (!best || finalInertia(run) < finalInertia(best)) best = run;
    }
    runs.push(best!);
  }
  return { inertia: runs.map(finalInertia), runs };
}

export const duration = (r: KRun) => T0 + r.n * P + HOLD;

/** Phase markers for the Timeline. */
export function markers(r: KRun, mode: Mode): { at: number; label: string }[] {
  if (mode === 'structure') return [{ at: 0, label: 'unlabelled' }, { at: T_REVEAL, label: 'groups found' }];
  if (r.n <= 1) return [{ at: 0, label: 'start' }, { at: T0, label: 'assign' }, { at: T0 + DA, label: 'update' }];
  const m = [{ at: T0, label: 'running' }];
  m.push({ at: T0 + (r.n - 1) * P, label: 'converged' });
  return m;
}

const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
/** Fixed pseudo-random delay in [0, 1) per point, so points change colour one after another rather than all at once. */
const stag = (i: number) => (i * 0.6180339887) % 1;

export type Phase = 'start' | 'assign' | 'update' | 'converged' | 'unlabelled' | 'groups';

export interface Frame {
  phase: Phase;
  /** Iteration being shown (0-based). */
  iter: number;
  /** Centroid positions at t (k × 2). */
  cent: Float64Array;
  /** Centroid opacity 0..1. */
  centA: number;
  /** Per point: previous cluster (-1 = grey/unassigned), new cluster, and colour-blend 0..1. */
  from: Int8Array; to: Uint8Array; mix: Float32Array;
  /** Inertia of exactly what is drawn (points -> their currently shown centroid); NaN before the first assignment. */
  inertia: number;
  /** x position on the inertia-vs-iteration chart (0 at start, 1 after the first update, ...). */
  cx: number;
  /** Points that changed cluster in this iteration (only meaningful in assign / update / converged). */
  moved: number;
}

const lerpC = (A: Float64Array, B: Float64Array, u: number) => { const o = new Float64Array(A.length); for (let i = 0; i < o.length; i++) o[i] = A[i] + (B[i] - A[i]) * u; return o; };

/** Everything needed to draw the algorithm at time t. Pure: scrubbing is free. */
export function frameAt(t: number, r: KRun): Frame {
  const { N, n } = r, from = new Int8Array(N).fill(-1), to = new Uint8Array(N), mix = new Float32Array(N);
  const u = t - T0;
  if (u < 0) {
    return { phase: 'start', iter: 0, cent: r.C[0].slice(), centA: ease(t / (T0 * 0.6)), from, to: r.a[0], mix, inertia: NaN, cx: 0, moved: 0 };
  }
  const done = u >= n * P, j = done ? n - 1 : Math.floor(u / P), rt = done ? P : u - j * P;
  const aj = r.a[j];
  to.set(aj);
  if (j > 0) for (let i = 0; i < N; i++) from[i] = r.a[j - 1][i];
  const assigning = !done && rt < DA, s = assigning ? rt / DA : 1;
  for (let i = 0; i < N; i++) mix[i] = j === 0 && !assigning ? 1 : ease(s * 1.6 - stag(i) * 0.6);
  let cent: Float64Array, cx: number, shown: Uint8Array | Int8Array, C: Float64Array;
  if (assigning) { cent = r.C[j]; cx = j; }
  else { const w = ease((rt - DA) / DU); cent = lerpC(r.C[j], r.C[j + 1], w); cx = j + w; for (let i = 0; i < N; i++) mix[i] = 1; }
  // inertia of what is drawn: each point measured to the centroid of its currently shown colour
  C = cent; shown = new Int8Array(N);
  for (let i = 0; i < N; i++) shown[i] = mix[i] >= 0.5 || from[i] < 0 ? aj[i] : from[i];
  let ine = 0;
  for (let i = 0; i < N; i++) ine += sq(r.pts[2 * i] - C[2 * shown[i]], r.pts[2 * i + 1] - C[2 * shown[i] + 1]);
  return { phase: done ? 'converged' : assigning ? 'assign' : 'update', iter: j, cent, centA: 1, from, to, mix, inertia: j === 0 && assigning ? NaN : ine, cx, moved: r.moved[j] };
}

/** Structure mode: grey points, then the converged clustering fades in (colours first, then centroids). */
export function structureFrame(t: number, r: KRun): Frame {
  const { N, n } = r, from = new Int8Array(N).fill(-1), to = r.a[n - 1], mix = new Float32Array(N);
  const s = (t - T_REVEAL) / REVEAL;
  for (let i = 0; i < N; i++) mix[i] = ease(s * 1.6 - stag(i) * 0.6);
  const centA = ease((t - T_REVEAL - REVEAL) / 0.8);
  return { phase: s > 0 ? 'groups' : 'unlabelled', iter: n - 1, cent: r.C[n], centA, from, to, mix, inertia: NaN, cx: 0, moved: 0 };
}

/** Smooth cluster sizes at a frame: index c < k is the (fractional) count in cluster c, index k is points still unassigned. */
export function sizes(f: Frame, k: number): Float64Array {
  const o = new Float64Array(k + 1);
  for (let i = 0; i < f.to.length; i++) {
    const m = f.mix[i];
    o[f.to[i]] += m; o[f.from[i] < 0 ? k : f.from[i]] += 1 - m;
  }
  return o;
}
