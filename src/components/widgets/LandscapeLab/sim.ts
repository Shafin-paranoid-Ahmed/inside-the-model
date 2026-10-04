export const DURATION = 10;

export type Mode = 'nonconvex' | 'saddle' | 'dimension' | 'coordinates' | 'residual';
export interface Point { x: number; y: number }
export interface PathPoint extends Point { loss: number; step: number }
export interface Marker { at: number; label: string }
export interface Block { input: number; branch: number; output: number; derivative: number }
export interface TinyNetStats {
  inputs: number[];
  plainLoss: number;
  residualLoss: number;
  plainJacobian: number[];
  residualJacobian: number[];
  plainOutput: number[];
  residualOutput: number[];
  alpha: number;
  /** Input x = 1 followed through the two residual blocks. */
  blocks: Block[];
  /** Input x = 1 followed through the two plain tanh blocks (branch = 0, output = tanh). */
  plainBlocks: Block[];
}
export interface Segment { a: Point; b: Point; negative: boolean }
export interface CurvePoint { x: number; y: number }
export interface Options { scale?: number; alpha?: number; side?: 'left' | 'right' }
export interface Frame {
  mode: Mode;
  t: number;
  phase: string;
  /** Double well: the trail so far, the displayed point, its loss and the completed update count. */
  path: PathPoint[];
  point: Point;
  loss: number;
  update: number;
  contours: Segment[];
  plotExtent: number;
  /** Saddle: the exact-center start and the displaced start's current position. */
  saddleStill: Point;
  saddleEscape: PathPoint;
  saddleUpdate: number;
  /** Sign counting. */
  dimension: number;
  patterns: number;
  allPositiveProbability: number;
  cellReveal: number;
  /** Coordinate rescaling: c eases from 1 to the chosen target. */
  coordinateScale: number;
  targetScale: number;
  theta: number;
  mappedThetaPrime: number;
  originalLoss: number;
  rescaledLoss: number;
  originalCurvature: number;
  rescaledCurvature: number;
  originalCurve: CurvePoint[];
  rescaledCurve: CurvePoint[];
  /** Residual blocks: alpha eases from 0 to the chosen target; the plot range covers both ends. */
  tinyNet: TinyNetStats;
  jacobianExtent: { low: number; high: number };
}

const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n));
const lerp = (a: number, b: number, p: number) => a + (b - a) * p;
const smooth = (p: number) => { const u = clamp(p, 0, 1); return u * u * (3 - 2 * u); };

/** A double-well objective with two equally deep minima at (±1, 0). */
export function nonconvexLoss({ x, y }: Point): number {
  return (x * x - 1) ** 2 + 0.35 * y * y;
}

export function nonconvexGradient({ x, y }: Point): Point {
  return { x: 4 * x * (x * x - 1), y: 0.7 * y };
}

/** Bounded quartic saddle; its Hessian at the origin is diag(2, -2). */
export function saddleLoss({ x, y }: Point): number {
  const r2 = x * x + y * y;
  return 0.25 * r2 * r2 + x * x - y * y;
}

export function saddleGradient({ x, y }: Point): Point {
  const r2 = x * x + y * y;
  return { x: x * r2 + 2 * x, y: y * r2 - 2 * y };
}

const SAMPLE_OFFSETS: Point[] = [
  { x: 0.22, y: -0.12 }, { x: -0.22, y: 0.12 },
  { x: 0.12, y: 0.22 }, { x: -0.12, y: -0.22 },
  { x: 0.18, y: 0.08 }, { x: -0.18, y: -0.08 },
  { x: 0.08, y: -0.18 }, { x: -0.08, y: 0.18 },
];

/** Eight fixed, zero-mean offsets stand in for mini-batch noise on the double-well path. */
export function sampledGradient(point: Point, sampleIndex: number): Point {
  const exact = nonconvexGradient(point), noise = SAMPLE_OFFSETS[sampleIndex % SAMPLE_OFFSETS.length];
  return { x: exact.x + noise.x, y: exact.y + noise.y };
}

export const NONCONVEX_STEPS = 100;
export const SADDLE_STEPS = 90;
export const NONCONVEX_RATE = 0.025;
export const SADDLE_RATE = 0.03;
export const NONCONVEX_START: Point = { x: 0.12, y: 1.35 };
export const SADDLE_START: Point = { x: 0, y: 0.06 };

export function makeNonconvexPath(steps = NONCONVEX_STEPS, learningRate = NONCONVEX_RATE): PathPoint[] {
  let point = { ...NONCONVEX_START };
  const path: PathPoint[] = [{ ...point, loss: nonconvexLoss(point), step: 0 }];
  for (let step = 1; step <= steps; step++) {
    const g = sampledGradient(point, step - 1);
    point = { x: point.x - learningRate * g.x, y: point.y - learningRate * g.y };
    path.push({ ...point, loss: nonconvexLoss(point), step });
  }
  return path;
}

export function makeSaddlePaths(steps = SADDLE_STEPS, learningRate = SADDLE_RATE): { still: PathPoint[]; escape: PathPoint[] } {
  const run = (start: Point) => {
    let point = { ...start };
    const path: PathPoint[] = [{ ...point, loss: saddleLoss(point), step: 0 }];
    for (let step = 1; step <= steps; step++) {
      const g = saddleGradient(point);
      point = { x: point.x - learningRate * g.x, y: point.y - learningRate * g.y };
      path.push({ ...point, loss: saddleLoss(point), step });
    }
    return path;
  };
  return { still: run({ x: 0, y: 0 }), escape: run(SADDLE_START) };
}

const NONCONVEX_PATH = makeNonconvexPath();
const SADDLE_PATHS = makeSaddlePaths();

function makeContours(mode: 'nonconvex' | 'saddle', extent: number): Segment[] {
  const n = 34, segments: Segment[] = [];
  const levels = mode === 'saddle' ? [-1.5, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 4] : [0.03, 0.08, 0.18, 0.35, 0.7, 1.2, 2];
  const objective = mode === 'saddle' ? saddleLoss : nonconvexLoss;
  for (const level of levels) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x0 = -extent + 2 * extent * i / n, x1 = -extent + 2 * extent * (i + 1) / n;
    const y0 = extent - 2 * extent * j / n, y1 = extent - 2 * extent * (j + 1) / n;
    const corners: [number, number, number][] = [[x0,y0,objective({ x: x0, y: y0 })],[x1,y0,objective({ x: x1, y: y0 })],[x1,y1,objective({ x: x1, y: y1 })],[x0,y1,objective({ x: x0, y: y1 })]];
    const hits: Point[] = [];
    for (let k = 0; k < 4; k++) {
      const a = corners[k], b = corners[(k + 1) % 4];
      if ((a[2] < level) !== (b[2] < level)) {
        const q = (level - a[2]) / (b[2] - a[2]);
        hits.push({ x: a[0] + (b[0] - a[0]) * q, y: a[1] + (b[1] - a[1]) * q });
      }
    }
    for (let k = 0; k + 1 < hits.length; k += 2) segments.push({ a: hits[k], b: hits[k + 1], negative: mode === 'saddle' && level < 0 });
  }
  return segments;
}

const NONCONVEX_CONTOURS = makeContours('nonconvex', 1.8);
const SADDLE_CONTOURS = makeContours('saddle', 1.65);

const INPUTS = [-1, -0.5, 0, 0.5, 1];
export const W1 = 1.2, W2 = 1.15, V = 0.5;
const TARGETS = INPUTS;

function plainNetwork(x: number) {
  const h1 = Math.tanh(W1 * x), d1 = W1 * (1 - h1 * h1);
  const h2 = Math.tanh(W2 * h1), d2 = W2 * (1 - h2 * h2);
  return { output: V * h2, jacobian: V * d2 * d1 };
}

function residualNetwork(x: number, alpha: number) {
  const a1 = W1 * x, h1 = x + alpha * Math.tanh(a1);
  const d1 = 1 + alpha * W1 * (1 - Math.tanh(a1) ** 2);
  const a2 = W2 * h1, h2 = h1 + alpha * Math.tanh(a2);
  const d2 = 1 + alpha * W2 * (1 - Math.tanh(a2) ** 2);
  return { output: V * h2, jacobian: V * d2 * d1 };
}

/** Two scalar tanh blocks with fixed weights w1, w2 and readout v, on five fixed inputs. */
export function tinyNetStats(alpha: number): TinyNetStats {
  const plain = INPUTS.map(plainNetwork), residual = INPUTS.map((x) => residualNetwork(x, alpha));
  let input = 1;
  const blocks = [W1, W2].map((weight) => {
    const activation = Math.tanh(weight * input), branch = alpha * activation;
    const block = { input, branch, output: input + branch, derivative: 1 + alpha * weight * (1 - activation * activation) };
    input = block.output;
    return block;
  });
  input = 1;
  const plainBlocks = [W1, W2].map((weight) => {
    const output = Math.tanh(weight * input);
    const block = { input, branch: 0, output, derivative: weight * (1 - output * output) };
    input = output;
    return block;
  });
  return {
    inputs: [...INPUTS],
    plainLoss: plain.reduce((sum, item, i) => sum + (item.output - TARGETS[i]) ** 2, 0) / INPUTS.length,
    residualLoss: residual.reduce((sum, item, i) => sum + (item.output - TARGETS[i]) ** 2, 0) / INPUTS.length,
    plainJacobian: plain.map((item) => item.jacobian),
    residualJacobian: residual.map((item) => item.jacobian),
    plainOutput: plain.map((item) => item.output),
    residualOutput: residual.map((item) => item.output),
    alpha,
    blocks,
    plainBlocks,
  };
}

/** Time at which a recorded update is shown exactly (the path timelines run at a constant update rate). */
const stepTime = (step: number, steps: number) => step / steps * DURATION;
const firstStep = (path: PathPoint[], test: (p: PathPoint) => boolean) => Math.max(0, path.findIndex(test));

/** Double well: the first update within 0.1 of the well centre. */
export const NONCONVEX_IN_WELL = firstStep(NONCONVEX_PATH, (p) => Math.abs(p.x - 1) < 0.1);
/** Saddle: the first update with y past 0.3, and the first with loss below -0.95 (the valleys sit at -1). */
export const SADDLE_LEAVES = firstStep(SADDLE_PATHS.escape, (p) => p.y > 0.3);
export const SADDLE_IN_VALLEY = firstStep(SADDLE_PATHS.escape, (p) => p.loss < -0.95);

export const DIMENSIONS = [2, 4, 8, 12];
const DIMENSION_SPAN = DURATION / DIMENSIONS.length;
const SCALE_FROM = 2.5, SCALE_TO = 7.5, ALPHA_FROM = 2.5, ALPHA_TO = 7.5;

/** Marker labels name the segment that starts at `at`, so the highlighted marker always describes the frame. */
export const MARKERS: Record<Mode, Marker[]> = {
  nonconvex: [
    { at: 0, label: 'Downhill' },
    { at: stepTime(NONCONVEX_IN_WELL, NONCONVEX_STEPS), label: 'In the well' },
  ],
  saddle: [
    { at: 0, label: 'Near center' },
    { at: stepTime(SADDLE_LEAVES, SADDLE_STEPS), label: 'Escaping' },
    { at: stepTime(SADDLE_IN_VALLEY, SADDLE_STEPS), label: 'In a valley' },
  ],
  dimension: DIMENSIONS.map((d, i) => ({ at: i * DIMENSION_SPAN, label: `${d} directions` })),
  coordinates: [
    { at: 0, label: 'Unscaled' },
    { at: SCALE_FROM, label: 'Stretching' },
    { at: SCALE_TO, label: 'Stretched' },
  ],
  residual: [
    { at: 0, label: 'Off' },
    { at: ALPHA_FROM, label: 'Growing' },
    { at: ALPHA_TO, label: 'Full' },
  ],
};

function phaseAt(t: number, markers: Marker[]) {
  let label = markers[0].label;
  for (const m of markers) if (m.at <= t + 1e-9) label = m.label;
  return label;
}

export function frameAt(t: number, mode: Mode, { scale = 4, alpha = 0.5, side = 'right' }: Options = {}): Frame {
  const time = clamp(t, 0, DURATION);
  const path = side === 'left' ? NONCONVEX_PATH.map((p) => ({ ...p, x: -p.x })) : NONCONVEX_PATH;
  const index = +(time / DURATION * NONCONVEX_STEPS).toFixed(9), lo = Math.min(NONCONVEX_STEPS, Math.floor(index)), hi = Math.min(NONCONVEX_STEPS, lo + 1), p = smooth(index - lo);
  const point = { x: lerp(path[lo].x, path[hi].x, p), y: lerp(path[lo].y, path[hi].y, p) };
  const trail = [...path.slice(0, lo + 1), { ...point, loss: nonconvexLoss(point), step: index }];

  const { still, escape } = SADDLE_PATHS;
  const sIndex = +(time / DURATION * SADDLE_STEPS).toFixed(9), si = Math.min(SADDLE_STEPS, Math.floor(sIndex)), sj = Math.min(SADDLE_STEPS, si + 1), sp = smooth(sIndex - si);
  const escapePoint = { x: lerp(escape[si].x, escape[sj].x, sp), y: lerp(escape[si].y, escape[sj].y, sp) };

  const segment = Math.min(DIMENSIONS.length - 1, Math.floor(time / DIMENSION_SPAN)), dimension = DIMENSIONS[segment];

  const targetScale = clamp(scale, 1, 4), c = lerp(1, targetScale, smooth((time - SCALE_FROM) / (SCALE_TO - SCALE_FROM)));
  const theta = 0.7, thetaPrime = c * theta;
  const curve = (width: number) => Array.from({ length: 81 }, (_, i) => {
    const x = -width + 2 * width * i / 80;
    return { x, y: (x / width) ** 2 };
  });

  const targetAlpha = clamp(alpha, -1.2, 1.2), shownAlpha = targetAlpha * smooth((time - ALPHA_FROM) / (ALPHA_TO - ALPHA_FROM));
  const ends = [tinyNetStats(0), tinyNetStats(targetAlpha)].flatMap((s) => [...s.plainJacobian, ...s.residualJacobian]);

  return {
    mode,
    t: time,
    phase: phaseAt(time, MARKERS[mode]),
    path: trail,
    point,
    loss: nonconvexLoss(point),
    update: lo,
    contours: mode === 'saddle' ? SADDLE_CONTOURS : mode === 'nonconvex' ? NONCONVEX_CONTOURS : [],
    plotExtent: mode === 'saddle' ? 1.65 : 1.8,
    saddleStill: { x: still[0].x, y: still[0].y },
    saddleEscape: { ...escapePoint, loss: saddleLoss(escapePoint), step: sIndex },
    saddleUpdate: si,
    dimension,
    patterns: 2 ** dimension,
    allPositiveProbability: 2 ** -dimension,
    cellReveal: smooth((time - segment * DIMENSION_SPAN) / 0.6),
    coordinateScale: c,
    targetScale,
    theta,
    mappedThetaPrime: thetaPrime,
    originalLoss: theta * theta,
    rescaledLoss: (thetaPrime / c) ** 2,
    originalCurvature: 2,
    rescaledCurvature: 2 / (c * c),
    originalCurve: curve(1),
    rescaledCurve: curve(c),
    tinyNet: tinyNetStats(shownAlpha),
    jacobianExtent: { low: Math.min(0, ...ends), high: Math.max(...ends) * 1.1 },
  };
}
