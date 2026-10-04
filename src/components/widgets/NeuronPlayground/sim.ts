/** One neuron with two inputs: weighted sum, dot product, activation, and a scripted fit. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';

export type Act = 'none' | 'step' | 'sigmoid';
export type Mode = 'weighted-sum' | 'dot-product' | 'activation' | 'playground';
export interface Pt { x1: number; x2: number; y: 0 | 1 }
export interface Params { w1: number; w2: number; b: number }

export const DOM = 3; // the input plane spans [-3, 3] on both axes
/** The neuron every step except the dot-product one shows: it separates the dataset. */
export const FINAL: Params = { w1: 1.5, w2: 1, b: -0.6 };
export const ACTS: Act[] = ['none', 'step', 'sigmoid'];

/** Twelve points per class, separable by FINAL with a margin, so the playground can end at 100%. */
export const POINTS: Pt[] = (() => {
  const rng = createRng(7), out: Pt[] = [], n = [0, 0];
  while (n[0] < 12 || n[1] < 12) {
    const x1 = rng.uniform(-2.6, 2.6), x2 = rng.uniform(-2.6, 2.6), s = FINAL.w1 * x1 + FINAL.w2 * x2 + FINAL.b;
    const y = s > 0 ? 1 : 0;
    if (Math.abs(s) > 1 && n[y] < 12) { out.push({ x1, x2, y }); n[y]++; }
  }
  return out;
})();

export const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
export const activate = (a: Act, z: number) => (a === 'none' ? z : a === 'step' ? (z > 0 ? 1 : 0) : sigmoid(z));
/** Display scale in [-1, 1] for colouring the plane: positive = the neuron leans to class 1. */
export const shade = (a: Act, z: number) => (a === 'none' ? Math.max(-1, Math.min(1, z / 2.2)) : 2 * activate(a, z) - 1);

export const preact = (p: Params, x1: number, x2: number) => p.w1 * x1 + p.w2 * x2 + p.b;
export const accuracy = (p: Params, pts: Pt[] = POINTS) => pts.reduce((s, q) => s + ((preact(p, q.x1, q.x2) > 0 ? 1 : 0) === q.y ? 1 : 0), 0) / pts.length;

const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;

// ---- dot-product step: fixed point x, weight vector of fixed length rotating, b = 0 ----
export const DOT_X = { x1: 1.6, x2: 1 };
export const W_LEN = 1.8;
const T_DOT = 12, REL0 = -Math.PI / 3;
const rel = (t: number) => REL0 + (2 * Math.PI * t) / T_DOT;
// ---- weighted-sum step: a point circling an ellipse ----
const T_SUM = 12, EC = { x: 0.2, y: 0 }, ER = { x: 2.1, y: 1.6 };
const sumPoint = (t: number) => { const a = Math.PI + (2 * Math.PI * t) / T_SUM; return { x1: EC.x + ER.x * Math.cos(a), x2: EC.y + ER.y * Math.sin(a) }; };
// ---- activation step: a point rides along w through the boundary; the activation changes every 4 s ----
const T_ACT = 12, SEG = 4, AMP = 1.1, X0 = { x1: 0.3, x2: 0.15 }, MIX = 0.5;
export const Z_RANGE = 2.5;
// ---- playground script ----
const START: Params = { w1: -0.8, w2: 0.5, b: 1.8 };
/** The playground's test point, shown going through the diagram. */
export const PROBE = { x1: 1.2, x2: 0.8 };
const P_HOLD = 2, P_TURN = 4, P_SLIDE = 3, P_END = 2.5;

export const duration = (m: Mode) => (m === 'weighted-sum' ? T_SUM : m === 'dot-product' ? T_DOT : m === 'activation' ? T_ACT : P_HOLD + P_TURN + P_SLIDE + P_END);

/** Times in [0, T) where the sign of f flips, found by scanning. */
function crossings(f: (t: number) => number, T: number) {
  const out: number[] = [];
  let prev = f(0) > 0;
  for (let i = 1; i <= 4000; i++) { const t = (T * i) / 4000, s = f(t) > 0; if (s !== prev) out.push(t); prev = s; }
  return out;
}
const SUM_CROSS = crossings((t) => { const p = sumPoint(t); return preact(FINAL, p.x1, p.x2); }, T_SUM);

export const markers = (m: Mode) =>
  m === 'weighted-sum' ? [{ at: 0, label: 'negative' }, { at: SUM_CROSS[0], label: 'positive' }, { at: SUM_CROSS[1], label: 'negative' }]
  : m === 'dot-product' ? [{ at: 0, label: 'positive' }, { at: (T_DOT * (Math.PI / 2 - REL0)) / (2 * Math.PI), label: 'negative' }, { at: (T_DOT * (1.5 * Math.PI - REL0)) / (2 * Math.PI), label: 'positive' }]
  : m === 'activation' ? [{ at: 0, label: 'no activation' }, { at: SEG, label: 'step' }, { at: 2 * SEG, label: 'sigmoid' }]
  : [{ at: 0, label: 'start' }, { at: P_HOLD, label: 'turn w' }, { at: P_HOLD + P_TURN, label: 'slide b' }, { at: P_HOLD + P_TURN + P_SLIDE, label: 'separated' }];

export interface Frame extends Params {
  /** The input shown going through the neuron. */
  x1: number; x2: number;
  z: number; p1: number; p2: number;
  act: Act; prev: Act; mix: number;
  /** Output of the neuron for the test point. */
  out: number;
  phase: string;
  /** Dot-product step only: angle between w and x (radians, in (-pi, pi]). */
  theta: number;
}

function build(p: Params, x: { x1: number; x2: number }, act: Act, prev: Act, mix: number, phase: string, theta = 0): Frame {
  const z = preact(p, x.x1, x.x2);
  return { ...p, x1: x.x1, x2: x.x2, z, p1: p.w1 * x.x1, p2: p.w2 * x.x2, act, prev, mix, out: activate(act, z), phase, theta };
}
const phaseOf = (m: Mode, t: number) => { const mk = markers(m); let l = mk[0].label; for (const k of mk) if (t >= k.at) l = k.label; return l; };

/** Everything needed to draw one frame. A pure function of t, so scrubbing is free. */
export function frameAt(t: number, mode: Mode, user?: Params, userAct: Act = 'sigmoid'): Frame {
  t = Math.max(0, Math.min(duration(mode), t));
  if (mode === 'weighted-sum') return build(FINAL, sumPoint(t), 'none', 'none', 1, phaseOf(mode, t));
  if (mode === 'dot-product') {
    const r = rel(t), a = Math.atan2(DOT_X.x2, DOT_X.x1) + r;
    const th = Math.atan2(Math.sin(r), Math.cos(r));
    return build({ w1: W_LEN * Math.cos(a), w2: W_LEN * Math.sin(a), b: 0 }, DOT_X, 'none', 'none', 1, phaseOf(mode, t), th);
  }
  if (mode === 'activation') {
    const i = Math.min(2, Math.floor(t / SEG)), mix = i === 0 ? 1 : ease((t - i * SEG) / MIX);
    const len = Math.hypot(FINAL.w1, FINAL.w2), s = AMP * Math.sin((2 * Math.PI * t) / SEG);
    return build(FINAL, { x1: X0.x1 + (s * FINAL.w1) / len, x2: X0.x2 + (s * FINAL.w2) / len }, ACTS[i], ACTS[Math.max(0, i - 1)], mix, phaseOf(mode, t));
  }
  const tp = Math.min(1, ease((t - P_HOLD) / P_TURN)), tb = ease((t - P_HOLD - P_TURN) / P_SLIDE);
  const p = user ?? { w1: lerp(START.w1, FINAL.w1, tp), w2: lerp(START.w2, FINAL.w2, tp), b: lerp(START.b, FINAL.b, tb) };
  return build(p, PROBE, userAct, userAct, 1, user ? 'manual' : phaseOf(mode, t));
}
