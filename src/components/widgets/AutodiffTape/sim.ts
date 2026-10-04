/**
 * Pure logic for AutodiffTape. The expression is built with the shared Value engine, flattened into a
 * Wengert list (tape), and differentiated two ways: reverse mode (the engine's own localBackward, one node
 * at a time) and forward mode (dual numbers walking the same tape).
 */
import { Value, prepareBackward, topo, type Op } from '../../../lib/nn';

export interface Params { w1: number; w2: number; b: number; x1: number; x2: number; t: number }
/** Same numbers as the backprop-mlp lesson where they overlap (x = (1, 0.5), target -0.5). */
export const DEFAULTS: Params = { w1: 0.5, w2: -0.6, b: 0.1, x1: 1, x2: 0.5, t: -0.5 };
export const PARAM_NAMES = ['w1', 'w2', 'b'] as const;

export interface Row {
  i: number; label: string; op: Op; args: number[]; value: number; exponent: number;
  /** Leaf rows: 'param' (we differentiate w.r.t. it) or 'const'. */
  kind: 'param' | 'const' | 'op';
}
export interface Tape {
  rows: Row[];
  N: number;
  /** adj[s][i]: adjoint of row i after s reverse steps (s = 0 is just the seed dL/dL = 1). */
  adj: number[][];
  root: number;
  /** Tangents d(row)/d(param) from a forward sweep seeded on the given parameter row. */
  tangents(seedRow: number): number[];
  paramRows: number[];
}

/** L = (tanh(w1 x1 + w2 x2 + b) - t)^2 */
export function buildTape(p: Params = DEFAULTS): Tape {
  const mk = (d: number, l: string) => { const v = new Value(d); v.label = l; return v; };
  const w1 = mk(p.w1, 'w1'), x1 = mk(p.x1, 'x1'), w2 = mk(p.w2, 'w2'), x2 = mk(p.x2, 'x2'), b = mk(p.b, 'b'), t = mk(p.t, 't');
  const lab = (v: Value, l: string) => { v.label = l; return v; };
  const m1 = lab(w1.mul(x1), 'm1'), m2 = lab(w2.mul(x2), 'm2');
  const s1 = lab(m1.add(m2), 's1'), s2 = lab(s1.add(b), 's2');
  const y = lab(s2.tanh(), 'y'), d = lab(y.sub(t), 'd'), L = lab(d.pow(2), 'L');

  const order = topo(L); // children before parents: already a valid tape
  const idx = new Map(order.map((v, i) => [v, i]));
  const params = new Set([w1, w2, b]);
  const rows: Row[] = order.map((v, i) => ({
    i, label: v.label ?? `v${i}`, op: v.op, args: v.children.map((c) => idx.get(c)!), value: v.data, exponent: v.exponent,
    kind: v.children.length ? 'op' : params.has(v) ? 'param' : 'const',
  }));

  // Reverse mode: the engine's own local rule, one node at a time, snapshotting every adjoint.
  const rev = prepareBackward(L); // reverse topological = rows from last to first
  const N = rows.length;
  const snap = () => order.map((v) => v.grad);
  const adj = [snap()];
  for (const v of rev) { v.localBackward(); adj.push(snap()); }

  const tangents = (seedRow: number): number[] => {
    const tan = new Array<number>(N).fill(0);
    for (const r of rows) {
      const [a, c] = r.args, va = a === undefined ? 0 : rows[a].value, vc = c === undefined ? 0 : rows[c].value;
      const ta = a === undefined ? 0 : tan[a], tc = c === undefined ? 0 : tan[c];
      switch (r.op) {
        case '': tan[r.i] = r.i === seedRow ? 1 : 0; break;
        case '+': tan[r.i] = ta + tc; break;
        case '-': tan[r.i] = ta - tc; break;
        case '*': tan[r.i] = ta * vc + va * tc; break;
        case '/': tan[r.i] = (ta * vc - va * tc) / (vc * vc); break;
        case 'neg': tan[r.i] = -ta; break;
        case 'pow': tan[r.i] = r.exponent * va ** (r.exponent - 1) * ta; break;
        case 'exp': tan[r.i] = r.value * ta; break;
        case 'log': tan[r.i] = ta / va; break;
        case 'tanh': tan[r.i] = (1 - r.value * r.value) * ta; break;
        case 'relu': tan[r.i] = va > 0 ? ta : 0; break;
        case 'sigmoid': tan[r.i] = r.value * (1 - r.value) * ta; break;
      }
    }
    return tan;
  };
  return { rows, N, adj, root: idx.get(L)!, tangents, paramRows: [w1, w2, b].map((v) => idx.get(v)!) };
}

/** Human-readable right-hand side, e.g. "w1 × x1" or "tanh(s2)". */
export function expr(tape: Tape, r: Row): string {
  const n = (k: number) => tape.rows[k].label;
  const [a, b] = r.args;
  switch (r.op) {
    case '': return r.kind === 'param' ? 'input' : 'constant';
    case '+': return `${n(a)} + ${n(b)}`;
    case '-': return `${n(a)} − ${n(b)}`;
    case '*': return `${n(a)} × ${n(b)}`;
    case '/': return `${n(a)} / ${n(b)}`;
    case 'neg': return `−${n(a)}`;
    case 'pow': return `${n(a)}^${r.exponent}`;
    default: return `${r.op}(${n(a)})`;
  }
}

/**
 * Row visits to get all n input derivatives, up to a small constant per row.
 * Forward mode: one shared value sweep plus one tangent sweep per input. Reverse mode: one value sweep plus one adjoint sweep.
 */
export const costs = (N: number, n: number) => ({ forward: N * (n + 1), reverse: 2 * N });

/** Timeline. */
export const REC = 4.5, B0 = 5.2, B1 = 11.2;
export const T_REVERSE = 12, T_FORWARD = 7;
export const durationOf = (mode: 'reverse' | 'forward') => (mode === 'reverse' ? T_REVERSE : T_FORWARD);
export const markersOf = (mode: 'reverse' | 'forward') => mode === 'reverse'
  ? [{ at: 0, label: 'record' }, { at: B0, label: 'backward' }, { at: B1, label: 'done' }]
  : [{ at: 0, label: 'trace + tangents' }, { at: REC + 0.5, label: 'done' }];

const ease = (p: number) => { const q = Math.min(1, Math.max(0, p)); return q * q * (3 - 2 * q); };

export interface Frame {
  /** Reveal (0..1) of each row's value. */
  show: number[];
  /** Adjoints as currently accumulated (reverse mode), tweened. */
  adj: number[];
  /** Tangents (forward mode), revealed with the values. */
  tan: number[];
  /** Row being processed (backward: the row pushing its adjoint down; recording: the row being written); -1 if none. */
  cursor: number;
  /** Progress 0..1 of the current backward step (arrow flowing from cursor to its args). */
  flow: number;
  phase: 'record' | 'backward' | 'done';
}

export function frameAt(t: number, tape: Tape, mode: 'reverse' | 'forward', tan: number[]): Frame {
  const N = tape.N;
  const c = Math.min(1, Math.max(0, t / REC)) * N;
  const show = tape.rows.map((_, i) => ease(c - i));
  const rec = t < REC;
  let adj = new Array<number>(N).fill(0), cursor = rec ? Math.min(N - 1, Math.floor(c)) : -1, flow = 0;
  let phase: Frame['phase'] = rec ? 'record' : 'done';
  if (mode === 'reverse' && t >= B0) {
    const k = Math.min(1, Math.max(0, (t - B0) / (B1 - B0))) * N;
    const s = Math.min(N - 1, Math.floor(k)), f = k >= N ? 1 : k - s;
    const a0 = tape.adj[s], a1 = tape.adj[s + 1], e = ease(f);
    adj = a0.map((v, i) => v + (a1[i] - v) * e);
    cursor = k >= N ? -1 : N - 1 - s; flow = f;
    phase = k >= N ? 'done' : 'backward';
  } else if (mode === 'reverse') {
    adj[tape.root] = t >= REC ? 1 : 0; // the seed is visible before the sweep starts
  }
  return { show, adj, tan: tan.map((v, i) => v * show[i]), cursor, flow, phase };
}
