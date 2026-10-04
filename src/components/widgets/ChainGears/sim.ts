/** Pure logic for ChainGears: a nudge to x travels through linked functions; each link multiplies it by its rate. */

export type ScenarioId = 'two' | 'long' | 'paths';
export interface GNode {
  name: string;
  label: string;
  inputs: number[];
  f: (v: number[]) => number;
  /** Partial derivatives with respect to each input. */
  d: (v: number[]) => number[];
  col: number;
  row: number;
}
export interface Scenario {
  id: ScenarioId;
  nodes: GNode[];
  x0: number;
  xRange: [number, number];
  /** One label per link level, then the closing label. */
  labels: string[];
  say: string[];
  closing: 'multiply' | 'add';
}
const X: GNode = { name: 'x', label: 'x', inputs: [], f: () => 0, d: () => [], col: 0.5, row: 0 };
const one = (name: string, label: string, f: (a: number) => number, df: (a: number) => number, row: number): GNode =>
  ({ name, label, inputs: [row - 1], f: (v) => f(v[0]), d: (v) => [df(v[0])], col: 0.5, row });

export const SCENARIOS: Record<ScenarioId, Scenario> = {
  // g = 3x, y = g²
  two: {
    id: 'two', x0: 1, xRange: [-2, 2], closing: 'multiply',
    nodes: [X, one('g', 'g = 3x', (a) => 3 * a, () => 3, 1), one('y', 'y = g²', (a) => a * a, (a) => 2 * a, 2)],
    labels: ['link 1', 'link 2', 'multiply'],
    say: [
      'Nudge x by a tiny amount. Follow the nudge down the chain.',
      'The first link multiplies the nudge by its rate, dg/dx.',
      'The second link multiplies that nudge again, by dy/dg.',
      'Multiply the two rates and you have dy/dx, the rate for the whole chain.',
    ],
  },
  // a = 2x, b = a², c = 1/(1+b), y = 4c
  long: {
    id: 'long', x0: 0.5, xRange: [-1, 1], closing: 'multiply',
    nodes: [
      X,
      one('a', 'a = 2x', (v) => 2 * v, () => 2, 1),
      one('b', 'b = a²', (v) => v * v, (v) => 2 * v, 2),
      one('c', 'c = 1/(1+b)', (v) => 1 / (1 + v), (v) => -1 / (1 + v) ** 2, 3),
      one('y', 'y = 4c', (v) => 4 * v, () => 4, 4),
    ],
    labels: ['link 1', 'link 2', 'link 3', 'link 4', 'multiply'],
    say: [
      'Nudge x. Four links now stand between x and y.',
      'Link 1 scales the nudge by da/dx.',
      'Link 2 scales it again, by db/da.',
      'Link 3 scales it again, by dc/db. A negative rate flips the direction of the nudge.',
      'Link 4 scales it once more.',
      'The rates along the chain multiply. Longer chain, same rule.',
    ],
  },
  // a = x², b = 3x, y = a·b
  paths: {
    id: 'paths', x0: 2, xRange: [-2, 3], closing: 'add',
    nodes: [
      X,
      { name: 'a', label: 'a = x²', inputs: [0], f: (v) => v[0] * v[0], d: (v) => [2 * v[0]], col: 0.27, row: 1 },
      { name: 'b', label: 'b = 3x', inputs: [0], f: (v) => 3 * v[0], d: () => [3], col: 0.73, row: 1 },
      { name: 'y', label: 'y = a·b', inputs: [1, 2], f: (v) => v[0] * v[1], d: (v) => [v[1], v[0]], col: 0.5, row: 2 },
    ],
    labels: ['two paths', 'they meet', 'add'],
    say: [
      'Now x feeds y through two separate paths. Nudge x.',
      'The nudge goes down both paths at once, each scaled by its own rate.',
      'Both scaled nudges arrive at y. The two contributions add.',
      'Multiply along each path, then add the paths: that is dy/dx.',
    ],
  },
};

export interface Edge { from: number; to: number; level: number }

export function levels(s: Scenario): number[] {
  const lv: number[] = [];
  s.nodes.forEach((n, i) => { lv[i] = n.inputs.length ? 1 + Math.max(...n.inputs.map((j) => lv[j])) : 0; });
  return lv;
}
export const edgesOf = (s: Scenario): Edge[] => {
  const lv = levels(s);
  return s.nodes.flatMap((n, i) => n.inputs.map((from) => ({ from, to: i, level: lv[i] })));
};

/** Node values, per-edge rates, and gain[i] = d node_i / dx. */
export function evalScenario(s: Scenario, x: number) {
  const vals: number[] = [], gain: number[] = [], rates: number[][] = [];
  s.nodes.forEach((n, i) => {
    if (!n.inputs.length) { vals[i] = x; gain[i] = 1; rates[i] = []; return; }
    const v = n.inputs.map((j) => vals[j]);
    vals[i] = n.f(v);
    rates[i] = n.d(v);
    gain[i] = rates[i].reduce((a, r, k) => a + r * gain[n.inputs[k]], 0);
  });
  const last = s.nodes.length - 1;
  return { vals, rates, gain, y: vals[last], dydx: gain[last] };
}

/** Product of rates along every path from x to y. Their sum is dy/dx. */
export function pathProducts(s: Scenario, x: number): number[][] {
  const { rates } = evalScenario(s, x);
  const out: number[][] = [];
  const walk = (i: number, acc: number[]) => {
    if (i === 0) { out.push(acc.slice().reverse()); return; }
    s.nodes[i].inputs.forEach((j, k) => walk(j, [...acc, rates[i][k]]));
  };
  walk(s.nodes.length - 1, []);
  return out;
}

export const numericDeriv = (s: Scenario, x: number, h = 1e-5) => (evalScenario(s, x + h).y - evalScenario(s, x - h).y) / (2 * h);

export const LEAD = 1.5, LINK = 2, END = 2.2;
export const nLevels = (s: Scenario) => Math.max(...levels(s));
export const duration = (s: Scenario) => LEAD + nLevels(s) * LINK + END;
export const linkStart = (l: number) => LEAD + (l - 1) * LINK;
export const markersOf = (s: Scenario) => [
  { at: 0, label: 'nudge' },
  ...s.labels.slice(0, -1).map((label, i) => ({ at: linkStart(i + 1), label })),
  { at: LEAD + nLevels(s) * LINK, label: s.labels[s.labels.length - 1] },
];

const clamp01 = (a: number) => Math.min(1, Math.max(0, a));
const ease = (a: number) => { const u = clamp01(a); return u * u * (3 - 2 * u); };

/** Cumulative stacked contributions arriving at each node, scaled by `k[i]` (0 to 1). */
function stack(s: Scenario, x: number, dx: number, k: number[]) {
  const e = evalScenario(s, x);
  return s.nodes.map((n, i) => {
    if (!n.inputs.length) return [0, dx * k[0]];
    const c = [0];
    n.inputs.forEach((j, m) => c.push(c[c.length - 1] + e.rates[i][m] * e.gain[j] * dx * k[i]));
    return c;
  });
}

/** Largest displayed nudge over all nodes when fully revealed: the shared magnification. */
export const extentOf = (s: Scenario, x: number, dx: number) =>
  Math.max(1e-12, ...stack(s, x, dx, s.nodes.map(() => 1)).flatMap((c) => c.map(Math.abs)));

/** Everything needed to draw at time t. `dx` is the size of the nudge to x. */
export function frameAt(t: number, s: Scenario, x: number, dx: number) {
  const e = evalScenario(s, x), edges = edgesOf(s), L = nLevels(s);
  const prog = edges.map((ed) => clamp01((t - linkStart(ed.level)) / LINK));
  // arrival of the nudge at a node: x grows during the lead-in, others at the end of their incoming link
  const reveal = s.nodes.map((_, i) => (i === 0 ? ease(t / (LEAD * 0.7)) : ease((prog[edges.findIndex((ed) => ed.to === i)] - 0.6) / 0.4)));
  const cum = stack(s, x, dx, reveal);
  const ms = markersOf(s);
  let k = 0;
  ms.forEach((m, i) => { if (t >= m.at - 1e-9) k = i; });
  return {
    phaseIdx: k, phase: ms[k].label, say: s.say[k] ?? '',
    reveal, amp: cum.map((c) => c[c.length - 1]), cum,
    edges: edges.map((ed, i) => ({
      ...ed, r: ease(prog[i] * 2), spin: prog[i],
      rate: e.rates[ed.to][s.nodes[ed.to].inputs.indexOf(ed.from)],
      flow: prog[i] > 0 && prog[i] < 1 ? prog[i] : -1,
    })),
    wig: Math.sin((t * 2 * Math.PI) / 1.6),
    total: ease((t - (LEAD + L * LINK)) / 0.8),
    e,
  };
}

export const fmt = (v: number) =>
  (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)).replace('-', '−');
/** Signed, for nudges. */
export const fmtSigned = (v: number) => (v >= 0 ? '+' : '') + fmt(v);

/** The closing equation and the honest check: the real change in y for the real nudge. */
export function summary(s: Scenario, x: number, dx: number) {
  const paths = pathProducts(s, x), e = evalScenario(s, x);
  const term = (p: number[]) => p.map(fmt).join(' × ');
  const rhs = s.closing === 'add' ? paths.map((p) => `(${term(p)})`).join(' + ') : term(paths[0]);
  const actualSlope = (evalScenario(s, x + dx).y - e.y) / dx;
  return { line: `dy/dx = ${rhs} = ${fmt(e.dydx)}`, actual: `Real change: Δy/Δx = ${actualSlope.toFixed(2).replace('-', '−')} for Δx = ${dx}`, dydx: e.dydx, actualSlope };
}
