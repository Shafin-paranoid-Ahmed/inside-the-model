/**
 * Backprop Visualizer logic. No DOM. Built on the shared `Value` autograd graph,
 * so every number the widget shows is the number the engine computed.
 *
 * Two models share one sequencer:
 *  - 'mlp':  2 inputs -> 3 hidden (tanh | relu) -> 1 linear output, L = (yhat - t)^2, no biases.
 *  - 'expr': L = (a*b + c)^2, every Value is a drawn node.
 *
 * A cycle is a list of Steps: forward (one drawn node each), loss, backward
 * (one `localBackward()` per Value, walking `prepareBackward()` order; pure
 * '+'/'-' Values are "silent" and run together with the step before them),
 * then one SGD update. Stepping past the end starts the next cycle from the
 * updated weights.
 *
 * On top of that, `makePlan` + `frameAt` (bottom of file) turn one cycle into a
 * continuous, scrubbable timeline: the widget only ever calls `frameAt(t, plan)`.
 */
import { Value, prepareBackward, backward } from '../../../lib/nn';

export type Mode = 'mlp' | 'expr';
export type Act = 'tanh' | 'relu';
export type Phase = 'forward' | 'loss' | 'backward' | 'update';
export const PHASES: Phase[] = ['forward', 'loss', 'backward', 'update'];

export interface Cfg {
  mode: Mode;
  act: Act;
  /** mlp: [x1, x2] */
  x: number[];
  target: number;
  /** initial parameters (mlp: 6 hidden weights w_ji at j*2+i, then v1..v3; expr: a, b, c) */
  init: number[];
  lr: number;
}

export const MLP_INIT = [0.5, -0.6, -0.7, 0.9, 0.3, 0.7, 0.8, -0.6, 0.5];
export const defaultCfg = (mode: Mode): Cfg =>
  mode === 'mlp'
    ? { mode, act: 'tanh', x: [1, 0.5], target: -0.5, init: [...MLP_INIT], lr: 0.1 }
    : { mode, act: 'tanh', x: [], target: 0, init: [2, -3, 10], lr: 0.02 };

// ---------- formatting ----------
const SUB = '₀₁₂₃₄₅₆₇₈₉';
export const sub = (n: number): string => String(n).replace(/\d/g, (d) => SUB[+d]);

/** 3 significant digits, true minus sign. */
export function fmt(x: number): string {
  if (Number.isNaN(x)) return 'NaN';
  if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
  if (x === 0) return '0.00';
  const a = Math.abs(x);
  let s = a < 1e-3 || a >= 1e5 ? a.toExponential(1) : a.toPrecision(3);
  if (s.includes('e') && a >= 1e-3) s = String(Math.round(a));
  return (x < 0 ? '−' : '') + s;
}

// ---------- local derivative (mirrors Value.localBackward) ----------
/** d(p)/d(p.children[k]) evaluated at the current data. */
export function ld(p: Value, k: number): number {
  const [a, b] = p.children;
  switch (p.op) {
    case '+': return 1;
    case '-': return k === 0 ? 1 : -1;
    case '*': return k === 0 ? b.data : a.data;
    case '/': return k === 0 ? 1 / b.data : -a.data / (b.data * b.data);
    case 'neg': return -1;
    case 'pow': return p.exponent * a.data ** (p.exponent - 1);
    case 'exp': return p.data;
    case 'log': return 1 / a.data;
    case 'tanh': return 1 - p.data * p.data;
    case 'relu': return p.data > 0 ? 1 : 0;
    case 'sigmoid': return p.data * (1 - p.data);
    default: return 0;
  }
}

// ---------- model ----------
export type Role = 'input' | 'param' | 'hidden' | 'out' | 'loss' | 'op';
export interface VNode {
  id: string; label: string; formula?: string; role: Role;
  value: Value; pre?: Value; preLabel?: string;
}
export interface VEdge {
  id: string; from: string; to: string;
  /** the Value whose localBackward pushes gradient along this edge, and which child */
  via: Value; k: number;
  /** true when the child is a trainable parameter drawn as the edge weight */
  param: boolean; sym: string;
}
export interface Model {
  nodes: VNode[]; edges: VEdge[]; params: Value[]; loss: Value;
  fwd: string[]; lossId: string;
  labels: Map<Value, string>; silent: Set<Value>;
}

export function buildMlp(w: number[], x: number[], target: number, act: Act): Model {
  const labels = new Map<Value, string>();
  const nodes: VNode[] = [], edges: VEdge[] = [];
  const W = w.slice(0, 6).map((d) => new Value(d));
  const X = x.map((d) => new Value(d));
  X.forEach((v, i) => {
    labels.set(v, `x${sub(i + 1)}`);
    nodes.push({ id: `x${i + 1}`, label: `x${sub(i + 1)}`, role: 'input', value: v });
  });
  const H: Value[] = [];
  for (let j = 0; j < 3; j++) {
    const p = [0, 1].map((i) => W[j * 2 + i].mul(X[i]));
    const z = p[0].add(p[1]);
    const h = act === 'tanh' ? z.tanh() : z.relu();
    H.push(h);
    labels.set(z, `z${sub(j + 1)}`);
    labels.set(h, `h${sub(j + 1)}`);
    nodes.push({
      id: `h${j + 1}`, label: `h${sub(j + 1)}`, role: 'hidden', value: h, pre: z, preLabel: `z${sub(j + 1)}`,
      formula: `${act}(z${sub(j + 1)}), z${sub(j + 1)} = w${sub(j + 1)}${sub(1)}x${sub(1)} + w${sub(j + 1)}${sub(2)}x${sub(2)}`,
    });
    [0, 1].forEach((i) => {
      const sym = `w${sub(j + 1)}${sub(i + 1)}`;
      labels.set(W[j * 2 + i], sym);
      edges.push({ id: `w${j + 1}${i + 1}`, from: `x${i + 1}`, to: `h${j + 1}`, via: p[i], k: 0, param: true, sym });
    });
  }
  const V = w.slice(6).map((d) => new Value(d));
  const q = V.map((v, j) => v.mul(H[j]));
  const yhat = q[0].add(q[1]).add(q[2]);
  V.forEach((v, j) => {
    const sym = `v${sub(j + 1)}`;
    labels.set(v, sym);
    edges.push({ id: `v${j + 1}`, from: `h${j + 1}`, to: 'yhat', via: q[j], k: 0, param: true, sym });
  });
  const d = yhat.sub(target);
  const L = d.pow(2);
  labels.set(yhat, 'ŷ');
  labels.set(L, 'L');
  nodes.push({ id: 'yhat', label: 'ŷ', role: 'out', value: yhat, formula: `ŷ = ${V.map((_, j) => `v${sub(j + 1)}h${sub(j + 1)}`).join(' + ')}` });
  nodes.push({ id: 'loss', label: 'L', role: 'loss', value: L, formula: 'L = (ŷ − t)²' });
  edges.push({ id: 'e-loss', from: 'yhat', to: 'loss', via: d, k: 0, param: false, sym: '' });

  const silent = new Set<Value>();
  const mark = (v: Value) => { if (v.op === '+' || v.op === '-') silent.add(v); v.children.forEach(mark); };
  mark(L);
  return { nodes, edges, params: [...W, ...V], loss: L, fwd: ['h1', 'h2', 'h3', 'yhat'], lossId: 'loss', labels, silent };
}

export function buildExpr(init: number[]): Model {
  const [a, b, c] = init.map((d) => new Value(d));
  const m = a.mul(b), s = m.add(c), L = s.pow(2);
  const labels = new Map<Value, string>([[a, 'a'], [b, 'b'], [c, 'c'], [m, 'm'], [s, 's'], [L, 'L']]);
  const nodes: VNode[] = [
    { id: 'a', label: 'a', role: 'param', value: a },
    { id: 'b', label: 'b', role: 'param', value: b },
    { id: 'c', label: 'c', role: 'param', value: c },
    { id: 'm', label: 'm', role: 'op', value: m, formula: 'm = a · b' },
    { id: 's', label: 's', role: 'op', value: s, formula: 's = m + c' },
    { id: 'L', label: 'L', role: 'loss', value: L, formula: 'L = s²' },
  ];
  const e = (from: string, to: string, via: Value, k: number): VEdge => ({ id: `${from}-${to}`, from, to, via, k, param: false, sym: '' });
  const edges = [e('a', 'm', m, 0), e('b', 'm', m, 1), e('m', 's', s, 0), e('c', 's', s, 1), e('s', 'L', L, 0)];
  return { nodes, edges, params: [a, b, c], loss: L, fwd: ['m', 's'], lossId: 'L', labels, silent: new Set() };
}

export const build = (cfg: Cfg, w: number[]): Model =>
  cfg.mode === 'mlp' ? buildMlp(w, cfg.x, cfg.target, cfg.act) : buildExpr(w);

// ---------- layout ----------
export interface Box { x: number; y: number; hw: number; hh: number }
export interface Layout { w: number; h: number; box: Record<string, Box>; vertical: boolean }

export function layout(mode: Mode, vertical: boolean): Layout {
  const B = (x: number, y: number, hw: number, hh: number): Box => ({ x, y, hw, hh });
  if (mode === 'mlp') {
    return vertical
      ? { w: 360, h: 620, vertical, box: {
          x1: B(110, 40, 36, 26), x2: B(250, 40, 36, 26),
          h1: B(60, 250, 54, 44), h2: B(180, 250, 54, 44), h3: B(300, 250, 54, 44),
          yhat: B(180, 440, 42, 26), loss: B(180, 565, 42, 26) } }
      : { w: 720, h: 400, vertical, box: {
          x1: B(66, 135, 36, 26), x2: B(66, 265, 36, 26),
          h1: B(300, 70, 54, 44), h2: B(300, 200, 54, 44), h3: B(300, 330, 54, 44),
          yhat: B(520, 200, 42, 26), loss: B(650, 200, 42, 26) } };
  }
  return vertical
    ? { w: 360, h: 500, vertical, box: {
        a: B(105, 50, 42, 26), b: B(255, 50, 42, 26), m: B(180, 165, 42, 26),
        c: B(285, 255, 42, 26), s: B(180, 335, 42, 26), L: B(180, 450, 42, 26) } }
    : { w: 720, h: 320, vertical, box: {
        a: B(64, 85, 42, 26), b: B(64, 205, 42, 26), m: B(250, 145, 42, 26),
        c: B(250, 265, 42, 26), s: B(440, 195, 42, 26), L: B(640, 195, 42, 26) } };
}

/** Straight-line endpoints from the near side of `from` to the near side of `to`. */
export function endpoints(e: VEdge, l: Layout): { x1: number; y1: number; x2: number; y2: number } {
  const f = l.box[e.from], t = l.box[e.to];
  return l.vertical
    ? { x1: f.x, y1: f.y + f.hh, x2: t.x, y2: t.y - t.hh }
    : { x1: f.x + f.hw, y1: f.y, x2: t.x - t.hw, y2: t.y };
}

// ---------- chain rule (generic path enumeration over the Value graph) ----------
export interface Factor { a: string; b: string; value: number }
export interface Rule { sym: string; paths: Factor[][]; value: number; grad: number; reached: boolean }
interface Hop { p: Value; k: number }

// ponytail: exhaustive path enumeration, fine for graphs of a few dozen nodes; memoise if graphs grow.
function hops(root: Value, target: Value): Hop[][] {
  if (root === target) return [[]];
  const out: Hop[][] = [];
  root.children.forEach((c, k) => { for (const rest of hops(c, target)) out.push([{ p: root, k }, ...rest]); });
  return out;
}

/** Every root->target path, its local derivatives multiplied between labelled Values. */
export function rulePaths(root: Value, target: Value, labels: Map<Value, string>): Factor[][] {
  return hops(root, target).map((path) => {
    const out: Factor[] = [];
    let prev = root, prod = 1;
    for (const { p, k } of path) {
      prod *= ld(p, k);
      const c = p.children[k];
      if (labels.has(c)) { out.push({ a: labels.get(prev)!, b: labels.get(c)!, value: prod }); prev = c; prod = 1; }
    }
    return out;
  });
}
export const ruleValue = (paths: Factor[][]): number =>
  paths.reduce((s, f) => s + f.reduce((p, x) => p * x.value, 1), 0);

// ---------- sequencer ----------
export interface Step {
  phase: Phase;
  /** what pressing Step will do */
  preview: string;
  /** drawn node to ring while this step is current */
  focus?: string;
  /** backward: Values to run localBackward on, in order */
  run?: Value[];
  /** edges the pulse travels along */
  edges: string[];
  /** text after the step has executed */
  narrate: () => string;
}

export interface Update { old: number[]; next: number[]; grads: number[]; lossBefore: number; lossAfter: number; lr: number }

export class Sim {
  cfg: Cfg;
  model!: Model;
  steps: Step[] = [];
  idx = 0;
  done = new Set<Value>();
  touched = new Set<Value>();
  upd: Update | null = null;
  /** start-of-cycle loss of every finished cycle */
  committed: number[] = [];
  cycleW: number[];
  gradScale = 1;
  last = '';

  constructor(cfg: Cfg) {
    this.cfg = { ...cfg, init: [...cfg.init], x: [...cfg.x] };
    this.cycleW = [...cfg.init];
    this.startCycle();
  }

  private startCycle(): void {
    this.model = build(this.cfg, this.cycleW);
    this.idx = 0; this.done.clear(); this.touched.clear(); this.upd = null; this.last = '';
    const m = this.model;
    const order = prepareBackward(m.loss);
    this.steps = this.makeSteps(order);
    const ref = build(this.cfg, this.cycleW);
    backward(ref.loss);
    this.gradScale = Math.max(1e-9, ...ref.edges.map((e) => Math.abs(e.via.grad * ld(e.via, e.k))));
  }

  private node(id: string): VNode { return this.model.nodes.find((n) => n.id === id)!; }

  private makeSteps(order: Value[]): Step[] {
    const m = this.model, lb = m.labels, out: Step[] = [];
    const cfg = this.cfg;
    for (const id of m.fwd) {
      const n = this.node(id);
      out.push({
        phase: 'forward', focus: id, preview: `Compute ${n.label}`,
        edges: m.edges.filter((e) => e.to === id).map((e) => e.id),
        narrate: () => n.pre
          ? `Forward: ${n.preLabel} = ${fmt(n.pre.data)}, ${n.label} = ${cfg.act}(${n.preLabel}) = ${fmt(n.value.data)}`
          : `Forward: ${n.label} = ${fmt(n.value.data)}${n.formula ? ` (${n.formula.replace(/^.* = /, '')})` : ''}`,
      });
    }
    const L = m.loss;
    out.push({
      phase: 'loss', focus: m.lossId, preview: 'Compute the loss',
      edges: m.edges.filter((e) => e.to === m.lossId).map((e) => e.id),
      narrate: () => cfg.mode === 'mlp'
        ? `Loss: L = (ŷ − t)² = (${fmt(m.nodes.find((n) => n.id === 'yhat')!.value.data)} − ${fmt(cfg.target)})² = ${fmt(L.data)}`
        : `Loss: L = s² = ${fmt(L.data)}`,
    });
    const groups: Value[][] = [];
    for (const n of order) {
      if (!n.children.length) continue;
      if (m.silent.has(n) && groups.length) groups[groups.length - 1].push(n);
      else groups.push([n]);
    }
    for (const run of groups) {
      const edges = m.edges.filter((e) => run.includes(e.via));
      const kids = [...new Set(run.flatMap((n) => n.children))]
        .filter((c) => lb.has(c) && !m.nodes.some((v) => v.role === 'input' && v.value === c));
      const syms = kids.map((c) => `∂L/∂${lb.get(c)}`);
      const vn = m.nodes.find((v) => v.value === run[0]) ?? this.node(edges[0]?.to ?? m.lossId);
      out.push({
        phase: 'backward', run, edges: edges.map((e) => e.id), focus: vn.id,
        preview: `Compute ${syms.join(', ') || 'gradients'}`,
        narrate: () => `Backward: ${kids.map((c) => `∂L/∂${lb.get(c)} = ${fmt(c.grad)}`).join(', ')}`,
      });
    }
    out.push({
      phase: 'update', preview: 'Apply one SGD step', edges: [],
      narrate: () => {
        const u = this.upd!;
        return `Update: loss ${fmt(u.lossBefore)} to ${fmt(u.lossAfter)} with learning rate ${u.lr}`;
      },
    });
    return out;
  }

  get complete(): boolean { return this.idx >= this.steps.length; }
  get current(): Step { return this.steps[Math.min(this.idx, this.steps.length - 1)]; }
  get phase(): Phase { return this.current.phase; }
  phaseStart(p: Phase): number { return this.steps.findIndex((s) => s.phase === p); }
  get lossNow(): number { return this.model.loss.data; }

  /** Loss values to plot: finished cycles, this cycle, and the post-update loss. */
  get history(): number[] {
    return [...this.committed, this.lossNow, ...(this.upd ? [this.upd.lossAfter] : [])];
  }

  private exec(s: Step): void {
    if (s.phase === 'backward') {
      for (const n of s.run!) {
        n.localBackward();
        this.done.add(n);
        this.touched.add(n);
        n.children.forEach((c) => this.touched.add(c));
      }
    } else if (s.phase === 'update') {
      const { params, loss } = this.model, lr = this.cfg.lr;
      const old = params.map((p) => p.data), grads = params.map((p) => p.grad);
      const next = old.map((v, i) => v - lr * grads[i]);
      this.upd = { old, next, grads, lossBefore: loss.data, lossAfter: build(this.cfg, next).loss.data, lr };
    }
  }

  /** Execute the next step (rolling into a new cycle after the update). */
  step(): Step {
    if (this.complete) this.nextCycle();
    const s = this.steps[this.idx++];
    this.exec(s);
    this.last = s.narrate();
    return s;
  }

  nextCycle(): void {
    if (!this.upd) return;
    this.committed.push(this.lossNow);
    this.cycleW = this.upd.next;
    this.startCycle();
  }

  /** Jump within the current cycle to `i` executed steps (rebuilds, then replays). */
  goto(i: number): void {
    this.startCycle();
    const n = Math.max(0, Math.min(i, this.steps.length));
    while (this.idx < n) this.step();
    if (this.idx === 0) this.last = '';
  }

  runToEnd(): void { while (!this.complete) this.step(); }

  /** Run `n` full training cycles with no animation; ends with the update executed. */
  train(n: number): void {
    for (let i = 0; i < n; i++) { this.runToEnd(); if (i < n - 1) this.nextCycle(); }
  }

  /** Values whose gradient a drawn node id or parameter-edge id stands for. */
  targets(id: string): Value[] {
    const m = this.model;
    const n = m.nodes.find((v) => v.id === id);
    if (n) return n.pre ? [n.value, n.pre] : [n.value];
    const e = m.edges.find((v) => v.id === id && v.param);
    return e ? [e.via.children[e.k]] : [];
  }

  /** Chain rules (with numbers) for a drawn node id or a parameter edge id. */
  rules(id: string): Rule[] {
    const m = this.model, lb = m.labels;
    return this.targets(id).map((t) => {
      const paths = rulePaths(m.loss, t, lb);
      return { sym: lb.get(t)!, paths, value: ruleValue(paths), grad: t.grad, reached: this.touched.has(t) || (t === m.loss && this.idx > this.phaseStart('backward')) };
    });
  }

  /** Drawn nodes and edges on every path from the loss to `id`. */
  route(id: string): { nodes: Set<string>; edges: Set<string> } {
    const m = this.model, nodes = new Set<string>(), edges = new Set<string>();
    for (const t of this.targets(id)) for (const path of hops(m.loss, t)) for (const { p, k } of path)
      for (const e of m.edges) if (e.via === p && (e.k === k || e.param)) { edges.add(e.id); nodes.add(e.from); nodes.add(e.to); }
    return { nodes, edges };
  }
  paramIndex(id: string): number {
    const e = this.model.edges.find((v) => v.id === id && v.param);
    if (e) return this.model.params.indexOf(e.via.children[e.k]);
    const n = this.model.nodes.find((v) => v.id === id);
    return n ? this.model.params.indexOf(n.value) : -1;
  }
}

// ---------- view (plain data for rendering) ----------
export interface NodeView {
  id: string; label: string; formula?: string; role: Role; hasPre: boolean; preLabel?: string;
  shown: boolean; value: string; gradShown: boolean; grad: string;
  preShown: boolean; pre: string; preGradShown: boolean; preGrad: string;
  focus: boolean; stale: boolean;
}
export interface EdgeView {
  id: string; from: string; to: string; param: boolean; sym: string;
  mode: 'weight' | 'local' | 'grad'; text: string; sign: -1 | 0 | 1; mag: number;
}
export interface View {
  nodes: NodeView[]; edges: EdgeView[];
  phase: Phase; idx: number; total: number; complete: boolean;
  preview: string; narration: string; history: number[]; lossNow: number;
  upd: Update | null;
}

export function view(s: Sim): View {
  const m = s.model, ex = s.steps.slice(0, s.idx);
  const fwdDone = new Set(ex.filter((x) => x.phase === 'forward').map((x) => x.focus));
  if (ex.some((x) => x.phase === 'loss')) fwdDone.add(m.lossId);
  const upd = s.upd;
  const cur = s.current;
  const nodes: NodeView[] = m.nodes.map((n) => {
    const isParam = n.role === 'param';
    const shown = n.role === 'input' || isParam || fwdDone.has(n.id);
    const pi = isParam ? m.params.indexOf(n.value) : -1;
    const showVal = upd && pi >= 0 ? upd.next[pi] : n.value.data;
    const gs = s.touched.has(n.value) || (n.value === m.loss && ex.some((x) => x.phase === 'backward'));
    return {
      id: n.id, label: n.label, formula: n.formula, role: n.role, hasPre: !!n.pre, preLabel: n.preLabel,
      shown, value: fmt(showVal), gradShown: gs, grad: fmt(n.value.grad),
      preShown: shown && !!n.pre, pre: n.pre ? fmt(n.pre.data) : '',
      preGradShown: !!n.pre && s.touched.has(n.pre), preGrad: n.pre ? fmt(n.pre.grad) : '',
      focus: !s.complete && cur.phase !== 'update' && cur.focus === n.id,
      stale: !!upd && !isParam && n.role !== 'input',
    };
  });
  const maxW = Math.max(1e-9, ...m.edges.filter((e) => e.param).map((e) => Math.abs(e.via.children[e.k].data)));
  const edges: EdgeView[] = m.edges.map((e) => {
    const child = e.via.children[e.k];
    const g = e.via.grad * ld(e.via, e.k);
    const gradMode = s.done.has(e.via) && !(upd && e.param);
    const pi = e.param ? m.params.indexOf(child) : -1;
    const w = upd && pi >= 0 ? upd.next[pi] : child.data;
    const sg = (x: number): -1 | 0 | 1 => (x > 0 ? 1 : x < 0 ? -1 : 0);
    if (gradMode) return { id: e.id, from: e.from, to: e.to, param: e.param, sym: e.sym, mode: 'grad', text: e.param || s.cfg.mode === 'expr' ? fmt(g) : '', sign: sg(g), mag: Math.min(1, Math.abs(g) / s.gradScale) };
    if (e.param) return { id: e.id, from: e.from, to: e.to, param: true, sym: e.sym, mode: 'weight', text: fmt(w), sign: sg(w), mag: Math.abs(w) / maxW };
    // expression graph: show the local derivative once the parent node exists
    const local = s.cfg.mode === 'expr' && fwdDone.has(e.to) ? fmt(ld(e.via, e.k)) : '';
    return { id: e.id, from: e.from, to: e.to, param: false, sym: '', mode: 'local', text: local, sign: 0, mag: 0.2 };
  });
  return {
    nodes, edges, phase: s.phase, idx: s.idx, total: s.steps.length, complete: s.complete,
    preview: s.complete ? 'Start the next cycle' : cur.preview, narration: s.last,
    history: s.history, lossNow: s.lossNow, upd,
  };
}

// ============================================================================
// Continuous timeline. One training iteration = one cycle of T seconds. `makePlan`
// runs the engine once (via Sim) and records every number plus a schedule; `frameAt`
// is a pure function of (t, plan), so scrubbing is free and nothing depends on history.
// ============================================================================
export const T = 6;
type Win = [number, number];
export const PH: Record<Phase, Win> = { forward: [0, 2.2], loss: [2.2, 3], backward: [3, 5.1], update: [5.1, 6] };
export const phaseAt = (t: number): Phase => (t < PH.loss[0] ? 'forward' : t < PH.backward[0] ? 'loss' : t < PH.update[0] ? 'backward' : 'update');

const lerp = (a: number, b: number, x: number): number => a * (1 - x) + b * x;
const sstep = (x: number): number => { x = x < 0 ? 0 : x > 1 ? 1 : x; return x * x * (3 - 2 * x); };
/** Linear progress of t through window w (exactly 0 / 1 outside it). */
const prog = (t: number, w: Win): number => (t >= w[1] ? 1 : t <= w[0] ? 0 : (t - w[0]) / (w[1] - w[0]));
const ease = (t: number, w: Win): number => sstep(prog(t, w));

export interface NodePlan {
  id: string; label: string; formula?: string; role: Role; hasPre: boolean; preLabel?: string;
  value: number; pre: number; grad: number; preGrad: number;
  /** parameter index for trainable nodes (expr mode), else -1 */
  pi: number;
  vw: Win; zw: Win; gw: Win; zgw: Win;
}
export interface EdgePlan {
  id: string; from: string; to: string; param: boolean; sym: string;
  pi: number; w0: number; w1: number;
  /** particle window (forward pass or loss) and gradient-sweep window (backward) */
  fw: Win; bw: Win;
  /** dL/dw for this edge, and the local derivative of the parent op */
  g: number; local: number;
}
export interface Plan {
  cfg: Cfg; sim: Sim; w0: number[]; w1: number[];
  lossBefore: number; lossAfter: number; maxW: number; gradScale: number;
  nodes: NodePlan[]; edges: EdgePlan[];
  /** weights morph over `wu`; derived numbers fade over `fade` */
  wu: Win; fade: Win;
}

/** Equal slot `i` of `n` inside [a, b]; the `rank`-th of `ranks` targets is staggered by up to 25% of the slot. */
function slot(a: number, b: number, i: number, n: number, rank: number, ranks: number): Win {
  const ls = lerp(a, b, i / n), le = lerp(a, b, (i + 1) / n), len = le - ls;
  const rr = ranks > 1 ? rank / (ranks - 1) : 1;
  return [ls + 0.25 * len * rr, le - 0.25 * len * (1 - rr)];
}

export function makePlan(cfg: Cfg, w: number[]): Plan {
  const sim = new Sim({ ...cfg, init: w });
  sim.runToEnd();
  const m = sim.model, u = sim.upd!;
  const depth: Record<string, number> = {};
  m.nodes.forEach((n) => (depth[n.id] = 0));
  for (let i = 0; i < m.nodes.length; i++) for (const e of m.edges) depth[e.to] = Math.max(depth[e.to], depth[e.from] + 1);
  const D = depth[m.lossId], dmax = Math.max(1, D - 1);
  const [F0, F1] = PH.forward, [L0, L1] = PH.loss, [B0, B1] = PH.backward, [U0, U1] = PH.update;
  const layerLen = (B1 - B0) / D;

  // rank of each target node within its layer (edges into one node share a window)
  const rankOf = (key: (e: VEdge) => number, sel: (e: VEdge) => boolean) => {
    const by = new Map<number, string[]>();
    for (const e of m.edges.filter(sel)) {
      const k = key(e), l = by.get(k) ?? [];
      if (!l.includes(e.to)) l.push(e.to);
      by.set(k, l);
    }
    return (e: VEdge): [number, number] => { const l = by.get(key(e))!; return [l.indexOf(e.to), l.length]; };
  };
  const fwdSel = (e: VEdge) => e.to !== m.lossId;
  const fRank = rankOf((e) => depth[e.to], fwdSel);
  const bRank = rankOf((e) => D - depth[e.to], () => true);

  const edges: EdgePlan[] = m.edges.map((e) => {
    const pi = e.param ? m.params.indexOf(e.via.children[e.k]) : -1;
    let fw: Win;
    if (fwdSel(e)) { const [r, n] = fRank(e); fw = slot(F0, F1, depth[e.to] - 1, dmax, r, n); }
    else fw = [L0, lerp(L0, L1, 0.65)];
    const [r, n] = bRank(e);
    return {
      id: e.id, from: e.from, to: e.to, param: e.param, sym: e.sym, pi,
      w0: pi >= 0 ? u.old[pi] : 0, w1: pi >= 0 ? u.next[pi] : 0,
      fw, bw: slot(B0, B1, D - depth[e.to], D, r, n), g: e.via.grad * ld(e.via, e.k), local: ld(e.via, e.k),
    };
  });

  const nodes: NodePlan[] = m.nodes.map((n) => {
    const ins = edges.filter((e) => e.to === n.id), outs = edges.filter((e) => e.from === n.id);
    const source = n.role === 'input' || n.role === 'param';
    let vw: Win = [0, 0], zw: Win = [0, 0];
    if (n.id === m.lossId) vw = [lerp(L0, L1, 0.45), L1];
    else if (!source && ins.length) {
      const [s, e] = ins[0].fw;
      vw = [lerp(s, e, n.pre ? 0.6 : 0.55), e];
      zw = [lerp(s, e, 0.4), lerp(s, e, 0.8)];
    }
    let gw: Win;
    if (n.id === m.lossId) gw = [B0, B0 + 0.5 * layerLen];
    else {
      const last = outs.reduce((a, b) => (b.bw[1] > a.bw[1] ? b : a));
      gw = [lerp(last.bw[0], last.bw[1], 0.5), last.bw[1]];
    }
    return {
      id: n.id, label: n.label, formula: n.formula, role: n.role, hasPre: !!n.pre, preLabel: n.preLabel,
      value: n.value.data, pre: n.pre ? n.pre.data : 0, grad: n.value.grad, preGrad: n.pre ? n.pre.grad : 0,
      pi: n.role === 'param' ? m.params.indexOf(n.value) : -1,
      vw, zw, gw, zgw: [gw[1], gw[1] + 0.3 * layerLen],
    };
  });

  const maxW = Math.max(1, ...u.old.map(Math.abs), ...u.next.map(Math.abs));
  return {
    cfg: sim.cfg, sim, w0: u.old, w1: u.next, lossBefore: u.lossBefore, lossAfter: u.lossAfter, maxW,
    gradScale: sim.gradScale, nodes, edges, wu: [U0, lerp(U0, U1, 0.85)], fade: [lerp(U0, U1, 0.3), U1],
  };
}

export interface NodeFrame {
  id: string;
  /** Exact computed numbers and their visual reveal progress 0..1. */
  value: number; valueP: number; pre: number; preP: number;
  grad: number; gradP: number; preGrad: number; preGradP: number;
  /** 1 while derived numbers are visible, fading to 0 at the end of Update */
  alpha: number;
}
export interface EdgeFrame {
  id: string;
  /** displayed weight (param edges), eased from old to new during Update */
  w: number;
  /** forward particle position along the edge (0 = from, 1 = to), or -1 when none is in flight */
  fwd: number;
  /** gradient sweep from head (to) to tail (from), 0..1, and its displayed value */
  fill: number; g: number;
  /** local derivative shown on expression edges, with its opacity */
  local: number; localA: number;
  alpha: number;
}
export interface Frame {
  t: number; phase: Phase; phaseP: number;
  /** Loss evaluated at the weights currently displayed during the update. */
  loss: number;
  nodes: NodeFrame[]; edges: EdgeFrame[];
  /** node the wave is currently at ('' during Update) */
  active: string;
  /** Update easing 0..1 (weights, loss tip) */
  upd: number;
}

export function frameAt(t: number, p: Plan): Frame {
  t = Math.max(0, Math.min(T, t));
  const phase = phaseAt(t), lossId = p.sim.model.lossId;
  const upd = ease(t, p.wu), alpha = 1 - ease(t, p.fade);
  const nodes: NodeFrame[] = p.nodes.map((n) => {
    const source = n.role === 'input' || n.role === 'param';
    const valueP = source ? 1 : ease(t, n.vw), preP = n.hasPre ? ease(t, n.zw) : 0;
    const gradP = ease(t, n.gw), preGradP = n.hasPre ? ease(t, n.zgw) : 0;
    return {
      id: n.id,
      value: n.pi >= 0 ? lerp(p.w0[n.pi], p.w1[n.pi], upd) : n.value,
      valueP, pre: n.pre, preP, grad: n.grad, gradP, preGrad: n.preGrad, preGradP, alpha,
    };
  });
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const edges: EdgeFrame[] = p.edges.map((e) => {
    const f = prog(t, e.fw), fill = ease(t, e.bw);
    return {
      id: e.id, w: e.param ? lerp(e.w0, e.w1, upd) : 0,
      fwd: f > 0 && f < 1 ? sstep(f) : -1,
      fill, g: e.g, local: e.local, localA: byId.get(e.to)!.valueP * alpha, alpha,
    };
  });
  // the node whose number is being computed right now (else the next one due, else the latest finished)
  const win = (n: NodePlan): Win | null =>
    phase === 'forward' ? (n.role === 'input' || n.role === 'param' || n.id === lossId ? null : n.vw)
      : phase === 'loss' ? (n.id === lossId ? n.vw : null)
      : phase === 'backward' ? n.gw : null;
  let active = '', best = Infinity, lastEnd = -1;
  for (const n of p.nodes) {
    const w = win(n);
    if (!w) continue;
    if (t < w[1] && w[1] < best) { best = w[1]; active = n.id; }
    else if (best === Infinity && t >= w[1] && w[1] > lastEnd) { lastEnd = w[1]; active = n.id; }
  }
  const loss = upd === 0 ? p.lossBefore : upd === 1 ? p.lossAfter
    : build(p.cfg, p.w0.map((w, i) => lerp(w, p.w1[i], upd))).loss.data;
  return { t, phase, phaseP: prog(t, PH[phase]), loss, nodes, edges, active, upd };
}
