/**
 * Scalar autograd node (micrograd-style) with an inspectable graph.
 *
 * Step-wise backprop design: each node stores `op` + `children` (no closures);
 * `localBackward()` applies the chain rule for THAT node only, adding
 * `out.grad * d(out)/d(child)` into each child's grad. `backwardOrder(root)` is
 * the reverse-topological node list. A widget animates by calling
 * `prepareBackward(root)` (zero grads, seed root=1, returns the order) then
 * `localBackward()` on each node of that order one at a time.
 * `backward(root)` does all of it at once.
 */
export type Op =
  | '' | '+' | '-' | '*' | '/' | 'neg' | 'pow' | 'exp' | 'log' | 'tanh' | 'relu' | 'sigmoid';

let nextId = 0;
export const resetIds = (): void => { nextId = 0; };

export class Value {
  grad = 0;
  readonly id = nextId++;
  /** exponent for op 'pow' */
  exponent = 0;
  constructor(
    public data: number,
    public readonly children: readonly Value[] = [],
    public readonly op: Op = '',
    public label?: string,
  ) {}

  add(o: Value | number) { return new Value(this.data + val(o).data, [this, val(o)], '+'); }
  sub(o: Value | number) { return new Value(this.data - val(o).data, [this, val(o)], '-'); }
  mul(o: Value | number) { return new Value(this.data * val(o).data, [this, val(o)], '*'); }
  div(o: Value | number) { return new Value(this.data / val(o).data, [this, val(o)], '/'); }
  neg() { return new Value(-this.data, [this], 'neg'); }
  pow(k: number) { const r = new Value(this.data ** k, [this], 'pow'); r.exponent = k; return r; }
  exp() { return new Value(Math.exp(this.data), [this], 'exp'); }
  log() { return new Value(Math.log(this.data), [this], 'log'); }
  tanh() { return new Value(Math.tanh(this.data), [this], 'tanh'); }
  relu() { return new Value(this.data > 0 ? this.data : 0, [this], 'relu'); }
  sigmoid() { return new Value(1 / (1 + Math.exp(-this.data)), [this], 'sigmoid'); }

  /** Chain rule at this node only: push this.grad into children's grads. */
  localBackward(): void {
    const [a, b] = this.children;
    const g = this.grad;
    switch (this.op) {
      case '+': a.grad += g; b.grad += g; break;
      case '-': a.grad += g; b.grad -= g; break;
      case '*': a.grad += b.data * g; b.grad += a.data * g; break;
      case '/': a.grad += g / b.data; b.grad -= (g * a.data) / (b.data * b.data); break;
      case 'neg': a.grad -= g; break;
      case 'pow': a.grad += this.exponent * a.data ** (this.exponent - 1) * g; break;
      case 'exp': a.grad += this.data * g; break;
      case 'log': a.grad += g / a.data; break;
      case 'tanh': a.grad += (1 - this.data * this.data) * g; break;
      case 'relu': a.grad += (this.data > 0 ? 1 : 0) * g; break;
      case 'sigmoid': a.grad += this.data * (1 - this.data) * g; break;
    }
  }
}

function val(x: Value | number): Value {
  return x instanceof Value ? x : new Value(x);
}

/** Nodes in topological order (children before parents, root last). */
export function topo(root: Value): Value[] {
  const seen = new Set<Value>([root]);
  const out: Value[] = [];
  // iterative DFS: deep graphs (RNN unrolls) must not overflow the stack
  const stack: [Value, number][] = [[root, 0]];
  while (stack.length) {
    const top = stack[stack.length - 1];
    const n = top[0];
    if (top[1] < n.children.length) {
      const c = n.children[top[1]++];
      if (!seen.has(c)) { seen.add(c); stack.push([c, 0]); }
    } else {
      out.push(n);
      stack.pop();
    }
  }
  return out;
}

/** Reverse topological order: the order in which to call localBackward(). */
export const backwardOrder = (root: Value): Value[] => topo(root).reverse();

/** Zero every grad in the graph and seed root.grad = 1. Returns the order to walk. */
export function prepareBackward(root: Value): Value[] {
  const order = backwardOrder(root);
  for (const n of order) n.grad = 0;
  root.grad = 1;
  return order;
}

export function backward(root: Value): void {
  for (const n of prepareBackward(root)) n.localBackward();
}

export const zeroGrad = (vs: Value[]): void => { for (const v of vs) v.grad = 0; };
