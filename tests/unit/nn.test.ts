import { describe, expect, test } from 'vitest';
import {
  Value, MLP, mse, binaryCrossEntropy, backward, backwardOrder, prepareBackward, topo, resetIds,
  sgd, momentum, rmsprop, adam, stepParams, step2D, zeroGrad, createRng,
  type Optimizer,
} from '../../src/lib/nn';

type F = (x: Value[]) => Value;
function gradCheck(f: F, xs: number[]) {
  const vs = xs.map((x) => new Value(x));
  backward(f(vs));
  const h = 1e-6;
  xs.forEach((_, i) => {
    const at = (d: number) => f(xs.map((x, j) => new Value(j === i ? x + d : x))).data;
    expect(vs[i].grad).toBeCloseTo((at(h) - at(-h)) / (2 * h), 6);
  });
}

describe('gradients', () => {
  const ops: [string, F, number[]][] = [
    ['add', ([a, b]) => a.add(b), [1.3, -0.7]],
    ['sub', ([a, b]) => a.sub(b), [1.3, -0.7]],
    ['mul', ([a, b]) => a.mul(b), [1.3, -0.7]],
    ['div', ([a, b]) => a.div(b), [1.3, -0.7]],
    ['neg', ([a]) => a.neg(), [0.4]],
    ['pow', ([a]) => a.pow(3), [0.9]],
    ['exp', ([a]) => a.exp(), [0.5]],
    ['log', ([a]) => a.log(), [1.7]],
    ['tanh', ([a]) => a.tanh(), [0.3]],
    ['relu+', ([a]) => a.relu(), [0.8]],
    ['relu-', ([a]) => a.relu(), [-0.8]],
    ['sigmoid', ([a]) => a.sigmoid(), [0.6]],
  ];
  test.each(ops)('%s', (_n, f, xs) => gradCheck(f, xs));

  test('composed', () =>
    gradCheck(([a, b, c]) => a.mul(b).add(c).pow(2).add(a.tanh().mul(c.exp())).div(b.pow(2).add(1)), [2, -3, 1]));

  test('node reuse accumulates', () => {
    const a = new Value(3);
    backward(a.add(a)); expect(a.grad).toBe(2);
    const b = new Value(3);
    backward(b.mul(b)); expect(b.grad).toBe(6);
  });

  test('step-wise equals backward', () => {
    const build = () => {
      const a = new Value(2), b = new Value(-3), c = new Value(10);
      return { l: a.mul(b).add(c).pow(2).add(a.mul(a)).tanh(), leaves: [a, b, c] };
    };
    const x = build(); backward(x.l);
    const y = build();
    for (const n of prepareBackward(y.l)) n.localBackward();
    expect(y.leaves.map((v) => v.grad)).toEqual(x.leaves.map((v) => v.grad));
    expect(backwardOrder(y.l)[0]).toBe(y.l);
    expect(topo(y.l).at(-1)).toBe(y.l);
  });

  test('resetIds and deep graph', () => {
    resetIds();
    expect(new Value(0).id).toBe(0);
    let v = new Value(0.5);
    for (let i = 0; i < 20000; i++) v = v.mul(1);
    expect(() => backward(v)).not.toThrow();
  });
});

describe('rng', () => {
  test('deterministic and in range', () => {
    const a = createRng(7), b = createRng(7);
    for (let i = 0; i < 100; i++) {
      const x = a.next();
      expect(x).toBe(b.next());
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
    }
    expect(createRng(1).next()).not.toBe(createRng(2).next());
    const r = createRng(3);
    const xs = Array.from({ length: 5000 }, () => r.normal(2, 0.5));
    expect(xs.reduce((s, x) => s + x) / xs.length).toBeCloseTo(2, 1);
  });
});

describe('mlp', () => {
  const mk = () => new MLP(2, [4, 1], { rng: createRng(42) });
  test('forward deterministic', () => {
    expect(mk().call([0.5, -0.5])[0].data).toBe(mk().call([0.5, -0.5])[0].data);
  });
  test('XOR trains with SGD', () => {
    const net = mk();
    const xs = [[0, 0], [0, 1], [1, 0], [1, 1]], ys = [0, 1, 1, 0];
    const p = net.parameters(), opt = sgd(0.3);
    let first = 0, last = 0;
    for (let i = 0; i < 500; i++) {
      const loss = mse(xs.map((x) => net.call(x)[0]), ys);
      if (i === 0) first = loss.data;
      last = loss.data;
      backward(loss);
      stepParams(opt, p);
    }
    expect(last).toBeLessThan(first * 0.2);
  });
  test('BCE value and gradient', () => {
    const p = new Value(0.8);
    const l = binaryCrossEntropy([p], [1]);
    expect(l.data).toBeCloseTo(-Math.log(0.8), 6);
    backward(l);
    expect(p.grad).toBeCloseTo(-1 / 0.8, 5);
  });
});

describe('optimizers', () => {
  const bowl = (x: Value, y: Value) => x.pow(2).add(y.pow(2).mul(3));
  const rosen = (x: Value, y: Value) => new Value(1).sub(x).pow(2).add(y.sub(x.pow(2)).pow(2).mul(100));
  const run = (o: Optimizer, f: typeof bowl, p: [number, number], n: number) => {
    for (let i = 0; i < n; i++) step2D(o, p, f);
    return f(new Value(p[0]), new Value(p[1])).data;
  };
  const cases: [string, Optimizer][] = [
    ['sgd', sgd(0.1)], ['momentum', momentum(0.02, 0.9)], ['rmsprop', rmsprop(0.05)], ['adam', adam(0.1)],
  ];
  test.each(cases)('%s reduces bowl loss', (_n, o) => {
    expect(run(o, bowl, [2, 2], 300)).toBeLessThan(0.1);
  });
  test('adam decreases Rosenbrock markedly', () => {
    const p: [number, number] = [-1.2, 1];
    const start = rosen(new Value(p[0]), new Value(p[1])).data;
    const end = run(adam(0.02), rosen, p, 5000);
    expect(end).toBeLessThan(start * 1e-3);
    expect(Math.hypot(p[0] - 1, p[1] - 1)).toBeLessThan(0.1);
  });
  test('zeroGrad', () => {
    const a = new Value(1);
    a.grad = 5;
    zeroGrad([a]);
    expect(a.grad).toBe(0);
  });
});
