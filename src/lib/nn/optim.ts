import { Value, backward } from './value';

/** Stateful in-place update: mutates x using gradient g. State is keyed by index. */
export interface Optimizer { step(x: number[], g: number[]): void }

export const sgd = (lr = 0.1): Optimizer => ({
  step(x, g) { for (let i = 0; i < x.length; i++) x[i] -= lr * g[i]; },
});

export const momentum = (lr = 0.1, beta = 0.9): Optimizer => {
  const v: number[] = [];
  return {
    step(x, g) {
      for (let i = 0; i < x.length; i++) {
        v[i] = beta * (v[i] ?? 0) + g[i];
        x[i] -= lr * v[i];
      }
    },
  };
};

export const rmsprop = (lr = 0.01, decay = 0.9, eps = 1e-8): Optimizer => {
  const s: number[] = [];
  return {
    step(x, g) {
      for (let i = 0; i < x.length; i++) {
        s[i] = decay * (s[i] ?? 0) + (1 - decay) * g[i] * g[i];
        x[i] -= (lr * g[i]) / (Math.sqrt(s[i]) + eps);
      }
    },
  };
};

export const adam = (lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8): Optimizer => {
  const m: number[] = [], v: number[] = [];
  let t = 0;
  return {
    step(x, g) {
      t++;
      for (let i = 0; i < x.length; i++) {
        m[i] = b1 * (m[i] ?? 0) + (1 - b1) * g[i];
        v[i] = b2 * (v[i] ?? 0) + (1 - b2) * g[i] * g[i];
        x[i] -= (lr * (m[i] / (1 - b1 ** t))) / (Math.sqrt(v[i] / (1 - b2 ** t)) + eps);
      }
    },
  };
};

/** Apply one optimizer step to Value params using their current .grad. */
export function stepParams(opt: Optimizer, params: Value[]): void {
  const x = params.map((p) => p.data);
  opt.step(x, params.map((p) => p.grad));
  params.forEach((p, i) => { p.data = x[i]; });
}

/** Engine-computed gradient of a 2D function at (x,y): returns [z, dz/dx, dz/dy]. */
export function grad2D(f: (x: Value, y: Value) => Value, x: number, y: number): [number, number, number] {
  const vx = new Value(x), vy = new Value(y);
  const z = f(vx, vy);
  backward(z);
  return [z.data, vx.grad, vy.grad];
}

/** One optimizer step on point p=[x,y] of f (gradient from the engine). Mutates and returns p. */
export function step2D(opt: Optimizer, p: [number, number], f: (x: Value, y: Value) => Value): [number, number] {
  const [, gx, gy] = grad2D(f, p[0], p[1]);
  opt.step(p, [gx, gy]);
  return p;
}
