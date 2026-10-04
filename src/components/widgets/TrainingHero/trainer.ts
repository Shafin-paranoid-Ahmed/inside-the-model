// Training loop core. Uses the shared autograd engine; runs inside the worker (DOM-free, unit-testable).
import { MLP, mse, backward, adam, stepParams, createRng, type Optimizer } from '../../../lib/nn';
import { SIZES, PARAMS, type Dataset } from './sim';

export interface Snapshot { weights: Float32Array; grads: Float32Array; loss: number; step: number }

export class Trainer {
  net!: MLP;
  opt!: Optimizer;
  step = 0;
  loss = 1;
  data!: Dataset;

  constructor(seed: number, data: Dataset) { this.reset(seed, data); }

  reset(seed: number, data: Dataset): void {
    this.net = new MLP(SIZES[0], SIZES.slice(1) as number[], { rng: createRng(seed), act: 'tanh', outAct: 'tanh' });
    this.opt = adam(0.008);
    this.step = 0;
    this.data = data;
  }

  train(): void {
    const { xs, ys } = this.data, n = ys.length;
    if (!n) return;
    const preds = [];
    for (let i = 0; i < n; i++) preds.push(this.net.call([xs[2 * i], xs[2 * i + 1]])[0]);
    const loss = mse(preds, Array.from(ys));
    backward(loss);
    this.loss = loss.data;
    stepParams(this.opt, this.net.parameters());
    this.step++;
  }

  /** Weights and last gradients as fresh Float32Arrays (safe to transfer). */
  snapshot(): Snapshot {
    const weights = new Float32Array(PARAMS), grads = new Float32Array(PARAMS);
    this.net.parameters().forEach((v, i) => { weights[i] = v.data; grads[i] = v.grad; });
    return { weights, grads, loss: this.loss, step: this.step };
  }
}
