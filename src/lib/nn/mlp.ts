import type { Rng } from '../math/rng';
import { Value } from './value';

export type Activation = 'tanh' | 'relu' | 'sigmoid' | 'linear';

const activate = (v: Value, act: Activation) =>
  act === 'tanh' ? v.tanh() : act === 'relu' ? v.relu() : act === 'sigmoid' ? v.sigmoid() : v;

export class Neuron {
  w: Value[];
  b: Value;
  constructor(nin: number, public act: Activation, rng: Rng) {
    // Xavier-ish uniform init
    const lim = Math.sqrt(6 / (nin + 1));
    this.w = Array.from({ length: nin }, () => new Value(rng.uniform(-lim, lim)));
    this.b = new Value(0);
  }
  call(x: (Value | number)[]): Value {
    let s = this.b;
    for (let i = 0; i < this.w.length; i++) s = s.add(this.w[i].mul(x[i]));
    return activate(s, this.act);
  }
  parameters(): Value[] { return [...this.w, this.b]; }
}

export class Layer {
  neurons: Neuron[];
  constructor(nin: number, nout: number, act: Activation, rng: Rng) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, act, rng));
  }
  call(x: (Value | number)[]): Value[] { return this.neurons.map((n) => n.call(x)); }
  parameters(): Value[] { return this.neurons.flatMap((n) => n.parameters()); }
}

export interface MLPOpts {
  rng: Rng;
  /** hidden-layer activation (default tanh) */
  act?: Activation;
  /** output-layer activation (default linear) */
  outAct?: Activation;
}

export class MLP {
  layers: Layer[];
  constructor(nin: number, nouts: number[], opts: MLPOpts) {
    const sizes = [nin, ...nouts];
    this.layers = nouts.map((n, i) =>
      new Layer(sizes[i], n, i === nouts.length - 1 ? (opts.outAct ?? 'linear') : (opts.act ?? 'tanh'), opts.rng));
  }
  call(x: (Value | number)[]): Value[] {
    let a: (Value | number)[] = x;
    for (const l of this.layers) a = l.call(a);
    return a as Value[];
  }
  parameters(): Value[] { return this.layers.flatMap((l) => l.parameters()); }
}
