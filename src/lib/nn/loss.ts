import { Value } from './value';

/** Mean squared error over paired predictions/targets. */
export function mse(preds: Value[], targets: (Value | number)[]): Value {
  let s = new Value(0);
  preds.forEach((p, i) => { s = s.add(p.sub(targets[i]).pow(2)); });
  return s.div(preds.length);
}

/** Mean BCE; preds are probabilities in (0,1), targets 0/1. */
export function binaryCrossEntropy(preds: Value[], targets: number[]): Value {
  const eps = 1e-12;
  let s = new Value(0);
  preds.forEach((p, i) => {
    const y = targets[i];
    s = s.add(p.add(eps).log().mul(y).add(p.neg().add(1 + eps).log().mul(1 - y)).neg());
  });
  return s.div(preds.length);
}
