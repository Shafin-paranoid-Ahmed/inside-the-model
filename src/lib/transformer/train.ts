import { forward, rotate, type ForwardOptions, type NormState, type Weights } from './model';

export interface Example {
  ids: readonly number[];
  targets: readonly number[]; // -1 skips a position
  targetProbs?: readonly Float64Array[]; // optional soft targets, same rows as ids
}
function add(a: Float64Array, b: Float64Array): Float64Array {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
  return a;
}

function projectBack(w: Weights, grad: Float64Array, name: string, x: Float64Array, dy: Float64Array): Float64Array {
  const tensor = w.offsets[`${name}.weight`], bias = w.offsets[`${name}.bias`];
  const dx = new Float64Array(x.length), T = x.length / tensor.cols;
  for (let t = 0; t < T; t++) for (let r = 0; r < tensor.rows; r++) {
    const dr = dy[t * tensor.rows + r];
    grad[bias.offset + r] += dr;
    for (let j = 0; j < tensor.cols; j++) {
      grad[tensor.offset + r * tensor.cols + j] += dr * x[t * tensor.cols + j];
      dx[t * tensor.cols + j] += dr * w.params[tensor.offset + r * tensor.cols + j];
    }
  }
  return dx;
}

function normBack(w: Weights, grad: Float64Array, name: string, state: NormState, dy: Float64Array): Float64Array {
  const d = w.config.d, go = w.offsets[`${name}.gain`].offset, bo = w.offsets[`${name}.bias`].offset;
  const dx = new Float64Array(dy.length);
  for (let t = 0; t < dy.length / d; t++) {
    let mean = 0, projected = 0;
    for (let j = 0; j < d; j++) {
      const i = t * d + j, dz = dy[i] * w.params[go + j];
      mean += dz; projected += dz * state.normalized[i];
      grad[go + j] += dy[i] * state.normalized[i]; grad[bo + j] += dy[i];
    }
    mean /= d; projected /= d;
    for (let j = 0; j < d; j++) {
      const i = t * d + j;
      dx[i] = state.invStd[t] * (dy[i] * w.params[go + j] - mean - state.normalized[i] * projected);
    }
  }
  return dx;
}

/** Accumulate summed cross-entropy gradients into grad; no scalar autograd graph. */
export function backward(w: Weights, example: Example, grad: Float64Array, opts: ForwardOptions = {}): {
  loss: number; tokens: number; residualGrads: Float64Array[];
} {
  if (opts.patch?.length) throw new Error('Patching is an inference intervention, not a training operation');
  if (example.targets.length !== example.ids.length || grad.length !== w.params.length) throw new Error('Training dimensions differ');
  const f = forward(w, example.ids, opts), c = w.config, d = c.d, V = c.vocab.length, T = example.ids.length, dk = d / c.heads;
  const du = new Float64Array(T * d);
  const uo = w.offsets[c.tied ? 'embedding' : 'unembedding'].offset, bo = w.offsets['logit.bias'].offset;
  let loss = 0, tokens = 0;
  for (let t = 0; t < T; t++) {
    if (example.targets[t] === -1) continue;
    const target = example.targets[t], softTarget = example.targetProbs?.[t];
    if (!Number.isInteger(target) || target < 0 || target >= V) throw new Error('Invalid target id');
    if (softTarget && (softTarget.length !== V || Math.abs(softTarget.reduce((s, p) => s + p, 0) - 1) > 1e-9 || softTarget.some((p) => p < 0 || !Number.isFinite(p)))) throw new Error('Invalid target distribution');
    tokens++;
    // Stable log-probabilities even for underflowed probabilities.
    let maxId = 0;
    for (let id = 1; id < V; id++) if (f.logits[t * V + id] > f.logits[t * V + maxId]) maxId = id;
    const logZ = f.logits[t * V + maxId] + Math.log(1 / f.probs[t * V + maxId]);
    for (let id = 0; id < V; id++) {
      const desired = softTarget ? softTarget[id] : id === target ? 1 : 0;
      if (desired) loss += desired * (logZ - f.logits[t * V + id]);
      const dy = f.probs[t * V + id] - desired;
      grad[bo + id] += dy;
      for (let j = 0; j < d; j++) {
        grad[uo + id * d + j] += dy * f.final.output[t * d + j];
        du[t * d + j] += dy * w.params[uo + id * d + j];
      }
    }
  }
  let dx = normBack(w, grad, 'final', f.final, du);
  const residualGrads: Float64Array[] = Array(c.layers + 1);
  residualGrads[c.layers] = dx.slice();
  for (let l = c.layers - 1; l >= 0; l--) {
    const b = f.layers[l];
    const ds2 = f.norm === 'post' ? normBack(w, grad, `${l}.ln2`, b.norm2, dx) : dx;
    let ds1: Float64Array = ds2.slice();
    if (c.ff) {
      const dh = projectBack(w, grad, `${l}.ff2`, b.hidden, ds2);
      for (let i = 0; i < dh.length; i++) if (b.hidden[i] <= 0) dh[i] = 0;
      const dm = projectBack(w, grad, `${l}.ff1`, b.mlpInput, dh);
      add(ds1, f.norm === 'pre' ? normBack(w, grad, `${l}.ln2`, b.norm2, dm) : dm);
    }
    if (f.norm === 'post') ds1 = normBack(w, grad, `${l}.ln1`, b.norm1, ds1);
    const dconcat = projectBack(w, grad, `${l}.o`, b.concat, ds1);
    const dqAll = new Float64Array(T * d), dkAll = new Float64Array(T * d), dvAll = new Float64Array(T * d);
    for (let h = 0; h < c.heads; h++) {
      if (opts.ablate?.some(([ll, hh]) => ll === l && hh === h)) continue;
      const dq = new Float64Array(T * dk), dkey = new Float64Array(T * dk), dv = new Float64Array(T * dk);
      for (let t = 0; t < T; t++) {
        const da = new Float64Array(T);
        let weighted = 0;
        for (let s = 0; s < T; s++) {
          let dot = 0;
          for (let j = 0; j < dk; j++) {
            const dy = dconcat[t * d + h * dk + j];
            dot += dy * b.v[h][s * dk + j];
            dv[s * dk + j] += b.weights[h][t * T + s] * dy;
          }
          da[s] = dot; weighted += b.weights[h][t * T + s] * dot;
        }
        for (let s = 0; s < T; s++) {
          const ds = b.weights[h][t * T + s] * (da[s] - weighted) / (opts.noScale ? 1 : Math.sqrt(dk));
          for (let j = 0; j < dk; j++) {
            dq[t * dk + j] += ds * b.k[h][s * dk + j];
            dkey[s * dk + j] += ds * b.q[h][t * dk + j];
          }
        }
      }
      if (c.positions === 'rope' && !opts.noPositions) { rotate(dq, dk, 0, opts, true); rotate(dkey, dk, 0, opts, true); }
      for (let t = 0; t < T; t++) for (let j = 0; j < dk; j++) {
        dqAll[t * d + h * dk + j] = dq[t * dk + j];
        dkAll[t * d + h * dk + j] = dkey[t * dk + j]; dvAll[t * d + h * dk + j] = dv[t * dk + j];
      }
    }
    const dn = projectBack(w, grad, `${l}.q`, b.attentionInput, dqAll);
    add(dn, projectBack(w, grad, `${l}.k`, b.attentionInput, dkAll));
    add(dn, projectBack(w, grad, `${l}.v`, b.attentionInput, dvAll));
    dx = add(ds1, f.norm === 'pre' ? normBack(w, grad, `${l}.ln1`, b.norm1, dn) : dn);
    residualGrads[l] = dx.slice();
  }
  for (let t = 0; t < T; t++) for (let j = 0; j < d; j++) {
    grad[w.offsets.embedding.offset + example.ids[t] * d + j] += dx[t * d + j];
    if (c.positions === 'learned' && !opts.noPositions) grad[w.offsets.position.offset + t * d + j] += (opts.positionScale ?? 1) * dx[t * d + j];
  }
  return { loss, tokens, residualGrads };
}

export function lossAndGrad(w: Weights, batch: readonly Example[], opts: ForwardOptions = {}): {
  loss: number; grad: Float64Array; residualGrads: Float64Array[];
} {
  if (!batch.length) throw new Error('Training needs a nonempty batch');
  const grad = new Float64Array(w.params.length);
  let loss = 0, tokens = 0, residualGrads: Float64Array[] = [];
  for (const example of batch) {
    const result = backward(w, example, grad, opts);
    loss += result.loss; tokens += result.tokens; residualGrads = result.residualGrads;
  }
  if (!tokens) throw new Error('No supervised positions in batch');
  for (let i = 0; i < grad.length; i++) grad[i] /= tokens;
  // These are the last example's residual gradients, scaled by the same batch-token mean.
  for (const g of residualGrads) for (let i = 0; i < g.length; i++) g[i] /= tokens;
  return { loss: loss / tokens, grad, residualGrads };
}

export interface AdamState { m: Float64Array; v: Float64Array; b1Product: number; b2Product: number; step: number }
export const makeAdam = (size: number): AdamState => ({ m: new Float64Array(size), v: new Float64Array(size), b1Product: 1, b2Product: 1, step: 0 });
export function adamStep(params: Float64Array, grad: Float64Array, state: AdamState, lr: number, clipNorm = 1): void {
  if (params.length !== grad.length || state.m.length !== params.length) throw new Error('Adam dimensions differ');
  let norm = 0; for (const value of grad) norm += value * value;
  const scale = Math.min(1, clipNorm / (Math.sqrt(norm) + 1e-30));
  state.step++; state.b1Product *= 0.9; state.b2Product *= 0.999;
  for (let i = 0; i < params.length; i++) {
    const g = grad[i] * scale;
    state.m[i] = 0.9 * state.m[i] + 0.1 * g; state.v[i] = 0.999 * state.v[i] + 0.001 * g * g;
    params[i] -= lr * (state.m[i] / (1 - state.b1Product)) / (Math.sqrt(state.v[i] / (1 - state.b2Product)) + 1e-8);
  }
}
