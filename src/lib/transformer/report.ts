import { allowedNext, encodeWords, SAMPLE_200, VOCAB, WORD_CLASS } from './grammar';
import { forward, type ForwardOptions, type Weights } from './model';
import { cosine } from './math';
import { lossAndGrad } from './train';

export interface HeadSummary { layer: number; head: number; previous: number; subjectDeterminer: number; verb: number; start: number }
export interface HeadDifference { pair: [number, number][]; meanTotalVariation: number }
/** Token-weighted unmasked probability mass on the oracle's legal next tokens.
 * All evaluations use the frozen SAMPLE_200, after decoding the shipped weight bytes.
 */
export function evaluate(w: Weights, opts: ForwardOptions = {}): {
  grammaticalMass: number; nextTokenLoss: number; tokens: number; heads: HeadSummary[]; distinctHeads: HeadDifference;
} {
  const c = w.config, V = c.vocab.length;
  const sums = Array.from({ length: c.layers * c.heads }, () => [0, 0, 0, 0]);
  const counts = Array.from({ length: sums.length }, () => [0, 0, 0, 0]);
  const pairDistances = new Float64Array(sums.length * sums.length);
  let mass = 0, loss = 0, tokens = 0, rows = 0;
  for (const sentence of SAMPLE_200) {
    const ids = sentence.slice(0, -1), f = forward(w, ids, opts), T = ids.length;
    const verb = ids.findIndex((id) => WORD_CLASS[id].kind === 'verb');
    for (let t = 0; t < T; t++) {
      for (const id of allowedNext(ids.slice(0, t + 1))) mass += f.probs[t * V + id];
      loss -= Math.log(f.probs[t * V + sentence[t + 1]]); tokens++;
      if (!t) continue;
      rows++;
      const weights = f.layers.flatMap((l) => l.weights);
      for (let h = 0; h < weights.length; h++) {
        const a = weights[h];
        sums[h][0] += a[t * T + t - 1]; counts[h][0]++;
        sums[h][1] += a[t * T + 1]; counts[h][1]++;
        if (verb >= 0 && t >= verb) { sums[h][2] += a[t * T + verb]; counts[h][2]++; }
        sums[h][3] += a[t * T]; counts[h][3]++;
      }
      for (let a = 0; a < weights.length; a++) for (let b = a + 1; b < weights.length; b++) {
        let distance = 0;
        for (let s = 0; s < T; s++) distance += Math.abs(weights[a][t * T + s] - weights[b][t * T + s]) / 2;
        pairDistances[a * weights.length + b] += distance;
      }
    }
  }
  const heads = sums.map((row, i) => ({ layer: Math.floor(i / c.heads), head: i % c.heads,
    previous: row[0] / counts[i][0], subjectDeterminer: row[1] / counts[i][1], verb: row[2] / counts[i][2], start: row[3] / counts[i][3] }));
  let best = 0, pair: [number, number][] = [];
  for (let a = 0; a < sums.length; a++) for (let b = a + 1; b < sums.length; b++) {
    if (Math.floor(a / c.heads) !== Math.floor(b / c.heads)) continue;
    const distance = pairDistances[a * sums.length + b] / rows;
    if (distance > best) { best = distance; pair = [[Math.floor(a / c.heads), a % c.heads], [Math.floor(b / c.heads), b % c.heads]]; }
  }
  return { grammaticalMass: mass / tokens, nextTokenLoss: loss / tokens, tokens, heads,
    distinctHeads: { pair, meanTotalVariation: best } };
}

export function probes(w: Weights): Record<string, unknown> {
  const d = w.config.d, eo = w.offsets.embedding.offset;
  const embedding = (word: string) => {
    const id = VOCAB.indexOf(word as typeof VOCAB[number]);
    return w.params.slice(eo + id * d, eo + (id + 1) * d);
  };
  const cat = embedding('cat'), cats = embedding('cats'), dog = embedding('dog'), dogs = embedding('dogs');
  const analogy = Float64Array.from(cat, (_, j) => cats[j] - cat[j] + dog[j]);
  const neighbours = VOCAB.filter((word) => !['cat', 'cats', 'dog'].includes(word)).map((word) => ({ word, cosine: cosine(analogy, embedding(word)) })).sort((a, b) => b.cosine - a.cosine);
  const sheepA = encodeWords('[start] this small sheep sees a red fox'), sheepB = encodeWords('[start] these small sheep see a red fox');
  const a = forward(w, sheepA), b = forward(w, sheepB), sheepIndex = 3;
  const at = (x: Float64Array, t: number) => x.slice(t * d, (t + 1) * d);
  const contextual = [cosine(at(a.x0, sheepIndex), at(b.x0, sheepIndex)), ...a.layers.map((l, i) => cosine(at(l.resid2, sheepIndex), at(b.layers[i].resid2, sheepIndex)))];
  const positions: { a: number; b: number; cosine: number }[] = [];
  const po = w.offsets.position?.offset;
  if (po !== undefined) for (let j = 1; j < w.config.context; j++) positions.push({ a: 0, b: j, cosine: cosine(w.params.subarray(po, po + d), w.params.subarray(po + j * d, po + (j + 1) * d)) });
  const example = encodeWords('[start] these small sheep chase a red fox');
  const f = forward(w, example), query = 3, T = example.length;
  const sheepAttention = f.layers.flatMap((l, layer) => l.weights.map((weights, head) => ({ layer, head,
    targets: example.map((id, t) => ({ word: VOCAB[id], position: t, weight: weights[query * T + t] })).sort((a, b) => b.weight - a.weight) })));
  return { analogy: { nearest: neighbours.slice(0, 5), dogsRank: neighbours.findIndex((n) => n.word === 'dogs') + 1,
    offsetCosine: cosine(Float64Array.from(cat, (_, j) => cats[j] - cat[j]), Float64Array.from(dog, (_, j) => dogs[j] - dog[j])) },
    contextualSheepCosines: contextual, learnedPositionCosines: positions, sheepAttention };
}

export function residualGradientProbe(w: Weights): Record<string, number[]> {
  const ids = encodeWords('[start] these small sheep chase a red fox');
  const ex = { ids: ids.slice(0, -1), targets: ids.slice(1) };
  const out: Record<string, number[]> = {};
  for (const norm of ['pre', 'post'] as const) out[norm] = lossAndGrad(w, [ex], { norm }).residualGrads.map((g) => Math.sqrt(g.reduce((s, x) => s + x * x, 0)));
  return out;
}
