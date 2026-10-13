import { createRng } from '../../../lib/math/rng';

export const SHAPES = ['circle', 'square', 'triangle', 'cross'] as const;
export type Shape = typeof SHAPES[number];
export type Mode = 'patches' | 'embed' | 'contrastive' | 'zeroshot' | 'vlm' | 'limits';
export const DURATION = 10;
export const WORDS = [...SHAPES, 'small', 'large', 'left', 'of', 'a'] as const;
const D = 8;
export function shapeImage(shape: Shape = 'circle', large = true, dx = 0, dy = 0): Float64Array {
  const r = large ? 5 : 3;
  return Float64Array.from({ length: 256 }, (_, i) => {
    const x = i % 16 - 7.5 - dx, y = Math.floor(i / 16) - 7.5 - dy;
    const hit = shape === 'circle' ? x * x + y * y <= r * r : shape === 'square' ? Math.max(Math.abs(x), Math.abs(y)) < r
      : shape === 'triangle' ? y >= -r && y < r && Math.abs(x) <= (y + r) / 2 : Math.max(Math.abs(x), Math.abs(y)) < r && Math.min(Math.abs(x), Math.abs(y)) < 1.5;
    return +hit;
  });
}
export function patches(image: Float64Array, p = 4): Float64Array[] {
  if (![2, 4, 8].includes(p)) throw new Error('Patch size must divide 16: choose 2, 4 or 8.');
  return Array.from({ length: (16 / p) ** 2 }, (_, k) => Float64Array.from({ length: p * p }, (_, j) => image[(Math.floor(k / (16 / p)) * p + Math.floor(j / p)) * 16 + (k % (16 / p)) * p + j % p]));
}
/** Shared, inspectable 16 -> 8 patch map. Output zero is the mean intensity. */
export function patchWeights(p = 4): Float64Array {
  return Float64Array.from({ length: 8 * p * p }, (_, k) => Math.floor(k / (p * p)) === 0 ? 1 / (p * p) : (((k % (p * p) + Math.floor(k / (p * p))) % 5) - 2) / (p * p));
}
export function project(x: Float64Array, w: Float64Array, width = 8): Float64Array {
  return Float64Array.from({ length: width }, (_, o) => x.reduce((s, v, j) => s + v * w[o * x.length + j], 0));
}
export function embedPatches(image: Float64Array, p = 4): Float64Array[] { return patches(image, p).map(x => project(x, patchWeights(p))); }
/** Additive position followed by a nonlinearity: unlike mean(x + p), this can bind content to position. */
export function poolPosition(tokens: Float64Array[], withPosition: boolean): Float64Array {
  return Float64Array.from({ length: 8 }, (_, d) => tokens.reduce((s, x, i) => s + (withPosition ? Math.tanh(x[d] + (i - 7.5) * (d + 1) / 40) : x[d]), 0) / tokens.length);
}
export function bag(text: string): Float64Array {
  const out = new Float64Array(WORDS.length), words = text.toLowerCase().split(/\s+/).filter(w => WORDS.includes(w as typeof WORDS[number]));
  for (const word of words) out[WORDS.indexOf(word as typeof WORDS[number])] += 1 / Math.max(1, words.length);
  return out;
}
export function dot(a: Float64Array, b: Float64Array): number { return a.reduce((s, v, i) => s + v * b[i], 0); }
function normalize(x: Float64Array): { v: Float64Array; norm: number } {
  const norm = Math.max(1e-9, Math.sqrt(dot(x, x)));
  return { v: x.map(v => v / norm), norm };
}
export function probabilities(scores: number[]): number[] {
  const max = Math.max(...scores), e = scores.map(x => Math.exp(x - max)), sum = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / sum);
}
export interface Model { image: Float64Array; text: Float64Array; logScale: number }
export interface Snapshot { step: number; loss: number; probability: number; scores: number[]; rows: number[]; scale: number }
export interface Run { model: Model; snapshots: Snapshot[]; accuracy: number[] }
/** Raster patches concatenated without losing their position; linear image encoder. */
export function imageInput(image: Float64Array): Float64Array { return Float64Array.from(patches(image).flatMap(x => Array.from(x))); }
export function initialize(): Model {
  const rng = createRng(1421);
  return { image: Float64Array.from({ length: 256 * D }, () => rng.uniform(-0.08, 0.08)), text: Float64Array.from({ length: WORDS.length * D }, () => rng.uniform(-0.2, 0.2)), logScale: Math.log(3) };
}
export const training = SHAPES.flatMap(shape => [false, true].map(large => ({ shape, large, caption: `${large ? 'large' : 'small'} ${shape}`, image: shapeImage(shape, large) })));
export function objective(model: Model, gradient = false) {
  const xs = training.map(x => imageInput(x.image)), ys = training.map(x => bag(x.caption));
  const im = xs.map(x => normalize(project(x, model.image))), tx = ys.map(x => normalize(project(x, model.text)));
  const n = xs.length, scale = Math.exp(model.logScale), scores = im.flatMap(a => tx.map(b => dot(a.v, b.v))), rows: number[] = [], ds = new Float64Array(n * n);
  let loss = 0, probability = 0;
  for (let i = 0; i < n; i++) {
    const row = probabilities(scores.slice(i * n, (i + 1) * n).map(v => v * scale));
    const col = probabilities(Array.from({ length: n }, (_, j) => scores[j * n + i] * scale));
    rows.push(...row); probability += row[i] / n;
    loss -= (Math.log(row[i]) + Math.log(col[i])) / (2 * n);
    for (let j = 0; j < n; j++) { ds[i * n + j] += (row[j] - +(i === j)) / (2 * n); ds[j * n + i] += (col[j] - +(i === j)) / (2 * n); }
  }
  const g: Model = { image: new Float64Array(model.image.length), text: new Float64Array(model.text.length), logScale: 0 };
  if (gradient) {
    const gi = im.map(() => new Float64Array(D)), gt = tx.map(() => new Float64Array(D));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const a = ds[i * n + j] * scale; g.logScale += a * scores[i * n + j];
      for (let d = 0; d < D; d++) { gi[i][d] += a * tx[j].v[d]; gt[j][d] += a * im[i].v[d]; }
    }
    for (let i = 0; i < n; i++) for (const [input, result, grad, out] of [[xs[i], im[i], gi[i], g.image], [ys[i], tx[i], gt[i], g.text]] as const) {
      const parallel = dot(grad, result.v);
      for (let d = 0; d < D; d++) for (let k = 0; k < input.length; k++) out[d * input.length + k] += (grad[d] - parallel * result.v[d]) / result.norm * input[k];
    }
  }
  return { loss, probability, scores, rows, scale, g };
}
export function similarities(model: Model, image: Float64Array, captions: string[]): number[] {
  const v = normalize(project(imageInput(image), model.image)).v;
  return captions.map(c => dot(v, normalize(project(bag(c), model.text)).v));
}
export function heldout(model: Model, prefix: string): number {
  let correct = 0, count = 0;
  for (const shape of SHAPES) for (const large of [false, true]) for (const dx of [-1, 1]) {
    const scores = similarities(model, shapeImage(shape, large, dx), SHAPES.map(s => `${prefix}${s}`));
    correct += +(SHAPES[scores.indexOf(Math.max(...scores))] === shape); count++;
  }
  return correct / count;
}
export function train(steps = 600): Run {
  const model = initialize(), snapshots: Snapshot[] = [];
  for (let step = 0; step <= steps; step++) {
    const r = objective(model, step < steps);
    if (step % 10 === 0 || step === steps) snapshots.push({ step, loss: r.loss, probability: r.probability, scores: r.scores, rows: r.rows, scale: r.scale });
    if (step < steps) {
      model.image = model.image.map((v, i) => v - 0.04 * r.g.image[i]);
      model.text = model.text.map((v, i) => v - 0.04 * r.g.text[i]);
      model.logScale = Math.min(Math.log(100), model.logScale - 0.04 * r.g.logScale);
    }
  }
  return { model, snapshots, accuracy: ['', 'a ', 'large '].map(p => heldout(model, p)) };
}
export function frameAt(t: number, mode: Mode, run: Run | null = null) {
  const progress = Math.min(1, Math.max(0, t / DURATION));
  return { progress, visible: Math.min(16, Math.floor(progress * 16) + 1), phase: mode === 'contrastive' ? 'Training updates' : mode === 'limits' ? 'Compare captions' : mode === 'vlm' ? 'Assemble context' : mode === 'embed' ? 'Project and position' : mode === 'zeroshot' ? 'Compare candidates' : 'Extract patches', snapshot: run?.snapshots[Math.min(run.snapshots.length - 1, Math.floor(progress * (run.snapshots.length - 1)))] ?? null };
}
/** Actual fixed projector for plumbing only; no learned language alignment or generation. */
export function assemble(image: Float64Array, p = 4) {
  const vision = embedPatches(image, p), w = Float64Array.from({ length: 12 * 8 }, (_, i) => ((i % 7) - 3) / 8);
  const projected = vision.map(v => project(v, w, 12));
  const text = ['Q', ':', 'what', 'shape', '?', 'A', ':'];
  return { projected, text, count: projected.length + text.length, share: projected.length / (projected.length + text.length) };
}
