/**
 * ClipLab simulation, pure and DOM-free: a shapes dataset, ViT-style patching, a patch projection with position vectors,
 * a tiny CLIP (image encoder + bag-of-words text encoder, symmetric InfoNCE loss, hand-written backprop), zero-shot
 * classification, VLM token plumbing and the order-blindness limit.
 *
 * Everything that feeds training uses exp / tanh / log / normal from lib/gen/mlp (built from + - * / only), so Chrome and Node
 * train bit-identically and the numbers in the lesson prose equal the tests.
 */
import { createRng, type Rng } from '../../../lib/math/rng';
import { exp, tanh, log, normal, adam, createAdam } from '../../../lib/gen/mlp';
import { conv2d, convLayer, map3 } from '../../../lib/cnn/conv';

// ------------------------------------------------------------------ the shapes dataset
export const SZ = 16;
export const SHAPES = ['circle', 'square', 'triangle', 'cross', 'bars'] as const;
export const SIZES = ['large', 'small'] as const;
export const POSITIONS = ['left', 'right', 'top', 'bottom'] as const;
export type Shape = (typeof SHAPES)[number];
export type Size = (typeof SIZES)[number];
export type Pos = (typeof POSITIONS)[number];

interface Obj { shape: Shape; cx: number; cy: number; r: number }
/** Both sizes share the four centres, which sit on the 4-pixel patch grid. */
const CENTER: Record<Pos, [number, number]> = { left: [4, 8], right: [12, 8], top: [8, 4], bottom: [8, 12] };
const RADIUS: Record<Size, number> = { large: 4, small: 2 };

function inside(o: Obj, px: number, py: number): boolean {
  const dx = px - o.cx, dy = py - o.cy, r = o.r;
  switch (o.shape) {
    case 'circle': return dx * dx + dy * dy <= r * r;
    case 'square': return Math.abs(dx) <= r * 0.85 && Math.abs(dy) <= r * 0.85;
    case 'triangle': return dy >= -r && dy <= r && Math.abs(dx) <= ((dy + r) / (2 * r)) * r;
    case 'cross': return (Math.abs(dx) <= r / 3 && Math.abs(dy) <= r) || (Math.abs(dy) <= r / 3 && Math.abs(dx) <= r);
    default: return Math.abs(dx) <= r && Math.abs(dy) <= r && Math.floor(px) % 2 === 0; // bars: vertical stripes
  }
}

/** Draw objects on a 16 x 16 grayscale image (1 = ink), shifted by (sx, sy) pixels, with optional Gaussian pixel noise. */
function render(objs: Obj[], sx = 0, sy = 0, rng?: Rng, noise = 0.08): Float64Array {
  const img = new Float64Array(SZ * SZ);
  for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) {
    let v = 0;
    for (const o of objs) if (inside(o, x + 0.5 - sx, y + 0.5 - sy)) v = 1;
    if (rng) v = Math.min(1, Math.max(0, v + noise * normal(rng)));
    img[y * SZ + x] = v;
  }
  return img;
}

export type Item =
  | { kind: 'single'; shape: Shape; size: Size; pos: Pos }
  | { kind: 'pair'; order: 0 | 1 }; // 0: circle left of square, 1: square left of circle

export const itemObjs = (it: Item): Obj[] => {
  if (it.kind === 'single') { const [cx, cy] = CENTER[it.pos]; return [{ shape: it.shape, cx, cy, r: RADIUS[it.size] }]; }
  const a: Shape = it.order === 0 ? 'circle' : 'square', b: Shape = it.order === 0 ? 'square' : 'circle';
  return [{ shape: a, cx: 4, cy: 8, r: 3 }, { shape: b, cx: 12, cy: 8, r: 3 }];
};
/** variant 0 is the clean image; v >= 1 shifts it by one pixel and adds noise (seeded by v). */
export function itemImage(it: Item, v = 0): Float64Array {
  if (v === 0) return render(itemObjs(it));
  const sh: [number, number][] = [[1, -1], [-1, 1], [1, 1], [-1, -1]];
  const [sx, sy] = sh[(v - 1) % 4];
  return render(itemObjs(it), sx, sy, createRng(1000 + v));
}
export const itemName = (it: Item): string => (it.kind === 'single' ? `${it.shape}-${it.size}-${it.pos}` : it.order === 0 ? 'circle-left-of-square' : 'square-left-of-circle');
export function parseItem(name: string): Item {
  if (name === 'circle-left-of-square') return { kind: 'pair', order: 0 };
  if (name === 'square-left-of-circle') return { kind: 'pair', order: 1 };
  const [shape, size, pos] = name.split('-');
  return { kind: 'single', shape: shape as Shape, size: size as Size, pos: pos as Pos };
}
/** "triangle-large-top" or "triangle-large-top-2" (a noisy, shifted variant) -> an image. */
export function namedImage(name: string): { item: Item; img: Float64Array } {
  const m = /^(.*)-(\d)$/.exec(name), item = parseItem(m ? m[1] : name);
  return { item, img: itemImage(item, m ? Number(m[2]) : 0) };
}

// ------------------------------------------------------------------ captions
export const WORDS = ['large', 'small', 'circle', 'square', 'triangle', 'cross', 'bars', 'left', 'right', 'top', 'bottom', 'of', 'a'] as const;
export const V = WORDS.length;
const WID = Object.fromEntries(WORDS.map((w, i) => [w, i])) as Record<string, number>;
export const wordIds = (text: string): number[] => text.split(' ').map((w) => WID[w]);
export const captionText = (it: Item, article = false, keepSize = true, keepPos = true): string =>
  (article ? 'a ' : '') + (it.kind === 'single' ? [keepSize ? it.size : '', it.shape, keepPos ? it.pos : ''].filter(Boolean).join(' ') : it.order === 0 ? 'circle left of square' : 'square left of circle');

/** The one (size, position) combination of each shape that never appears in training. */
export const HELD_OUT: Record<Shape, [Size, Pos][]> = {
  circle: [['large', 'right']],
  square: [['small', 'left']],
  triangle: [['large', 'top']],
  cross: [['small', 'right']],
  bars: [['large', 'bottom']],
};
const isHeld = (s: Shape, z: Size, p: Pos) => HELD_OUT[s].some(([a, b]) => a === z && b === p);
export const TRAIN_ITEMS: Item[] = [];
export const HELD_ITEMS: Item[] = [];
for (const shape of SHAPES) for (const size of SIZES) for (const pos of POSITIONS)
  (isHeld(shape, size, pos) ? HELD_ITEMS : TRAIN_ITEMS).push({ kind: 'single', shape, size, pos });
/** The pair items are part of training: that is how the model meets "circle left of square" and its mirror caption. */
export const PAIR_ITEMS: Item[] = [{ kind: 'pair', order: 0 }, { kind: 'pair', order: 1 }];
/** 20 held-out images: each held-out combination as the clean image and as three shifted, noisy variants. */
export const HELD_IMAGES = HELD_ITEMS.flatMap((it) => [0, 1, 2, 3].map((v) => ({ item: it, name: itemName(it) + (v ? `-${v}` : ''), img: itemImage(it, v) })));

// ------------------------------------------------------------------ ViT patching
/** Cut an image into P x P patches in raster order; token i is the P*P pixels of patch i, row by row. */
export function patchify(img: Float64Array, P: number): Float64Array[] {
  const g = SZ / P, out: Float64Array[] = [];
  for (let gy = 0; gy < g; gy++) for (let gx = 0; gx < g; gx++) {
    const p = new Float64Array(P * P);
    for (let y = 0; y < P; y++) for (let x = 0; x < P; x++) p[y * P + x] = img[(gy * P + y) * SZ + gx * P + x];
    out.push(p);
  }
  return out;
}
export const tokenCount = (side: number, patch: number): number => (side / patch) * (side / patch);

// ------------------------------------------------------------------ the tiny CLIP
export const P = 4, D = 8, NT = 16, KP = 16; // patch size, width, tokens, pixels per patch
const O_E = 0, O_POS = O_E + D * KP, O_WI = O_POS + NT * D, O_BI = O_WI + D * D, O_EMB = O_BI + D, O_WT = O_EMB + V * D, O_BT = O_WT + D * D, O_S = O_BT + D;
export const NPARAMS = O_S + 1;
export const MAX_SCALE = log(100);
export const slice = { E: O_E, POS: O_POS };

export function initParams(seed: number): Float64Array {
  const r = createRng(seed), p = new Float64Array(NPARAMS);
  const fill = (o: number, n: number, a: number) => { for (let i = 0; i < n; i++) p[o + i] = r.uniform(-a, a); };
  fill(O_E, D * KP, Math.sqrt(3 / KP)); fill(O_POS, NT * D, 0.5); fill(O_WI, D * D, Math.sqrt(3 / D));
  fill(O_EMB, V * D, 1); fill(O_WT, D * D, Math.sqrt(3 / D));
  p[O_S] = log(1 / 0.07);
  return p;
}
export const PROJECTION_SEED = 5;

/** Token vectors before the nonlinearity: z_i = E x_i + p_i (with usePos=false, z_i = E x_i). */
export function projectPatch(p: Float64Array, x: ArrayLike<number>, i: number, usePos = true, out = new Float64Array(D)): Float64Array {
  for (let j = 0; j < D; j++) {
    let s = 0;
    for (let k = 0; k < KP; k++) s += p[O_E + j * KP + k] * x[k];
    out[j] = s + (usePos ? p[O_POS + i * D + j] : 0);
  }
  return out;
}
/** Mean over tokens of tanh(z_i). `perm[i]` says which patch sits in slot i (default: identity). */
export function pooled(p: Float64Array, patches: Float64Array[], usePos = true, perm?: number[]): Float64Array {
  const m = new Float64Array(D), z = new Float64Array(D);
  for (let i = 0; i < NT; i++) {
    projectPatch(p, patches[perm ? perm[i] : i], i, usePos, z);
    for (let j = 0; j < D; j++) m[j] += tanh(z[j]) / NT;
  }
  return m;
}
export function shufflePerm(seed = 3): number[] {
  const r = createRng(seed), a = Array.from({ length: NT }, (_, i) => i);
  for (let i = NT - 1; i > 0; i--) { const j = Math.floor(r.next() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
export const distance = (a: ArrayLike<number>, b: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s); };

/** Patch embedding done as a strided convolution (kernel = stride = patch size): same numbers as projectPatch without positions. */
export function patchEmbedConv(p: Float64Array, img: Float64Array): Float64Array[] {
  const L = convLayer(1, D, P, P, 0);
  L.weight.set(p.subarray(O_E, O_E + D * KP));
  const y = conv2d(map3(1, SZ, SZ, img), L), out: Float64Array[] = [];
  for (let i = 0; i < NT; i++) { const v = new Float64Array(D); for (let j = 0; j < D; j++) v[j] = y.data[j * NT + i]; out.push(v); }
  return out;
}

const unit = (a: Float64Array): { u: Float64Array; n: number } => {
  let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  const n = Math.sqrt(s) + 1e-12, u = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) u[i] = a[i] / n;
  return { u, n };
};
const matVec = (p: Float64Array, o: number, v: Float64Array, bo: number): Float64Array => {
  const out = new Float64Array(D);
  for (let j = 0; j < D; j++) { let s = p[bo + j]; for (let k = 0; k < D; k++) s += p[o + j * D + k] * v[k]; out[j] = s; }
  return out;
};
export const embedImage = (p: Float64Array, patches: Float64Array[]): Float64Array => unit(matVec(p, O_WI, pooled(p, patches), O_BI)).u;
const textMean = (p: Float64Array, words: number[]): Float64Array => {
  const q = new Float64Array(D);
  for (const w of words) for (let j = 0; j < D; j++) q[j] += p[O_EMB + w * D + j] / words.length;
  return q;
};
export const embedText = (p: Float64Array, words: number[]): Float64Array => unit(matVec(p, O_WT, textMean(p, words), O_BT)).u;
export const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

export interface LossOut { loss: number; cos: Float64Array; probRow: Float64Array; diag: number }
/** Symmetric InfoNCE over a batch: L = (CE over rows + CE over columns) / 2 of softmax(scale * cos). Adds gradients into grad when given. */
export function clipLoss(p: Float64Array, imgs: Float64Array[][], caps: number[][], grad?: Float64Array): LossOut {
  const B = imgs.length, scale = exp(Math.min(p[O_S], MAX_SCALE));
  // forward, keeping what backward needs
  const ms: Float64Array[] = [], hs: Float64Array[][] = [], as: Float64Array[] = [], es: Float64Array[] = [], an: number[] = [];
  const qs: Float64Array[] = [], rs: Float64Array[] = [], fs: Float64Array[] = [], rn: number[] = [];
  for (let b = 0; b < B; b++) {
    const hb: Float64Array[] = [], m = new Float64Array(D);
    for (let i = 0; i < NT; i++) {
      const z = projectPatch(p, imgs[b][i], i), h = new Float64Array(D);
      for (let j = 0; j < D; j++) { h[j] = tanh(z[j]); m[j] += h[j] / NT; }
      hb.push(h);
    }
    const a = matVec(p, O_WI, m, O_BI), { u, n } = unit(a);
    hs.push(hb); ms.push(m); as.push(a); es.push(u); an.push(n);
    const q = textMean(p, caps[b]), r = matVec(p, O_WT, q, O_BT), f = unit(r);
    qs.push(q); rs.push(r); fs.push(f.u); rn.push(f.n);
  }
  const cos = new Float64Array(B * B), probRow = new Float64Array(B * B), probCol = new Float64Array(B * B);
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) cos[i * B + j] = dot(es[i], fs[j]);
  let loss = 0, diag = 0;
  for (let i = 0; i < B; i++) {
    let mx = -Infinity; for (let j = 0; j < B; j++) mx = Math.max(mx, scale * cos[i * B + j]);
    let s = 0; for (let j = 0; j < B; j++) s += exp(scale * cos[i * B + j] - mx);
    for (let j = 0; j < B; j++) probRow[i * B + j] = exp(scale * cos[i * B + j] - mx) / s;
    loss += (mx + log(s) - scale * cos[i * B + i]) / (2 * B); diag += probRow[i * B + i] / B;
  }
  for (let j = 0; j < B; j++) {
    let mx = -Infinity; for (let i = 0; i < B; i++) mx = Math.max(mx, scale * cos[i * B + j]);
    let s = 0; for (let i = 0; i < B; i++) s += exp(scale * cos[i * B + j] - mx);
    for (let i = 0; i < B; i++) probCol[i * B + j] = exp(scale * cos[i * B + j] - mx) / s;
    loss += (mx + log(s) - scale * cos[j * B + j]) / (2 * B);
  }
  if (!grad) return { loss, cos, probRow, diag };

  // backward
  const dS = new Float64Array(B * B);
  let dscale = 0;
  for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) {
    const d = (probRow[i * B + j] + probCol[i * B + j] - 2 * (i === j ? 1 : 0)) / (2 * B);
    dS[i * B + j] = d; dscale += d * cos[i * B + j];
  }
  if (p[O_S] < MAX_SCALE) grad[O_S] += dscale * scale;
  for (let b = 0; b < B; b++) {
    const de = new Float64Array(D), df = new Float64Array(D);
    for (let k = 0; k < B; k++) for (let j = 0; j < D; j++) { de[j] += dS[b * B + k] * scale * fs[k][j]; df[j] += dS[k * B + b] * scale * es[k][j]; }
    // normalisation backward: da = (de - e (e . de)) / |a|
    const ed = dot(es[b], de), da = new Float64Array(D); for (let j = 0; j < D; j++) da[j] = (de[j] - es[b][j] * ed) / an[b];
    const fd = dot(fs[b], df), dr = new Float64Array(D); for (let j = 0; j < D; j++) dr[j] = (df[j] - fs[b][j] * fd) / rn[b];
    const dm = new Float64Array(D), dq = new Float64Array(D);
    for (let j = 0; j < D; j++) {
      grad[O_BI + j] += da[j]; grad[O_BT + j] += dr[j];
      for (let k = 0; k < D; k++) { grad[O_WI + j * D + k] += da[j] * ms[b][k]; dm[k] += p[O_WI + j * D + k] * da[j]; grad[O_WT + j * D + k] += dr[j] * qs[b][k]; dq[k] += p[O_WT + j * D + k] * dr[j]; }
    }
    for (const w of caps[b]) for (let j = 0; j < D; j++) grad[O_EMB + w * D + j] += dq[j] / caps[b].length;
    for (let i = 0; i < NT; i++) for (let j = 0; j < D; j++) {
      const du = (dm[j] / NT) * (1 - hs[b][i][j] ** 2);
      grad[O_POS + i * D + j] += du;
      for (let k = 0; k < KP; k++) grad[O_E + j * KP + k] += du * imgs[b][i][k];
    }
  }
  return { loss, cos, probRow, diag };
}

// ------------------------------------------------------------------ training
export const BATCH = 8, STEPS = 3000, SNAP_EVERY = 50, LR = 0.01, KEEP = 0.3, TRAIN_SEED = 12;
export interface ClipRun { steps: number[]; snaps: Float64Array[]; trainLoss: number[]; evalLoss: number[]; evalDiag: number[]; scale: number[] }

/** A fixed batch of eight different captions drawn from the training combinations, clean images, shown in the matrix. */
export const EVAL_ITEMS: Item[] = [0, 5, 10, 14, 19, 23, 27, 31].map((i) => TRAIN_ITEMS[i % TRAIN_ITEMS.length]);
export const EVAL_ARTICLE = false;
const evalPatches = EVAL_ITEMS.map((it) => patchify(itemImage(it), P)), evalCaps = EVAL_ITEMS.map((it) => wordIds(captionText(it)));
export const evalBatch = (p: Float64Array): LossOut => clipLoss(p, evalPatches, evalCaps);

export function trainClip(onProgress?: (f: number) => void): ClipRun {
  const rng = createRng(TRAIN_SEED), p = initParams(PROJECTION_SEED), st = createAdam(NPARAMS), grad = new Float64Array(NPARAMS);
  const pool: Item[] = [...TRAIN_ITEMS, ...PAIR_ITEMS];
  const run: ClipRun = { steps: [], snaps: [], trainLoss: [], evalLoss: [], evalDiag: [], scale: [] };
  const snap = (s: number) => { const e = evalBatch(p); run.steps.push(s); run.snaps.push(p.slice()); run.evalLoss.push(e.loss); run.evalDiag.push(e.diag); run.scale.push(exp(Math.min(p[O_S], MAX_SCALE))); };
  snap(0);
  for (let s = 1; s <= STEPS; s++) {
    // a batch of eight items with different captions. Captions often leave out the size or the position, so that a bare
    // shape word ("circle") is something the model has met. The two pair captions share one bag of words, so at most one of them.
    const idx = pool.map((_, i) => i); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(rng.next() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    const batch: Item[] = [], texts: string[] = [], seen = new Set<string>();
    for (const i of idx) {
      const it = pool[i], txt = captionText(it, rng.next() < 0.5, rng.next() < KEEP, rng.next() < KEEP), key = it.kind === 'pair' ? 'pair' : txt.replace(/^a /, '');
      if (seen.has(key)) continue;
      seen.add(key); batch.push(it); texts.push(txt);
      if (batch.length === BATCH) break;
    }
    const imgs = batch.map((it) => patchify(itemImage(it, rng.next() < 0.5 ? 0 : 1 + Math.floor(rng.next() * 8)), P));
    const caps = texts.map(wordIds);
    grad.fill(0);
    const { loss } = clipLoss(p, imgs, caps, grad);
    adam(p, grad, st, LR);
    if (s % 10 === 0) run.trainLoss.push(loss);
    if (s % SNAP_EVERY === 0) { snap(s); onProgress?.(s / STEPS); }
  }
  return run;
}

// ------------------------------------------------------------------ zero-shot
export type Template = 'plain' | 'article' | 'sized';
export const TEMPLATES: { id: Template; label: string; text: (shape: Shape) => string }[] = [
  { id: 'plain', label: 'circle', text: (s) => s },
  { id: 'article', label: 'a circle', text: (s) => `a ${s}` },
  { id: 'sized', label: 'a large circle', text: (s) => `a large ${s}` },
];
export function classEmbeddings(p: Float64Array, tpl: Template): Float64Array[] {
  const t = TEMPLATES.find((x) => x.id === tpl)!;
  return SHAPES.map((s) => embedText(p, wordIds(t.text(s))));
}
export interface ZeroShot { cos: number[]; prob: number[]; pred: number }
export function classify(p: Float64Array, img: Float64Array, cls: Float64Array[]): ZeroShot {
  const e = embedImage(p, patchify(img, P)), cos = cls.map((c) => dot(e, c)), sc = exp(Math.min(p[O_S], MAX_SCALE));
  const mx = Math.max(...cos), ex = cos.map((c) => exp(sc * (c - mx))), s = ex.reduce((a, b) => a + b, 0);
  let pred = 0; cos.forEach((c, i) => { if (c > cos[pred]) pred = i; });
  return { cos, prob: ex.map((v) => v / s), pred };
}
export const shapeIndex = (it: Item): number => (it.kind === 'single' ? SHAPES.indexOf(it.shape) : -1);
export function heldOutAccuracy(p: Float64Array, tpl: Template): { correct: boolean[]; acc: number } {
  const cls = classEmbeddings(p, tpl);
  const correct = HELD_IMAGES.map((h) => classify(p, h.img, cls).pred === shapeIndex(h.item));
  return { correct, acc: correct.filter(Boolean).length / correct.length };
}

// ------------------------------------------------------------------ order-blindness (limits)
export type PairKind = 'order' | 'size';
export interface PairSet { items: [Item, Item]; captions: [string, string] }
export const PAIRS: Record<PairKind, PairSet> = {
  order: { items: [{ kind: 'pair', order: 0 }, { kind: 'pair', order: 1 }], captions: ['circle left of square', 'square left of circle'] },
  size: { items: [{ kind: 'single', shape: 'circle', size: 'large', pos: 'left' }, { kind: 'single', shape: 'circle', size: 'small', pos: 'left' }], captions: ['large circle left', 'small circle left'] },
};
/** cos[i][j] = cosine of image i and caption j. */
export function pairCosines(p: Float64Array, kind: PairKind): number[][] {
  const { items, captions } = PAIRS[kind], caps = captions.map((c) => embedText(p, wordIds(c)));
  return items.map((it) => { const e = embedImage(p, patchify(itemImage(it), P)); return caps.map((c) => dot(e, c)); });
}

// ------------------------------------------------------------------ VLM token plumbing
export const LM_WIDTH = 16;
export const VLM_TEXT = ['Q:', 'what', 'shape', '?', 'A:'];
export interface VlmTokens { n: number; patchDim: number; enc: Float64Array[]; proj: Float64Array[]; text: Float64Array[]; share: number }
/** Patch embeddings (width 8), a linear projector into the language model's width (16), and the question's token embeddings. Seeded weights. */
export function vlmTokens(img: Float64Array, patch: number): VlmTokens {
  const r = createRng(21), pd = patch * patch, g = SZ / patch, n = g * g;
  const E = Float64Array.from({ length: D * pd }, () => r.uniform(-Math.sqrt(3 / pd), Math.sqrt(3 / pd)));
  const W = Float64Array.from({ length: LM_WIDTH * D }, () => r.uniform(-Math.sqrt(3 / D), Math.sqrt(3 / D)));
  const T = VLM_TEXT.map(() => Float64Array.from({ length: LM_WIDTH }, () => r.uniform(-1, 1)));
  const enc = patchify(img, patch).map((x) => { const z = new Float64Array(D); for (let j = 0; j < D; j++) { let s = 0; for (let k = 0; k < pd; k++) s += E[j * pd + k] * x[k]; z[j] = s; } return z; });
  const proj = enc.map((z) => { const o = new Float64Array(LM_WIDTH); for (let j = 0; j < LM_WIDTH; j++) { let s = 0; for (let k = 0; k < D; k++) s += W[j * D + k] * z[k]; o[j] = s; } return o; });
  return { n, patchDim: pd, enc, proj, text: T, share: n / (n + VLM_TEXT.length) };
}

// ------------------------------------------------------------------ timing and frames
export type Mode = 'patches' | 'embed' | 'contrastive' | 'zeroshot' | 'vlm' | 'limits';
export interface Seg { at: number; label: string }
export const MODE_TIME: Record<Mode, { dur: number; markers: Seg[] }> = {
  patches: { dur: 9, markers: [{ at: 0, label: 'Grid' }, { at: 2, label: 'Cut out' }, { at: 6, label: 'Flatten' }] },
  embed: { dur: 13, markers: [{ at: 0, label: 'Multiply' }, { at: 4.5, label: 'Add position' }, { at: 6.5, label: 'All patches' }, { at: 9, label: 'Shuffle test' }] },
  contrastive: { dur: 12, markers: [{ at: 0, label: 'Early' }, { at: 4, label: 'Middle' }, { at: 8, label: 'Late' }] },
  zeroshot: { dur: 16, markers: [{ at: 0, label: 'Compare' }, { at: 5, label: 'Winner' }, { at: 6, label: 'Held-out set' }] },
  vlm: { dur: 10, markers: [{ at: 0, label: 'Embed' }, { at: 3, label: 'Project' }, { at: 5, label: 'Add text' }, { at: 7.5, label: 'Share' }] },
  limits: { dur: 8, markers: [{ at: 0, label: 'Score' }, { at: 4, label: 'Compare' }] },
};
export const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));
export const ease = (x: number): number => { const u = clamp01(x); return u * u * (3 - 2 * u); };

/** patches: grid draw progress, per-patch lift and flatten progress. */
export function patchesFrame(t: number, n: number) {
  return {
    grid: clamp01(t / 2),
    lift: (i: number) => ease((t - (2 + (3.2 * i) / n)) / 0.8),
    flat: (i: number) => clamp01((t - (6 + (2.4 * i) / n)) / 0.6),
    phase: t < 2 ? 'grid' : t < 6 ? 'cut' : 'flatten',
  };
}
/** embed: which column of E is being multiplied, how many are done, and which stage we are in. */
export function embedFrame(t: number) {
  const phase = t < 4.5 ? 'multiply' : t < 6.5 ? 'position' : t < 9 ? 'tokens' : 'shuffle';
  return {
    phase,
    col: t < 4 ? Math.min(D - 1, Math.floor(t / 0.5)) : -1,
    done: t < 4 ? Math.floor(t / 0.5) : D,
    posShown: t >= 4.5, sumShown: t >= 5.5,
    tokens: t < 6.5 ? 1 : Math.min(NT, 1 + Math.floor(((t - 6.5) / 2) * NT)),
    shuffled: t >= 9, pooledShown: t >= 10.5,
  };
}
export const trainSnapshot = (t: number, K: number, dur = MODE_TIME.contrastive.dur): number => Math.min(K - 1, Math.floor((t / dur) * K));
export function zeroShotFrame(t: number, nHeld = HELD_IMAGES.length) {
  if (t < 6) return { held: -1, bars: t < 5 ? Math.min(SHAPES.length, t) : SHAPES.length, winner: t >= 5 };
  return { held: Math.min(nHeld - 1, Math.floor((t - 6) / 0.5)), bars: SHAPES.length, winner: true };
}
export const limitsFrame = (t: number) => ({ bars: Math.min(4, t), compare: t >= 4 });
export function vlmFrame(t: number, n: number) {
  return {
    enc: clamp01(t / 3), proj: clamp01((t - 3) / 2), imgChips: clamp01((t - 3.5) / 1.5),
    textShown: t < 5 ? 0 : Math.min(VLM_TEXT.length, Math.floor((t - 5) / 0.5) + 1), share: clamp01((t - 7.5) / 2), n,
  };
}
