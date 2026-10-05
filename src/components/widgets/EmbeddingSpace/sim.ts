/** Embedding lessons: lookup, training checkpoints, similarity, 2D map, offsets, contextual vectors. Pure logic, no DOM. */
import { createRng } from '../../../lib/math/rng';
import { cosine } from '../../../lib/transformer/math';
import { forward, type Weights } from '../../../lib/transformer/model';
import { fromFloat16 } from '../../../lib/transformer/load';
import { VOCAB, WORD_CLASS, SAMPLE_200, encodeWords } from '../../../lib/transformer/grammar';
import { covariance, eigSym } from '../../../lib/math/linalg';

export type Mode = 'lookup' | 'training' | 'similarity' | 'projection' | 'arithmetic' | 'contextual';
export type Pt = [number, number];
export type Metric = 'cosine' | 'dot';
export const D = 16;
export const DEFAULT_SENTENCE = '[start] these small sheep chase a red fox';
export const PROBES = ['cat', 'eats', 'these', 'red'];
export const PRESETS: readonly [string, string, string][] = [
  ['cats', 'cat', 'dog'], ['foxes', 'fox', 'bird'], ['apples', 'apple', 'seed'], ['chase', 'chases', 'sees'], ['these', 'this', 'that'],
];

const clamp01 = (u: number) => (u < 0 ? 0 : u > 1 ? 1 : u);
export const ease = (u: number) => { const x = clamp01(u); return x * x * (3 - 2 * x); };
const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
export const idOf = (word: string) => VOCAB.indexOf(word as (typeof VOCAB)[number]);
export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const norm = (a: ArrayLike<number>) => Math.sqrt(dot(a, a));
export const sub = (a: number[], b: number[]) => a.map((x, i) => x - b[i]);
export const fmt = (x: number, n = 3) => (x < 0 ? '−' : '') + Math.abs(x).toFixed(n);

// ------------------------------------------------------------------ data

export interface SnapMeta { checkpoints: { step: number; byteOffset: number }[]; shape: [number, number] }
export interface Data { w: Weights; table: number[][]; snaps: number[][][]; steps: number[] }

const rowsAt = (get: (i: number) => number, rows: number, cols: number) =>
  Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => get(i * cols + j)));

/** The trained embedding table (34 rows by 16 columns) and the eight embedding checkpoints recorded during training. */
export function makeData(w: Weights, meta: SnapMeta, bytes: Uint8Array): Data {
  const { offset, rows, cols } = w.offsets.embedding;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return {
    w,
    table: rowsAt((k) => w.params[offset + k], rows, cols),
    snaps: meta.checkpoints.map((c) => rowsAt((k) => fromFloat16(view.getUint16(c.byteOffset + k * 2, true)), meta.shape[0], meta.shape[1])),
    steps: meta.checkpoints.map((c) => c.step),
  };
}

// ------------------------------------------------------------------ word classes

export const CLASS_NAMES = ['Determiner', 'Adjective', 'Animal noun', 'Food noun', 'Verb', 'Start / full stop'] as const;
/** --viz-cat slot for each class (identity colours; slot 6 is skipped because it is close to slot 3). */
export const CLASS_TOKENS = ['--viz-cat-1', '--viz-cat-2', '--viz-cat-3', '--viz-cat-4', '--viz-cat-5', '--viz-cat-7'] as const;
export const classOf = (i: number): number => {
  const c = WORD_CLASS[i];
  return c.kind === 'det' ? 0 : c.kind === 'adj' ? 1 : c.kind === 'noun' ? (c.category === 'animal' ? 2 : 3) : c.kind === 'verb' ? 4 : 5;
};

// ------------------------------------------------------------------ PCA (reuses PCAProjector)

export interface Pca { mean: number[]; axes: number[][]; share: number }
export function pca2(rows: number[][]): Pca {
  const eig = eigSym(covariance(rows)), n = rows.length;
  const mean = Array.from({ length: rows[0].length }, (_, j) => rows.reduce((s, r) => s + r[j], 0) / n);
  const total = eig.values.reduce((a, b) => a + b, 0);
  return { mean, axes: [eig.vectors[0], eig.vectors[1]], share: (eig.values[0] + eig.values[1]) / total };
}
export const project = (p: { mean: number[]; axes: number[][] }, row: number[]): Pt =>
  [0, 1].map((a) => row.reduce((s, x, j) => s + (x - p.mean[j]) * p.axes[a][j], 0)) as Pt;

/** Cosine neighbours of word `i` among all other words (optionally excluding more), best first. */
export function neighbours(rows: number[][], i: number, k: number, exclude: number[] = [], metric: Metric = 'cosine') {
  return rows.map((r, j) => ({ word: VOCAB[j], id: j, value: metric === 'cosine' ? cosine(rows[i], r) : dot(rows[i], r) }))
    .filter((x) => x.id !== i && !exclude.includes(x.id)).sort((a, b) => b.value - a.value).slice(0, k);
}

/** Fit points (and pad) into a square-pixel-friendly data box: returns [x0, x1, y0, y1]. */
export function fitBox(points: Pt[], pad = 0.12): [number, number, number, number] {
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = (x1 - x0 || 1) * pad, py = (y1 - y0 || 1) * pad;
  return [x0 - px, x1 + px, y0 - py, y1 + py];
}

/** Single-linkage groups of points closer than `gap` (in box fractions); each group's label is its words. */
export function clusterWords(points: Pt[], box: [number, number, number, number], gap: number): number[][] {
  const sx = box[1] - box[0], sy = box[3] - box[2], group = points.map((_, i) => i);
  const root = (i: number): number => (group[i] === i ? i : (group[i] = root(group[i])));
  for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
    if (Math.hypot((points[i][0] - points[j][0]) / sx, (points[i][1] - points[j][1]) / sy) < gap) group[root(j)] = root(i);
  }
  const out = new Map<number, number[]>();
  points.forEach((_, i) => { const r = root(i); out.set(r, [...(out.get(r) ?? []), i]); });
  return [...out.values()];
}

/** Stage index and eased position for a timeline with holds and short glides centred on each boundary k * S. */
export function stagePos(t: number, n: number, S: number, glide: number) {
  let p = 0;
  for (let k = 1; k < n; k++) p += ease((t - (k * S - glide / 2)) / glide);
  return { p, idx: Math.min(n - 1, Math.floor(t / S + 1e-9)) };
}

export interface Marker { at: number; label: string }
export interface BaseFrame { phase: string; status: string }

// ------------------------------------------------------------------ lookup

const SEG = 1.3, LOOK = 'Look up each token', DONE = 'The finished matrix';
export interface LookupScene { mode: 'lookup'; ids: number[]; table: number[][]; maxAbs: number; T: number; markers: Marker[] }
export interface LookupFrame extends BaseFrame { mode: 'lookup'; active: number; slide: number; pointer: number; done: number }

function lookupScene(data: Data, sentence: string): LookupScene {
  const ids = encodeWords(sentence);
  return {
    mode: 'lookup', ids, table: data.table, maxAbs: Math.max(...data.table.flat().map(Math.abs)),
    T: ids.length * SEG + 2.4,
    markers: [{ at: 0, label: LOOK }, { at: ids.length * SEG, label: DONE }],
  };
}
function lookupFrame(t: number, s: LookupScene): LookupFrame {
  const n = s.ids.length, k = Math.floor(t / SEG);
  if (k >= n) return { mode: 'lookup', active: -1, slide: 1, pointer: 1, done: n, phase: DONE,
    status: `The sentence is now a ${n} by ${D} matrix: one row of ${D} numbers per token. The table itself is unchanged.` };
  const u = t / SEG - k, id = s.ids[k], row = s.table[id].slice(0, 4).map((x) => fmt(x, 2)).join(', ');
  return { mode: 'lookup', active: k, pointer: ease(u / 0.35), slide: ease((u - 0.4) / 0.5), done: k, phase: LOOK,
    status: `Token ${k + 1} of ${n}: "${VOCAB[id]}" has ID ${id}, so the model copies row ${id} of the table. Its first four numbers are ${row}.` };
}

// ------------------------------------------------------------------ training

const STEP_T = 1.4, HOLD = 1.4, LEAD = 1.6; // LEAD: time spent on the random table before training starts
export interface TrainingScene {
  mode: 'training'; pts: Pt[][]; box: [number, number, number, number]; steps: number[]; cat: number; dog: number;
  cosines: number[]; T: number; markers: Marker[];
}
export interface TrainingFrame extends BaseFrame { mode: 'training'; points: Pt[]; idx: number; step: number; cosine: number }

function trainingScene(data: Data): TrainingScene {
  const basis = pca2(data.snaps.at(-1)!); // one fixed basis (the trained table's), so motion is real movement
  const pts = data.snaps.map((rows) => rows.map((r) => project(basis, r)));
  const cat = idOf('cat'), dog = idOf('dog'), n = data.snaps.length;
  return {
    mode: 'training', pts, box: fitBox(pts.flat(), 0.1), steps: data.steps, cat, dog,
    cosines: data.snaps.map((rows) => cosine(rows[cat], rows[dog])), T: LEAD + (n - 1) * STEP_T + HOLD,
    // Segments follow the snapped checkpoint: the marker sits where the readout changes (halfway to the next checkpoint).
    markers: [{ at: 0, label: 'Random' }, { at: LEAD + 0.5 * STEP_T, label: 'Early' }, { at: LEAD + 2.5 * STEP_T, label: 'Middle' }, { at: LEAD + 5.5 * STEP_T, label: 'Trained' }],
  };
}
function trainingFrame(t: number, s: TrainingScene): TrainingFrame {
  const n = s.pts.length, p = Math.max(0, Math.min(n - 1, (t - LEAD) / STEP_T)), k = Math.min(n - 2, Math.floor(p)), u = p - k;
  const idx = Math.min(n - 1, Math.round(p));
  const points = s.pts[k].map((a, i) => [lerp(a[0], s.pts[k + 1][i][0], u), lerp(a[1], s.pts[k + 1][i][1], u)] as Pt);
  const phase = idx === 0 ? 'Random' : idx <= 2 ? 'Early' : idx <= 5 ? 'Middle' : 'Trained';
  return { mode: 'training', points, idx, step: s.steps[idx], cosine: s.cosines[idx], phase,
    status: `Training step ${s.steps[idx].toLocaleString('en-US')}: the angle between "cat" and "dog" has cosine ${fmt(s.cosines[idx])}.` };
}

// ------------------------------------------------------------------ similarity

const SIM_S = 3.4, SIM_TR = 1.0;
export interface SimilarityScene {
  mode: 'similarity'; probes: number[]; metric: Metric; domain: number; T: number; markers: Marker[];
  /** per probe: words sorted best first, with values */
  sorted: { id: number; value: number }[][];
}
export interface SimRow { id: number; value: number; rank: number }
export interface SimilarityFrame extends BaseFrame { mode: 'similarity'; probe: number; rows: SimRow[]; top: { word: string; value: number }[] }

function similarityScene(data: Data, probes: string[], metric: Metric): SimilarityScene {
  const ids = probes.map(idOf);
  const sorted = ids.map((p) => data.table.map((r, j) => ({ id: j, value: metric === 'cosine' ? cosine(data.table[p], r) : dot(data.table[p], r) }))
    .filter((x) => x.id !== p).sort((a, b) => b.value - a.value));
  const biggest = Math.max(...sorted.flat().map((x) => Math.abs(x.value)));
  return {
    mode: 'similarity', probes: ids, metric, sorted, T: ids.length * SIM_S, domain: metric === 'cosine' ? 1 : Math.ceil(biggest * 4) / 4,
    markers: ids.map((id, i) => ({ at: i ? i * SIM_S + SIM_TR / 2 : 0, label: VOCAB[id] })),
  };
}
function similarityFrame(t: number, s: SimilarityScene): SimilarityFrame {
  const n = s.probes.length, k = Math.min(n - 1, Math.floor(t / SIM_S)), local = t - k * SIM_S;
  const u = k > 0 ? ease(local / SIM_TR) : 1, from = s.sorted[Math.max(0, k - 1)], to = s.sorted[k];
  const rank = new Map(from.map((x, i) => [x.id, i])), value = new Map(from.map((x) => [x.id, x.value]));
  const shown = u >= 0.5 ? k : Math.max(0, k - 1);
  const rows = to.map((x, i) => ({ id: x.id, value: lerp(value.get(x.id) ?? x.value, x.value, u), rank: lerp(rank.get(x.id) ?? i, i, u) }));
  const probe = s.probes[shown];
  return { mode: 'similarity', probe, rows, phase: VOCAB[probe], top: s.sorted[shown].slice(0, 3).map((x) => ({ word: VOCAB[x.id], value: x.value })),
    status: `Probe "${VOCAB[probe]}": closest are ${s.sorted[shown].slice(0, 3).map((x) => `${VOCAB[x.id]} ${fmt(x.value)}`).join(', ')}.` };
}

// ------------------------------------------------------------------ projection

const ROT = 3.2, HI = 2.2;
export const GROUPS: { label: string; has: (i: number) => boolean; says: string }[] = [
  { label: 'Animals', has: (i) => classOf(i) === 2,
    says: 'Animals: cat, dog, fox and bird sit together, their plurals form a second group, and sheep sits apart.' },
  { label: 'Foods', has: (i) => classOf(i) === 3,
    says: 'Foods: apple and seed sit together, apples and seeds form their own group, and fish lies near sheep.' },
  { label: 'Verbs', has: (i) => classOf(i) === 4,
    says: 'Verbs: sees, chases, eats and likes form one group and see, chase, eat and like another, split by number like the nouns.' },
  { label: 'Small words', has: (i) => [0, 1, 5].includes(classOf(i)),
    says: 'Determiners, adjectives, [start] and the full stop: each class sits in its own corner, away from the nouns and verbs.' },
];
export interface ProjectionScene {
  mode: 'projection'; table: number[][]; basis: Pca; start: number[][]; box: [number, number, number, number];
  clusters: number[][]; T: number; markers: Marker[]; share: number;
}
export interface ProjectionFrame extends BaseFrame { mode: 'projection'; points: Pt[]; group: number; settle: number }

function orthonormal(rows: number[][]): number[][] {
  const out: number[][] = [];
  for (const r of rows) {
    const v = r.slice();
    for (const u of out) { const c = dot(v, u); for (let j = 0; j < v.length; j++) v[j] -= c * u[j]; }
    const len = norm(v) || 1; out.push(v.map((x) => x / len));
  }
  return out;
}
const mixBasis = (a: number[][], b: number[][], s: number) => (s <= 0 ? a : s >= 1 ? b : orthonormal(a.map((r, i) => r.map((x, j) => lerp(x, b[i][j], s)))));

function projectionScene(data: Data): ProjectionScene {
  const basis = pca2(data.table), rng = createRng(31);
  const start = orthonormal([Array.from({ length: D }, () => rng.normal()), Array.from({ length: D }, () => rng.normal())]);
  const at = (s: number) => data.table.map((r) => project({ mean: basis.mean, axes: mixBasis(start, basis.axes, s) }, r));
  const box = fitBox([0, 0.25, 0.5, 0.75, 1].flatMap(at), 0.1), final = at(1);
  return {
    mode: 'projection', table: data.table, basis, start, box, clusters: clusterWords(final, box, 0.04), share: basis.share,
    T: ROT + GROUPS.length * HI + 1,
    markers: [{ at: 0, label: 'Rotate' }, ...GROUPS.map((g, i) => ({ at: ROT + i * HI, label: g.label }))],
  };
}
function projectionFrame(t: number, s: ProjectionScene): ProjectionFrame {
  const rot = ease(t / ROT), axes = mixBasis(s.start, s.basis.axes, rot);
  const points = s.table.map((r) => project({ mean: s.basis.mean, axes }, r));
  const g = t < ROT ? -1 : Math.min(GROUPS.length - 1, Math.floor((t - ROT) / HI));
  return { mode: 'projection', points, group: g, settle: ease((t - ROT + 0.4) / 0.8), phase: g < 0 ? 'Rotate' : GROUPS[g].label,
    status: g < 0 ? 'The view rotates from an arbitrary angle to the angle that keeps the most spread.' : GROUPS[g].says };
}

// ------------------------------------------------------------------ arithmetic

const AR = { offset: 1.6, copy: 3.4, land: 4.4, T: 7 };
export interface ArithmeticScene {
  mode: 'arithmetic'; T: number; markers: Marker[];
  each: {
    names: [string, string, string]; box: [number, number, number, number]; plane: Record<'a' | 'b' | 'c' | 'land', Pt>;
    near: { word: string; value: number; at: Pt }[]; offsetCos: number;
  }[];
}
export interface ArithmeticFrame extends BaseFrame {
  mode: 'arithmetic'; k: number; draw: number; slide: number; ring: number; shown: number; arrow: [Pt, Pt];
}

function planeOf(data: Data, names: [string, string, string]) {
  const [a, b, c] = names.map((w) => data.table[idOf(w)]);
  const u0 = sub(a, b), len = norm(u0), u = u0.map((x) => x / len);
  const cb = sub(c, b), pu = dot(cb, u), v0 = cb.map((x, j) => x - pu * u[j]), v = v0.map((x) => x / (norm(v0) || 1));
  const at = (r: number[]): Pt => { const d = sub(r, b); return [dot(d, u), dot(d, v)]; };
  const target = a.map((x, j) => x - b[j] + c[j]);
  const near = data.table.map((r, i) => ({ id: i, value: cosine(target, r) })).filter((x) => !names.map(idOf).includes(x.id))
    .sort((x, y) => y.value - x.value).slice(0, 4).map((x) => ({ word: VOCAB[x.id], value: x.value, at: at(data.table[x.id]) }));
  const cc = at(c), landing: Pt = [cc[0] + len, cc[1]];
  return { names, len, near, plane: { a: at(a), b: at(b), c: cc, land: landing }, at, target };
}
function arithmeticScene(data: Data, presets: [string, string, string][]): ArithmeticScene {
  return {
    mode: 'arithmetic', T: AR.T * presets.length,
    markers: presets.length === 1
      ? [{ at: 0, label: 'Offset' }, { at: AR.offset, label: 'Copy' }, { at: AR.copy, label: 'Land' }, { at: AR.land, label: 'Nearest' }]
      : presets.map((p, i) => ({ at: i * AR.T, label: `${p[1]} to ${p[2]}` })),
    each: presets.map((names) => {
      const pl = planeOf(data, names), [a, b, c] = names.map(idOf);
      // The offset from a pair of the same kind: nouns pair with their own plural, so quote cosine(a-b, near-c) for the best match.
      const best = pl.near[0], bi = idOf(best.word);
      const offsetCos = cosine(sub(data.table[a], data.table[b]), sub(data.table[bi], data.table[c]));
      const pts = [pl.plane.a, pl.plane.b, pl.plane.c, pl.plane.land, ...pl.near.slice(0, 3).map((x) => x.at)];
      return { names, box: fitBox(pts, 0.25), plane: pl.plane, near: pl.near, offsetCos };
    }),
  };
}
function arithmeticFrame(t: number, s: ArithmeticScene): ArithmeticFrame {
  const k = Math.min(s.each.length - 1, Math.floor(t / AR.T)), e = s.each[k], local = t - k * AR.T;
  const { a, b, c } = e.plane, [A, B, C] = e.names;
  const draw = ease(local / (AR.offset - 0.2)), slide = ease((local - AR.offset - 0.2) / (AR.copy - AR.offset - 0.4));
  const ring = ease((local - AR.copy) / 0.7), shown = local < AR.land ? 0 : Math.min(e.near.length, 1 + Math.floor((local - AR.land) / 0.55));
  const from: Pt = [lerp(b[0], c[0], slide), lerp(b[1], c[1], slide)], head: Pt = [from[0] + (a[0] - b[0]) * draw, from[1] + (a[1] - b[1]) * draw];
  const phase = local < AR.offset ? 'Offset' : local < AR.copy ? 'Copy' : local < AR.land ? 'Land' : 'Nearest';
  const label = s.each.length === 1 ? phase : `${B} to ${C}`;
  const status = local < AR.offset ? `The arrow from "${B}" to "${A}" is the difference ${A} minus ${B}.`
    : local < AR.copy ? `Copy that same arrow so it starts at "${C}".`
    : local < AR.land ? `The arrow ends at the point ${A} − ${B} + ${C}.`
    : `Nearest words to that point, leaving out ${A}, ${B} and ${C}: ${e.near.slice(0, shown).map((x) => `${x.word} ${fmt(x.value)}`).join(', ')}${shown < e.near.length ? ' …' : '.'}`;
  return { mode: 'arithmetic', k, draw, slide, ring, shown, arrow: [from, head], phase: label, status };
}

// ------------------------------------------------------------------ contextual

const CS = 3.2, CG = 1.0;
export const STAGES = ['Embedding + position', 'After block 1', 'After block 2'] as const;
const sentenceFor = (det: string) => encodeWords(`[start] ${det} small sheep sees a red fox .`);
export interface ContextScene {
  mode: 'contextual'; T: number; markers: Marker[]; dets: string[]; row: number[];
  /** per stage: normalised cloud points and per-determiner sheep points */
  cloud: Pt[][]; plural: boolean[]; sheep: Pt[][]; cos: number[][]; // sheep[stage][det], cos[stage][det] vs "this"
  /** cosine of the third sheep with "these" at each stage (empty when there is no third) */
  cosThese: number[];
}
export interface ContextFrame extends BaseFrame {
  mode: 'contextual'; stage: number; cloud: Pt[]; sheep: Pt[]; cosThisThese: number;
}

const stagesOf = (tr: ReturnType<typeof forward>) => [tr.x0, tr.layers[0].resid2, tr.layers[1].resid2];
interface Cloud { vecs: number[][][]; plural: boolean[] }
const clouds = new WeakMap<Data, Cloud>();
function addSentence(data: Data, ids: readonly number[], c: Cloud) {
  const st = stagesOf(forward(data.w, ids));
  let number: string | undefined = 'either';
  ids.forEach((tok, pos) => {
    const k = WORD_CLASS[tok];
    if (k.kind === 'det') number = k.number;
    if (k.kind !== 'noun') return;
    const n = k.number !== 'either' ? k.number : number;
    if (n === 'either') return; // "the sheep" / "the fish": the number is not fixed by the words before it
    c.vecs.push(st.map((a) => Array.from(a.slice(pos * D, pos * D + D))));
    c.plural.push(n === 'pl');
  });
}
/** Noun vectors of the 200 sample sentences at each stage, computed once per page. */
function cloudOf(data: Data): Cloud {
  let c = clouds.get(data);
  if (!c) { c = { vecs: [], plural: [] }; for (const ids of SAMPLE_200) addSentence(data, ids, c); clouds.set(data, c); }
  return c;
}
/** Same as cloudOf, but yields to the browser between small batches so no task blocks input for long. */
export async function prepareContext(data: Data, batch = 20): Promise<void> {
  if (clouds.has(data)) return;
  const c: Cloud = { vecs: [], plural: [] };
  for (let i = 0; i < SAMPLE_200.length; i += batch) {
    for (const ids of SAMPLE_200.slice(i, i + batch)) addSentence(data, ids, c);
    await new Promise((r) => setTimeout(r, 0));
  }
  if (!clouds.has(data)) clouds.set(data, c);
}

export function contextScene(data: Data, third: string | null = null): ContextScene {
  const dets = ['this', 'these', ...(third ? [third] : [])];
  const { vecs, plural } = cloudOf(data);
  const vec = dets.map((d) => stagesOf(forward(data.w, sentenceFor(d))).map((a) => Array.from(a.slice(3 * D, 3 * D + D))));
  const cloud: Pt[][] = [], sheep: Pt[][] = [];
  let prev: Pt[] | null = null;
  for (let st = 0; st < 3; st++) {
    const basis = pca2(vecs.map((v) => v[st])), raw = vecs.map((v) => project(basis, v[st]));
    // Keep axes consistent from stage to stage so the glide is a real move, not a mirror flip.
    // Stage 0: the vertical axis points towards the singular nouns.
    const sign = [0, 1].map((a) => {
      const agree = prev ? raw.reduce((x, p, i) => x + p[a] * prev![i][a], 0) : a ? raw.reduce((x, p, i) => x + p[1] * (plural[i] ? -1 : 1), 0) : 1;
      return agree < 0 ? -1 : 1;
    });
    const flip = (p: Pt): Pt => [p[0] * sign[0], p[1] * sign[1]];
    const pts = raw.map(flip), sheepRaw = dets.map((_, d) => flip(project(basis, vec[d][st])));
    // Each axis is scaled to fit [-1, 1]: this view shows which group a point joins, not distances.
    const all = [...pts, ...sheepRaw], sx = Math.max(...all.map((p) => Math.abs(p[0]))), sy = Math.max(...all.map((p) => Math.abs(p[1])));
    const nz = (p: Pt): Pt => [p[0] / sx, p[1] / sy];
    cloud.push(pts.map(nz)); sheep.push(sheepRaw.map(nz)); prev = pts;
  }
  return {
    mode: 'contextual', T: 3 * CS, dets, row: data.table[idOf('sheep')], cloud, plural, sheep,
    cos: [0, 1, 2].map((st) => dets.map((_, d) => cosine(vec[0][st], vec[d][st]))),
    cosThese: dets.length > 2 ? [0, 1, 2].map((st) => cosine(vec[2][st], vec[1][st])) : [],
    markers: [{ at: 0, label: 'Embedding' }, { at: CS, label: 'Block 1' }, { at: 2 * CS, label: 'Block 2' }],
  };
}
function contextFrame(t: number, s: ContextScene): ContextFrame {
  const { p, idx } = stagePos(t, 3, CS, CG), k = Math.min(1, Math.floor(p)), u = p - k;
  const mix = (a: Pt, b: Pt): Pt => [lerp(a[0], b[0], u), lerp(a[1], b[1], u)];
  const cloud = s.cloud[k].map((c, i) => mix(c, s.cloud[k + 1][i]));
  const sheep = s.sheep[k].map((a, d) => mix(a, s.sheep[k + 1][d]));
  const c = s.cos[idx][1];
  return { mode: 'contextual', stage: idx, cloud, sheep, cosThisThese: c, phase: ['Embedding', 'Block 1', 'Block 2'][idx],
    status: `${idx ? STAGES[idx] : 'After the embedding table and position'}: the two "sheep" vectors have cosine ${fmt(c)}.` };
}

// ------------------------------------------------------------------ dispatch

export type Scene = LookupScene | TrainingScene | SimilarityScene | ProjectionScene | ArithmeticScene | ContextScene;
export type Frame = LookupFrame | TrainingFrame | SimilarityFrame | ProjectionFrame | ArithmeticFrame | ContextFrame;

export interface SceneOptions { sentence?: string; probes?: string[]; metric?: Metric; presets?: [string, string, string][]; third?: string | null }
export function buildScene(mode: Mode, data: Data, o: SceneOptions = {}): Scene {
  switch (mode) {
    case 'lookup': return lookupScene(data, o.sentence ?? DEFAULT_SENTENCE);
    case 'training': return trainingScene(data);
    case 'similarity': return similarityScene(data, o.probes ?? PROBES, o.metric ?? 'cosine');
    case 'projection': return projectionScene(data);
    case 'arithmetic': return arithmeticScene(data, o.presets ?? [PRESETS[0]]);
    case 'contextual': return contextScene(data, o.third ?? null);
  }
}
export function frameAt(t: number, s: Scene): Frame {
  const tt = Math.min(Math.max(t, 0), s.T);
  switch (s.mode) {
    case 'lookup': return lookupFrame(tt, s);
    case 'training': return trainingFrame(tt, s);
    case 'similarity': return similarityFrame(tt, s);
    case 'projection': return projectionFrame(tt, s);
    case 'arithmetic': return arithmeticFrame(tt, s);
    case 'contextual': return contextFrame(tt, s);
  }
}
