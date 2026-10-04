/** Byte-level BPE. Token ids 0..255 are literal bytes; learned ids start at 256. */
export type Mode = 'granularity' | 'merges' | 'apply' | 'playground' | 'vocab' | 'quirks';
export const MAX_MERGES = 160;
export const DEFAULT_TEXT = 'the unhappiest foxes outran the dogs';
export const PLAYGROUND_TEXT = 'the water is low';
export const APPLY_TEXT = 'lowering';
// Original English teaching corpus, not model-training text. Repeated words give visible pair counts.
export const CORPUS = `the low dog sees the lower fox. the lowest bird sees the slow dog.
the dogs see the foxes. the foxes see the birds. the birds see the dogs.
the water is low. the water is lower. the water is lowest in the old well.
the happy dog is running. the happier fox is walking. the happiest bird is singing.
the small cat likes the red apple. the old dog likes the big apple.
this small sheep eats the seeds. these old sheep eat the seeds.
the farmer is planting and watering. the gardener is working and singing.
the runner is running slowly. the walker is walking slowly. the bird is flying.
in 2026 the farm has 20 dogs and 26 birds. in 2020 the farm had 20 cats.
the low wall is lower than the gate. the lowest gate is near the water.
the water flows under the lower wall. the dog walks along the low wall.
the foxes ran past the dogs. the dogs ran past the birds. the birds flew past the cats.
the happy farmer brings water to the small garden. the garden has red apples.
the old farmer brings seeds to the birds. the birds like seeds and water.
the lower field has water. the lowest field has more water. the low field has grass.
the small fox is hiding. the old dog is waiting. the red bird is singing.
the cat is sleeping near the water. the sheep is eating near the low wall.
the gardener is planting seeds and picking apples. the farmer is watering the garden.
the water is cold in the morning. the water is warm in the evening.
the dogs are happy in the garden. the foxes are happier in the field.
the birds are happiest near the water. the cats are happy near the wall.
the low dog sees the small fox. the lower wall hides the old cat.
the lowest branch holds the red bird. the slow farmer brings the water.
the children are walking and talking. the children are working and playing.
the farmer is reading in the garden. the gardener is writing near the field.
the water is low in the well. the water is lower in the stream.
the lowest path runs along the water. the lower path runs past the garden.
the fox likes water and apples. the dog likes water and seeds.
in 2026 the children planted 20 seeds. in 2020 they planted 26 seeds.
the dog is running to the garden. the fox is walking to the field.
the bird is flying to the lower branch. the cat is sleeping by the low wall.
the happy children bring water. the happier farmer brings apples.
the happiest gardener brings seeds to the birds near the lowest wall.`;
export const HELD_OUT = 'a fox drinks water beside a quiet stream. children walk below an open window. the garden grows greener after rain.';
export const QUIRKS = [['the', ' the', 'The'], ['2026', '12345', '1,234,567'], ['water', 'পানি']];
const utf8 = new TextEncoder();
const textDecoder = new TextDecoder('utf-8', { fatal: true });
export interface Pair { left: number; right: number; count: number }
export interface Merge extends Pair { id: number; rank: number; top: Pair[] }
export interface Bpe { vocab: number[][]; merges: Merge[] }
export const bytesOf = (text: string): number[] => Array.from(utf8.encode(text));

export function mergeIds(ids: readonly number[], left: number, right: number, id: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < ids.length; i++) {
    if (ids[i] === left && ids[i + 1] === right) { out.push(id); i++; }
    else out.push(ids[i]);
  }
  return out;
}

export function countPairs(ids: readonly number[]): Pair[] {
  const pairs = new Map<string, Pair>();
  for (let i = 0; i + 1 < ids.length; i++) {
    const key = `${ids[i]},${ids[i + 1]}`;
    const found = pairs.get(key);
    if (found) found.count++;
    else pairs.set(key, { left: ids[i], right: ids[i + 1], count: 1 });
  }
  return [...pairs.values()];
}

function compareBytes(a: readonly number[], b: readonly number[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

/** Equal counts: compare the left token's bytes, then the right token's bytes. */
export function trainBpe(corpus: string, maxMerges: number): Bpe {
  if (!Number.isInteger(maxMerges) || maxMerges < 0) throw new RangeError('Merge count must be a nonnegative integer');
  const vocab = Array.from({ length: 256 }, (_, i) => [i]);
  const merges: Merge[] = [];
  let ids = bytesOf(corpus);
  for (let rank = 0; rank < maxMerges; rank++) {
    const pairs = countPairs(ids).sort((a, b) => b.count - a.count ||
      compareBytes(vocab[a.left], vocab[b.left]) || compareBytes(vocab[a.right], vocab[b.right]));
    if (!pairs.length || pairs[0].count < 2) break;
    const winner = pairs[0], id = vocab.length;
    vocab.push([...vocab[winner.left], ...vocab[winner.right]]);
    merges.push({ ...winner, id, rank, top: pairs.slice(0, 4) });
    ids = mergeIds(ids, winner.left, winner.right, id);
  }
  return { vocab, merges };
}

export interface EncodingStep { ids: number[]; merge?: Merge }
export function encodingSteps(text: string, model: Bpe, limit = model.merges.length): EncodingStep[] {
  let ids = bytesOf(text);
  const steps: EncodingStep[] = [{ ids }];
  for (const m of model.merges.slice(0, Math.max(0, Math.floor(limit)))) {
    const next = mergeIds(ids, m.left, m.right, m.id);
    if (next.length !== ids.length) { ids = next; steps.push({ ids, merge: m }); }
  }
  return steps;
}
/** Public tokenizer for later lessons: deterministic ids, no unknown-token fallback. */
export function encode(text: string, model: Bpe = defaultModel(), limit = model.merges.length): number[] {
  return encodingSteps(text, model, limit).at(-1)!.ids;
}
export function decode(ids: readonly number[], model: Bpe = defaultModel()): string {
  const bytes: number[] = [];
  for (const id of ids) {
    if (!Number.isInteger(id) || !model.vocab[id]) throw new RangeError(`Unknown token id ${id}`);
    bytes.push(...model.vocab[id]);
  }
  return textDecoder.decode(new Uint8Array(bytes));
}
/** Partial UTF-8 tokens are hex, never replacement characters masquerading as text. */
export function tokenLabel(bytes: readonly number[]): string {
  try {
    return textDecoder.decode(new Uint8Array(bytes)).replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\t', '⇥');
  } catch { return bytes.map((b) => b.toString(16).toUpperCase().padStart(2, '0')).join(' '); }
}
let cached: Bpe | undefined;
export const defaultModel = (): Bpe => cached ??= trainBpe(CORPUS, MAX_MERGES);
export const CHAR_VOCAB = [...new Set(Array.from(CORPUS))];
export const WORD_VOCAB = [...new Set(CORPUS.match(/\S+/g))];
export const wordPieces = (text: string): string[] => (text.match(/\S+/g) ?? []).map((word) => WORD_VOCAB.includes(word) ? word : '[unknown]');
export interface CurvePoint { merges: number; vocab: number; tokens: number; per100: number }
let cachedCurve: CurvePoint[] | undefined;
export function vocabCurve(): CurvePoint[] {
  if (cachedCurve) return cachedCurve;
  let ids = bytesOf(HELD_OUT);
  const points = [{ merges: 0, vocab: 256, tokens: ids.length, per100: 100 * ids.length / Array.from(HELD_OUT).length }];
  for (const m of defaultModel().merges) {
    ids = mergeIds(ids, m.left, m.right, m.id);
    points.push({ merges: m.rank + 1, vocab: m.id + 1, tokens: ids.length, per100: 100 * ids.length / Array.from(HELD_OUT).length });
  }
  return cachedCurve = points;
}

export interface Chip { label: string; id?: number; bytes: number[]; highlight?: boolean }
export interface Row { label: string; chips: Chip[]; total: number; revealed: number; visibility: number; detail: string }
export interface Frame {
  phase: string; status: string; rows: Row[]; progress: number;
  pair?: Merge; completed: number; curve?: CurvePoint; pointIndex?: number; join: number;
}
const chip = (id: number, model: Bpe, highlight = false): Chip => ({ id, bytes: model.vocab[id], label: tokenLabel(model.vocab[id]), highlight });
const row = (label: string, chips: Chip[], detail: string, fraction = 1): Row => ({
  label, chips, total: chips.length, revealed: Math.ceil(chips.length * Math.min(1, Math.max(0, fraction))), detail,
  visibility: chips.length * Math.min(1, Math.max(0, fraction)),
});
export function durationOf(mode: Mode, text: string): number {
  if (mode === 'merges') return 36;
  if (mode === 'apply') return (encodingSteps(text, defaultModel()).length + 1) * 3;
  return mode === 'granularity' || mode === 'quirks' ? 18 : 12;
}
export function markersOf(mode: Mode, text: string): { at: number; label: string }[] {
  if (mode === 'merges') return Array.from({ length: 12 }, (_, i) => ({ at: i * 3, label: `Merge ${i + 1}` }));
  if (mode === 'apply') return encodingSteps(text, defaultModel()).map((s, i) => ({ at: i * 3, label: s.merge ? `Rank ${s.merge.rank + 1}` : 'Bytes' })).concat([{ at: (encodingSteps(text, defaultModel()).length) * 3, label: 'IDs' }]);
  if (mode === 'granularity') return ['Characters', 'Words', 'Subwords'].map((label, i) => ({ at: i * 6, label }));
  if (mode === 'quirks') return ['Spaces', 'Numbers', 'Bengali'].map((label, i) => ({ at: i * 6, label }));
  if (mode === 'vocab') return [0, 50, 100, defaultModel().merges.length].map((n, i, stops) => ({ at: n / defaultModel().merges.length * 12, label: `${n}${i + 1 < stops.length ? `–${stops[i + 1] - 1}` : ''} merges` }));
  return [{ at: 0, label: 'Read text' }, { at: 6, label: 'Read IDs' }];
}
export function frameAt(t: number, mode: Mode, text: string): Frame {
  const model = defaultModel(), duration = durationOf(mode, text);
  const time = Math.max(0, Math.min(duration, Number.isFinite(t) ? t : 0));
  const progress = time / duration;
  const result: Frame = { phase: '', status: '', rows: [], progress, completed: 0, join: 1 };
  if (mode === 'granularity') {
    const stage = Math.min(2, Math.floor(time / 6));
    const all = [Array.from(text).map((c) => ({ label: tokenLabel(bytesOf(c)), bytes: bytesOf(c) })),
      wordPieces(text).map((c) => ({ label: c, bytes: bytesOf(c), highlight: c === '[unknown]' })), encode(text, model).map((id) => chip(id, model))];
    const labels = ['Characters', 'Words', 'Subwords'];
    result.rows = all.map((chips, i) => row(labels[i], chips, `${[CHAR_VOCAB.length, WORD_VOCAB.length + 1, model.vocab.length][i]} vocabulary entries${i === 1 ? ' (includes unknown)' : ''}`, i < stage ? 1 : i > stage ? 0 : (time - stage * 6) / 4));
    result.phase = labels[stage];
    result.status = `${labels[stage]}: ${result.rows[stage].revealed} of ${result.rows[stage].total} pieces. The word row cannot spell unseen words.`;
  } else if (mode === 'merges' || mode === 'apply') {
    const index = mode === 'merges' ? Math.min(11, Math.floor(time / 3)) : Math.min(encodingSteps(text, model).length, Math.floor(time / 3));
    const local = (time - index * 3) / 3;
    let before: number[], after: number[], m: Merge | undefined;
    if (mode === 'merges') {
      const selected = model.merges[index];
      m = selected;
      const corpusIds = encode(CORPUS, model, index);
      const hit = corpusIds.findIndex((id, i) => id === selected.left && corpusIds[i + 1] === selected.right);
      // Excerpt boundaries follow existing tokens, so every displayed merge has a real match.
      const start = Math.max(0, Math.min(hit - 3, corpusIds.length - 8));
      before = corpusIds.slice(start, start + 8);
      after = mergeIds(before, m.left, m.right, m.id);
      result.completed = index + (local >= 0.55 ? 1 : 0);
      result.phase = `Merge ${index + 1}`;
    } else {
      const steps = encodingSteps(text, model);
      const current = steps[Math.min(index, steps.length - 1)];
      m = index > 0 && index < steps.length ? current.merge : undefined;
      before = steps[Math.max(0, Math.min(index - 1, steps.length - 1))].ids;
      after = current.ids;
      result.completed = Math.max(0, Math.min(steps.length - 1, index - (local < 0.55 && !!m ? 1 : 0)));
      result.phase = index === 0 ? 'Bytes' : index === steps.length ? 'IDs' : `Rank ${m!.rank + 1}`;
    }
    result.pair = m;
    result.join = m ? Math.min(1, Math.max(0, (local - 0.55) / 0.3)) : 1;
    result.rows = [row(mode === 'merges' ? 'Corpus excerpt' : m ? 'Before the rule' : 'Input pieces', before.map((id, i) => chip(id, model, !!m && (id === m.left && before[i + 1] === m.right || id === m.right && before[i - 1] === m.left))), m ? 'Before this merge' : 'Each piece has its own ID'),
      row(m ? 'After the merge' : 'Current encoding', (local >= 0.55 || !m ? after : before).map((id) => chip(id, model, id === m?.id)), local >= 0.55 || !m ? 'These pieces recover the input' : 'Highlighted neighbours will join')];
    result.status = m ? `${result.phase}: ${tokenLabel(model.vocab[m.left])} + ${tokenLabel(model.vocab[m.right])} → ${tokenLabel(model.vocab[m.id])}, ID ${m.id}. ${mode === 'merges' ? `${m.count} adjacent pairs in the full corpus; vocabulary ${256 + result.completed}.` : `${result.rows[1].total} pieces after ${result.completed} applied merges.`}` : `${result.phase}: ${after.length} pieces. IDs: ${after.join(', ')}.`;
  } else if (mode === 'playground') {
    const ids = encode(text, model), chars = Array.from(text).length;
    result.phase = time < 6 ? 'Read text' : 'Read IDs';
    result.rows = [row('Tokens sent to the model', ids.map((id) => chip(id, model)), `${chars} characters · ${ids.length} token${ids.length === 1 ? '' : 's'} · ${ids.length ? (chars / ids.length).toFixed(2) : '0.00'} characters/token`, time / 8)];
    result.status = `${result.rows[0].revealed} of ${ids.length} tokens revealed. · means a space. Hex labels mean part of a UTF-8 character.`;
  } else if (mode === 'quirks') {
    const group = Math.min(2, Math.floor(time / 6));
    result.phase = ['Spaces', 'Numbers', 'Bengali'][group];
    result.rows = QUIRKS[group].map((sample) => row(sample.startsWith(' ') ? '·the' : sample, encode(sample, model).map((id) => chip(id, model)), `${Array.from(sample).length} Unicode code points · ${bytesOf(sample).length} UTF-8 bytes`, Math.min(1, (time - group * 6) / 3 + 0.15)));
    result.status = `${result.phase}: ${result.rows.map((r) => `${r.label} = ${r.total} token${r.total === 1 ? '' : 's'}`).join('; ')}.`;
  } else {
    const curve = vocabCurve(), index = Math.min(curve.length - 1, Math.floor(progress * (curve.length - 1)));
    result.curve = curve[index]; result.pointIndex = index; result.completed = index;
    result.phase = markersOf(mode, text).filter((m) => m.at <= time).at(-1)!.label;
    result.status = `${index} merges: ${result.curve.vocab} vocabulary entries, ${result.curve.tokens} tokens on ${Array.from(HELD_OUT).length} held-out characters (${result.curve.per100.toFixed(1)} per 100). Embedding rows × 16 = ${result.curve.vocab * 16}; rows × 768 = ${(result.curve.vocab * 768).toLocaleString('en-US')} parameters.`;
  }
  return result;
}

export interface PositionedChip extends Chip { x: number; y: number; width: number; opacity: number }
/** Wrapped SVG chips; hide long-input tails, while the total/IDs remain complete below. */
export function layoutChips(row: Row, width: number, y: number, maxShown = 42, join = 1): { chips: PositionedChip[]; bottom: number; hidden: number } {
  let x = 12, line = y;
  const chips = row.chips.slice(0, maxShown).map((c, i) => {
    const w = Math.min(width - 24, Math.max(c.id === undefined ? 28 : 60, Math.min(16, c.label.length) * 7 + 16));
    if (x + w > width - 12) { x = 12; line += 47; }
    const opacity = Math.max(0.12, Math.min(1, row.visibility - i)) * (c.highlight ? 0.25 + 0.75 * Math.min(1, Math.max(0, join)) : 1);
    const out = { ...c, x, y: line, width: w, opacity };
    x += w + 5;
    return out;
  });
  return { chips, bottom: line + 47, hidden: Math.max(0, row.total - maxShown) };
}
