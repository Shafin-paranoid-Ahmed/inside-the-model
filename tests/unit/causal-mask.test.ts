import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseModel, decodeIds, encodeWords, type Manifest } from '../../src/lib/transformer';
import { buildScene, candidates, flowChips, frameAt, layoutOf, SENTENCE, type EditScene, type LeakScene, type NegInfScene, type ParallelScene, type Models } from '../../src/components/widgets/CausalMask/sim';

const read = (name: string) => {
  const manifest: Manifest = JSON.parse(readFileSync(`public/models/${name}.json`, 'utf8'));
  return { manifest, weights: parseModel(manifest, new Uint8Array(readFileSync(`public/models/${name}.bin`))) };
};
const c = read('tiny-lm'), f = read('tiny-lm-nomask');
const models: Models = { causal: c.weights, free: f.weights };
const curves = { masked: c.manifest.lossCurve, free: f.manifest.lossCurve };

describe('CausalMask sim', () => {
  test('mask grid: T(T+1)/2 allowed cells, counted row by row', () => {
    const s = buildScene('mask', models);
    expect(s.mode === 'mask' && s.words.length).toBe(8);
    const counts = [0, 1, 2, 3, 4, 5, 6, 7].map((r) => { const fr = frameAt(r + 0.5, s); return fr.kind === 'mask' ? fr.count : -1; });
    expect(counts).toEqual([1, 3, 6, 10, 15, 21, 28, 36]);
    const end = frameAt(s.duration, s);
    expect(end.kind === 'mask' && end.total).toBe(64);
    expect(end.kind === 'mask' && end.count).toBe(36);
  });

  test('encoder sees every word, decoder only the past', () => {
    const s = buildScene('encdec', models);
    const fr = frameAt(3.5, s);
    expect(fr.kind === 'encdec' && [fr.row, fr.enc, fr.dec]).toEqual([3, 8, 4]);
  });

  test('neginf: masked row sums to 1, zero on the future; unmasked leaks to later words', () => {
    const s = buildScene('neginf', models) as NegInfScene;
    expect(s.q).toBe(3);
    expect(s.masked.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(s.masked.slice(4)).toEqual([0, 0, 0, 0]);
    expect(s.unmasked.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(s.futureShare).toBeCloseTo(0.712, 3);
    // Masking by -inf equals renormalising the allowed unmasked weights.
    const allowed = s.unmasked.slice(0, 4), tot = allowed.reduce((a, b) => a + b, 0);
    allowed.forEach((u, i) => expect(s.masked[i]).toBeCloseTo(u / tot, 9));
    const mid = frameAt(7.4, s), end = frameAt(11, s);
    expect(mid.kind === 'neginf' && mid.sum).toBeCloseTo(1, 12);
    expect(end.kind === 'neginf' && end.sum).toBeCloseTo(tot, 9);
    expect(tot).toBeLessThan(0.3);
    const wr = frameAt(8.5, s);
    expect(wr.kind === 'neginf' && wr.sum).toBeCloseTo(tot, 9);
    expect(wr.kind === 'neginf' && wr.labels[0]).toBeCloseTo(s.unmasked[0], 12);
    expect(tot.toFixed(3)).toBe('0.288');
  });

  test('parallel: one pass gives every position; average loss and pass count', () => {
    const s = buildScene('parallel', models) as ParallelScene;
    expect(s.rows.map((r) => r.target)).toEqual(['these', 'small', 'sheep', 'chase', 'a', 'red', 'fox', '.']);
    expect(s.avg).toBeCloseTo(1.697, 3);
    expect(Math.max(...s.rows.map((r) => r.loss))).toBeCloseTo(s.rows[2].loss, 12);
    expect(s.rows[7].loss).toBeLessThan(1e-3);
    const one = frameAt(2.5, s), seq = frameAt(s.duration - 1, s);
    expect(one.kind === 'parallel' && one.passes).toBe(1);
    expect(one.kind === 'parallel' && one.avg).toBeCloseTo(s.avg, 12);
    expect(seq.kind === 'parallel' && seq.passes).toBe(8);
    expect(seq.kind === 'parallel' && seq.avg).toBeCloseTo(s.avg, 12);
  });

  test('edit: earlier outputs of the masked model are bit-identical; the unmasked model leaks', () => {
    const s = buildScene('edit', models, { editAt: 4 }) as EditScene;
    expect(s.variants.map((v) => v.word)).toEqual(['chase', 'see', 'like']);
    for (const v of s.variants) {
      expect(v.maxBefore).toBe(0);
      v.rows.slice(0, 4).forEach((r) => expect(r.change).toBe(0));
    }
    expect(s.variants[1].maxFrom).toBeGreaterThan(0.05);
    expect(s.variants[1].freeBefore).toBeGreaterThan(0.9);
    expect(s.variants[0].maxFrom).toBe(0);
    // Every editable position leaves the earlier rows untouched.
    for (let pos = 1; pos < 8; pos++) {
      const e = buildScene('edit', models, { editAt: pos }) as EditScene;
      expect(e.variants.length).toBeGreaterThan(1);
      e.variants.forEach((v) => expect(v.maxBefore).toBe(0));
    }
  });

  test('edit candidates keep the sentence grammatical', () => {
    const ids = encodeWords(SENTENCE);
    expect(candidates(ids, 3).map((i) => decodeIds([i]))).toEqual(['cats', 'dogs', 'foxes']);
  });

  test('leak: curves end far apart; the unmasked model writes an ungrammatical sentence', () => {
    const s = buildScene('leak', models, {}, curves) as LeakScene;
    expect(curves.masked.at(-1)!.loss).toBeGreaterThan(1.5);
    expect(curves.free.at(-1)!.loss).toBeLessThan(2e-4);
    expect(s.prompt.join(' ')).toBe('[start] these big birds');
    expect(s.gens.masked.words.join(' ')).toBe('chase a sheep .');
    expect(s.gens.masked.bad.some(Boolean)).toBe(false);
    expect(s.gens.free.words.join(' ')).toBe('apples .');
    expect(s.gens.free.bad[0]).toBe(true);
    const start = frameAt(0, s), end = frameAt(s.duration, s);
    expect(start.kind === 'leak' && start.step).toBe(1);
    expect(end.kind === 'leak' && end.step).toBe(6000);
    expect(end.kind === 'leak' && end.shown).toEqual({ masked: 4, free: 2 });
  });

  test('frameAt is deterministic and layouts are fixed per mode', () => {
    for (const m of ['mask', 'encdec', 'neginf', 'parallel', 'edit'] as const) {
      const s = buildScene(m, models);
      expect(frameAt(1.234, s)).toEqual(frameAt(1.234, s));
      expect(layoutOf(m, true).w).toBe(288);
    }
    const { chips, lines } = flowChips(['a', 'bb', 'ccc'], 60, 7, 24);
    expect(lines).toBe(2);
    expect(chips[0].y).toBe(0);
  });
});
