import { describe, expect, it } from 'vitest';
import { APPLY_TEXT, CHAR_VOCAB, CORPUS, DEFAULT_TEXT, HELD_OUT, MAX_MERGES, PLAYGROUND_TEXT, QUIRKS, WORD_VOCAB, bytesOf, countPairs, decode, defaultModel, durationOf, encode, encodingSteps, frameAt, layoutChips, markersOf, mergeIds, tokenLabel, trainBpe, vocabCurve, wordPieces, type Mode } from '../../src/components/widgets/TokenizerLab/sim';

describe('byte-level BPE', () => {
  it('counts overlapping neighbours and replaces non-overlapping pairs left to right', () => {
    expect(countPairs(bytesOf('aaab aab')).find((p) => p.left === 97 && p.right === 97)?.count).toBe(3);
    expect(mergeIds(bytesOf('aaab aab'), 97, 97, 256)).toEqual([256, 97, 98, 32, 256, 98]);
  });
  it('breaks frequency ties by expanded byte order, independent of encounter order', () => {
    const a = trainBpe(' cd ab cd ab ', 1), b = trainBpe(' ab cd ab cd ', 1);
    expect(a.vocab[256]).toEqual([32, 97]);
    expect(b.vocab[256]).toEqual(a.vocab[256]);
  });
  it('handles empty input, one byte, exhausted pairs and invalid merge counts', () => {
    expect(trainBpe('', 12).merges).toEqual([]);
    expect(trainBpe('a', 12).merges).toEqual([]);
    expect(trainBpe('ab', 12).merges).toEqual([]);
    expect(() => trainBpe('abc', -1)).toThrow(RangeError);
    expect(() => trainBpe('abc', 1.5)).toThrow(RangeError);
    expect(encode('')).toEqual([]);
  });
  it('retains all 256 bytes and only combines previously defined tokens', () => {
    const m = defaultModel();
    expect(m.merges).toHaveLength(MAX_MERGES);
    expect(m.vocab.slice(0, 256)).toEqual(Array.from({ length: 256 }, (_, i) => [i]));
    for (const merge of m.merges) {
      expect(merge.left).toBeLessThan(merge.id);
      expect(merge.right).toBeLessThan(merge.id);
      expect(m.vocab[merge.id]).toEqual([...m.vocab[merge.left], ...m.vocab[merge.right]]);
    }
  });
  it('reproduces the full learned model deterministically', () => {
    expect(trainBpe(CORPUS, MAX_MERGES)).toEqual(defaultModel());
  });
  it('preserves arbitrary valid Unicode, whitespace, punctuation and case', () => {
    for (const sample of [CORPUS, HELD_OUT, DEFAULT_TEXT, 'পানি', '🌱🙂', 'café e\u0301 中文', '\n\t  A\0B', ...QUIRKS.flat()]) {
      for (const limit of [0, 12, 50, MAX_MERGES]) expect(decode(encode(sample, defaultModel(), limit))).toBe(sample);
    }
  });
  it('encodes every Unicode scalar in a generated multilingual range without unknown IDs', () => {
    for (let cp = 0; cp < 0x10ffff; cp += 7919) {
      if (cp >= 0xd800 && cp <= 0xdfff) continue;
      const text = String.fromCodePoint(cp);
      expect(decode(encode(text))).toBe(text);
    }
  });
  it('uses hex for partial UTF-8 bytes and decodes only after concatenation', () => {
    expect(tokenLabel([0xe0])).toBe('E0');
    expect(tokenLabel(bytesOf('পানি'))).toBe('পানি');
    expect(tokenLabel(bytesOf(' \n\t'))).toBe('·↵⇥');
    expect(() => decode([9999])).toThrow(RangeError);
    expect(() => decode([1.5])).toThrow(RangeError);
  });
  it('matches an independent lowest-rank encoder', () => {
    const m = defaultModel();
    for (const text of [CORPUS, HELD_OUT, APPLY_TEXT, DEFAULT_TEXT, ...QUIRKS.flat()]) {
      let ids = bytesOf(text);
      while (true) {
        const next = m.merges.find((merge) => ids.some((id, i) => id === merge.left && ids[i + 1] === merge.right));
        if (!next) break;
        ids = mergeIds(ids, next.left, next.right, next.id);
      }
      expect(encode(text, m)).toEqual(ids);
    }
  });
  it('pins the prose examples to their measured values', () => {
    const m = defaultModel();
    expect(m.merges.slice(0, 3).map((s) => [tokenLabel(m.vocab[s.left]), tokenLabel(m.vocab[s.right]), s.count])).toEqual([
      ['e', '·', 126], ['t', 'h', 119], ['th', 'e·', 113],
    ]);
    expect([Array.from(DEFAULT_TEXT).length, wordPieces(DEFAULT_TEXT).length, encode(DEFAULT_TEXT).length]).toEqual([36, 6, 12]);
    expect([CHAR_VOCAB.length, WORD_VOCAB.length + 1, m.vocab.length]).toEqual([29, 120, 416]);
    expect(wordPieces(DEFAULT_TEXT).filter((p) => p === '[unknown]')).toHaveLength(2);
    expect(new RegExp(`\\b${APPLY_TEXT}\\b`).test(CORPUS)).toBe(false);
    expect(encode(APPLY_TEXT)).toEqual([272, 262, 263]);
    expect(encodingSteps(APPLY_TEXT, m).slice(1).map((s) => s.merge!.rank + 1)).toEqual([6, 7, 8, 14, 17]);
    expect(encode(PLAYGROUND_TEXT)).toHaveLength(3);
    expect(QUIRKS.flat().map((s) => encode(s).length)).toEqual([2, 3, 3, 2, 5, 9, 1, 12]);
  });
  it('computes the held-out curve from actual encodings and only lowers sequence length', () => {
    const curve = vocabCurve();
    for (const p of curve) {
      expect(p.tokens).toBe(encode(HELD_OUT, defaultModel(), p.merges).length);
      expect(p.vocab).toBe(256 + p.merges);
      expect(p.per100).toBe(100 * p.tokens / Array.from(HELD_OUT).length);
      if (p.merges) expect(p.tokens).toBeLessThanOrEqual(curve[p.merges - 1].tokens);
    }
    expect([curve[0].tokens, curve[50].tokens, curve[100].tokens, curve[160].tokens]).toEqual([114, 77, 73, 65]);
  });
});

describe('tokenizer timeline and chip layout', () => {
  const modes: Mode[] = ['granularity', 'merges', 'apply', 'playground', 'vocab', 'quirks'];
  for (const mode of modes) {
    it(`${mode}: deterministic, finite, seekable endpoints and matching segment labels`, () => {
      const text = mode === 'apply' ? APPLY_TEXT : DEFAULT_TEXT;
      const duration = durationOf(mode, text);
      const markers = markersOf(mode, text);
      for (const t of [0, duration * 0.25, duration * 0.6, duration]) {
        const f = frameAt(t, mode, text);
        expect(f).toEqual(frameAt(t, mode, text));
        expect(f.progress).toBeGreaterThanOrEqual(0);
        expect(f.progress).toBeLessThanOrEqual(1);
        const segment = markers.filter((m) => m.at <= t).at(-1)!;
        expect(f.phase).toBe(segment.label);
        for (const r of f.rows) {
          expect(r.total).toBe(r.chips.length);
          expect(r.revealed).toBeLessThanOrEqual(r.total);
          for (const w of [288, 720]) {
            const layout = layoutChips(r, w, 20);
            for (const chip of layout.chips) {
              expect(chip.x).toBeGreaterThanOrEqual(12);
              expect(chip.x + chip.width).toBeLessThanOrEqual(w - 12);
              expect(chip.y + 39).toBeLessThanOrEqual(layout.bottom);
            }
          }
        }
      }
      expect(frameAt(-1, mode, text)).toEqual(frameAt(0, mode, text));
      expect(frameAt(Infinity, mode, text)).toEqual(frameAt(0, mode, text));
      expect(frameAt(duration + 1, mode, text)).toEqual(frameAt(duration, mode, text));
    });
  }
  it('keeps merge counts and visible states synchronized at the join boundary', () => {
    expect(frameAt(1.64, 'merges', DEFAULT_TEXT).completed).toBe(0);
    expect(frameAt(1.66, 'merges', DEFAULT_TEXT).completed).toBe(1);
    expect(frameAt(1.66, 'merges', DEFAULT_TEXT).rows[1].total).toBeLessThan(frameAt(1.64, 'merges', DEFAULT_TEXT).rows[1].total);
    expect(frameAt(36, 'merges', DEFAULT_TEXT).completed).toBe(12);
    expect(frameAt(durationOf('apply', APPLY_TEXT), 'apply', APPLY_TEXT).rows[1].chips.map((c) => c.id)).toEqual(encode(APPLY_TEXT));
    for (let index = 0; index < 12; index++) {
      const f = frameAt(index * 3 + 2, 'merges', DEFAULT_TEXT);
      expect(f.rows[0].chips.some((c) => c.highlight)).toBe(true);
      expect(f.rows[1].total).toBeLessThan(f.rows[0].total);
      expect(f.rows[1].chips.some((c) => c.id === f.pair!.id)).toBe(true);
    }
  });
  it('bounds long Unicode chip drawings but keeps exact totals', () => {
    const f = frameAt(12, 'playground', '🙂'.repeat(120));
    expect(f.rows[0].total).toBe(480);
    const layout = layoutChips(f.rows[0], 288, 20);
    expect(layout.chips).toHaveLength(42);
    expect(layout.hidden).toBe(438);
  });
  it('never overlaps token rectangles during any of the twelve merges', () => {
    for (const width of [288, 720]) for (let index = 0; index < 12; index++) {
      for (const offset of [0, 1.64, 1.65, 1.66, 1.8, 2, 2.54, 2.99]) {
        const f = frameAt(index * 3 + offset, 'merges', DEFAULT_TEXT);
        for (const [ri, row] of f.rows.entries()) {
          const layout = layoutChips(row, width, 0, 42, ri === 1 ? f.join : 1);
          for (let i = 0; i < layout.chips.length; i++) for (let j = i + 1; j < layout.chips.length; j++) {
            const a = layout.chips[i], b = layout.chips[j];
            const overlap = a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + 39 && b.y < a.y + 39;
            expect(overlap).toBe(false);
          }
        }
      }
    }
  });
  it('keeps merge excerpts compact and the last row height constant', () => {
    for (const width of [288, 720]) {
      const heights = new Set<number>();
      for (let index = 0; index < 12; index++) for (const offset of [0, 2.8]) {
        const f = frameAt(index * 3 + offset, 'merges', DEFAULT_TEXT);
        expect(f.rows[0].total).toBe(8);
        expect(f.rows[1].total).toBe(offset === 0 ? 8 : 7);
        heights.add(layoutChips(f.rows[1], width, 0).bottom);
      }
      expect(heights.size).toBe(1);
    }
  });
});
