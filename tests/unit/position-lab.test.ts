import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseModel, type Manifest } from '../../src/lib/transformer';
import * as S from '../../src/components/widgets/PositionLab/sim';

const w = parseModel(JSON.parse(readFileSync('public/models/tiny-lm.json', 'utf8')) as Manifest, new Uint8Array(readFileSync('public/models/tiny-lm.bin')));
const modes: S.Mode[] = ['shuffle', 'ablate', 'sinusoidal', 'learned', 'rope', 'relative'];

describe('PositionLab sim', () => {
  test('shuffling words only reorders attention outputs', () => {
    for (const seed of [0, 1, 2]) {
      const d = S.makeShuffle(w, seed);
      expect(d.maxDiff).toBeLessThan(1e-12);
      expect([...d.perm].sort()).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
      const f = S.shuffleFrameAt(0, d), g = S.shuffleFrameAt(4.5, d);
      expect(f.rows.every((r) => r.yR === r.i)).toBe(true);
      expect(g.rows.every((r) => r.yR === d.slotOf[r.i])).toBe(true);
    }
  });
  test('removing positions reproduces the report numbers', () => {
    const a = S.makeAblate(w);
    expect(a.sampleWith).toBeCloseTo(0.997887330707, 9);
    expect(a.sampleWithout).toBeCloseTo(0.869001, 6);
    expect(S.ablateFrameAt(0, a).mass).toEqual(a.withPos);
    expect(S.ablateFrameAt(7, a).mass).toEqual(a.without);
    expect(a.without[3]).toBeLessThan(0.01);
    expect(a.without[7]).toBeLessThan(0.01);
  });
  test('sinusoidal wavelengths and table', () => {
    expect(S.wavelength(0)).toBeCloseTo(2 * Math.PI, 12);
    expect(S.wavelength(7)).toBeCloseTo(2 * Math.PI * 10000 ** (14 / 16), 6);
    expect(S.sinPE(0)[0]).toBe(0);
    expect(S.sinPE(0)[1]).toBe(1);
  });
  test('learned cosines match the report; position 9 has no summary', () => {
    const d = S.makeLearned(w);
    expect(d.cos[0][1]).toBeCloseTo(-0.489122, 5);
    expect(d.cos[0][9]).toBeCloseTo(-0.076716, 5);
    expect(S.learnedFrameAt(9.5, d).neighbour).toBeNaN();
    expect(S.learnedFrameAt(10.5, d).row).toBe(-1);
  });
  test('rope score depends only on the gap', () => {
    const base = S.ropeFrameAt(0).score;
    for (let m = 2; m < 30; m += 3) expect(S.ropeFrameAt(0, { m, n: m - 2 }).score).toBeCloseTo(base, 10);
    expect(S.ropeFrameAt(0, { m: 3, n: 0 }).score).not.toBeCloseTo(base, 3);
  });
  test('relative grid: RoPE constant along diagonals, additive is not', () => {
    const d = S.makeRelative();
    for (const c of [-7, 0, 4]) expect(S.spread(S.diagonal(d.rope, c))).toBeLessThan(1e-10);
    expect(S.spread(S.diagonal(d.additive, 0))).toBeGreaterThan(1);
  });
  test('timelines: every frame is defined at the ends and markers are inside', () => {
    for (const m of modes) {
      const T = S.durationOf(m);
      expect(T).toBeGreaterThan(0);
      for (const k of S.markersOf(m)) { expect(k.at).toBeGreaterThanOrEqual(0); expect(k.at).toBeLessThan(T); }
    }
    const sd = S.makeSin(), rd = S.makeRelative();
    for (const t of [0, S.durationOf('sinusoidal') - 0.01]) expect(S.sinFrameAt(t, sd).index).toBeLessThan(32);
    expect(S.relFrameAt(0, rd).c).toBe(-15);
    expect(S.relFrameAt(S.durationOf('relative') - 0.01, rd).c).toBe(15);
  });
});
