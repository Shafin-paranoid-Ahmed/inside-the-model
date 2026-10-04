import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { createRng } from '../../src/lib/math/rng';
import { TINY_CONFIG, initWeights, parseModel, forward, encodeWords, type Manifest } from '../../src/lib/transformer';
import { residualGradientProbe } from '../../src/lib/transformer/report';
import {
  NORM_PRE, NORM_POST, NORM_SEED, PRESETS, approxParams, buildData, countParams, frameAt, norm2, rowOf, spread, timelineOf, tinyCounts, topK, probOf,
  type Mode,
} from '../../src/components/widgets/BlockFlow/sim';

const manifest: Manifest = JSON.parse(readFileSync('public/models/tiny-lm.json', 'utf8'));
const w = parseModel(manifest, new Uint8Array(readFileSync('public/models/tiny-lm.bin')));
const data = buildData(w);

describe('BlockFlow sim', () => {
  test('stream stages end at the real model output and the lens uses the real trace', () => {
    const f = forward(w, encodeWords('[start] these small sheep chase a red fox'));
    expect(data.stages).toHaveLength(5);
    const last = data.stages[4].probs;
    for (let i = 0; i < 34; i++) expect(Math.abs(last[i] - f.probs[3 * 34 + i])).toBeLessThan(1e-12);
    expect(data.words[data.q]).toBe('sheep');
    expect(topK(last, 1)[0].p).toBeGreaterThan(0.2);
    expect(probOf(data.stages[4].probs, 'chase')).toBeGreaterThan(probOf(data.stages[4].probs, 'chases'));
  });
  test('residual sums and active counts come from the trace', () => {
    for (const B of data.blocks) for (let t = 0; t < data.T; t++) {
      const x = rowOf(B.input, t, 16), a = rowOf(B.attn, t, 16), s = rowOf(B.r1, t, 16);
      for (let j = 0; j < 16; j++) expect(s[j]).toBeCloseTo(x[j] + a[j], 12);
      const pre = rowOf(B.pre, t, 64), h = rowOf(B.hidden, t, 64);
      for (let j = 0; j < 64; j++) expect(h[j]).toBeCloseTo(Math.max(0, pre[j]), 9);
      expect(B.active[t]).toBe(Array.from(h).filter((v) => v > 0).length);
    }
    // quoted in the lesson (block 2 for the residual step, block 1 for the MLP step)
    expect(norm2(rowOf(data.blocks[1].input, 3, 16))).toBeCloseTo(12.391, 2);
    expect(norm2(rowOf(data.blocks[1].attn, 3, 16))).toBeCloseTo(8.422, 2);
    expect(data.blocks[0].active[3]).toBe(24); expect(data.blocks[0].active[7]).toBe(30);
  });
  test('gradient table equals the manifest and a fresh random-init 12-block probe', () => {
    const m = manifest.metrics.randomInit12BlockGradientNorms as { pre: number[]; post: number[] };
    expect(NORM_PRE).toEqual(m.pre); expect(NORM_POST).toEqual(m.post);
    const fresh = residualGradientProbe(initWeights({ ...TINY_CONFIG, layers: 12 }, createRng(NORM_SEED)));
    fresh.pre.forEach((v, i) => expect(v).toBeCloseTo(NORM_PRE[i], 10));
    fresh.post.forEach((v, i) => expect(v).toBeCloseTo(NORM_POST[i], 10));
    expect(spread(NORM_PRE)).toBeLessThan(spread(NORM_POST));
  });
  test('parameter counts: exact TinyLM, GPT-2 small and XL', () => {
    expect(tinyCounts().total).toBe(7330); expect(tinyCounts().total).toBe(manifest.parameterCount);
    expect(countParams(PRESETS[0])).toEqual(tinyCounts());
    expect(countParams(PRESETS[1]).total).toBe(124_439_808);
    expect(countParams(PRESETS[2]).total).toBe(1_557_611_200);
    expect(approxParams(PRESETS[1])).toBe(123_532_032);
    const a = countParams({ ...PRESETS[1] }).perBlock, b = countParams({ ...PRESETS[1], d: 1536 }).perBlock;
    expect(b / a).toBeGreaterThan(3.9);
  });
  test('stream numbers snap to the real stage while bars ease', () => {
    const fr = frameAt(8.3, 'stream', data);
    if (fr.mode !== 'stream') throw new Error('mode');
    expect(fr.rows[4].chase).toBe(probOf(data.stages[4].probs, 'chase'));
    expect(fr.top.map((e) => e.p)).toEqual(topK(data.stages[4].probs, 3).map((e) => e.p));
    expect(fr.rows[4].chaseBar).not.toBe(fr.rows[4].chase);
  });
  const modes: Mode[] = ['overview', 'residual', 'mlp', 'norm', 'stream', 'count'];
  test('frames are deterministic and defined at both ends of every timeline', () => {
    for (const mode of modes) {
      const { duration, markers } = timelineOf(mode);
      for (const t of [0, duration / 3, duration - 1e-6]) {
        const a = frameAt(t, mode, data), b = frameAt(t, mode, data);
        expect(a).toEqual(b); expect(a.status.length).toBeGreaterThan(10);
      }
      expect(markers[0].at).toBe(0);
    }
    const end = frameAt(timelineOf('count').duration - 1e-6, 'count', null);
    expect(end.mode === 'count' && end.counts.total).toBe(countParams(PRESETS[2]).total);
    const norm = frameAt(7, 'norm', null);
    expect(norm.mode === 'norm' && norm.reveal).toBe(13);
  });
});
