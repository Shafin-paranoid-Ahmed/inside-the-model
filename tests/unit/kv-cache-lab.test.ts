import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseModel, forward } from '../../src/lib/transformer';
import { macsForward } from '../../src/lib/lm/cost';
import * as S from '../../src/components/widgets/KVCacheLab/sim';

const w = parseModel(JSON.parse(readFileSync('public/models/house.json', 'utf8')), new Uint8Array(readFileSync('public/models/house.bin')));
const run = await S.buildRun(w);

describe('KVCacheLab sim', () => {
  test('cached and uncached logits agree at every step; the run is deterministic', async () => {
    expect(run.P).toBe(16);
    expect(run.ids.length).toBe(64);
    expect(Math.max(...run.diff)).toBeLessThan(1e-9);
    expect(S.textOf(run).startsWith(S.PROMPT)).toBe(true);
    expect((await S.buildRun(w)).ids).toEqual(run.ids);
  });
  test('stored cache rows equal the keys of a full forward pass', () => {
    const tr = forward(w, run.ids.slice(0, 63)), d = w.config.d, dk = d / w.config.heads;
    for (let l = 0; l < 2; l++) for (const r of [0, 15, 40, 62]) for (let j = 0; j < d; j++)
      expect(Math.abs(run.keys[l][r * d + j] - tr.layers[l].k[Math.floor(j / dk)][r * dk + (j % dk)])).toBeLessThan(1e-9);
  });
  test('work totals: without a cache grows with the square, with a cache about linearly', () => {
    const a = S.stepScene('nocache', run, 64), b = S.stepScene('nocache', run, 32);
    expect(a.totalNo).toBe(67758912); expect(a.totalCache).toBe(2336040);
    expect(a.totalNo / b.totalNo).toBeGreaterThan(4);
    expect(a.totalCache / b.totalCache).toBeLessThan(2.5);
    expect(a.repeated).toBeCloseTo(1 - a.totalCache / a.totalNo, 12);
    expect(a.totalCache).toBe(a.stepNo[a.G - 1]); // cached run = one full pass over the final text
    expect(a.stepNo[10]).toBe(macsForward(w.config, 26, 26));
  });
  test('stepFrameAt endpoints and counters', () => {
    const sc = S.stepScene('cache', run, 64), end = S.stepFrameAt(sc.duration, sc), start = S.stepFrameAt(0, sc);
    expect(end.s).toBe(sc.G - 1); expect(end.macsCache).toBe(sc.totalCache); expect(end.macsNo).toBe(sc.totalNo);
    expect(start.macsCache).toBe(0); expect(start.newFrom).toBe(0);
    const loop = S.stepScene('loop', run, 28), e = S.stepFrameAt(loop.duration, loop);
    expect(e.positions).toBe(258); expect(e.L).toBe(27);
  });
  test('memory: per-token bytes follow 2 x layers x kvDim x bytes', () => {
    const m = S.memoryScene(w, 1);
    expect(m.rows.map((r) => r.perToken)).toEqual([288, 524288, 131072]);
    expect(S.memoryFrameAt(0, m).ctx).toBe(1);
    expect(S.memoryFrameAt(m.duration, m).ctx).toBe(131072);
    expect(S.fmtBytes(524288 * 4096)).toBe('2 GiB');
    expect(S.crossing(m.rows[1], S.memoryScene(w, 8))).toBeCloseTo(80e9 / (524288 * 8), 6);
  });
  test('phases: prefill is compute-dense, decode is about two operations per weight byte', () => {
    const p = S.phasesScene(w, 32);
    expect(p.intensity[0]).toBeGreaterThan(50);
    for (const x of p.intensity.slice(1)) { expect(x).toBeGreaterThan(2); expect(x).toBeLessThan(2.4); }
    expect(S.phasesFrameAt(0, p).stage).toBe('prefill'); expect(S.phasesFrameAt(p.duration, p).stage).toBe('decode');
    expect(S.phasesFrameAt(p.duration, p).bars).toBe(p.D + 1);
  });
  test('batch: roofline ridge, per-user speed flat then falling, largest batch that fits', () => {
    const b = S.batchScene('llama2_7b', 4096);
    expect(b.ridge).toBeCloseTo(989.5e12 / 3.35e12, 6);
    expect(S.batchPoint(b, 1).bound).toBe('memory'); expect(S.batchPoint(b, 512).bound).toBe('compute');
    expect(S.batchPoint(b, 8).perUser).toBeCloseTo(S.batchPoint(b, 1).perUser, 6);
    expect(S.batchPoint(b, 512).perUser).toBeLessThan(S.batchPoint(b, 1).perUser);
    expect(b.fitMax).toBe(30); expect(S.batchPoint(b, 30).fits).toBe(true); expect(S.batchPoint(b, 31).fits).toBe(false);
    expect(S.batchAt(0)).toBe(1); expect(S.batchAt(b.duration)).toBe(512);
    expect(b.markers[0].at).toBe(0);
    expect(S.batchScene('llama3_8b', 4096).fitMax).toBeGreaterThan(b.fitMax);
  });
});
