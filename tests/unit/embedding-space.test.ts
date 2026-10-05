import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseModel, type Manifest } from '../../src/lib/transformer';
import { cosine } from '../../src/lib/transformer/math';
import {
  makeData, buildScene, frameAt, idOf, pca2, neighbours, stagePos, prepareContext, PRESETS,
  type Data, type LookupFrame, type TrainingFrame, type SimilarityFrame, type ProjectionFrame, type ArithmeticFrame, type ContextFrame,
  type TrainingScene, type SimilarityScene, type ProjectionScene, type ArithmeticScene, type ContextScene,
} from '../../src/components/widgets/EmbeddingSpace/sim';

const read = (f: string) => readFileSync(`public/models/${f}`);
const manifest: Manifest = JSON.parse(read('tiny-lm.json').toString('utf8'));
const w = parseModel(manifest, new Uint8Array(read('tiny-lm.bin')));
const data: Data = makeData(w, JSON.parse(read('tiny-lm-snapshots.json').toString('utf8')), new Uint8Array(read('tiny-lm-snapshots.bin')));
const near = (a: number, b: number, tol = 5e-4) => expect(Math.abs(a - b)).toBeLessThan(tol);

describe('EmbeddingSpace sim', () => {
  test('table is 34 x 16 and equals the last checkpoint; eight checkpoints', () => {
    expect(data.table.length).toBe(34); expect(data.table[0].length).toBe(16);
    expect(data.snaps.length).toBe(8); expect(data.steps).toEqual([0, 857, 1714, 2571, 3429, 4286, 5143, 6000]);
    expect(data.snaps[7]).toEqual(data.table);
  });
  test('lookup: one token at a time, then the finished matrix', () => {
    const s = buildScene('lookup', data);
    if (s.mode !== 'lookup') throw new Error();
    expect(s.ids.length).toBe(8);
    const f0 = frameAt(0, s) as LookupFrame, fEnd = frameAt(s.T, s) as LookupFrame;
    expect(f0.active).toBe(0); expect(fEnd.active).toBe(-1); expect(fEnd.done).toBe(8);
    expect(frameAt(1.3 * 3 + 0.1, s)).toMatchObject({ active: 3, done: 3 });
    expect(frameAt(1.3 * 3 + 0.1, s).status).toContain('"sheep" has ID 20');
  });
  test('training: fixed final basis, snapped readout, cat-dog cosine at the ends', () => {
    const s = buildScene('training', data) as TrainingScene;
    const a = frameAt(0, s) as TrainingFrame, z = frameAt(s.T, s) as TrainingFrame;
    expect(a.step).toBe(0); expect(z.step).toBe(6000);
    near(a.cosine, cosine(data.snaps[0][idOf('cat')], data.snaps[0][idOf('dog')]), 1e-12);
    expect(a.cosine).toBeLessThan(0); expect(z.cosine).toBeGreaterThan(0.98);
    // the last frame's points are exactly the projection of the shipped table
    const final = pca2(data.table); const x = data.table[5].reduce((acc, v, j) => acc + (v - final.mean[j]) * final.axes[0][j], 0);
    near(z.points[5][0], x, 1e-12);
    // between checkpoints the points are a blend, the readout is the nearer checkpoint
    const mid = frameAt(1.6 + 1.4 * 0.4, s) as TrainingFrame; expect(mid.step).toBe(0);
    expect((frameAt(1.6 + 1.4 * 0.6, s) as TrainingFrame).step).toBe(857);
  });
  test('similarity: cat neighbours, sorted, eased transition keeps rows', () => {
    const s = buildScene('similarity', data) as SimilarityScene;
    const f = frameAt(0, s) as SimilarityFrame;
    expect(f.rows.length).toBe(33); expect(f.top.map((x) => x.word).sort()).toEqual(['bird', 'dog', 'fox']);
    near(f.top[0].value, 0.996, 1e-3);
    near(cosine(data.table[idOf('cat')], data.table[idOf('eats')]), -0.231, 1e-3);
    expect(f.rows.map((r) => Math.round(r.rank)).sort((a, b) => a - b)).toEqual([...Array(33).keys()]);
    const g = frameAt(3.4 + 0.1, s) as SimilarityFrame; expect(g.probe).toBe(idOf('cat')); // not yet halfway through the glide
    expect((frameAt(3.4 + 0.6, s) as SimilarityFrame).probe).toBe(idOf('eats'));
    const dotScene = buildScene('similarity', data, { metric: 'dot' }) as SimilarityScene;
    expect(dotScene.domain).toBeGreaterThan(1);
    near(dotScene.sorted[0][0].value, 0.776, 1e-3); near(dotScene.sorted[1][0].value, 1.346, 1e-3);
  });
  test('projection: share of variance and the rotation endpoints', () => {
    const s = buildScene('projection', data) as ProjectionScene;
    near(s.share, 0.435, 2e-3);
    const first = frameAt(0, s) as ProjectionFrame, last = frameAt(s.T, s) as ProjectionFrame;
    expect(first.group).toBe(-1); expect(last.group).toBe(3); expect(last.status).toContain('each class sits in its own corner');
    const x = data.table[3].reduce((acc, v, j) => acc + (v - s.basis.mean[j]) * s.basis.axes[0][j], 0);
    near(last.points[3][0], x, 1e-12);
    expect(first.points[3][0]).not.toBeCloseTo(last.points[3][0], 2);
    expect(neighbours(data.table, idOf('sheep'), 3).length).toBe(3);
  });
  test('arithmetic: cats - cat + dog lands on dogs; planes contain the arrow exactly', () => {
    for (const p of PRESETS) {
      const s = buildScene('arithmetic', data, { presets: [p] }) as ArithmeticScene;
      expect(s.each[0].near[0].word).toBe({ cats: 'dogs', foxes: 'birds', apples: 'seeds', chase: 'see', these: 'those' }[p[0]]);
      const pl = s.each[0].plane;
      near(pl.land[0] - pl.c[0], pl.a[0] - pl.b[0], 1e-12); expect(pl.a[1]).toBeCloseTo(0, 12);
    }
    const s = buildScene('arithmetic', data) as ArithmeticScene;
    near(s.each[0].near[0].value, 0.985441, 1e-6); near(s.each[0].offsetCos!, 0.991701, 1e-6);
    const end = frameAt(s.T, s) as ArithmeticFrame; expect(end.shown).toBe(4); expect(end.status).toMatch(/sheep 0\.460\.$/);
    expect(frameAt(0, s).status).toContain('cats minus cat');
  });
  test('contextual: sheep after this vs these separates stage by stage', () => {
    const s = buildScene('contextual', data) as ContextScene;
    const at = (t: number) => (frameAt(t, s) as ContextFrame).cosThisThese;
    near(at(0), 1, 1e-9); near(at(3.2 + 0.1), 0.880533, 1e-6); near(at(6.4 + 0.1), 0.172188, 1e-6);
    const withThe = buildScene('contextual', data, { third: 'the' }) as ContextScene;
    expect(withThe.sheep[2].length).toBe(3);
    expect(withThe.cloud[0].length).toBe(390);
    near(withThe.cos[2][2], 0.735, 1e-3); near(withThe.cosThese[2], 0.774, 1e-3);
    // deterministic
    expect(JSON.stringify(frameAt(5, s))).toBe(JSON.stringify(frameAt(5, buildScene('contextual', data))));
  });
  test('contextual: the batched page-load path gives the same scene as the direct one', async () => {
    const fresh = makeData(w, JSON.parse(read('tiny-lm-snapshots.json').toString('utf8')), new Uint8Array(read('tiny-lm-snapshots.bin')));
    await prepareContext(fresh, 7);
    expect(JSON.stringify(buildScene('contextual', fresh))).toBe(JSON.stringify(buildScene('contextual', data)));
  });
  test('stagePos switches the stage exactly at the marker', () => {
    expect(stagePos(3.19, 3, 3.2, 1).idx).toBe(0); expect(stagePos(3.2, 3, 3.2, 1).idx).toBe(1);
    expect(stagePos(3.2, 3, 3.2, 1).p).toBeCloseTo(0.5, 9);
  });
});
