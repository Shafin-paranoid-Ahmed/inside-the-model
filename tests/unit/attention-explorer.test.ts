import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseModel, type Manifest } from '../../src/lib/transformer/load';
import { encodeWords } from '../../src/lib/transformer/grammar';
import { evaluate } from '../../src/lib/transformer/report';
import {
  MODES, ABLATE_LABELS, DEFAULT_SENTENCE, REACH_SENTENCE, HEADS, ablationMasses, ablatedMass, argmax, defaultScene, frameAt, headData,
  maskedSoftmax, mixCosine, pipelineRow, previousWordMean, resolveQuery, rowDisagreement, sampleStats, settledTime, startTime, strongestOffDiagonal,
  headStats, sampleMass, suggest, sweepQuery, timeline, validate, weightMatrix, fullVector, type Mode, type ScoresFrame, type SoftmaxFrame, type MixFrame, type ArcsFrame,
  type ReachFrame, type SplitFrame, type ConcatFrame, type AblateFrame,
} from '../../src/components/widgets/AttentionExplorer/sim';

const manifest: Manifest = JSON.parse(readFileSync('public/models/tiny-lm.json', 'utf8'));
const w = parseModel(manifest, new Uint8Array(readFileSync('public/models/tiny-lm.bin')));
const close = (a: number, b: number, tol = 1e-9) => expect(Math.abs(a - b)).toBeLessThan(tol);
// Layer 1, head 2 (zero-based 0, 1) is the head whose "sheep" row puts most weight on "these".
const scene = defaultScene(w, DEFAULT_SENTENCE, 0, 1);
const opts = (mode: Mode, s = scene) => ({ mode, query: 'sheep', mass: mode === 'ablate' ? ablationMasses(w, s.ids) : undefined });

describe('scene and sentence input', () => {
  test('default sentence is 8 tokens with 8-wide heads and the sheep row of layer 1 head 2', () => {
    expect(scene.T).toBe(8); expect(scene.dk).toBe(8); expect(resolveQuery(scene, 'sheep')).toBe(3);
    const row = headData(scene).A.subarray(3 * 8, 4 * 8);
    expect(argmax(row)).toBe(1); close(row[1], 0.556897, 2e-5);
    for (let j = 4; j < 8; j++) expect(row[j]).toBe(0);
  });
  test('validation: adds [start], flags unknown words with suggestions, long input and off-grammar sentences', () => {
    expect(validate('these small sheep chase a red fox').ids).toEqual(encodeWords(DEFAULT_SENTENCE));
    const bad = validate('the purple sheep chase');
    expect(bad.ids).toBeNull(); expect(bad.unknown.map((u) => u.word)).toEqual(['purple']);
    expect(suggest('sheeps')).toContain('sheep'); expect(suggest('chasez')).toEqual(['chase', 'chases']); expect(suggest('purple')).toEqual([]); expect(suggest('fo').slice(0, 2)).toEqual(['fox', 'foxes']);
    expect(validate('a a a a a a a a a a a').tooLong).toBe(true);
    const off = validate('the sheep chase a seed');
    expect(off.ids).not.toBeNull(); expect(off.offGrammar).toEqual({ index: 5, word: 'seed' });
    expect(validate(DEFAULT_SENTENCE).offGrammar).toBeNull();
  });
});

describe('frames', () => {
  test('every mode returns finite numbers and a status at several playheads, deterministically', () => {
    for (const mode of MODES) {
      const s = mode === 'reach' ? defaultScene(w, REACH_SENTENCE) : scene, tl = timeline(mode, s, 'sheep');
      expect(tl.duration).toBeGreaterThan(1); expect(tl.markers.length).toBeGreaterThan(1);
      const o = opts(mode, s);
      for (const f of [0, 0.13, 0.4, 0.77, 1].map((x) => x * tl.duration)) {
        const a = frameAt(f, s, o), b = frameAt(f, s, o);
        expect(JSON.stringify(a)).toBe(JSON.stringify(b));
        expect(a.status.length).toBeGreaterThan(8);
        expect(a.phase.length).toBeGreaterThan(0);
        for (const v of Object.values(a)) if (typeof v === 'number') expect(Number.isFinite(v)).toBe(true);
      }
    }
  });
  test('arcs: each settled row is the real attention row; the future is exactly 0', () => {
    const { A } = headData(scene);
    for (let q = 0; q < 8; q++) {
      const f = frameAt(settledTime('arcs', scene, q), scene, { mode: 'arcs' }) as ArcsFrame;
      expect(f.q).toBe(q); for (let j = 0; j < 8; j++) close(f.target[j], A[q * 8 + j]);
    }
    const f = frameAt(settledTime('arcs', scene, 3), scene, { mode: 'arcs' });
    expect(f.status).toContain('these'); expect(f.status).toContain('0.56');
    expect(sweepQuery('arcs', scene, 1.8 * 3 + 0.1)).toBe(3); expect(startTime('arcs', scene, 'sheep')).toBeGreaterThan(5.4);
  });
  test('scores: the detail sum counts up to the true q.k and the grid ends equal to Q K^T', () => {
    const { S, q, k } = headData(scene), tl = timeline('scores', scene, 'sheep');
    let dot = 0; for (let d = 0; d < 8; d++) dot += q[3 * 8 + d] * k[1 * 8 + d];
    close(dot, S[3 * 8 + 1], 1e-9);
    const mid = frameAt(0.9 * 1 + 0.9 * 0.97, scene, opts('scores')) as ScoresFrame;
    expect(mid.detail?.j).toBe(1); close(mid.detail!.sum, S[3 * 8 + 1], 1e-6);
    const end = frameAt(tl.duration, scene, opts('scores')) as ScoresFrame;
    for (let c = 0; c < 64; c++) { expect(end.vis[c]).toBe(1); close(end.val[c], S[c]); }
    expect(end.detail).toBeNull();
  });
  test('softmax: weights add to 1, masked entries are 0, scaling by 1/sqrt(8) changes the largest weight', () => {
    const tl = timeline('softmax', scene, 'sheep'), at = (x: number) => frameAt(x, scene, opts('softmax')) as SoftmaxFrame;
    const scaled = at(6.9), plain = at(tl.duration);
    close(scaled.sum, 1, 1e-9); close(plain.sum, 1, 1e-9);
    for (let j = 4; j < 8; j++) { expect(scaled.w[j]).toBe(0); expect(plain.w[j]).toBe(0); }
    close(scaled.maxW, 0.556897, 2e-5); expect(scaled.maxIdx).toBe(1);
    expect(plain.maxW).toBeGreaterThan(0.9); expect(plain.maxIdx).toBe(1);
    close(plain.f, 1); close(scaled.f, 1 / Math.sqrt(8));
    expect(at(0).alpha[2]).toBe(0); expect(at(0).sum).toBe(0);
    const row = maskedSoftmax([1, 2, 3, 99, 99], 2, 5); close(row.reduce((a, b) => a + b, 0), 1); expect(row[3]).toBe(0);
  });
  test('mix: the running total ends at the head output and the output points toward the value of "these"', () => {
    const tl = timeline('mix', scene, 'sheep'), f = frameAt(tl.duration, scene, opts('mix')) as MixFrame;
    const { out } = headData(scene); expect(f.done).toBe(true);
    for (let d = 0; d < 8; d++) close(f.acc[d], out[3 * 8 + d], 1e-12);
    const m = mixCosine(scene, 3); expect(m.top).toBe(1); expect(m.cos).toBeGreaterThan(0.9);
    const half = frameAt(1.2 * 1.5, scene, opts('mix')) as MixFrame; expect(half.j).toBe(1);
  });
  test('explorer pipeline row matches the trace, with and without scaling', () => {
    const p = pipelineRow(scene, 3), { S } = headData(scene);
    for (let j = 0; j < 8; j++) close(p.raw[j], S[3 * 8 + j]); close(p.scaled[1], p.raw[1] / Math.sqrt(8));
    expect(pipelineRow(scene, 3, 0, 1, { noScale: true }).scaled[1]).toBe(p.raw[1]);
    close(p.w.reduce((a, b) => a + b, 0), 1);
  });
  test('reach: seven recurrent hops against one attention hop, T squared scores', () => {
    const s = defaultScene(w, REACH_SENTENCE), tl = timeline('reach', s), at = (x: number) => frameAt(x, s, { mode: 'reach' }) as ReachFrame;
    const mid = at(tl.markers[1].at - 0.01); expect(mid.rnnHops).toBe(7); expect(mid.attnHops).toBe(1); expect(s.T).toBe(9);
    expect(at(0.1).rnnHops).toBe(0); expect(at(0.1).attnHops).toBe(0);
    const end = at(tl.duration); expect(end.scores).toBe(81); expect(end.Tn).toBe(9);
    expect(at(tl.markers[1].at + 0.01).scores).toBe(1);
  });
  test('split: parameter count is 1,024 for any number of heads; gaps follow the head count', () => {
    const f = (x: number) => frameAt(x, scene, { mode: 'split' }) as SplitFrame;
    expect(f(2).count).toBe(1024); expect(f(0).count).toBe(0);
    expect(f(2).h).toBe(1); expect(f(5).h).toBe(2); expect(f(8.9).h).toBe(4);
    expect(f(5.5).gaps[7]).toBe(1); expect(f(5.5).gaps[3]).toBe(0); expect(f(8.9).gaps[3]).toBe(1); expect(f(8.9).gaps[7]).toBe(1);
    const x = fullVector(scene, 0, 'q', 3), M = weightMatrix(w, 0, 'q'), bias = w.offsets['0.q.bias'].offset;
    const ln = scene.trace.layers[0].ln1.subarray(3 * 16, 4 * 16);
    for (let r = 0; r < 16; r++) { let v = w.params[bias + r]; for (let j = 0; j < 16; j++) v += ln[j] * M[j * 16 + r]; close(v, x[r], 1e-9); }
  });
  test('concat: head outputs stacked then W_O give the attention output', () => {
    const L = scene.trace.layers[0], f = frameAt(9, scene, { mode: 'concat', query: 'sheep' }) as ConcatFrame;
    const M = weightMatrix(w, 0, 'o'), bias = w.offsets['0.o.bias'].offset, cc = L.concat.subarray(3 * 16, 4 * 16);
    for (let r = 0; r < 16; r++) { let v = w.params[bias + r]; for (let j = 0; j < 16; j++) v += cc[j] * M[j * 16 + r]; close(v, L.attnOut[3 * 16 + r], 1e-9); }
    for (let d = 0; d < 8; d++) { close(cc[d], L.headOut[0][3 * 8 + d]); close(cc[8 + d], L.headOut[1][3 * 8 + d]); }
    expect(f.norms.input).toBeGreaterThan(0); expect(f.norms.attn).toBeGreaterThan(0);
  });
  test('heads: the two layer-1 heads disagree most on a definite row; previous-word means are finite', () => {
    const d = rowDisagreement(scene); expect(d[0]).toBe(0); expect(Math.max(...d)).toBeGreaterThan(0.5);
    for (const [l, h] of HEADS) { const m = previousWordMean(scene, l, h); expect(m).toBeGreaterThan(0); expect(m).toBeLessThan(1); }
    expect(strongestOffDiagonal(scene, 0, 1).weight).toBeGreaterThan(0.8);
  });
});

describe('statistics over the 200-sentence sample reproduce the core report', () => {
  test('head means and ablation masses equal evaluate() and the plan table', async () => {
    const t = performance.now(), st = await sampleStats(w), rep = evaluate(w);
    for (const [k, [l, h]] of HEADS.entries()) {
      const r = rep.heads.find((x) => x.layer === l && x.head === h)!;
      close(st.heads[k].previous, r.previous); close(st.heads[k].determiner, r.subjectDeterminer); close(st.heads[k].verb, r.verb); close(st.heads[k].start, r.start);
    }
    close(st.mass[0], rep.grammaticalMass, 1e-12); close(st.mass[1], 0.861865, 2e-6); close(st.mass[2], 0.860681, 2e-6);
    close(st.mass[3], 0.992661, 2e-6); close(st.mass[4], 0.983285, 2e-6);
    expect((await headStats(w))).toBe(st.heads); process.stdout.write(`stats ms ${(performance.now() - t).toFixed(0)}\n`);
  });
  test('ablate frames: the first segment is the full model; switching a head off lowers the average', () => {
    const masses = ablationMasses(w, scene.ids), tl = timeline('ablate', scene);
    expect(tl.markers.map((m) => m.label)).toEqual(ABLATE_LABELS);
    const f0 = frameAt(2.3, scene, { mode: 'ablate', mass: masses }) as AblateFrame; expect(f0.off.some(Boolean)).toBe(false); close(f0.avg, f0.avgFull);
    const f1 = frameAt(2.4 * 1 + 2.3, scene, { mode: 'ablate', mass: masses }) as AblateFrame; expect(f1.off).toEqual([true, false, false, false]); expect(f1.avg).toBeLessThan(f1.avgFull);
    close(ablatedMass(w, scene.ids, [true, false, false, false])[3], masses[1][3]);
    const custom = frameAt(0, scene, { mode: 'ablate', mass: masses, toggles: [false, true, true, false], customMass: ablatedMass(w, scene.ids, [false, true, true, false]) }) as AblateFrame;
    expect(custom.state).toBe(-1); expect(custom.off).toEqual([false, true, true, false]);
  });
  test('sampleMass matches the single-head table and handles two heads off', async () => {
    close(await sampleMass(w, [true, false, false, false]), 0.861865, 2e-6);
    const both = await sampleMass(w, [true, false, false, true]);
    expect(both).toBeGreaterThan(0.5); expect(both).toBeLessThan(0.9985);
    expect(await sampleMass(w, [false, false, false, false])).toBeGreaterThan(0.9978);
  });
  test('numbers quoted in the multi-head lesson', () => {
    const l1 = defaultScene(w, DEFAULT_SENTENCE, 0, 0), T = 8, row = (h: number, i: number) => Array.from(headData(l1, 0, h).A.subarray(i * T, i * T + T));
    const near = (a: number, b: number) => expect(Math.abs(a - b)).toBeLessThan(0.005);
    near(row(0, 3)[0], 0.53); near(row(0, 3)[1], 0.03); near(row(0, 3)[3], 0.35); near(row(1, 3)[1], 0.56); near(row(1, 3)[3], 0.23); near(row(1, 3)[0], 0.13);
    near(row(0, 5)[1], 0.32); near(row(0, 5)[4], 0.28); near(row(0, 5)[5], 0.23); near(row(1, 5)[5], 0.94);
    const dis = rowDisagreement(l1); expect(argmax(dis)).toBe(5); near(dis[5], 0.71);
    near(previousWordMean(l1, 0, 0), 0.24); near(previousWordMean(l1, 0, 1), 0.31);
    const L = l1.trace.layers[0], q = 3, n = (a: ArrayLike<number>) => Math.sqrt(Array.from(a).reduce((x, y) => x + y * y, 0));
    near(n(L.input.subarray(q * 16, q * 16 + 16)), 1.28); near(n(L.attnOut.subarray(q * 16, q * 16 + 16)), 1.87); near(n(L.resid1.subarray(q * 16, q * 16 + 16)), 2.27);
    const g = defaultScene(w, '[start] a big fox eats the red apples'), G = 8, at = (l: number, h: number, i: number, j: number) => headData(g, l, h).A[i * G + j];
    near(at(0, 1, 2, 1), 0.70); near(at(0, 1, 6, 5), 0.90); near(at(1, 0, 5, 4), 1.0); near(at(1, 0, 6, 4), 1.0); near(at(1, 1, 5, 4), 0.98); near(at(1, 1, 6, 4), 0.98);
    const ms = ablationMasses(w, l1.ids), avg = (m: number[]) => m.filter((x) => !Number.isNaN(x)).reduce((a, b) => a + b, 0) / m.length;
    expect(ms.map((m) => +avg(m).toFixed(3))).toEqual([0.998, 0.961, 0.909, 0.995, 0.991]);
    expect(ms.map((m) => +m[3].toFixed(3))).toEqual([0.998, 0.837, 0.945, 0.996, 0.939]);
    near(ms[2][5], 0.670); near(ms[2][6], 0.788);
  });
  test('names, numbers and narration switch at the segment boundary; previous bars cross-fade out', () => {
    const { A } = headData(scene), at = (mode: Mode, t: number) => frameAt(t, scene, { mode });
    const a = at('arcs', 1.8 * 4 + 0.001) as ArcsFrame;       // start of "chase"
    for (let j = 0; j < 8; j++) close(a.target[j], A[4 * 8 + j]);
    expect(a.status).toContain('chase'); expect(a.fade).toBeLessThan(0.1); close(a.prev[3], A[3 * 8 + 3]);
    const c = at('compare', 2.0 * 4 + 0.001) as { targets: Float64Array[]; prevs: Float64Array[]; fade: number; status: string; argmax: number[] };
    expect(c.status.startsWith('“chase”')).toBe(true);
    close(c.targets[1][2], headData(scene, 0, 1).A[4 * 8 + 2]); expect(c.fade).toBeLessThan(0.1); close(c.prevs[1][1], headData(scene, 0, 1).A[3 * 8 + 1]);
  });
});
