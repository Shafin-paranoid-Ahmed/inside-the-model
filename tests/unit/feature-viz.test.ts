import { describe, expect, test } from 'vitest';
import {
  DEFAULT_CFG, N, STEPS, STAGES, ALL_TARGETS, CLASSES, NCLS, ascend, ascentDuration, argmax, filterDuration, filterFrameAt, filterMarkers,
  forward, galleryDuration, galleryFrameAt, initNet, inputGrad, makeSet, pearson, randomizeUpTo, roughness, segmentAt, softmax, targetKey,
  train, trainAll, saliencyDuration, type Cfg, type Target,
} from '../../src/components/widgets/FeatureViz/sim';
import { createRng } from '../../src/lib/math/rng';

// Small config so the whole file stays well under the default 5 s timeout.
const CFG: Cfg = { ...DEFAULT_CFG, nTrain: 96, nTest: 48, epochs: 6, snaps: 8 };
const targets: Target[] = [{ kind: 'class', index: 3 }, { kind: 'l2', index: 0 }];
const tr = trainAll(CFG, targets);

describe('FeatureViz data', () => {
  test('seeded and balanced, with pixel values in a sane range', () => {
    const a = makeSet(40, createRng(1)), b = makeSet(40, createRng(1));
    expect(Array.from(a.x[7])).toEqual(Array.from(b.x[7]));
    expect(Array.from(a.y).filter((v) => v === 2)).toHaveLength(10);
    a.clean.forEach((im) => { expect(Math.max(...im)).toBeGreaterThan(0.5); expect(Math.min(...im)).toBeGreaterThanOrEqual(0); expect(Math.max(...im)).toBeLessThanOrEqual(1); });
  });
  test('the four classes are different pictures', () => {
    const set = makeSet(4, createRng(2), 0);
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) expect(pearson(set.clean[i], set.clean[j])).toBeLessThan(0.9);
  });
});

describe('FeatureViz network', () => {
  test('forward gives four scores whose softmax sums to 1', () => {
    const net = initNet(createRng(3)), c = forward(net, makeSet(1, createRng(4)).x[0]);
    expect(c.z).toHaveLength(NCLS);
    expect(softmax(c.z).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect([c.p1.h, c.p1.w, c.r2.h, c.p2.c, c.p2.h]).toEqual([8, 8, 8, 8, 1]);
  });
  test('input gradients match finite differences (class score and a layer 2 channel)', () => {
    const net = initNet(createRng(5)), img = makeSet(1, createRng(6)).x[0];
    for (const t of targets) {
      const { g } = inputGrad(net, img, t);
      for (const i of [20, 100, 133, 200]) {
        const e = 1e-6, up = img.slice(), dn = img.slice();
        up[i] += e; dn[i] -= e;
        const num = (inputGrad(net, up, t).score - inputGrad(net, dn, t).score) / (2 * e);
        expect(g[i]).toBeCloseTo(num, 4);
      }
    }
  });
  test('training is deterministic and learns', () => {
    const a = train(CFG), b = train(CFG);
    expect(Array.from(a.net.c1.weight)).toEqual(Array.from(b.net.c1.weight));
    const first = a.snaps[0], last = a.snaps[a.snaps.length - 1];
    expect(first.iter).toBe(0);
    expect(last.loss).toBeLessThan(first.loss);
    expect(tr.finalAcc).toBeGreaterThan(0.6);
  });
});

describe('FeatureViz analyses', () => {
  test('gradient ascent raises the activation and stays inside [0, 1]', () => {
    for (const t of targets) {
      const A = tr.ascents[targetKey(t)];
      for (const k of ['plain', 'reg'] as const) {
        expect(A[k].act[STEPS]).toBeGreaterThan(A[k].act[0] + 0.5);
        expect(Math.min(...A[k].frames)).toBeGreaterThanOrEqual(0);
        expect(Math.max(...A[k].frames)).toBeLessThanOrEqual(1);
      }
    }
    expect(tr.ascents[targetKey(targets[0])].plain.frames).toHaveLength((STEPS + 1) * N);
  });
  test('the regularised ascent is smoother than the plain one (full-size net)', () => {
    const { net } = train({ ...DEFAULT_CFG, snaps: 2 }), last = (a: Float32Array) => a.subarray(STEPS * N);
    const t: Target = { kind: 'l2', index: 0 };
    expect(roughness(last(ascend(net, t, 'reg').frames))).toBeLessThan(0.5 * roughness(last(ascend(net, t, 'plain').frames)));
  }, 20000);
  test('ascend is deterministic', () => {
    const { net } = train({ ...CFG, epochs: 1, snaps: 1 });
    expect(Array.from(ascend(net, targets[0], 'reg').act)).toEqual(Array.from(ascend(net, targets[0], 'reg').act));
  });
  test('gallery: sorted, six per channel, counts consistent', () => {
    const G = tr.gallery;
    expect(G.l1).toHaveLength(6); expect(G.l2).toHaveLength(8);
    [...G.l1, ...G.l2].forEach((ex) => { expect(ex).toHaveLength(G.top); for (let i = 1; i < ex.length; i++) expect(ex[i - 1].act).toBeGreaterThanOrEqual(ex[i].act); });
    [...G.stat1, ...G.stat2].forEach((s) => { expect(s.n).toBeGreaterThan(0); expect(s.n).toBeLessThanOrEqual(s.of); expect(CLASSES[s.cls]).toBeTruthy(); });
  });
  test('saliency: stage 0 is the trained net, and randomising layers changes the map', () => {
    const S = tr.saliency;
    expect(S.meanCorrTrained[0]).toBeCloseTo(1, 6);
    expect(S.meanCorrTrained[3]).toBeLessThan(0.6);
    expect(S.cases).toHaveLength(NCLS);
    S.cases.forEach((c) => { expect(c.maps).toHaveLength(STAGES.length); expect(c.byClass).toHaveLength(NCLS); expect(c.logits).toHaveLength(NCLS); });
    const net = initNet(createRng(8)), fresh = initNet(createRng(9));
    expect(Array.from(randomizeUpTo(net, fresh, 0).c1.weight)).toEqual(Array.from(net.c1.weight));
    expect(Array.from(randomizeUpTo(net, fresh, 3).c1.weight)).toEqual(Array.from(fresh.c1.weight));
    expect(Array.from(randomizeUpTo(net, fresh, 1).c2.weight)).toEqual(Array.from(net.c2.weight));
    expect(argmax([1, 3, 2])).toBe(1);
  });
  test('every target of the full widget is a class score or a layer 2 channel', () => {
    expect(ALL_TARGETS).toHaveLength(NCLS + 8);
  });
});

describe('FeatureViz timeline', () => {
  test('filter frames hit the first and last snapshot and markers are ordered', () => {
    expect(filterFrameAt(0, tr).i).toBe(0);
    expect(filterFrameAt(filterDuration, tr).i).toBe(tr.snaps.length - 1);
    expect(filterFrameAt(filterDuration * 5, tr).i).toBe(tr.snaps.length - 1);
    const m = filterMarkers(tr);
    for (let i = 1; i < m.length; i++) expect(m[i].at).toBeGreaterThanOrEqual(m[i - 1].at);
  });
  test('segments partition the timeline', () => {
    expect([0, 1.99, 2, 5.9, 8].map((t) => segmentAt(t, saliencyDuration, 4))).toEqual([0, 0, 1, 2, 3]);
  });
  test('gallery reveals layer 1 first, then layer 2', () => {
    expect(galleryFrameAt(0)).toMatchObject({ n1: 0, n2: 0, layer: 1 });
    expect(galleryFrameAt(galleryDuration / 2)).toMatchObject({ n1: 6, layer: 2 });
    expect(galleryFrameAt(galleryDuration / 2).n2).toBeGreaterThan(0);
    expect(galleryFrameAt(galleryDuration)).toMatchObject({ n1: 6, n2: 6 });
    expect(ascentDuration).toBeGreaterThan(0);
  });
});
