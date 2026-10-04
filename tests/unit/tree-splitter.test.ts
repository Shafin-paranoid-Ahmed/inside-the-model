import { describe, it, expect } from 'vitest';
import {
  impurity, impurityP, evalSplit, bestSplit, fitTree, predict, accuracy, leavesOf, nodesOf,
  TRAIN, VAL, buildStory, frameAt, nodeAlpha, isSplit, depthStudy, depthAt, DMAX, T_DEPTH, impurityAt, BAG_RANK, BAG, T_IMP,
} from '../../src/components/widgets/TreeSplitter/sim';

describe('impurity', () => {
  it('matches closed forms', () => {
    expect(impurity([5, 5], 'gini')).toBeCloseTo(0.5, 12);
    expect(impurity([5, 5], 'entropy')).toBeCloseTo(1, 12);
    expect(impurity([10, 0], 'gini')).toBe(0);
    expect(impurity([10, 0], 'entropy')).toBe(0);
    expect(impurity([3, 1], 'gini')).toBeCloseTo(1 - (0.75 ** 2 + 0.25 ** 2), 12);
    expect(impurity([3, 1], 'entropy')).toBeCloseTo(-(0.75 * Math.log2(0.75) + 0.25 * Math.log2(0.25)), 12);
    expect(impurity([1, 1, 1, 1], 'gini')).toBeCloseTo(0.75, 12);
    expect(impurityP(0.3, 'gini')).toBeCloseTo(2 * 0.3 * 0.7, 12);
  });
});

describe('splits', () => {
  const X = [[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]], y = [0, 0, 0, 1, 1, 1], idx = [0, 1, 2, 3, 4, 5];
  it('a clean separable split has gain equal to the parent impurity', () => {
    const e = evalSplit(X, y, idx, 0, 0.5);
    expect(e.nL).toBe(3); expect(e.nR).toBe(3); expect(e.weighted).toBe(0);
    expect(e.gain).toBeCloseTo(0.5, 12);
    const b = bestSplit(X, y, idx)!;
    expect(b.threshold).toBeCloseTo(0.5, 12); expect(b.gain).toBeCloseTo(0.5, 12);
  });
  it('bestSplit agrees with brute force over a fine grid (real data)', () => {
    const all = TRAIN.X.map((_, i) => i), b = bestSplit(TRAIN.X, TRAIN.y, all)!;
    let bestGain = -1;
    for (const f of [0, 1]) for (let k = 1; k < 1000; k++) bestGain = Math.max(bestGain, evalSplit(TRAIN.X, TRAIN.y, all, f, k / 1000).gain);
    expect(b.gain).toBeGreaterThanOrEqual(bestGain - 1e-12);
    expect(b.gain).toBeCloseTo(evalSplit(TRAIN.X, TRAIN.y, all, b.feature, b.threshold).gain, 12);
  });
  it('minLeaf is respected', () => {
    const t = fitTree(TRAIN.X, TRAIN.y, { maxDepth: 8, minLeaf: 4 });
    for (const l of leavesOf(t)) expect(l.n).toBeGreaterThanOrEqual(4);
  });
});

describe('fitTree', () => {
  it('reaches 100% on separable data and respects maxDepth', () => {
    const X = [[0.1, 0.1], [0.2, 0.9], [0.8, 0.2], [0.9, 0.8]], y = [0, 0, 1, 1];
    const t = fitTree(X, y, { maxDepth: 3 });
    expect(accuracy(t, X, y)).toBe(1);
    expect(predict(t, X)).toEqual(y);
    for (const n of nodesOf(fitTree(TRAIN.X, TRAIN.y, { maxDepth: 2 }))) expect(n.depth).toBeLessThanOrEqual(2);
  });
  it('depth 0 is the majority class; leaf samples partition the data', () => {
    const t0 = fitTree(TRAIN.X, TRAIN.y, { maxDepth: 0 });
    expect(t0.left).toBeUndefined();
    expect(t0.pred).toBe(TRAIN.y.filter((v) => v === 1).length * 2 > TRAIN.y.length ? 1 : 0);
    const t = fitTree(TRAIN.X, TRAIN.y, { maxDepth: 4 });
    expect(leavesOf(t).reduce((s, l) => s + l.n, 0)).toBe(TRAIN.y.length);
  });
  it('training accuracy never falls as depth grows; a full tree memorises the noise', () => {
    const acc = Array.from({ length: DMAX + 1 }, (_, d) => accuracy(fitTree(TRAIN.X, TRAIN.y, { maxDepth: d }), TRAIN.X, TRAIN.y));
    for (let d = 1; d < acc.length; d++) expect(acc[d]).toBeGreaterThanOrEqual(acc[d - 1] - 1e-12);
    expect(acc[DMAX]).toBe(1);
  });
  it('is deterministic', () => {
    expect(JSON.stringify(fitTree(TRAIN.X, TRAIN.y, { maxDepth: 3 }))).toBe(JSON.stringify(fitTree(TRAIN.X, TRAIN.y, { maxDepth: 3 })));
  });
});

describe('depth study', () => {
  it('validation accuracy peaks early and then falls while training reaches 1', () => {
    const s = depthStudy();
    const best = s.val.indexOf(Math.max(...s.val));
    expect(best).toBeGreaterThanOrEqual(2); expect(best).toBeLessThanOrEqual(3);
    expect(s.val[DMAX]).toBeLessThan(s.val[best] - 0.05);
    expect(s.train[DMAX]).toBe(1);
    expect(VAL.y.length).toBeGreaterThan(300);
  });
  it('depthAt sits on integers at the ends and moves monotonically', () => {
    expect(depthAt(0).v).toBe(0); expect(depthAt(T_DEPTH).v).toBe(DMAX);
    let prev = -1;
    for (let t = 0; t <= T_DEPTH; t += 0.05) { const { v } = depthAt(t); expect(v).toBeGreaterThanOrEqual(prev - 1e-12); prev = v; }
  });
});

describe('story', () => {
  for (const mode of ['split', 'gain', 'grow'] as const) {
    it(`${mode}: frame endpoints, continuity and determinism`, () => {
      const st = buildStory(mode);
      expect(JSON.stringify(frameAt(1.3, st))).toBe(JSON.stringify(frameAt(1.3, st)));
      const f0 = frameAt(0, st), fT = frameAt(st.T, st);
      expect(f0.stage).toBe('intro'); expect(f0.splitCount).toBe(0);
      expect(fT.splitCount).toBe(st.events.length); expect(fT.reveal).toBe(1);
      expect(st.markers.every((m) => m.at >= 0 && m.at <= st.T)).toBe(true);
      let prev = frameAt(0, st);
      for (let t = 0.02; t <= st.T; t += 0.02) {
        const f = frameAt(t, st);
        expect(f.splitCount).toBeGreaterThanOrEqual(prev.splitCount);
        if (f.event === prev.event) for (let i = 0; i < f.sweeps.length; i++) expect(Math.abs(f.sweeps[i].thr - prev.sweeps[i].thr)).toBeLessThan(0.12);
        prev = f;
      }
      for (const n of st.nodes) expect(nodeAlpha(st, fT, n.id)).toBe(1);
      expect(isSplit(st, fT, 0)).toBe(true);
    });
  }
  it('the winner lands on the tree threshold and its gain is the largest', () => {
    const st = buildStory('gain');
    const E = st.events[0], f = frameAt(st.T, st), w = E.sweeps[E.win];
    expect(f.sweeps[E.win].thr).toBeCloseTo(st.tree.threshold!, 12);
    expect(f.sweeps[E.win].ev.gain).toBeCloseTo(st.tree.gain!, 12);
    for (const s of E.sweeps) expect(s.best.gain).toBeLessThanOrEqual(w.best.gain + 1e-12);
    expect(w.feature).toBe(0);
  });
  it('the sampled curve never beats the exact best split', () => {
    const st = buildStory('gain'), E = st.events[0];
    for (const s of E.sweeps) expect(Math.min(...s.curve)).toBeGreaterThanOrEqual(st.nodes[0].impurity - s.best.gain - 1e-9);
  });
  it('gini and entropy pick the same root here', () => {
    expect(buildStory('split', 'entropy').tree.threshold).toBeCloseTo(buildStory('split', 'gini').tree.threshold!, 12);
  });
});

describe('impurity study', () => {
  it('counts dots, p and both measures consistently', () => {
    expect(impurityAt(0)).toMatchObject({ k: 0, p: 0, gini: 0, entropy: 0 });
    const m = impurityAt(T_IMP / 2);
    expect(m.p).toBe(0.5); expect(m.gini).toBeCloseTo(0.5, 12); expect(m.entropy).toBeCloseTo(1, 12);
    expect(impurityAt(T_IMP).p).toBe(1);
    expect(new Set(BAG_RANK).size).toBe(BAG);
    for (let t = 0; t <= T_IMP; t += 0.3) { const { k } = impurityAt(t); expect(BAG_RANK.filter((r) => r < k).length).toBe(k); }
  });
});
