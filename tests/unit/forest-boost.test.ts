import { describe, it, expect } from 'vitest';
import * as S from '../../src/components/widgets/ForestBoost/sim';
import { predictOne } from '../../src/components/widgets/TreeSplitter/sim';

describe('regression tree', () => {
  it('fits a step function exactly and respects depth', () => {
    const x = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8], r = [1, 1, 1, -2, -2, -2];
    const t = S.fitReg(x, r, 1);
    expect(t.thr).toBeCloseTo(0.45, 12);
    expect(S.predictReg(t, 0.2)).toBe(1); expect(S.predictReg(t, 0.9)).toBe(-2);
    const lv = S.leavesReg(t);
    expect(lv.map((l) => l.v)).toEqual([1, -2]); expect(lv[0].hi).toBeCloseTo(0.45, 12); expect(lv[1].lo).toBe(lv[0].hi);
    expect(S.fitReg(x, r, 0).value).toBeCloseTo(-0.5, 12);
  });
});

describe('gradient boosting', () => {
  it('pseudo-residual is the negative gradient of the loss (numerical check)', () => {
    const h = 1e-6;
    for (const [y, F] of [[1.3, 0.2], [-0.4, 0.9]]) {
      const d = (L: (f: number) => number) => (L(F + h) - L(F - h)) / (2 * h);
      expect(S.pseudoResidual(y, F, 'sq')).toBeCloseTo(-d((f) => 0.5 * (y - f) ** 2), 6);
      expect(S.pseudoResidual(y, F, 'abs')).toBeCloseTo(-d((f) => Math.abs(y - f)), 6);
    }
  });
  it('training loss falls every round and the data fits the plot range', () => {
    const r = S.boostRun({ lr: 0.3, depth: 2, rounds: 30, loss: 'sq' });
    for (let m = 1; m <= 30; m++) expect(r.lossTrain[m]).toBeLessThan(r.lossTrain[m - 1]);
    expect(r.lossTrain[30]).toBeLessThan(0.05);
    for (const v of S.REG_TRAIN.y) { expect(v).toBeGreaterThan(S.Y_LO); expect(v).toBeLessThan(S.Y_HI); }
  });
  it('F_m + update equals F_{m+1}, on points and on the grid', () => {
    const r = S.boostRun({ lr: 0.5, depth: 2, rounds: 5, loss: 'sq' });
    for (let m = 0; m < 5; m++) {
      r.F[m].forEach((v, i) => expect(v + r.H[m][i]).toBeCloseTo(r.F[m + 1][i], 12));
      r.Fg[m].forEach((v, i) => expect(v + r.Hg[m][i]).toBeCloseTo(r.Fg[m + 1][i], 12));
    }
    expect(r.G).toHaveLength(6);
  });
  it('a high learning rate overfits: validation error ends above its best', () => {
    const r = S.boostRun({ lr: 1, depth: 2, rounds: 30, loss: 'sq' });
    expect(r.lossVal[30]).toBeGreaterThan(Math.min(...r.lossVal) + 0.01);
  });
  it('frames: endpoints, tween and determinism', () => {
    const r = S.boostRun({ lr: 0.3, depth: 2, rounds: 30, loss: 'sq' }), T = S.boostT(30);
    expect(S.boostFrame(0, 30)).toMatchObject({ m: 0, stage: 'residual', w: 0 });
    expect(S.boostFrame(T, 30)).toMatchObject({ m: 30, stage: 'done' });
    const f = S.boostFrame(S.roundStart(2) + 0.99 * (S.roundStart(3) - S.roundStart(2)), 30);
    expect(f.stage).toBe('update'); expect(f.w).toBeGreaterThan(0.99);
    expect(S.lossNow(r, S.boostFrame(T, 30)).train).toBe(r.lossTrain[30]);
    expect(S.lossNow(r, S.boostFrame(0, 30)).train).toBe(r.lossTrain[0]);
    expect(S.boostRun({ lr: 0.3, depth: 2, rounds: 30, loss: 'sq' }).lossVal).toEqual(r.lossVal);
  });
});

describe('bagging and random forests', () => {
  const bag = S.buildCls(2, 8), rf = S.buildCls(1, 16), plain = S.buildCls(2, 16);
  it('bootstrap draws match counts and leave about a third out', () => {
    for (let m = 0; m < 8; m++) {
      const c = new Uint8Array(S.N_TRAIN);
      bag.draws[m].forEach((i) => c[i]++);
      expect(Array.from(c)).toEqual(Array.from(bag.count[m]));
      const oob = bag.count[m].filter((v) => v === 0).length / S.N_TRAIN;
      expect(oob).toBeGreaterThan(0.25); expect(oob).toBeLessThan(0.5);
    }
    const c = new Uint8Array(S.N_TRAIN);
    expect(S.picksAt(bag, 0, S.N_TRAIN, c)).toEqual(bag.count[0]);
  });
  it('trees are full depth: each fits its own bootstrap sample at least 90%', () => {
    expect(bag.stats[1].singleTrain).toBeGreaterThan(0.9);
  });
  it('averaging beats one tree on validation data', () => {
    expect(bag.stats[8].valAcc).toBeGreaterThan(bag.stats[1].valAcc + 0.03);
    expect(plain.stats[16].valAcc).toBeGreaterThan(plain.stats[1].valAcc + 0.03);
  });
  it('feature randomisation makes trees disagree more, with the same bootstrap samples', () => {
    expect(rf.draws[3]).toEqual(plain.draws[3]);
    expect(rf.stats[16].disagree).toBeGreaterThan(plain.stats[16].disagree);
    expect(Number.isNaN(rf.stats[1].disagree)).toBe(true);
  });
  it('a one-feature-per-split forest still classifies with its trees; votes add up', () => {
    const k = 5, v = rf.gridVotes[k], g = rf.treeGrid;
    for (let i = 0; i < v.length; i += 97) expect(v[i]).toBe(g.slice(0, k).reduce((s, a) => s + a[i], 0));
    expect(predictOne(rf.trees[0], [0.9, 0.9])).toBe(1);
  });
  it('clsFrame: endpoints, stages and monotone tree count', () => {
    const tm = S.BAG_TIMING, n = S.N_TRAIN;
    expect(S.clsFrame(0, n, tm, 8)).toMatchObject({ m: 0, stage: 'draw', k: 0, picked: 0 });
    expect(S.clsFrame(tm.per * 8, n, tm, 8)).toMatchObject({ stage: 'done', k: 8, picked: n });
    let last = 0;
    for (let t = 0; t <= tm.per * 8; t += 0.1) { const k = S.clsFrame(t, n, tm, 8).k; expect(k).toBeGreaterThanOrEqual(last); last = k; }
  });
});

describe('forest against boosting', () => {
  const c = S.buildCompare();
  it('forest error flattens, fast boosting overfits, slow boosting keeps improving for longer', () => {
    expect(c.rfVal[100]).toBeLessThan(c.rfVal[1]);
    expect(Math.abs(c.rfVal[100] - c.rfVal[50])).toBeLessThan(0.01);
    expect(c.hiBest).toBeLessThan(15);
    expect(c.hiVal[100]).toBeGreaterThan(c.hiVal[c.hiBest] + 0.015);
    expect(c.hiTrain[100]).toBeLessThan(0.001);
    expect(c.loBest).toBeGreaterThan(c.hiBest + 20);
    expect(c.loVal[25]).toBeGreaterThan(c.loVal[50]);
  });
  it('count is monotone and hits the ends', () => {
    expect(S.cmpCount(0)).toBe(0); expect(S.cmpCount(S.T_CMP)).toBe(S.CMP_N);
  });
});
