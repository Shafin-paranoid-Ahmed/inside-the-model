import { describe, it, expect } from 'vitest';
import { createRng } from '../../src/lib/math/rng';
import {
  makeData, truth, fitBoost, predictBoost, fitTree, treeClass, rules, initMlp, lossGrad, fitMlp, mlpParams, computeCurves, computeInspect,
  recommend, raceAt, knotT, SIZES, duration, raceMarkers, inspectAt, checklistAt, CHOICE_DEFAULT, type Choice,
} from '../../src/components/widgets/ModelChooser/sim';

describe('ModelChooser sim', () => {
  it('data is deterministic and the label noise is about 6 percent', () => {
    const a = makeData('tabular', 2000, createRng(1)), b = makeData('tabular', 2000, createRng(1));
    expect(Array.from(a.y)).toEqual(Array.from(b.y));
    let flips = 0; for (let i = 0; i < a.n; i++) flips += truth('tabular', a.X, i * a.F) !== a.y[i] ? 1 : 0;
    expect(flips / a.n).toBeGreaterThan(0.04); expect(flips / a.n).toBeLessThan(0.08);
  });
  it('MLP backprop matches finite differences', () => {
    const d = makeData('pixels', 30, createRng(2)), m = initMlp(d.F, 5, createRng(3)), { grad } = lossGrad(m, d), h = 1e-6;
    const W = m.W1, k = 3; const w0 = W[k];
    W[k] = w0 + h; const up = lossGrad(m, d).loss; W[k] = w0 - h; const dn = lossGrad(m, d).loss; W[k] = w0;
    expect(grad[k]).toBeCloseTo((up - dn) / (2 * h), 6);
    const b2 = m.b2; m.b2 = b2 + h; const u2 = lossGrad(m, d).loss; m.b2 = b2 - h; const d2 = lossGrad(m, d).loss;
    expect(grad[grad.length - 1]).toBeCloseTo((u2 - d2) / (2 * h), 6);
    expect(grad.length).toBe(mlpParams(d.F, 5));
  });
  it('Adam lowers the loss and the boosted trees learn the tabular rule', () => {
    const d = makeData('tabular', 300, createRng(4)), m0 = initMlp(d.F, 16, createRng(5));
    expect(fitMlp(d, createRng(5), 100).loss).toBeLessThan(lossGrad(m0, d).loss);
    const { model } = fitBoost(d), test = makeData('tabular', 500, createRng(6));
    let ok = 0; for (let i = 0; i < test.n; i++) ok += (predictBoost(model, test.X, i * test.F) >= 0.5 ? 1 : 0) === test.y[i] ? 1 : 0;
    expect(ok / test.n).toBeGreaterThan(0.85);
  });
  it('a depth-3 tree is readable: at most 8 rules, counts add up', () => {
    const d = makeData('tabular', 400, createRng(7)), t = fitTree(d, 3), r = rules(t, ['a', 'b', 'c', 'd']);
    expect(r.length).toBeLessThanOrEqual(8);
    expect(r.reduce((s, x) => s + x.n, 0)).toBe(400);
    expect(treeClass(t)).toBeGreaterThanOrEqual(0);
  });
  it('learning curves: trees lead on tabular, the net leads on stripes, deterministic', () => {
    const tab = computeCurves('tabular'), pix = computeCurves('pixels');
    const last = (c: typeof tab) => c.points[c.points.length - 1];
    expect(last(tab).accTree).toBeGreaterThan(last(tab).accNet);
    expect(last(pix).accNet).toBeGreaterThan(last(pix).accTree + 0.05);
    expect(tab.points[0].accNet).toBeLessThan(tab.points[0].accTree + 0.05);
    expect(last(tab).accTree).toBeLessThanOrEqual(tab.ceiling + 0.02);
    expect(last(tab).opsNet).toBeGreaterThan(5 * last(tab).opsTree);
    expect(computeCurves('tabular').points[3].accTree).toBe(tab.points[3].accTree);
    expect(tab.points[0].gridTree.length).toBe(24 * 24);
  }, 30000);
  it('inspect: snapshots start untrained and improve', () => {
    const i = computeInspect();
    expect(i.snaps[0].epoch).toBe(0);
    expect(i.snaps.at(-1)!.loss).toBeLessThan(i.snaps[0].loss);
    expect(i.treeAcc).toBeGreaterThan(0.8);
  });
  it('checklist recommendations are sensible', () => {
    const c = (o: Partial<Choice>) => recommend({ ...CHOICE_DEFAULT, ...o });
    expect(c({ explain: 'yes' }).pick).toMatch(/linear|tree/i);
    expect(c({ explain: 'no' }).pick).toMatch(/boosted/i);
    expect(c({ data: 'media', rows: 'few' }).pick).toMatch(/pretrained/i);
    expect(c({ data: 'media', rows: 'many', explain: 'no' }).pick).toBe('A neural network');
    expect(c({ explain: 'no' }).path).toEqual(['data', 'rows', 'explain', 'latency']);
  });
  it('frames: endpoints, monotone size sweep, markers inside the timeline', () => {
    expect(raceAt(0).n).toBe(SIZES[0]);
    expect(raceAt(duration('race')).n).toBe(SIZES.at(-1));
    let prev = 0; for (let t = 0; t <= duration('race'); t += 0.1) { const n = raceAt(t).n; expect(n).toBeGreaterThanOrEqual(prev); prev = n; }
    for (let j = 0; j < SIZES.length; j++) expect(raceAt(knotT(j)).n).toBe(SIZES[j]);
    expect(raceMarkers.every((m) => m.at >= 0 && m.at <= duration('race'))).toBe(true);
    expect(inspectAt(0, 10).depth).toBe(0); expect(inspectAt(12, 10).depth).toBe(3);
    expect(checklistAt(0).verdict).toBe(0); expect(checklistAt(10).verdict).toBe(1);
  });
});
