import { describe, it, expect } from 'vitest';
import { createRng } from '../../src/lib/math/rng';
import {
  initParams, lossGrad, lossOf, makeSpirals, newAdam, adamStep, runNet, degradationFrame, STEPS, SNAP, GRID, DEG_T,
  makeBlock, blockAt, BLOCK_D, historyAt, ARCHS, NETS, type NetSpec, type NetId,
} from '../../src/components/widgets/ResidualTrainer/sim';

describe('gradients', () => {
  for (const spec of [
    { depth: 3, width: 4, skip: false }, { depth: 4, width: 4, skip: true }, { depth: 4, width: 4, skip: true, act: 'relu' as const },
  ] satisfies NetSpec[]) {
    it(`matches numerical gradients (${spec.depth} layers, skip ${spec.skip}, ${spec.act ?? 'tanh'})`, () => {
      const d = makeSpirals(12, 1, 0.02, createRng(3)), p = initParams(spec, createRng(9)), g = new Float64Array(p.length);
      lossGrad(spec, p, d, g);
      let worst = 0;
      for (let k = 0; k < p.length; k += 3) {
        const h = 1e-5, o = p[k]; p[k] = o + h; const a = lossOf(spec, p, d); p[k] = o - h; const b = lossOf(spec, p, d); p[k] = o;
        worst = Math.max(worst, Math.abs((a - b) / (2 * h) - g[k]) / (1e-6 + Math.abs(g[k]) + Math.abs((a - b) / (2 * h))));
      }
      expect(worst).toBeLessThan(1e-4);
    });
  }
  it('reduces the loss with Adam', () => {
    const spec = { depth: 2, width: 6, skip: false }, d = makeSpirals(40, 1, 0.02, createRng(1)), p = initParams(spec, createRng(2));
    const g = new Float64Array(p.length), a = newAdam(p.length), l0 = lossGrad(spec, p, d, g);
    for (let i = 0; i < 100; i++) { lossGrad(spec, p, d, g); adamStep(p, g, a, 0.02); }
    expect(lossOf(spec, p, d)).toBeLessThan(l0);
  });
  it('a skip block with zero weights is the identity', () => {
    const spec = { depth: 2, width: 4, skip: true }, p = initParams(spec, createRng(1)), d = makeSpirals(8, 1, 0, createRng(1));
    const before = lossOf(spec, p, d);
    p.fill(0, 12, 52); // stem is 2*4+4 = 12 numbers; two hidden layers are 20 each
    const flat = { depth: 0, width: 4, skip: false }, q = new Float64Array(12 + 5);
    q.set(p.subarray(0, 12)); q.set(p.subarray(52), 12);
    expect(lossOf(spec, p, d)).toBeCloseTo(lossOf(flat, q, d), 10);
    expect(lossOf(spec, p, d)).not.toBeCloseTo(before, 6);
  });
});

describe('degradation experiment', () => {
  const runs = {} as Record<NetId, ReturnType<typeof runNet>>;
  it('is deterministic and the plain deep net trains worse than the shallow one', () => {
    for (const n of NETS) runs[n.id] = runNet(n.id, 2);
    expect(runNet('shallow', 2).loss[STEPS]).toBe(runs.shallow.loss[STEPS]);
    expect(runs.deep.loss[STEPS]).toBeGreaterThan(10 * runs.shallow.loss[STEPS]);
    expect(Math.min(...runs.deep.loss.slice(50))).toBeGreaterThan(0.6); // stays near the coin-flip loss ln 2 = 0.693
    expect(Math.max(...runs.deep.loss.slice(50))).toBeLessThan(0.71);
    expect(runs.resnet.loss[STEPS]).toBeLessThan(runs.shallow.loss[STEPS]);
    // lesson prose: at the start the plain net's first-layer gradient is about a fifth of its last-layer gradient
    expect(runs.deep.gradFirst / runs.deep.gradLast).toBeGreaterThan(0.15);
    expect(runs.deep.gradFirst / runs.deep.gradLast).toBeLessThan(0.3);
    expect(runs.deep.snaps.length).toBe((STEPS / SNAP + 1) * GRID * GRID);
  }, 60000);
  it('frameAt endpoints', () => {
    const a = degradationFrame(0, runs), b = degradationFrame(DEG_T, runs);
    expect(a.step).toBe(0); expect(b.step).toBe(STEPS); expect(b.snap).toBe(STEPS / SNAP - 1); expect(b.f).toBeCloseTo(1, 9);
    expect(b.loss[2]).toBe(runs.resnet.loss[STEPS]);
  });
});

describe('block', () => {
  const b = makeBlock();
  it('at zero weights the residual block returns its input and the plain layer returns zeros', () => {
    const f = blockAt(8, b);
    expect(f.w).toBe(0); expect(f.residGap).toBe(0); expect(f.plain.every((v) => v === 0)).toBe(true);
    expect(f.plainGap).toBeCloseTo(Math.hypot(...b.x), 9); expect(f.x.length).toBe(BLOCK_D);
  });
  it('full weights give a nonzero branch', () => { expect(blockAt(0, b).residGap).toBeGreaterThan(0.1); });
});

describe('history', () => {
  it('reveals networks in order', () => {
    expect(historyAt(0)).toEqual([0, 0, 0, 0]);
    expect(historyAt(100).every((v) => v === 1)).toBe(true);
    expect(ARCHS.map((a) => a.year)).toEqual([...ARCHS.map((a) => a.year)].sort());
  });
});
