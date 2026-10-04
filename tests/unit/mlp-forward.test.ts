import { describe, it, expect } from 'vitest';
import {
  buildNet, trace, forward, paramCount, breakdown, columns, f2, frameAt, markersOf, restAt,
  matmulPlan, matmulFrame, forwardPlan, forwardFrame, paramsPlan, paramsFrame, dwPlan, dwFrame, makeMoons,
  accuracy, DATA, DW_ARCHS, DW_T, MM_SEG, TRAINED, archKey, GX, GY, FW_LEAD, FW_LAYER, PA_SEG,
} from '../../src/components/widgets/MLPForward/sim';
import { trainNet, round4 } from '../../src/components/widgets/MLPForward/train';

describe('networks from the shared engine', () => {
  it('trace agrees with the plain forward pass', () => {
    const { net, mlp } = buildNet([2, 5, 4, 1], 9);
    const x = [0.3, -0.8];
    const tr = trace(mlp, x);
    expect(tr.h[3][0]).toBeCloseTo(forward(net, x)[0], 12);
    expect(tr.h[1].map((h, j) => Math.abs(h - Math.tanh(tr.z[0][j])) < 1e-12).every(Boolean)).toBe(true);
    expect(tr.h[0]).toEqual(x);
  });
  it('is seeded and deterministic', () => {
    expect(buildNet([2, 3, 1], 4).net.W).toEqual(buildNet([2, 3, 1], 4).net.W);
    expect(buildNet([2, 3, 1], 4).net.W).not.toEqual(buildNet([2, 3, 1], 5).net.W);
  });
  it('set overwrites the init', () => {
    const { net } = buildNet([2, 1], 1, 'tanh', { W: [[[1, 2]]], b: [[3]] });
    expect(forward(net, [1, 1])[0]).toBe(6);
  });
});

describe('parameter counting', () => {
  it('matches n_in*n_out + n_out per layer and the engine parameter list', () => {
    for (const sizes of [[2, 8, 1], [2, 4, 4, 1], [2, 3, 3, 3, 1], [2, 8, 8, 8, 8, 1]]) {
      expect(paramCount(sizes)).toBe(buildNet(sizes, 1).mlp.parameters().length);
      expect(breakdown(sizes).reduce((s, l) => s + l.total, 0)).toBe(paramCount(sizes));
    }
    expect(paramCount([2, 8, 1])).toBe(33);
    expect(paramCount([2, 4, 4, 1])).toBe(37);
    expect(paramCount([2, 3, 3, 3, 1])).toBe(37);
    expect(breakdown([2, 4, 4, 1]).map((l) => [l.w, l.b])).toEqual([[8, 4], [16, 4], [4, 1]]);
  });
});

describe('matmul mode', () => {
  const P = matmulPlan();
  it('has the numbers of the lesson text', () => {
    expect(P.z.map(f2)).toEqual(['0.30', '−0.45', '0.85']);
    expect(P.h.map(f2)).toEqual(['0.29', '−0.42', '0.69']);
    expect(P.h[1]).toBeCloseTo(Math.tanh(-0.45), 12);
    expect(P.W[1]).toEqual([-0.7, 0.9]); // second row of the backprop lessons' W1
  });
  it('is the matrix product W x + b, row by row', () => {
    P.W.forEach((row, j) => expect(row[0] * P.x[0] + row[1] * P.x[1] + P.b[j]).toBeCloseTo(P.z[j], 12));
  });
  it('frame endpoints and the counting', () => {
    const f0 = matmulFrame(0, P);
    expect(f0.rows.every((r) => r.zA === 0 && r.hA === 0)).toBe(true);
    const end = matmulFrame(P.T, P);
    expect(end.all).toBe(true);
    end.rows.forEach((r, j) => { expect(r.zShown).toBeCloseTo(P.z[j], 12); expect(r.hShown).toBeCloseTo(P.h[j], 12); });
    // neuron 2 midway through its dot product: neuron 1 is finished, neuron 3 untouched
    const mid = matmulFrame(MM_SEG + 0.25 * MM_SEG, P);
    expect(mid.active).toBe(1); expect(mid.sub).toBe('dot');
    expect(mid.rows[0].hShown).toBeCloseTo(P.h[0], 12);
    expect(mid.rows[2].zA).toBe(0);
    // The negative first term arrives before the positive second term cancels part of it.
    expect(mid.rows[1].zShown).toBeCloseTo(P.terms[1][0] * mid.rows[1].tA[0] + P.terms[1][1] * mid.rows[1].tA[1], 12);
    expect(mid.rows[1].tA[1]).toBe(0);
    expect(matmulFrame(MM_SEG * 1.9, P).sub).toBe('act');
  });
});

describe('forward mode', () => {
  const P = forwardPlan();
  it('has the numbers of the lesson text', () => {
    expect(P.tr.z[0].map(f2)).toEqual(['0.30', '−0.45']);
    expect(P.tr.h[1].map(f2)).toEqual(['0.29', '−0.42']);
    expect(P.tr.z[1].map(f2)).toEqual(['0.64', '−0.04']);
    expect(P.tr.h[2].map(f2)).toEqual(['0.57', '−0.04']);
    expect(f2(P.tr.h[3][0])).toBe('0.64');
    expect(P.tr.z[2][0]).toBe(P.tr.h[3][0]); // linear output
  });
  it('hand check of layer 2', () => {
    const [a, b] = P.tr.h[1];
    expect(P.tr.z[1][0]).toBeCloseTo(0.8 * a - 0.5 * b + 0.2, 12);
    expect(P.tr.z[1][1]).toBeCloseTo(0.4 * a + 0.6 * b + 0.1, 12);
  });
  it('first layer reuses the backprop lessons weights', () => {
    expect(P.net.W[0]).toEqual([[0.5, -0.6], [-0.7, 0.9]]);
  });
  it('frames: layer index follows the clock, values end exact', () => {
    expect(forwardFrame(0.5, P).layer).toBe(0);
    expect(forwardFrame(FW_LEAD + 0.1, P).layer).toBe(1);
    expect(forwardFrame(FW_LEAD + FW_LAYER * 1.5, P).layer).toBe(2);
    expect(forwardFrame(FW_LEAD + FW_LAYER * 2.5, P).layer).toBe(3);
    const end = forwardFrame(P.T, P);
    expect(end.layer).toBe(4);
    expect(end.nodes[3][0].h).toBeCloseTo(P.tr.h[3][0], 12);
    expect(end.nodes[2].map((n) => n.z)).toEqual(P.tr.z[1].map((z) => expect.closeTo(z, 12)));
    const s = forwardFrame(0, P);
    expect(s.nodes[1].every((n) => n.zA === 0 && n.hA === 0)).toBe(true);
    expect(s.nodes[0].map((n) => n.h)).toEqual([1, 0.5]);
  });
  it('the pulse travels the layer that is running, and only that one', () => {
    const f = forwardFrame(FW_LEAD + 0.25 * FW_LAYER, P);
    expect(f.pulse[0].flat().every((p) => p > 0 && p < 1)).toBe(true);
    expect(f.pulse[1].flat().every((p) => p === -1)).toBe(true);
    expect(f.labelA[0]).toBeGreaterThan(0.9);
    expect(f.labelA[1]).toBe(0);
  });
});

describe('params mode', () => {
  it('totals and frame endpoints', () => {
    const P = paramsPlan(2, 4);
    expect(P.sizes).toEqual([2, 4, 4, 1]);
    expect(P.total).toBe(37);
    const f0 = paramsFrame(0, P), end = paramsFrame(P.T, P);
    expect(f0.total).toBe(0);
    expect(end.total).toBe(37);
    expect(end.active).toBe(-1);
    expect(end.layers.map((l) => [l.w, l.b])).toEqual([[8, 4], [16, 4], [4, 1]]);
  });
  it('the running total never decreases and ends on the layer sums', () => {
    const P = paramsPlan(4, 8);
    let prev = -1;
    for (let t = 0; t <= P.T; t += 0.05) { const f = paramsFrame(t, P); expect(f.total).toBeGreaterThanOrEqual(prev); prev = f.total; }
    expect(prev).toBe(249);
    expect(paramsFrame(PA_SEG * 1.2, P).active).toBe(1);
  });
  it('markers cover every layer plus the total', () => {
    const P = paramsPlan(3, 5);
    expect(markersOf(P).length).toBe(P.layers.length + 1);
    expect(markersOf(P).at(-1)!.label).toBe('Total');
  });
});

describe('depth-width mode', () => {
  it('trained weights are in place and fit the data', () => {
    const P = dwPlan();
    expect(P.archs.map((a) => a.params)).toEqual([33, 37, 37]);
    for (const a of P.archs) expect(a.acc).toBeGreaterThanOrEqual(0.95);
    expect(P.archs.every((a) => a.grid.length === GX * GY)).toBe(true);
  });
  it('hardcoded weights equal a fresh deterministic training run', () => {
    for (const a of DW_ARCHS) {
      const fresh = round4(trainNet(a.sizes, DATA, 3, 1500)), stored = TRAINED[archKey(a.sizes)];
      fresh.W.flat(2).forEach((w, i) => expect(w).toBeCloseTo(stored.W.flat(2)[i], 2));
      expect(accuracy(buildNet(a.sizes, 3, 'tanh', stored).net)).toBeGreaterThanOrEqual(0.95);
    }
  });
  it('mounting is cheap and cached', () => {
    const t0 = performance.now();
    const P = dwPlan();
    expect(performance.now() - t0).toBeLessThan(50);
    expect(dwPlan()).toBe(P);
  });
  it('the moons are seeded and balanced', () => {
    expect(makeMoons()).toEqual(makeMoons());
    expect(DATA.y.reduce((s, y) => s + y, 0)).toBe(DATA.y.length / 2);
  });
  it('frames cycle through the architectures', () => {
    const P = dwPlan();
    expect(dwFrame(0, P).idx).toBe(0);
    expect(dwFrame(DW_T / 3 + 0.1, P).idx).toBe(1);
    expect(dwFrame(DW_T, P).idx).toBe(2);
    const f = dwFrame(DW_T / 3 + 4 * 0.9, P);
    expect(f.paramsShown).toBe(37);
    expect(dwFrame(0.01, P).paramsShown).toBeGreaterThanOrEqual(33);
    expect(f.thumbA[0]).toBe(1);
  });
});

describe('generic', () => {
  it('columns spaces layers across the rect', () => {
    const c = columns([2, 3, 1], { x: 10, y: 0, w: 100, h: 100 });
    expect(c[0][0].x).toBe(10); expect(c[2][0].x).toBe(110);
    expect(c[1].map((p) => p.y)).toEqual([16, 50, 84]);
    expect(columns([2, 3], { x: 0, y: 0, w: 50, h: 80 }, true)[1][0].y).toBe(80);
  });
  it('frameAt is deterministic and pure; restAt is a valid time', () => {
    for (const P of [matmulPlan(), forwardPlan(), paramsPlan(2, 4), dwPlan()]) {
      expect(JSON.stringify(frameAt(1.7, P))).toBe(JSON.stringify(frameAt(1.7, P)));
      expect(restAt(P)).toBeGreaterThan(0); expect(restAt(P)).toBeLessThanOrEqual(P.T);
      expect(markersOf(P).every((m, i, a) => m.at < P.T && (i === 0 || m.at > a[i - 1].at))).toBe(true);
    }
  });
});
