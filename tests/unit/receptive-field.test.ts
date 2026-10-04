import { describe, expect, test } from 'vitest';
import {
  EFF, EFF_T, buildEffective, buildScene, centerPick, durationOf, effDepthAt, effStats, effectiveMap, frameAt, makeEffNet,
  readoutAt, region, rfTable, ringMass, sizes, stackRf, unitNear,
} from '../../src/components/widgets/ReceptiveField/sim';
import { conv2d, map3, relu } from '../../src/lib/cnn/conv';

describe('receptive field recurrence', () => {
  test('stacked 3x3 stride-1 layers give 1 + L (k - 1)', () => {
    const t = rfTable(buildScene('stack').specs);
    t.forEach((e, l) => expect(e.r).toBe(stackRf(l)));
    expect(t[6].r).toBe(13);
    expect(rfTable(buildScene('one').specs)[1].r).toBe(3);
  });

  test('a pooling layer makes later layers count double', () => {
    expect(rfTable(buildScene('stride').specs).map((e) => e.r)).toEqual([1, 3, 4, 8, 12, 16]);
    expect(rfTable(buildScene('stride', true).specs).map((e) => e.r)).toEqual([1, 3, 5, 7, 9, 11]);
  });

  test('traced region has the same width as r at the input, for every top unit', () => {
    for (const mode of ['one', 'stack', 'stride'] as const)
      for (const plain of [false, true]) {
        const sc = buildScene(mode, plain), L = sc.specs.length, r = rfTable(sc.specs)[L].r;
        for (let u = 0; u < sizes(sc)[L]; u++) {
          const reg = region(sc, u);
          expect(reg[0][1] - reg[0][0] + 1).toBe(r);
          expect(reg[0][1]).toBeLessThan(sc.n);
        }
      }
  });

  test('field found by brute force: perturbing outside the traced region never changes the unit', () => {
    const sc = buildScene('stride');
    const run = (x: number[]) => { // 1-D conv with weights 1, pool = max with k = 2
      let v = x;
      for (const s of sc.specs) {
        const out: number[] = [];
        for (let i = 0; i + s.k <= v.length; i += s.stride) out.push(s.label.startsWith('pool') ? Math.max(...v.slice(i, i + s.k)) : v.slice(i, i + s.k).reduce((a, b) => a + b, 0));
        v = out;
      }
      return v;
    };
    const base = run(Array(sc.n).fill(1)), u = 1, [lo, hi] = region(sc, u)[0];
    for (let i = 0; i < sc.n; i++) {
      const x = Array(sc.n).fill(1); x[i] = 5;
      const changed = run(x)[u] !== base[u];
      expect(changed).toBe(i >= lo && i <= hi);
    }
  });

  test('unitNear returns the unit closest to a pixel and clamps', () => {
    const sc = buildScene('stack');
    expect(unitNear(sc, 6)).toBe(0);
    expect(unitNear(sc, 8)).toBe(2);
    expect(unitNear(sc, -5)).toBe(0);
    expect(unitNear(sc, 99)).toBe(4);
    expect(centerPick(sc)).toEqual({ col: 2, row: 2 });
  });
});

describe('frameAt', () => {
  const sc = buildScene('stack');
  test('endpoints: only the chosen unit at t = 0, the whole field at the end', () => {
    const a = frameAt(0, sc), b = frameAt(durationOf(sc), sc);
    expect(a.phase).toBe('pick'); expect(a.back).toBe(0);
    expect(a.reveal.slice(0, 6).every((v) => v === 0)).toBe(true);
    expect(b.phase).toBe('done'); expect(b.back).toBe(6);
    expect(b.reveal.every((v) => v === 1)).toBe(true);
  });
  test('readout cells follow 1 + 2 j', () => {
    for (let j = 0; j <= 6; j++) {
      const fr = frameAt(1.2 + j + 0.001, sc);
      expect(readoutAt(sc, centerPick(sc), fr).cells).toBe(1 + 2 * j);
    }
  });
  test('deterministic', () => expect(frameAt(3.3, sc)).toEqual(frameAt(3.3, sc)));
});

describe('effective receptive field', () => {
  const net = makeEffNet();
  test('gradient of the real stack matches a finite difference at a few pixels', () => {
    const depth = 3, n = EFF.n, c = n >> 1, img = net.images[0];
    const out = (x: typeof img) => {
      let v = x;
      for (let d = 0; d < depth; d++) { v = conv2d(v, net.layers[d]); if (d < depth - 1) v = relu(v); }
      return v.data[c * n + c];
    };
    const g = effectiveMap({ layers: net.layers, images: [img] }, depth);
    for (const [y, x] of [[c, c], [c - 1, c + 2], [c + 2, c - 2]]) {
      const h = 1e-6, a = map3(1, n, n, img.data.slice()), b = map3(1, n, n, img.data.slice());
      a.data[y * n + x] += h; b.data[y * n + x] -= h;
      const num = (out(a) - out(b)) / (2 * h);
      expect(Math.abs(num)).toBeCloseTo(g[y * n + x], 5);
    }
  });

  test('gradient is exactly zero outside the theoretical square', () => {
    const n = EFF.n, c = n >> 1;
    for (const depth of [1, 4, 9]) {
      const g = effectiveMap(net, depth);
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++)
        if (Math.max(Math.abs(x - c), Math.abs(y - c)) > depth) expect(g[y * n + x]).toBe(0);
    }
  });

  test('effective core is concentrated: it stays well inside the theoretical square', () => {
    const e = buildEffective();
    expect(e.stats.map((s) => s.theory)).toEqual(Array.from({ length: EFF.maxDepth }, (_, i) => 2 * i + 3));
    expect(e.stats.every((s) => s.core <= s.theory)).toBe(true);
    expect(e.stats[0].core).toBe(3);
    const last = e.stats[EFF.maxDepth - 1];
    expect(last.theory).toBe(25);
    expect(last.core).toBeLessThan(0.7 * last.theory);
    // the core grows more slowly than the theory
    expect(last.core / e.stats[0].core).toBeLessThan(last.theory / e.stats[0].theory);
    // deterministic
    expect(buildEffective().stats).toEqual(e.stats);
  });

  test('ringMass is cumulative and effStats picks the first ring reaching 90%', () => {
    const n = 9, m = new Float64Array(n * n).fill(1);
    const r = ringMass(m, n);
    expect(r[4]).toBe(81); expect(r[0]).toBe(1); expect(r[1]).toBe(9);
    expect(effStats(m, 4, 0.9, n).core).toBe(9); // 49 of 81 is only 60 percent
  });

  test('timeline depth steps 1..max', () => {
    expect(effDepthAt(0)).toBe(1);
    expect(effDepthAt(EFF_T)).toBe(EFF.maxDepth);
  });
});
