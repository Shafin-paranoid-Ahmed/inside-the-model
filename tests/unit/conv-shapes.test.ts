import { describe, expect, test } from 'vitest';
import {
  CH_IN, CH_LAYER, CH_OUT, DEFAULT_SIZE, ST_IN, coverage, durationOf, frameAt, outDim, paramCount,
  partialSum, poolOutput, strideOutput, sweep, type Mode,
} from '../../src/components/widgets/ConvShapes/sim';
import { conv2d, outSize } from '../../src/lib/cnn/conv';

describe('ConvShapes sim', () => {
  test('partial sums over channels plus bias equal conv2d', () => {
    for (let f = 0; f < 2; f++) for (let oy = 0; oy < 4; oy++) for (let ox = 0; ox < 4; ox++) {
      const sum = [0, 1, 2].reduce((a, c) => a + partialSum(f, c, oy, ox), CH_LAYER.bias[f]);
      expect(sum).toBe(CH_OUT.data[(f * 4 + oy) * 4 + ox]);
    }
    expect(CH_OUT.c).toBe(2);
    expect(CH_OUT.h).toBe(4);
  });

  test('parameter count', () => {
    expect(paramCount(3, 2, 3)).toEqual({ perFilter: 27, withBias: 28, total: 56 });
    expect(paramCount(1, 1, 3).perFilter).toBe(9);
  });

  test('stride 2 outputs are every other stride 1 output', () => {
    const a = strideOutput(1), b = strideOutput(2);
    expect([a.h, b.h]).toEqual([5, 3]);
    for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) expect(b.data[y * 3 + x]).toBe(a.data[2 * y * 5 + 2 * x]);
    expect(ST_IN.h).toBe(7);
  });

  test('coverage: corner 1 vs 4, centre 9; counts sum to windows x 9 minus clipped', () => {
    const c0 = coverage(5, 3, 1, 0), c1 = coverage(5, 3, 1, 1);
    expect([c0[0], c0[12], c1[0], c1[12]]).toEqual([1, 9, 4, 9]);
    expect(c0.reduce((a, b) => a + b, 0)).toBe(9 * 9);
  });

  test('size formula and floor', () => {
    expect(outDim(8, 3, 2, 0)).toBe(outSize(8, 3, 2, 0));
    expect(outDim(8, 3, 2, 0)).toBe(3);
    expect(outDim(8, 3, 1, 1)).toBe(8);
    expect(outDim(3, 7, 1, 0)).toBe(0);
    const f = frameAt(1, 'size', DEFAULT_SIZE);
    expect(f.calc.lines[2].text).toBe('floor(2.5) + 1 = 3');
    expect(f.grids.find((g) => g.id === 'in')!.cells.filter((c) => c.kind === 'drop').length).toBe(8 * 8 - 7 * 7);
  });

  test('pooling halves the map and matches the core', () => {
    const mx = poolOutput('max');
    expect([mx.h, mx.w]).toEqual([3, 3]);
    expect(Array.from(mx.data)).toEqual([7, 1, 9, 6, 8, 5, 2, 9, 7]);
  });

  test('sweep glides then holds, and ends on the last window', () => {
    expect(sweep(0, 1, 4).cur).toBe(0);
    const g = sweep(1.1, 1, 4);
    expect([g.cur, g.to]).toEqual([0, 1]);
    expect(g.e).toBeLessThan(1);
    expect(sweep(1.6, 1, 4).cur).toBe(1);
    expect(sweep(99, 1, 4).cur).toBe(3);
  });

  test('frames are deterministic and well-formed for every mode', () => {
    for (const m of ['channels', 'stride', 'padding', 'size', 'pool'] as Mode[]) {
      const T = durationOf(m);
      for (const [t, narrow] of [[0, false], [T / 3, true], [T, false], [T, true]] as [number, boolean][]) {
        const a = frameAt(t, m, undefined, narrow), b = frameAt(t, m, undefined, narrow);
        expect(JSON.stringify(a)).toBe(JSON.stringify(b));
        for (const g of a.grids) expect(g.cells.length).toBe(g.rows * g.cols);
        for (const g of a.grids) { expect(g.x + g.cols * g.cell).toBeLessThanOrEqual(a.w); expect(g.y + g.rows * g.cell).toBeLessThanOrEqual(a.h); }
      }
    }
    expect(conv2d(CH_IN, CH_LAYER).data).toEqual(CH_OUT.data);
  });
});
