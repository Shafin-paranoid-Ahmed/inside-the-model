import { describe, it, expect } from 'vitest';
import { lasso, ridge, G2, G6, W2_OLS, PATHS, N, P, LMAX, T, frameAt, ellipse, excess2 } from '../../src/components/widgets/RegularizationPaths/sim';

const Gw = (g: typeof G6, w: number[]) => g.G.map((r) => r.reduce((s, v, k) => s + v * w[k], 0));
const row = (pen: 'l1' | 'l2', i: number, m: number, key: 'w6' | 'w2') => Array.from(PATHS[pen][key].slice(i * m, i * m + m));

describe('solvers', () => {
  it('ridge satisfies the stationarity condition (G + lam I) w = c', () => {
    const w = ridge(G6, 0.7), gw = Gw(G6, w);
    gw.forEach((v, j) => expect(v + 0.7 * w[j]).toBeCloseTo(G6.c[j], 9));
  });
  it('lasso satisfies the KKT conditions', () => {
    const lam = 0.5, w = lasso(G6, lam), gw = Gw(G6, w);
    w.forEach((v, j) => {
      const r = G6.c[j] - gw[j];
      if (v !== 0) expect(r).toBeCloseTo(lam * Math.sign(v), 7);
      else expect(Math.abs(r)).toBeLessThanOrEqual(lam + 1e-9);
    });
  });
  it('lambda = 0 gives least squares; lambda = LMAX zeroes every lasso weight', () => {
    lasso(G2, 0).forEach((v, j) => expect(v).toBeCloseTo(W2_OLS[j], 6));
    expect(lasso(G6, LMAX + 1e-9).every((v) => v === 0)).toBe(true);
  });
});

describe('paths', () => {
  it('L1 snaps the 2-weight solution onto an axis; L2 never does', () => {
    const onAxis = (pen: 'l1' | 'l2') => Array.from({ length: N }, (_, i) => row(pen, i, 2, 'w2')).filter((w) => (w[0] === 0) !== (w[1] === 0)).length;
    expect(onAxis('l1')).toBeGreaterThan(5);
    expect(onAxis('l2')).toBe(0);
  });
  it('L1 zeroes some 6-feature weights exactly before the end; L2 keeps all non-zero', () => {
    const zerosAt = (pen: 'l1' | 'l2', i: number) => row(pen, i, P, 'w6').filter((v) => v === 0).length;
    expect(zerosAt('l1', Math.floor(N / 2))).toBeGreaterThan(0);
    expect(zerosAt('l1', N - 1)).toBe(P);
    expect(zerosAt('l2', N - 1)).toBe(0);
  });
  it('weight norm shrinks monotonically with lambda', () => {
    for (const pen of ['l1', 'l2'] as const) {
      let prev = Infinity;
      for (let i = 0; i < N; i++) {
        const n = row(pen, i, P, 'w6').reduce((s, v) => s + v * v, 0);
        expect(n).toBeLessThanOrEqual(prev + 1e-9);
        prev = n;
      }
    }
  });
  it('validation error has an interior minimum: some penalty beats none, too much is worse', () => {
    for (const pen of ['l1', 'l2'] as const) {
      const p = PATHS[pen];
      expect(p.best).toBeGreaterThan(0);
      expect(p.best).toBeLessThan(N - 1);
      expect(p.val[p.best]).toBeLessThan(p.val[0]);
      expect(p.val[p.best]).toBeLessThan(p.val[N - 1]);
      expect(p.train[N - 1]).toBeGreaterThan(p.train[0]);
    }
  });
});

describe('frameAt', () => {
  it('endpoints and determinism', () => {
    const a = frameAt(0, 'l1'), b = frameAt(T, 'l1');
    expect(a.lam).toBe(0);
    expect(a.w2[0]).toBeCloseTo(W2_OLS[0], 6);
    expect(b.lam).toBeCloseTo(LMAX, 9);
    expect(b.zeros).toBe(P);
    expect(frameAt(5.3, 'l2')).toEqual(frameAt(5.3, 'l2'));
  });
  it('the 2-weight solution lies on the constraint boundary it reports', () => {
    const f = frameAt(6, 'l2');
    expect(f.radius).toBeCloseTo(Math.hypot(...f.w2), 12);
  });
  it('ellipse points all have the requested excess loss', () => {
    const e = ellipse(0.7);
    for (let i = 0; i < e.length; i += 2) expect(excess2([e[i], e[i + 1]])).toBeCloseTo(0.7, 9);
  });
});
