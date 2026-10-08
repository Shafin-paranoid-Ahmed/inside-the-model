import { describe, expect, it } from 'vitest';
import * as MK from '../../src/components/widgets/ModelKnobs/sim';
import * as DT from '../../src/components/widgets/DataTable/sim';
import * as LP from '../../src/components/widgets/LossPlayground/sim';
import * as LF from '../../src/components/widgets/LineFit/sim';
import * as GD from '../../src/components/widgets/GradientDescent2D/sim';
import { contour, splitPanels } from '../../src/components/widgets/F1Shared/plot';

const numGrad = (f: (x: number, y: number) => number, x: number, y: number, h = 1e-5): [number, number] => [
  (f(x + h, y) - f(x - h, y)) / (2 * h), (f(x, y + h) - f(x, y - h)) / (2 * h),
];

describe('ModelKnobs', () => {
  it('predicts w*x+b', () => expect(MK.predict(2, 35, 100)).toBe(235));
  it('sweep starts and ends on the good fit in every phase, so it loops seamlessly', () => {
    for (const k of [1, 2] as const) {
      for (let p = 0; p < k * 1.5 + 1 && p * MK.SWEEP < MK.duration(k); p++) {
        const f = MK.frameAt(p * MK.SWEEP, k);
        expect(f.w).toBeCloseTo(MK.GOOD.w);
        expect(f.b).toBeCloseTo(MK.GOOD.b);
      }
      const end = MK.frameAt(MK.duration(k) - 1e-9, k);
      expect(end.w).toBeCloseTo(MK.GOOD.w, 3);
      expect(end.b).toBeCloseTo(MK.GOOD.b, 1);
    }
  });
  it('the good knobs fit the houses far better than a bad setting', () => expect(MK.mse(2, 35)).toBeLessThan(0.05 * MK.mse(0.5, 0)));
  it('sweep stays inside the slider ranges', () => {
    for (let t = 0; t < 18; t += 0.1) {
      const f = MK.frameAt(t, 2);
      expect(f.w).toBeGreaterThanOrEqual(MK.W_RANGE[0]);
      expect(f.w).toBeLessThanOrEqual(MK.W_RANGE[1]);
      expect(f.b).toBeGreaterThanOrEqual(MK.B_RANGE[0]);
      expect(f.b).toBeLessThanOrEqual(MK.B_RANGE[1]);
    }
  });
});

describe('DataTable', () => {
  it('one-hot encodes a category', () => expect(DT.oneHot('B')).toEqual([0, 1, 0]));
  it('age grouping finds three groups that are exactly the neighborhoods', () => {
    const g = DT.groups();
    expect(new Set(g).size).toBe(3);
    for (const h of DT.HOODS) expect(new Set(DT.ROWS.map((r, i) => (r.hood === h ? g[i] : -1)).filter((x) => x >= 0)).size).toBe(1);
  });
  it('at similar area, neighborhood C costs more than A (the one-hot columns carry signal)', () => {
    const of = (h: string) => DT.ROWS.filter((r) => r.hood === h);
    let pairs = 0;
    for (const a of of('A')) for (const c of of('C')) if (Math.abs(a.area - c.area) <= 30) { pairs++; expect(c.price).toBeGreaterThan(a.price + 50); }
    expect(pairs).toBeGreaterThanOrEqual(3);
  });
  it('all rows arrive by the end and none at the start', () => {
    expect(DT.frameAt(0).arrive.every((a) => a === 0)).toBe(true);
    expect(DT.frameAt(DT.duration()).arrive.every((a) => a === 1)).toBe(true);
    expect(DT.frameAt(DT.duration()).active).toBe(DT.ROWS.length - 1);
  });
  it('every value sits inside its axis range', () => {
    for (const r of DT.ROWS) {
      for (const f of DT.FEATURES) { expect(r[f.key]).toBeGreaterThanOrEqual(f.min); expect(r[f.key]).toBeLessThanOrEqual(f.max); }
      expect(r.price).toBeGreaterThanOrEqual(DT.PRICE.min);
      expect(r.price).toBeLessThanOrEqual(DT.PRICE.max);
    }
  });
});

describe('LossPlayground', () => {
  const pts = LP.CLEAN;
  it('mse and mae match hand computation', () => {
    const P = [{ x: 0, y: 1 }, { x: 1, y: 3 }];
    expect(LP.mse(1, 0, P)).toBeCloseTo((1 + 4) / 2); // residuals 1, 2
    expect(LP.mae(1, 0, P)).toBeCloseTo(1.5);
  });
  it('an outlier hurts MSE much more than MAE', () => {
    const w = 0.6, b = LP.bestB(w, pts);
    const r = (k: 'mse' | 'mae') => LP.lossValue(k, w, b, 1) / LP.lossValue(k, w, b, 0);
    expect(r('mse')).toBeGreaterThan(5 * r('mae'));
  });
  it('the MSE minimiser moves further than the MAE minimiser when the outlier appears', () => {
    const shift = (k: 'mse' | 'mae') => Math.abs(LP.curve(k, 1, 1).best - LP.curve(k, 1, 0).best);
    expect(shift('mse')).toBeGreaterThan(shift('mae'));
  });
  it('the sampled MSE minimum agrees with the closed form for fixed b', () => {
    // for fixed b, dMSE/dw = 0 at w = sum x (y - b) / sum x^2
    const b = 1, wStar = pts.reduce((s, p) => s + p.x * (p.y - b), 0) / pts.reduce((s, p) => s + p.x * p.x, 0);
    expect(LP.curve('mse', b, 0, 601).best).toBeCloseTo(wStar, 1);
  });
  it('log loss is 0.693 at p = 0.5 and grows for confident mistakes', () => {
    expect(LP.logLoss1(1, 0.5)).toBeCloseTo(Math.LN2);
    expect(LP.logLoss1(1, 0.01)).toBeGreaterThan(4);
    expect(LP.logLoss1(1, 1)).toBeCloseTo(0, 6);
    expect(Number.isFinite(LP.logLoss1(1, 0))).toBe(true);
  });
  it('logistic loss at w=0,b=0 is ln 2 for any labels', () => expect(LP.logloss(0, 0)).toBeCloseTo(Math.LN2));
  it('every view loops and starts with no outlier', () => {
    for (const v of ['residuals', 'squares', 'curve', 'outlier', 'classify'] as const) {
      const a = LP.frameAt(v, 0), z = LP.frameAt(v, LP.duration(v));
      expect(a.amt).toBe(0);
      expect(z.w).toBeCloseTo(a.w);
      expect(z.amt).toBeCloseTo(a.amt);
    }
  });
  it('only the outlier view moves the outlier, and it holds the line still', () => {
    for (const v of ['residuals', 'squares', 'curve'] as const) for (let t = 0; t < 6; t += 0.5) expect(LP.frameAt(v, t).amt).toBe(0);
    expect(LP.frameAt('outlier', 5).amt).toBeCloseTo(1);
    expect(LP.frameAt('outlier', 5).w).toBe(LP.frameAt('outlier', 0).w);
  });
});

describe('LineFit', () => {
  it('expands predictions, signed errors and MSE with hand-checkable values', () => {
    const c = LF.calculation(1, 1, [{ x: 2, y: 5 }, { x: 0, y: 0 }]);
    expect(c.rows.map((p) => [p.prediction, p.error, p.squared])).toEqual([[3, -2, 4], [1, 1, 1]]);
    expect(c.sum).toBe(5);
    expect(c.mse).toBe(2.5);
    expect(c.gw).toBe(-4);
    expect(c.gb).toBe(-1);
    expect(c.next.w).toBeCloseTo(1.32);
    expect(c.next.b).toBeCloseTo(1.08);
    expect(c.next.loss).toBeCloseTo((1.28 ** 2 + 1.08 ** 2) / 2);
  });
  it('the slow single update reaches exactly the first full-batch MSE update', () => {
    for (const start of [LF.START, { w: 2.5, b: -3 }]) {
      const path = LF.gdPath(start);
      expect(LF.frameAt(3, 'update', path).w).toBe(start.w);
      const end = LF.frameAt(8, 'update', path);
      expect(end.w).toBeCloseTo(path[1].w, 12);
      expect(end.b).toBeCloseTo(path[1].b, 12);
      expect(end.loss).toBeCloseTo(path[1].loss, 12);
      const middle = LF.frameAt(6, 'update', path);
      expect(middle.loss).toBeCloseTo(LF.calculation(middle.w, middle.b).mse, 12);
    }
  });
  it('the prediction and error demonstrations raise every prediction by one with fixed w', () => {
    for (const mode of ['prediction', 'errors'] as const) {
      const start = LF.frameAt(0, mode), end = LF.frameAt(10, mode);
      expect(end.w).toBe(start.w);
      const a = LF.calculation(start.w, start.b), z = LF.calculation(end.w, end.b);
      z.rows.forEach((row, i) => expect(row.prediction - a.rows[i].prediction).toBeCloseTo(1));
      expect(end.loss).toBeCloseTo(z.mse, 12);
      const edge = LF.gdPath({ w: 1, b: LF.B_DOM[1] });
      expect(LF.frameAt(10, mode, edge).b).toBe(LF.B_DOM[1]);
    }
  });
  it('gradient matches numerical', () => {
    for (const [w, b] of [[0, 0], [-1.4, -2.2], [2, 1]]) {
      const [gw, gb] = LF.grad(w, b), [nw, nb] = numGrad(LF.loss, w, b);
      expect(gw).toBeCloseTo(nw, 5);
      expect(gb).toBeCloseTo(nb, 5);
    }
  });
  it('closed form has zero gradient and beats nearby lines', () => {
    const { w, b } = LF.BEST, [gw, gb] = LF.grad(w, b);
    expect(Math.abs(gw)).toBeLessThan(1e-9);
    expect(Math.abs(gb)).toBeLessThan(1e-9);
    for (const d of [[0.05, 0], [-0.05, 0], [0, 0.05], [0, -0.05]]) expect(LF.loss(w + d[0], b + d[1])).toBeGreaterThan(LF.loss(w, b));
  });
  it('closed form recovers an exact line', () => {
    const P = [-1, 0, 2, 5].map((x) => ({ x, y: 3 * x - 2 }));
    const { w, b } = LF.closedForm(P);
    expect(w).toBeCloseTo(3);
    expect(b).toBeCloseTo(-2);
  });
  it('gradient descent converges to the closed form', () => {
    const end = LF.gdPath(LF.START, LF.LR, 400).at(-1)!;
    expect(end.w).toBeCloseTo(LF.BEST.w, 4);
    expect(end.b).toBeCloseTo(LF.BEST.b, 4);
  });
  it('the shipped path is near the solution and loss decreases monotonically', () => {
    for (let i = 1; i < LF.PATH.length; i++) expect(LF.PATH[i].loss).toBeLessThan(LF.PATH[i - 1].loss);
    expect(LF.PATH.at(-1)!.loss - LF.loss(LF.BEST.w, LF.BEST.b)).toBeLessThan(0.02);
  });
  it('every mode replays from a user-chosen start (path[0])', () => {
    const s = { w: 2.5, b: -3 }, path = LF.gdPath(s);
    for (const m of ['iterative', 'closed', 'tour'] as const) {
      const f = LF.frameAt(0, m, m === 'iterative' ? path : [{ ...s, loss: LF.loss(s.w, s.b) }]);
      expect(f.w).toBeCloseTo(s.w); expect(f.b).toBeCloseTo(s.b);
    }
    expect(LF.frameAt(LF.duration('iterative'), 'iterative', path).loss).toBeCloseTo(LF.loss(LF.BEST.w, LF.BEST.b), 2);
  });
  it('descent from any corner of the map reaches the minimum (the lesson says so)', () => {
    const min = LF.loss(LF.BEST.w, LF.BEST.b);
    for (const w of LF.W_DOM) for (const b of LF.B_DOM) expect(LF.gdPath({ w, b }).at(-1)!.loss - min).toBeLessThan(0.01);
  });
  it('the tour ends on the best line', () => {
    const f = LF.frameAt(LF.duration('tour'), 'tour');
    expect(f.w).toBeCloseTo(LF.BEST.w); expect(f.b).toBeCloseTo(LF.BEST.b);
  });
  it('frameAt endpoints', () => {
    expect(LF.frameAt(0, 'iterative').w).toBeCloseTo(LF.START.w);
    const T = LF.duration('iterative');
    expect(LF.frameAt(T, 'iterative').w).toBeCloseTo(LF.PATH.at(-1)!.w);
    expect(LF.frameAt(LF.duration('closed'), 'closed').w).toBeCloseTo(LF.BEST.w);
    expect(LF.frameAt(0, 'closed').b).toBeCloseTo(LF.START.b);
  });
});

describe('GradientDescent2D', () => {
  it('the regression surface and updates are exactly the preceding lesson MSE', () => {
    const s = GD.SURFACES.regression, tr = GD.trajectory(s, s.start, s.lr);
    tr.slice(0, LF.PATH.length).forEach((p, i) => {
      expect(p.x).toBe(LF.PATH[i].w); expect(p.y).toBe(LF.PATH[i].b);
      expect(p.f).toBe(LF.PATH[i].loss);
    });
    const mid = GD.frameAt(GD.DT / 2, tr, s.lr, s);
    expect(mid.f).toBeCloseTo(LF.loss(mid.x, mid.y), 12);
    expect(mid.f).not.toBeCloseTo((tr[0].f + tr[1].f) / 2, 2);
    const end = GD.frameAt(GD.DT, tr.slice(0, 2), s.lr, s);
    expect(end.x).toBe(tr[1].x); expect(end.f).toBeCloseTo(tr[1].f, 12);
    const boundary = GD.frameAt(4.8, tr, s.lr, s);
    expect(boundary.k).toBe(24); expect(boundary.u).toBe(0);
    expect(boundary.x).toBe(tr[24].x);
  });
  it('regression SGD uses actual sample gradients and their average equals the full gradient', () => {
    const s = GD.SURFACES.regression, tr = GD.trajectory(s, s.start, s.lr, true, 3);
    expect(tr).toEqual(GD.trajectory(s, s.start, s.lr, true, 3));
    tr.forEach((p) => {
      const g = LF.grad(p.x, p.y, [LF.POINTS[p.sample!]]);
      expect([p.gx, p.gy]).toEqual(g);
      const avg = LF.POINTS.map(q => LF.grad(p.x, p.y, [q])).reduce((v,g) => [v[0] + g[0] / 12, v[1] + g[1] / 12], [0, 0]);
      expect(avg[0]).toBeCloseTo(p.tx, 10); expect(avg[1]).toBeCloseTo(p.ty, 10);
    });
    expect(tr[0].sample).toBe(8);
    expect(tr[1].f).toBeCloseTo(10.56510395246707, 10);
    expect(tr.some((p, i) => i > 0 && p.f > tr[i - 1].f)).toBe(true);
  });
  it('the lesson learning rates straddle the regression stability boundary', () => {
    const s = GD.SURFACES.regression;
    expect(2 / s.hmax!).toBeCloseTo(0.272396, 6);
    const end = (lr: number) => GD.trajectory(s, s.start, lr).at(-1)!.f;
    expect(end(0.01)).toBeCloseTo(1.295606, 6);
    expect(end(0.26)).toBeCloseTo(0.202170, 6);
    expect(end(0.28)).toBeGreaterThan(3800);
  });
  it('analytic gradients match numerical ones on every surface', () => {
    for (const s of Object.values(GD.SURFACES)) {
      for (const [x, y] of [[0.3, -0.7], [1.9, 1.1], [-2.2, 0.4], [3, 2]]) {
        const [gx, gy] = s.grad(x, y), [nx, ny] = numGrad(s.f, x, y);
        expect(gx).toBeCloseTo(nx, 4);
        expect(gy).toBeCloseTo(ny, 4);
      }
    }
  });
  it('full-batch descent lowers the loss on a small step', () => {
    const tr = GD.trajectory(GD.SURFACES.bowl, [3, 1.3], 0.1);
    for (let i = 1; i < tr.length; i++) expect(tr[i].f).toBeLessThanOrEqual(tr[i - 1].f);
    expect(tr.at(-1)!.f).toBeLessThan(0.05);
  });
  it('quadratic diverges exactly when lr > 2/L', () => {
    const s = GD.SURFACES.bowl, crit = 2 / s.hmax!;
    expect(crit).toBeCloseTo(0.25);
    const long = (lr: number) => GD.trajectory(s, [3, 1.3], lr, false, 1, 300).at(-1)!.f;
    expect(long(crit * 0.99)).toBeLessThan(1e-3); // oscillates, but shrinks
    expect(long(crit * 1.01)).toBeGreaterThan(1e3); // grows without bound
    expect(GD.trajectory(s, [3, 1.3], 0.3, false, 1, 300).at(-1)!.diverged).toBe(true);
  });
  it('lr = 1/L takes the steep direction to the minimum in one step', () => {
    const tr = GD.trajectory(GD.SURFACES.bowl, [0, 2], 1 / 8);
    expect(tr[1].y).toBeCloseTo(0, 10);
  });
  it('starting side decides the minimum on the two-valley surface', () => {
    const s = GD.SURFACES.multi;
    const right = GD.trajectory(s, [1.2, 1], 0.2, false, 1, 300).at(-1)!, left = GD.trajectory(s, [-1.2, 1], 0.2, false, 1, 300).at(-1)!;
    expect(right.x).toBeGreaterThan(1.5);
    expect(left.x).toBeLessThan(-1.5);
    expect(left.f).toBeLessThan(right.f);
  });
  it('a start on the saddle ridge slides off it and reaches a minimum', () => {
    const end = GD.trajectory(GD.SURFACES.saddle, GD.SURFACES.saddle.start, 0.2, false, 1, 200).at(-1)!;
    expect(Math.abs(end.y)).toBeGreaterThan(2);
    expect(end.f).toBeCloseTo(-1.25, 2);
  });
  it('SGD is seeded and noisy: same seed same path, noise stops it settling exactly', () => {
    const a = GD.trajectory(GD.SURFACES.bowl, [3, 1.3], 0.1, true, 3), b = GD.trajectory(GD.SURFACES.bowl, [3, 1.3], 0.1, true, 3);
    expect(a).toEqual(b);
    const c = GD.trajectory(GD.SURFACES.bowl, [3, 1.3], 0.1, true, 4);
    expect(c[10].x).not.toBe(a[10].x);
    expect(a.at(-1)!.f).toBeGreaterThan(1e-4);
  });
  it('frameAt hits the trajectory points at step boundaries and is deterministic', () => {
    const s = GD.SURFACES.bowl, tr = GD.trajectory(s, s.start, s.lr);
    const f0 = GD.frameAt(0, tr, s.lr), f5 = GD.frameAt(5 * GD.DT, tr, s.lr);
    expect(f0.x).toBeCloseTo(s.start[0]);
    expect(f5.x).toBeCloseTo(tr[5].x);
    expect(GD.frameAt(GD.duration(), tr, s.lr).x).toBeCloseTo(tr.at(-1)!.x);
    expect(GD.frameAt(1.234, tr, s.lr)).toEqual(GD.frameAt(1.234, tr, s.lr));
  });
  it('the slice along -g has the ball at s = 0 and slopes down to the right', () => {
    const s = GD.SURFACES.bowl, [gx, gy] = s.grad(3, 1.3), sl = GD.slice(s, 3, 1.3, gx, gy, 1, 3);
    expect(sl.s[1]).toBe(0);
    expect(sl.v[2]).toBeLessThan(sl.v[1]);
  });
});

describe('F1Shared plot', () => {
  it('contour of a circle stays on the circle', () => {
    const n = 41, v = (i: number, j: number) => Math.hypot(i - 20, j - 20);
    const seg = contour(v, n, n, 10);
    expect(seg.length).toBeGreaterThan(100);
    for (let i = 0; i < seg.length; i += 2) expect(Math.hypot(seg[i] - 20, seg[i + 1] - 20)).toBeCloseTo(10, 0);
  });
  it('splits panels side by side when wide and stacked when narrow', () => {
    expect(splitPanels(720, 400).stacked).toBe(false);
    expect(splitPanels(360, 560).stacked).toBe(true);
  });
});
