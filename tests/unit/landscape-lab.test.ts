import { describe, expect, it } from 'vitest';
import {
  DIMENSIONS,
  DURATION,
  MARKERS,
  NONCONVEX_IN_WELL,
  frameAt,
  makeNonconvexPath,
  makeSaddlePaths,
  nonconvexGradient,
  nonconvexLoss,
  saddleGradient,
  saddleLoss,
  sampledGradient,
  tinyNetStats,
} from '../../src/components/widgets/LandscapeLab/sim';

function finiteDifference(fn: (x: number, y: number) => number, x: number, y: number, eps = 1e-5) {
  return {
    x: (fn(x + eps, y) - fn(x - eps, y)) / (2 * eps),
    y: (fn(x, y + eps) - fn(x, y - eps)) / (2 * eps),
  };
}

describe('LandscapeLab simulation', () => {
  it('matches closed-form gradients for both toy objectives', () => {
    const point = { x: 0.37, y: -0.62 };
    const doubleWell = finiteDifference((x, y) => nonconvexLoss({ x, y }), point.x, point.y);
    const saddle = finiteDifference((x, y) => saddleLoss({ x, y }), point.x, point.y);
    expect(nonconvexGradient(point).x).toBeCloseTo(doubleWell.x, 7);
    expect(nonconvexGradient(point).y).toBeCloseTo(doubleWell.y, 7);
    expect(saddleGradient(point).x).toBeCloseTo(saddle.x, 7);
    expect(saddleGradient(point).y).toBeCloseTo(saddle.y, 7);
    expect(saddleGradient({ x: 0, y: 0 })).toEqual({ x: 0, y: 0 });
  });

  it('keeps fixed sampled gradients deterministic and paths at their endpoints', () => {
    const p = { x: 0.2, y: -0.4 };
    expect(sampledGradient(p, 3)).toEqual(sampledGradient(p, 3));
    expect(makeNonconvexPath()).toEqual(makeNonconvexPath());
    const { still, escape } = makeSaddlePaths();
    expect(still.at(-1)).toEqual({ x: 0, y: 0, loss: 0, step: 90 });
    expect(Math.abs(escape.at(-1)!.y)).toBeGreaterThan(0.06);
    const first = makeNonconvexPath()[0], last = makeNonconvexPath().at(-1)!;
    expect(first.step).toBe(0);
    expect(last.step).toBe(100);
    expect(last.loss).toBeCloseTo(nonconvexLoss(last), 12);
  });

  it('returns exact timeline endpoints and a deterministic interpolated frame', () => {
    const start = frameAt(0, 'nonconvex');
    const end = frameAt(DURATION, 'nonconvex');
    expect(start.t).toBe(0);
    expect(start.path).toHaveLength(2);
    expect(end.t).toBe(DURATION);
    expect(end.path.at(-1)?.step).toBe(100);
    expect(frameAt(4.25, 'nonconvex')).toEqual(frameAt(4.25, 'nonconvex'));
    expect(frameAt(-2, 'nonconvex').t).toBe(0);
    expect(frameAt(20, 'nonconvex').t).toBe(DURATION);
  });

  it('preserves function values while coordinate curvature changes', () => {
    expect(frameAt(0, 'coordinates').coordinateScale).toBe(1);
    const frame = frameAt(DURATION, 'coordinates', { scale: 4 });
    expect(frame.mappedThetaPrime).toBeCloseTo(2.8, 12);
    expect(frame.originalLoss).toBeCloseTo(frame.rescaledLoss, 12);
    expect(frame.originalCurvature).toBe(2);
    expect(frame.rescaledCurvature).toBeCloseTo(0.125, 12);
    expect(frame.rescaledCurve[80].x).toBe(4);
    expect(frame.rescaledCurve[80].y).toBe(1);
  });

  it('shows residual identity branches in exact tiny-network Jacobians', () => {
    const baseline = tinyNetStats(0), amplified = tinyNetStats(0.5), cancelling = tinyNetStats(-1 / 1.2);
    expect(baseline.residualOutput).toEqual([-0.5, -0.25, 0, 0.25, 0.5]);
    expect(baseline.residualJacobian).toEqual([0.5, 0.5, 0.5, 0.5, 0.5]);
    expect(amplified.residualJacobian.at(-1)).toBeGreaterThan(baseline.plainJacobian.at(-1)!);
    expect(cancelling.residualJacobian[2]).toBeCloseTo(0, 12);
    expect(frameAt(DURATION, 'residual').tinyNet.alpha).toBe(0.5);
    expect(frameAt(1, 'residual').tinyNet.alpha).toBe(0);
    const x1 = tinyNetStats(0.5);
    expect(0.5 * x1.plainBlocks[0].derivative * x1.plainBlocks[1].derivative).toBeCloseTo(x1.plainJacobian[4], 12);

    const inputs = [-1, -0.5, 0, 0.5, 1];
    const plainOutput = (x: number) => 0.5 * Math.tanh(1.15 * Math.tanh(1.2 * x));
    const residualOutput = (x: number, alpha: number) => {
      const h1 = x + alpha * Math.tanh(1.2 * x);
      return 0.5 * (h1 + alpha * Math.tanh(1.15 * h1));
    };
    const diff = (fn: (x: number) => number, x: number) => (fn(x + 1e-5) - fn(x - 1e-5)) / 2e-5;
    for (const alpha of [0, 0.5, -1 / 1.2]) {
      const stats = tinyNetStats(alpha);
      const expectedPlain = inputs.map(plainOutput);
      const expectedResidual = inputs.map((x) => residualOutput(x, alpha));
      const mse = (values: number[]) => values.reduce((sum, output, i) => sum + (output - inputs[i]) ** 2, 0) / inputs.length;
      inputs.forEach((x, i) => {
        expect(stats.plainOutput[i]).toBeCloseTo(expectedPlain[i], 12);
        expect(stats.residualOutput[i]).toBeCloseTo(expectedResidual[i], 12);
        expect(stats.plainJacobian[i]).toBeCloseTo(diff(plainOutput, x), 7);
        expect(stats.residualJacobian[i]).toBeCloseTo(diff((value) => residualOutput(value, alpha), x), 7);
      });
      expect(stats.plainLoss).toBeCloseTo(mse(expectedPlain), 12);
      expect(stats.residualLoss).toBeCloseTo(mse(expectedResidual), 12);
    }
  });

  it('computes the independent-sign toy probability exactly', () => {
    expect(frameAt(DURATION, 'dimension').allPositiveProbability).toBe(1 / 4096);
    for (const [i, m] of MARKERS.dimension.entries()) {
      const f = frameAt(m.at, 'dimension');
      expect(f.dimension).toBe(DIMENSIONS[i]);
      expect(f.phase).toBe(`${DIMENSIONS[i]} directions`);
      expect(f.allPositiveProbability).toBe(1 / f.patterns);
    }
  });

  it('names timeline segments where the state changes', () => {
    const settle = MARKERS.nonconvex[1].at, path = makeNonconvexPath();
    expect(frameAt(settle, 'nonconvex').update).toBe(NONCONVEX_IN_WELL);
    expect(Math.abs(path[NONCONVEX_IN_WELL].x - 1)).toBeLessThan(0.1);
    expect(Math.abs(path[NONCONVEX_IN_WELL - 1].x - 1)).toBeGreaterThanOrEqual(0.1);
    expect(frameAt(settle - 0.01, 'nonconvex').phase).toBe('Downhill');
    expect(frameAt(MARKERS.saddle[2].at, 'saddle').saddleEscape.loss).toBeLessThan(-0.95);
    expect(frameAt(MARKERS.saddle[2].at - 0.2, 'saddle').saddleEscape.loss).toBeGreaterThan(-0.95);
    expect(frameAt(MARKERS.coordinates[1].at, 'coordinates').coordinateScale).toBe(1);
  });
});
