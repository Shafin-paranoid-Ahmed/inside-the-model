import { describe, expect, it } from 'vitest';
import { backward, Value } from '../../src/lib/nn';
import {
  BASE, DEPTH, WIDTH, T, frameAt, presetScale, signalStats, symmetryExperiment,
} from '../../src/components/widgets/InitializationLab/sim';

function sharedEngineGradients(weights: number[]) {
  const x = [new Value(1), new Value(0.5)];
  const params = weights.map((w) => new Value(w));
  const hidden = [0, 1, 2].map((j) => params[j * 2].mul(x[0]).add(params[j * 2 + 1].mul(x[1])).tanh());
  const prediction = params[6].mul(hidden[0]).add(params[7].mul(hidden[1])).add(params[8].mul(hidden[2]));
  const loss = prediction.sub(0.8).pow(2).mul(0.5);
  backward(loss);
  return { params, loss: loss.data };
}

function objective(weights: number[]) {
  const hidden = [0, 1, 2].map((j) => Math.tanh(weights[j * 2] + weights[j * 2 + 1] * 0.5));
  const prediction = weights[6] * hidden[0] + weights[7] * hidden[1] + weights[8] * hidden[2];
  return 0.5 * (prediction - 0.8) ** 2;
}

describe('InitializationLab simulation', () => {
  it('uses the equal-width Xavier and He variance targets', () => {
    expect(presetScale('xavier')).toBe(1);
    expect(presetScale('he')).toBe(Math.SQRT2);
    const xavier = signalStats(BASE, 'tanh', presetScale('xavier'));
    const he = signalStats(BASE, 'relu', presetScale('he'));
    expect(xavier.theoreticalWeightVariance).toBeCloseTo(1 / WIDTH, 14);
    expect(he.theoreticalWeightVariance).toBeCloseTo(2 / WIDTH, 14);
    for (let layer = 0; layer < DEPTH; layer++) {
      expect(he.measuredWeightVariance[layer] / xavier.measuredWeightVariance[layer]).toBeCloseTo(2, 12);
    }
  });

  it('reports real, deterministic activation variances and gradient norms', () => {
    const a = signalStats(BASE, 'tanh', 1);
    const b = signalStats(BASE, 'tanh', 1);
    expect(a.activationVariance).toHaveLength(DEPTH + 1);
    expect(a.gradientNorm).toHaveLength(DEPTH + 1);
    expect(a.activationVariance).toEqual(b.activationVariance);
    expect(a.gradientNorm).toEqual(b.gradientNorm);
    expect(a.gradientNorm[DEPTH]).toBeCloseTo(1, 12);
    expect(a.activationVariance.slice(1).some((v) => v > 0)).toBe(true);
  });

  it('zero scale gives zero hidden activations and early gradients in the bias-free stack', () => {
    const stats = signalStats(BASE, 'tanh', 0);
    expect(stats.activationVariance.slice(1).every((v) => v === 0)).toBe(true);
    expect(stats.gradientNorm.slice(0, DEPTH).every((v) => v === 0)).toBe(true);
    expect(stats.gradientNorm[DEPTH]).toBeCloseTo(1, 12);
  });

  it('keeps matched nonzero hidden rows matched and distinguishes a perturbed row', () => {
    const matched = symmetryExperiment('matched');
    expect(matched.before.hidden[0]).toBe(matched.before.hidden[1]);
    expect(matched.before.hiddenGradient[0]).toBe(matched.before.hiddenGradient[2]);
    expect(matched.before.hiddenGradient[1]).toBe(matched.before.hiddenGradient[3]);
    expect(matched.updatedWeights[0]).toBe(matched.updatedWeights[2]);
    expect(matched.updatedWeights[1]).toBe(matched.updatedWeights[3]);
    expect(matched.updatedWeights[6]).toBe(matched.updatedWeights[7]);

    const perturbed = symmetryExperiment('perturbed');
    expect(perturbed.before.hiddenGradient[0]).not.toBe(perturbed.before.hiddenGradient[2]);
  });

  it('has zero gradients only in the specified all-zero, both-layer tanh example', () => {
    const zero = symmetryExperiment('zero');
    expect(zero.before.prediction).toBe(0);
    expect(zero.before.hidden.every((h) => h === 0)).toBe(true);
    expect(zero.before.hiddenGradient.every((g) => g === 0)).toBe(true);
    expect(zero.before.outputGradient.every((g) => g === 0)).toBe(true);
    expect(zero.updatedWeights.every((w) => w === 0)).toBe(true);
  });

  it('matches the shared autograd engine and centered finite differences', () => {
    const exp = symmetryExperiment('matched');
    const weights = exp.initialWeights;
    const engine = sharedEngineGradients(weights);
    expect(engine.loss).toBeCloseTo(exp.before.loss, 14);
    const epsilon = 1e-6;
    for (let i = 0; i < weights.length; i++) {
      const expected = i < 6 ? exp.before.hiddenGradient[i] : exp.before.outputGradient[i - 6];
      expect(engine.params[i].grad).toBeCloseTo(expected, 12);
      const plus = [...weights], minus = [...weights];
      plus[i] += epsilon; minus[i] -= epsilon;
      const finiteDifference = (objective(plus) - objective(minus)) / (2 * epsilon);
      expect(expected).toBeCloseTo(finiteDifference, 7);
    }
  });

  it('maps timeline boundaries deterministically and clamps outside time', () => {
    const signalStart = frameAt(0, 'signals');
    const signalBack = frameAt(3, 'signals');
    const signalEnd = frameAt(T, 'signals');
    expect(signalStart.forwardReveal).toBe(0);
    expect(signalBack.phase).toBe('backward');
    expect(signalBack.backwardReveal).toBe(0);
    expect(signalEnd.backwardReveal).toBe(1);
    expect(frameAt(-2, 'signals')).toEqual(signalStart);
    expect(frameAt(T + 2, 'signals')).toEqual(signalEnd);

    expect(frameAt(2, 'symmetry').phase).toBe('backward');
    expect(frameAt(4, 'symmetry').phase).toBe('update');
    expect(frameAt(4, 'symmetry').updateProgress).toBe(0);
    expect(frameAt(T, 'symmetry').updateProgress).toBe(1);
  });

  it('recomputes metrics from interpolated symmetry weights and clamps progress', () => {
    const start = symmetryExperiment('matched', 0);
    const finish = symmetryExperiment('matched', 1);
    expect(start.shown.loss).toBe(start.before.loss);
    expect(finish.shown.loss).toBe(finish.after.loss);
    expect(symmetryExperiment('matched', -1).updateProgress).toBe(0);
    expect(symmetryExperiment('matched', 2).updateProgress).toBe(1);
  });
});
