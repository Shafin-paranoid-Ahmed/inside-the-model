import { describe, expect, it } from 'vitest';
import {
  INPUT, MARKERS, T, augmentation, decayUpdates, earlyStopRun, enumerateMasks, expectation, forward,
  frameAt, sampleMasks,
} from '../../src/components/widgets/DropoutLab/sim';

describe('DropoutLab simulations', () => {
  it('enumerates 16 Bernoulli masks with probabilities summing to one', () => {
    const masks = enumerateMasks(0.35);
    expect(masks).toHaveLength(16);
    expect(new Set(masks.map(({ mask }) => mask.join(''))).size).toBe(16);
    expect(masks.reduce((sum, mask) => sum + mask.probability, 0)).toBeCloseTo(1, 12);
  });

  it('computes a reproducible Bernoulli training sequence distinct from exhaustive enumeration', () => {
    const first = sampleMasks(0.5), second = sampleMasks(0.5);
    expect(first).toEqual(second);
    expect(first).toHaveLength(16);
    expect(first.map(({ mask }) => mask.join(''))).not.toEqual(enumerateMasks(0.5).map(({ mask }) => mask.join('')));
  });

  it('preserves the linear readout expectation under inverted dropout', () => {
    for (const keep of [0.2, 0.5, 0.8, 1]) {
      const result = expectation(keep);
      expect(result.probabilityTotal).toBeCloseTo(1, 12);
      expect(result.linear).toBeCloseTo(result.full, 12);
    }
  });

  it('does not assert equality after a nonlinear output transform', () => {
    const result = expectation(0.5);
    expect(Math.abs(result.nonlinear - result.sigmoidFull)).toBeGreaterThan(1e-5);
  });

  it('rejects invalid keep probabilities instead of dividing by zero', () => {
    expect(() => forward(INPUT, [1, 0, 1, 0], 0)).toThrow(RangeError);
    expect(() => enumerateMasks(0)).toThrow(RangeError);
    expect(() => sampleMasks(1.01)).toThrow(RangeError);
  });

  it('shows SGD equivalence with matched L2 and decay, but distinct Adam updates', () => {
    const result = decayUpdates();
    expect(result.sgd).toBeCloseTo(0.96, 12);
    expect(result.sgdL2).toBeCloseTo(0.94, 12);
    expect(result.sgdL2).toBeCloseTo(result.theta * (1 - result.learningRate * result.lambda) - result.learningRate * result.gradient, 12);
    expect(result.adamCoupled).toBeCloseTo(0.9, 7);
    expect(result.adamW).toBeCloseTo(0.88, 7);
    expect(result.adamW).not.toBeCloseTo(result.adamCoupled, 3);
  });

  it('selects an actual best validation snapshot and stops after patience', () => {
    const run = earlyStopRun();
    expect(run.bestEpoch).toBe(4);
    expect(run.stopEpoch).toBe(7);
    expect(run.history[run.bestEpoch]).toEqual(run.selected);
    expect(run.stopEpoch).toBeGreaterThan(run.bestEpoch);
    expect(run.history[run.stopEpoch].validation).toBeGreaterThan(run.selected.validation);
    const at = (label: string) => MARKERS['early-stop'].find((m) => m.label === label)!.at;
    const beforeWait = frameAt(at('Waiting') - 0.01, 'early-stop');
    expect([beforeWait.visibleEpoch, beforeWait.phase, beforeWait.checkpoint.epoch]).toEqual([run.bestEpoch, 'Improving', run.bestEpoch]);
    const wait = frameAt(at('Waiting'), 'early-stop');
    expect([wait.visibleEpoch, wait.phase, wait.staleChecks]).toEqual([run.bestEpoch + 1, 'Waiting', 1]);
    const beforeStop = frameAt(at('Stop') - 0.01, 'early-stop');
    expect([beforeStop.visibleEpoch, beforeStop.phase]).toEqual([run.stopEpoch - 1, 'Waiting']);
    for (const time of [at('Stop'), T]) {
      const stop = frameAt(time, 'early-stop');
      expect([stop.visibleEpoch, stop.phase, stop.staleChecks]).toEqual([run.stopEpoch, 'Stop', 3]);
    }
  });

  it('reveals values only once they are drawn, in step with the timeline markers', () => {
    const early = frameAt(1, 'scaling');
    expect([early.showUnscaledMean, early.invertedShown, early.showFull]).toEqual([false, 0, false]);
    const inverted = frameAt(4.7, 'scaling');
    expect([inverted.phase, inverted.showUnscaledMean, inverted.showInvertedMean, inverted.showFull]).toEqual(['Inverted', true, true, false]);
    expect(frameAt(T, 'scaling').showFull).toBe(true);
    for (const [i, m] of MARKERS.decay.entries()) {
      expect(frameAt(m.at, 'decay').decayRows).toBe(i + 1);
      expect(frameAt(m.at, 'decay').phase).toBe(m.label);
    }
    expect(frameAt(T, 'decay').decayGrow).toBe(1);
    expect(frameAt(3.9, 'ensemble').showMean).toBe(false);
    expect(frameAt(3.9, 'ensemble').ensembleShown).toBe(16);
    expect(frameAt(4, 'augment').showAfterLabel).toBe(false);
    expect(frameAt(5, 'augment').flip).toBe(1);
  });

  it('starts the training animation on a typical mask', () => {
    expect(sampleMasks(0.5)[0].mask).toEqual([1, 0, 1, 0]);
    expect(frameAt(0, 'mask').network.output).toBeCloseTo(1.0233, 4);
  });

  it('applies a real horizontal flip and reports whether the target is invariant', () => {
    const presence = augmentation('presence');
    expect(presence.flipped).toEqual([[0, 1], [0, 0]]);
    expect(presence.labelPreserved).toBe(true);
    const side = augmentation('side');
    expect(side.labelBefore).toContain('left');
    expect(side.labelAfter).toContain('right');
    expect(side.labelPreserved).toBe(false);
    expect(side.wrongLabel).toBe(side.labelBefore);
  });

  it('clamps timeline time and keeps the computed frame deterministic', () => {
    expect(frameAt(-1, 'mask')).toEqual(frameAt(0, 'mask'));
    expect(frameAt(T + 2, 'ensemble')).toEqual(frameAt(T, 'ensemble'));
    expect(frameAt(2.1, 'mask')).toEqual(frameAt(2.1, 'mask'));
  });
});
