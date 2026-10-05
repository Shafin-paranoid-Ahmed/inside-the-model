import { describe, it, expect } from 'vitest';
import { STATE, POS_SEQ, POSITION, CONTRIB, CONTRIB_SUM_OF_NORMS, CONTRIB_NORM_OF_SUM, FACTOR_S, FACTOR, factorAt, stateSteps, TIMING } from '../../src/components/widgets/SeqChart/sim';
import { makeWeights, run, tokens } from '../../src/components/widgets/RNNUnroller/sim';

const close = (a: readonly number[], b: readonly number[]) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 3));

describe('SeqChart data matches RNNUnroller', () => {
  const w1 = makeWeights('rnn', 1);
  it('state', () => {
    run(w1, tokens('hello', 8)).h.forEach((r, k) => close(STATE.hello[k], r));
    run(w1, tokens('help', 8)).h.forEach((r, k) => close(STATE.help[k], r));
  });
  it('position', () => close(POSITION, run(w1, tokens(POS_SEQ, 12)).jac));
  it('contrib', () => {
    const tr = run(w1, tokens('hello', 12));
    close(CONTRIB, tr.contrib);
    expect(CONTRIB_SUM_OF_NORMS).toBeCloseTo(tr.contrib.reduce((a, c) => a + c, 0), 3);
    expect(CONTRIB_NORM_OF_SUM).toBeCloseTo(Math.hypot(...tr.grads.Wh.h.flat()), 3);
  });
  it('factor', () => {
    for (const cell of ['rnn', 'lstm'] as const) close(FACTOR[cell], FACTOR_S.map((s) => run(makeWeights(cell, s), tokens('abcdefghijkl', 12)).factor));
  });
});

describe('SeqChart timeline', () => {
  it('state grows from 1 to 5 characters', () => {
    expect(stateSteps(0)).toBe(1);
    expect(stateSteps(TIMING.state.grow)).toBe(5);
  });
  it('factor interpolates between measured scales', () => {
    expect(factorAt(FACTOR.rnn, 0.5)).toBe(0.419);
    expect(factorAt(FACTOR.rnn, 2)).toBe(0.98);
    expect(factorAt(FACTOR.lstm, 1.05)).toBeCloseTo(0.812, 3);
  });
});
