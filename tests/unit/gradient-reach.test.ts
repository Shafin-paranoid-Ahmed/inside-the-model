import { describe, it, expect } from 'vitest';
import { SCALES, N_MIN, N_MAX, SEQ, REACH, reachAt, share, lengthAt, GROW } from '../../src/components/widgets/GradientReach/sim';
import { makeWeights, run, tokens } from '../../src/components/widgets/RNNUnroller/sim';

describe('GradientReach', () => {
  it('the precomputed table matches the RNNUnroller network', () => {
    SCALES.forEach((s, i) => {
      const w = makeWeights('rnn', s);
      for (let n = N_MIN; n <= N_MAX; n++) {
        const tr = run(w, tokens(SEQ, n));
        expect(REACH[i][n - N_MIN]).toBeCloseTo(Math.log10(tr.dhN[0] / tr.dhN[n - 1]), 2);
      }
    });
  });
  it('agrees with the lesson table at length 11 ("hello world")', () => {
    expect(share(reachAt(0, 11))).toBe('0.033%');
    expect(share(reachAt(1, 11))).toBe('2.8%');
    expect(share(reachAt(4, 11))).toBe('13×');
  });
  it('the timeline grows from 2 to 30 and holds', () => {
    expect(lengthAt(0)).toBe(N_MIN);
    expect(lengthAt(GROW)).toBe(N_MAX);
    expect(reachAt(2, 30)).toBeCloseTo(-3.459, 3);
  });
});
