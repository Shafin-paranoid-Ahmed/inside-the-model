/**
 * Plain charts that break up Day 8's unrolled-chain figures. Pure logic, no DOM.
 * Every number comes from RNNUnroller's network (seed 11); tests/unit/seq-chart.test.ts recomputes them with its sim,
 * so the page ships no autograd.
 */

export type Kind = 'state' | 'position' | 'contrib' | 'factor';

/** state: hidden states h_1..h_n (4 units) for "hello" and "help", plain RNN, scale 1. */
export const STATE = {
  hello: [[-0.068, 0.66, -0.279, -0.128], [0.786, -0.012, -0.294, -0.357], [0.627, -0.581, 0.422, 0.157], [-0.065, -0.312, 0.78, 0.262], [-0.635, 0.744, 0.146, 0.238]],
  help: [[-0.068, 0.66, -0.279, -0.128], [0.786, -0.012, -0.294, -0.357], [0.627, -0.581, 0.422, 0.157], [-0.752, 0.046, 0.478, 0.431]],
};

/** position: ‖∂h_12/∂h_t‖ for "twelve steps" (scale 1): how much a nudge to the state after character t moves the final state. */
export const POS_SEQ = 'twelve steps';
export const POSITION = [0.226, 0.243, 0.281, 0.311, 0.344, 0.395, 0.456, 0.527, 0.595, 0.762, 0.91, 1];

/** contrib: ‖δ_t h_{t-1}ᵀ‖ for "hello", loss at the last step (scale 1), and the norms of their sum. */
export const CONTRIB = [0, 0.085, 0.148, 0.197, 0.198];
export const CONTRIB_SUM_OF_NORMS = 0.629, CONTRIB_NORM_OF_SUM = 0.4;

/** factor: geometric-mean change of the gradient per step going backward over "abcdefghijkl", against the recurrent scale s. */
export const FACTOR_S = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2];
export const FACTOR = {
  rnn: [0.419, 0.496, 0.572, 0.643, 0.709, 0.768, 0.817, 0.853, 0.867, 0.844, 0.756, 0.865, 0.911, 0.935, 0.956, 0.98],
  lstm: [0.81, 0.81, 0.81, 0.81, 0.811, 0.812, 0.812, 0.813, 0.814, 0.815, 0.816, 0.817, 0.818, 0.819, 0.82, 0.821],
};

export const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };

/** Timeline per kind: `grow` seconds of drawing, then a hold. */
export const TIMING: Record<Kind, { grow: number; T: number; markers: { at: number; label: string }[] }> = {
  state: { grow: 6, T: 8.5, markers: [{ at: 0, label: 'reading' }, { at: 3.6, label: 'l or p' }, { at: 6, label: 'whole word' }] },
  position: { grow: 6, T: 8.5, markers: [{ at: 0, label: 'last steps' }, { at: 6, label: 'all twelve' }] },
  contrib: { grow: 7, T: 9.5, markers: [{ at: 0, label: 'backward' }, { at: 5, label: 'add them up' }] },
  factor: { grow: 6, T: 8.5, markers: [{ at: 0, label: 'sweep s' }, { at: 6, label: 's = 2' }] },
};

/** state: steps drawn so far (1..5, fractional). */
export const stateSteps = (t: number) => 1 + 4 * ease(t / TIMING.state.grow);
/** position: bars revealed from the last step backward; returns how many (0..12, fractional). */
export const positionShown = (t: number) => 12 * ease(t / TIMING.position.grow);
/** contrib: the backward pass reaches step 5, 4, … 1 over the first 5 s, then the two totals fade in. */
export const contribShown = (t: number) => ({ bars: 5 * ease(t / 5), totals: ease((t - 5) / 1.5) });
/** factor: current s while sweeping 0.5 → 2. */
export const factorS = (t: number) => 0.5 + 1.5 * ease(t / TIMING.factor.grow);
/** Linear interpolation of a factor row at scale s. */
export function factorAt(row: readonly number[], s: number): number {
  const x = (Math.min(2, Math.max(0.5, s)) - 0.5) / 0.1, k = Math.min(row.length - 2, Math.floor(x)), f = x - k;
  return row[k] + (row[k + 1] - row[k]) * f;
}
