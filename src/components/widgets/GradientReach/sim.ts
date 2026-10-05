/** How much gradient reaches the first step of a plain tanh RNN, as the sequence grows. Pure logic, no DOM. */

export const SCALES = [0.5, 0.8, 1, 1.5, 2] as const;
export const N_MIN = 2, N_MAX = 30;
/** The sequence the rows were measured on (cut to n characters), and the network: RNNUnroller's, cell 'rnn', seed 11. */
export const SEQ = 'hello world hello world hello world';

/**
 * REACH[i][n - N_MIN] = log10(‖∂L/∂h_1‖ / ‖∂L/∂h_n‖) for weight scale SCALES[i], loss at the last step.
 * Precomputed so the page ships no autograd; tests/unit/gradient-reach.test.ts recomputes every value with RNNUnroller's sim.
 */
export const REACH: readonly (readonly number[])[] = [
  [-0.308, -0.706, -1.15, -1.563, -1.735, -2.005, -2.655, -2.852, -3.308, -3.485, -3.972, -4.446, -4.811, -5.185, -5.39, -5.962, -6.142, -6.512, -6.925, -7.286, -7.512, -7.881, -8.31, -8.767, -9.033, -9.495, -9.631, -10.179, -10.331],
  [-0.109, -0.313, -0.543, -0.799, -0.835, -0.879, -1.483, -1.311, -1.558, -1.548, -2.052, -2.395, -2.367, -2.863, -2.652, -3.412, -3.113, -3.209, -3.638, -3.625, -3.718, -3.853, -4.402, -4.629, -4.521, -5.285, -4.919, -5.597, -5.266],
  [-0.019, -0.138, -0.263, -0.436, -0.423, -0.404, -1.029, -0.792, -0.746, -0.832, -1.221, -1.61, -1.47, -1.731, -1.59, -2.375, -2.243, -2.06, -2.554, -2.312, -2.342, -2.484, -2.913, -3.109, -2.775, -3.215, -3.407, -3.944, -3.459],
  [0.132, 0.149, 0.192, 0.206, 0.305, 0.181, -0.169, -0.077, 0.454, 0.331, 0.441, 0.51, 0.573, 0.764, 0.769, 0.69, 0.546, 0.352, 0.422, 0.652, -0.051, 0.349, -0.221, 0.511, 0.486, 0.059, 0.228, 0.138, 0.602],
  [0.234, 0.32, 0.45, 0.549, 0.758, 0.386, 0.285, 0.362, 0.493, 1.119, 0.812, 0.447, 0.12, 0.456, 0.647, 0.025, 0.659, 0.431, 0.333, 0.618, 0.307, 0.144, 0.317, 0.214, 0.576, -0.771, 0.336, -0.691, 0.67],
];

/** Timeline: the sequence grows from N_MIN to N_MAX over GROW seconds, then holds. */
export const GROW = 7, T = 9;
export const MARKERS = [{ at: 0, label: 'longer sequences' }, { at: GROW, label: 'length 30' }];

const ease = (u: number) => { const c = Math.min(1, Math.max(0, u)); return c * c * (3 - 2 * c); };
/** Sequence length drawn at time t (fractional while growing). */
export const lengthAt = (t: number) => N_MIN + (N_MAX - N_MIN) * ease(t / GROW);

/** log10 reach for scale row i at a fractional length n, linear between measured lengths. */
export function reachAt(i: number, n: number): number {
  const x = Math.min(N_MAX, Math.max(N_MIN, n)) - N_MIN, k = Math.min(REACH[i].length - 2, Math.floor(x)), f = x - k;
  return REACH[i][k] + (REACH[i][k + 1] - REACH[i][k]) * f;
}

/** A reach as plain words: a percentage below 1, a multiple above. */
export function share(log: number): string {
  const v = 10 ** log;
  if (v >= 1) return `${v < 10 ? v.toFixed(1) : Math.round(v)}×`;
  const p = v * 100;
  return p >= 10 ? `${Math.round(p)}%` : p >= 0.01 ? `${p.toPrecision(2)}%` : `10${sup(Math.round(log))}`;
}
const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
/** Axis label for 10^e: "1×" at 0, else 10 with a superscript exponent. */
export const pow10 = (e: number) => (e === 0 ? '1×' : `10${sup(e)}`);
const sup = (n: number) => String(n).split('').map((ch) => SUP[ch]).join('');
