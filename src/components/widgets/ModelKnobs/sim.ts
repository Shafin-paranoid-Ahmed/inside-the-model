/** A model is a function with knobs: price = w · area + b. Pure logic, no DOM. */

export const HOUSES = [
  { x: 45, y: 130 }, { x: 62, y: 165 }, { x: 78, y: 190 }, { x: 90, y: 215 },
  { x: 110, y: 255 }, { x: 125, y: 280 }, { x: 150, y: 335 }, { x: 175, y: 390 },
];
/** The three houses whose predictions the readout shows. */
export const EXAMPLES = [60, 100, 140];
export const X_MAX = 200, Y_MAX = 450;
export const W_RANGE: [number, number] = [0, 4], B_RANGE: [number, number] = [-50, 150];
/** The knob values the sweep is centred on (a decent fit to HOUSES). */
export const GOOD = { w: 2, b: 35 };
export const SWEEP = 6; // seconds per phase

export const predict = (w: number, b: number, x: number) => w * x + b;
export const mse = (w: number, b: number) => HOUSES.reduce((s, h) => s + (predict(w, b, h.x) - h.y) ** 2, 0) / HOUSES.length;

export type Phase = 'slope' | 'intercept' | 'both';
export const duration = (knobs: 1 | 2) => (knobs === 1 ? SWEEP : 3 * SWEEP);
export const markers = (knobs: 1 | 2) =>
  knobs === 1 ? [{ at: 0, label: 'turn w' }] : [{ at: 0, label: 'turn w' }, { at: SWEEP, label: 'turn b' }, { at: 2 * SWEEP, label: 'both' }];

/** The auto-sweep. Every phase starts and ends at GOOD, so the loop is seamless. */
export function frameAt(t: number, knobs: 1 | 2): { w: number; b: number; phase: Phase } {
  const T = duration(knobs);
  const tt = ((t % T) + T) % T;
  const ph = Math.min(2, Math.floor(tt / SWEEP));
  const a = 2 * Math.PI * ((tt - ph * SWEEP) / SWEEP);
  if (ph === 0) return { w: GOOD.w + 1.5 * Math.sin(a), b: GOOD.b, phase: 'slope' };
  if (ph === 1) return { w: GOOD.w, b: GOOD.b + 70 * Math.sin(a), phase: 'intercept' };
  return { w: GOOD.w + 1.2 * Math.sin(a), b: GOOD.b + 55 * Math.sin(2 * a), phase: 'both' };
}
