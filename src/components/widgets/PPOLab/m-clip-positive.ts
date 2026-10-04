/**
 * clip-positive (ppo step 3) and, through clipMode, clip-negative (step 4): the clipped surrogate for one advantage.
 * The ratio cursor sweeps from the left end of `ratios` to the right and rests at each of `stops` (hold), while both
 * curves trace. For A > 0 the clipped curve flattens above 1 + eps; for A < 0 it flattens below 1 - eps.
 * Props: advantage, epsilon (0.2), ratios ("0.5,1.6" = the plotted range), stops (positive "1.2,1.5", negative "0.6,1.4").
 * Exact: A 1, eps 0.2 gives 1.2 at both rho 1.2 and rho 1.5; A -1 gives -0.8 at rho 0.6 and -1.4 at rho 1.4.
 */
import { type Frame, type ModeImpl, type Opts, nums } from '../_rl/kit';
import { fmt, type Env } from '../_rl/svg';
import { surrogatePlot } from './draw';
import { clippedSurrogate, curveOpts } from './sim';

const SWEEP = 7, REST = 1.6, HOLD = 2;
type Scene = ReturnType<typeof curveOpts> & { stops: number[] };
interface F extends Frame { rho: number }

const rhoAt = (t: number, s: Scene): number => {
  let x = s.lo;
  for (const st of [...s.stops, s.hi]) {
    const d = (SWEEP * (st - x)) / (s.hi - s.lo);
    if (t <= d) return x + (st - x) * (d ? t / d : 1);
    t -= d;
    if (st === s.hi) break;
    if (t <= REST) return st;
    t -= REST;
    x = st;
  }
  return s.hi;
};
/** Time at which the cursor first reaches ratio r. */
const arrive = (r: number, s: Scene) => (SWEEP * (r - s.lo)) / (s.hi - s.lo) + REST * s.stops.filter((x) => x < r - 1e-9).length;

export const clipMode = (A0: number, stops0: string): ModeImpl<Scene, F> => ({
  setup: (o: Opts) => {
    const c = curveOpts(o, A0);
    return { ...c, stops: nums(o.stops ?? stops0, []).filter((x) => x > c.lo && x < c.hi).sort((a, b) => a - b) };
  },
  timeline: (s) => ({
    duration: SWEEP + REST * s.stops.length + HOLD,
    markers: [{ at: 0, label: 'Below the band' }, { at: arrive(1 - s.eps, s), label: 'Inside the band' }, { at: arrive(1 + s.eps, s) + 1e-3, label: 'Above the band' }],
  }),
  frameAt(t, s) {
    const rho = rhoAt(t, s), c = clippedSurrogate(rho, s.A, s.eps), r2 = fmt(rho, 2), rr = Math.round(rho * 100) / 100, hiB = 1 + s.eps, loB = 1 - s.eps, A = s.A;
    const un = fmt(rho * A, 2), val = fmt(c.value, 2), good = A > 0 ? 'good' : 'bad';
    let status: string;
    if (rr > hiB + 1e-9) {
      status = A > 0
        ? `ρ = ${r2} is above 1 + ε = ${fmt(hiB, 2)}: the clipped objective stays at ${val} while ρ·A would give ${un}. Raising this good action further earns nothing, so the gradient is 0.`
        : `ρ = ${r2} is above the band, but the advantage is negative: the minimum picks ρ·A = ${val}, which keeps getting worse as the bad action grows more likely. The gradient is still ${fmt(c.dRho, 0)}.`;
    } else if (rr < loB - 1e-9) {
      status = A > 0
        ? `ρ = ${r2} is below the band: the minimum picks ρ·A = ${val}, so making a good action rarer still costs objective. The gradient is still ${fmt(c.dRho, 0)}.`
        : `ρ = ${r2} is below 1 − ε = ${fmt(loB, 2)}: the clipped objective stays at ${val} while ρ·A would give ${un}. Making this ${good} action even rarer earns nothing more, so the gradient is 0.`;
    } else status = `ρ = ${r2} is inside the band: both curves agree at ${val} and the gradient is A = ${fmt(A, 0)}.`;
    return { rho, status, readouts: [['Ratio ρ', r2], ['ρ·A', fmt(rho * A, 3)], ['Clipped objective', fmt(c.value, 3)], ['Slope d/dρ', fmt(c.dRho, 0)]] };
  },
  draw(E: Env, f, s) {
    const b = E.narrow ? { x: 46, y: 26, w: E.W - 60, h: E.H - 76 } : { x: 56, y: 28, w: E.W - 80, h: E.H - 76 };
    surrogatePlot(E, b, { ...s, rho: f.rho, upto: f.rho });
  },
  label: (s) => `The PPO clipped objective for an advantage of ${s.A} against the probability ratio from ${s.lo} to ${s.hi}: it is flat ${s.A > 0 ? 'above ' + fmt(1 + s.eps, 2) : 'below ' + fmt(1 - s.eps, 2)}.`,
});

const mode = clipMode(1, '1.2,1.5');
export default mode;
