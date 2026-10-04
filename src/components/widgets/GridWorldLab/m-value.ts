/**
 * value (rl-basics step 5): iterative policy evaluation of the book's optimal policy at gamma 1. Every sweep replaces
 * each cell's value by the chance-weighted average of (reward + old value of where the move lands). The panel shows that
 * sum for one focus cell (the start by default); the grid cross-fades to the new values at the start of each sweep.
 * Props: focus ("1,1"), sweeps (20), gamma (1), stepReward (-0.04), slip.
 * Measured: V(start) by sweep 1..12 = -0.040 -0.080 -0.120 -0.160 0.128 0.326 0.482 0.579 0.636 0.669 0.686 0.696, final 0.705.
 */
import { ARROWS, START, cellName, transitions, type Grid } from '../../../lib/rl/gridworld';
import { optimal, backupTerms, deterministic, policyEvaluation } from '../../../lib/rl/tabular';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, ease, fmt, text, type Env } from '../_rl/svg';
import { focusOf, gridOf } from './sim';
import { drawGrid, drawOutcomes, drawValues, gridIn } from './draw';

const LEAD = 0.9, SLOW = 1.5, FAST = 0.5, SLOW_UNTIL = 12, HOLD = 3;
interface Scene { g: Grid; V: Float64Array[]; delta: number[]; focus: number; action: number; K: number; final: number; starts: number[]; n1: number }
interface F extends Frame { k: number; m: number }
const startOf = (k: number) => (k <= 0 ? 0 : LEAD + (Math.min(k - 1, SLOW_UNTIL) * SLOW + Math.max(0, k - 1 - SLOW_UNTIL) * FAST));
const len = (k: number) => (k <= SLOW_UNTIL ? SLOW : FAST);

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const g = gridOf(o, 1), K = int(o.sweeps, 20), focus = focusOf(o, START), pol = optimal(g, g.gamma).policy;
    const { V, delta } = policyEvaluation(g, deterministic(pol), K), final = optimal(g, g.gamma).V[focus];
    let n1 = 0;
    for (let k = 1; k <= K; k++) { if (V[k][focus] > V[k - 1][focus]) { n1 = k - 1; break; } }
    return { g, V, delta, focus, action: pol[focus], K, final, starts: Array.from({ length: K + 1 }, (_, k) => startOf(k)), n1 };
  },
  timeline(s) {
    const m = [{ at: 0, label: 'All zero' }];
    if (s.n1 >= 1) m.push({ at: s.starts[1], label: s.n1 === 1 ? 'Sweep 1' : `Sweeps 1 to ${s.n1}` });
    const settled = s.V.findIndex((v, k) => k > 0 && s.V.slice(k).every((w) => Math.abs(w[s.focus] - s.final) < 0.005));
    if (s.n1 >= 1 && s.n1 + 1 < settled) m.push({ at: s.starts[s.n1 + 1], label: `Sweeps ${s.n1 + 1} to ${settled - 1}` });
    if (settled > 0 && settled <= s.K) m.push({ at: s.starts[settled], label: `Sweeps ${settled} on` });
    return { duration: s.starts[s.K] + len(s.K) + HOLD, markers: m };
  },
  frameAt(t, s) {
    let k = 0;
    while (k < s.K && t >= s.starts[k + 1]) k++;
    const u = k === 0 ? 0 : t - s.starts[k], m = k === 0 ? 1 : ease(u / 0.3), v = s.V[k][s.focus];
    const status = k === 0
      ? 'Every cell starts with value 0, as if nothing were known. Each sweep replaces every value using the values from the sweep before.'
      : `Sweep ${k}: the value of ${cellName(s.focus)} goes from ${fmt(s.V[k - 1][s.focus], 3)} to ${fmt(v, 3)}, ` +
        (k <= s.n1 ? 'because the goal is still too far away to matter: each sweep only adds the cost of one more move.' : v > s.V[k - 1][s.focus] ? 'as news of the goal travels back along the route.' : 'as the values settle.');
    return { k, m, status, readouts: [['Sweep', `${k} of ${s.K}`], [`Value of ${cellName(s.focus)}`, fmt(v, 3)], ['Largest change in this sweep', k ? fmt(s.delta[k - 1], 3) : '–']] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, gb = nar ? { x: 8, y: 8, w: E.W - 16, h: E.H * 0.5 } : { x: 8, y: 8, w: E.W * 0.56, h: E.H - 16 };
    const G = gridIn(gb), k = f.k;
    drawGrid(E, G, { highlight: s.focus });
    drawValues(E, G, s.V[Math.max(0, k - 1)], { digits: 2 });
    if (k > 0) drawValues(E, G, s.V[k], { digits: 2, show: () => f.m });
    drawOutcomes(E, G, s.focus, transitions(s.g, s.focus, s.action), { labels: false, maxWidth: 6, op: 0.9 });
    // the backup panel: the sum that produced this sweep's value of the focus cell
    const tx = nar ? 12 : G.x0 + G.w + 24, ty = nar ? gb.y + gb.h + 26 : 30, bt = backupTerms(s.g, s.V[Math.max(1, k) - 1], s.focus, s.action);
    text(E, `Sweep ${Math.max(1, k)}: new value of ${cellName(s.focus)}, moving ${ARROWS[s.action]}`, tx, ty, { weight: 650 });
    const cols = nar ? [0, 44, 100, 214] : [0, 46, 106, 222], y0 = ty + 26;
    ['Chance', 'Lands in', 'Reward + old value', 'Adds'].forEach((h, j) => text(E, h, tx + cols[j], y0, { size: 11, fill: MUTED }));
    bt.terms.forEach((o, i) => {
      const y = y0 + 22 + i * 21;
      text(E, fmt(o.p, 1), tx + cols[0], y, { mono: true });
      text(E, o.s2 === s.focus ? 'stays' : cellName(o.s2), tx + cols[1], y);
      text(E, `${fmt(o.r, 2)} + ${fmt(o.v, 3)}`, tx + cols[2], y, { mono: true });
      text(E, fmt(o.term, 3), tx + cols[3], y, { mono: true });
    });
    text(E, `New value = ${fmt(bt.sum, 3)}`, tx + cols[3] - 70, y0 + 28 + bt.terms.length * 21 + 8, { weight: 650, mono: true });
  },
  label: (s) => `The 4 by 3 grid with the value of every cell after each sweep of policy evaluation; after ${s.K} sweeps the value of ${cellName(s.focus)} is ${s.V[s.K][s.focus].toFixed(3)}.`,
};
export default mode;
