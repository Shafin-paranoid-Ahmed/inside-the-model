/**
 * qvalues (q-learning step 1): the converged Q*(s, a) of the 4x3 world. Each action cell is split into four triangles
 * (up, right, down, left) coloured by Q; cells fade in one by one (start cell first), the best triangle is outlined,
 * and a side panel lists the four numbers. A last segment visits the cells where the top two actions nearly tie.
 * Props: gamma (1), stepReward (-0.04), slip (0.2).
 */
import { ACTIONS, cellName } from '../../../lib/rl/gridworld';
import { type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, ease, fmt, text, type Env } from '../_rl/svg';
import { gridOf } from './sim';
import { drawGrid, drawQ, gridIn } from './draw';
import { START_FIRST, actionBars, qName, qStar, rank } from './q-common';

const CELL = 1.5, FADE = 0.6, TIE = 0.02, TIE_T = 2.2;
interface Scene { Q: Float64Array; ties: number[]; pi: Int8Array }
interface F extends Frame { cur: number; shown: Set<number>; fade: number; tieSeg: boolean }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const g = gridOf(o), q = qStar(kit, g);
    return { Q: q.Q, pi: q.policy, ties: START_FIRST.filter((s) => rank(q.Q, s).gap < TIE) };
  },
  timeline(s) {
    const walk = START_FIRST.length * CELL;
    return {
      duration: walk + s.ties.length * TIE_T + 1.5,
      markers: [{ at: 0, label: 'Start cell' }, { at: CELL, label: 'Other cells' }, { at: walk, label: 'Near ties' }],
    };
  },
  frameAt(t, s) {
    const n = START_FIRST.length, walk = n * CELL, tieSeg = t >= walk;
    const i = Math.min(n - 1, Math.floor(t / CELL));
    const cur = tieSeg ? s.ties[Math.min(s.ties.length - 1, Math.floor((t - walk) / TIE_T))] : START_FIRST[i];
    const shown = new Set(tieSeg ? START_FIRST : START_FIRST.slice(0, i + 1));
    const fade = tieSeg ? 1 : ease((t - i * CELL) / FADE);
    const r = rank(s.Q, cur), name = cellName(cur), q = (a: number) => fmt(s.Q[cur * 4 + a], 3);
    let status: string;
    if (!tieSeg) {
      status = i === 0
        ? `At the start cell ${name}, going ${ACTIONS[r.best]} is worth ${q(r.best)}: the best you can expect from here, counting the step costs and the slips. The runner-up, ${ACTIONS[r.second]}, is worth ${q(r.second)}.`
        : `At ${name} the best action is ${ACTIONS[r.best]} (${q(r.best)}); the runner-up is ${ACTIONS[r.second]} (${q(r.second)}), ${fmt(r.gap, 3)} behind.`;
    } else {
      status = `${name} is a near tie: ${ACTIONS[r.best]} is worth ${q(r.best)} and ${ACTIONS[r.second]} ${q(r.second)}, only ${fmt(r.gap, 3)} apart. That is less than half the price of one move (0.04), so a slightly different route or a few unlucky slips could swap them.`;
    }
    return { cur, shown, fade, tieSeg, status, readouts: [['Cell', name], ['Best action', ACTIONS[r.best]], ['Lead over runner-up', fmt(r.gap, 3)]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, gb = nar ? { x: 8, y: 6, w: E.W - 16, h: E.W * 0.74 } : { x: 10, y: 10, w: E.W * 0.58, h: E.H - 20 };
    const G = gridIn(gb);
    drawGrid(E, G, { highlight: f.cur });
    drawQ(E, G, s.Q, { digits: 3, show: (c) => (c === f.cur ? f.fade : f.shown.has(c) ? 1 : 0), tip: (c, a) => `${qName(c, a)}: Q = ${fmt(s.Q[c * 4 + a], 3)}` });
    const pb = nar ? { x: 14, y: G.y0 + G.h + 14, w: E.W - 28, h: 130 } : { x: G.x0 + G.w + 28, y: 30, w: E.W - (G.x0 + G.w + 28) - 14, h: 130 };
    const op = f.tieSeg ? 1 : f.fade;
    const yEnd = actionBars(E, pb, s.Q, f.cur, { op });
    if (f.tieSeg) { const r = rank(s.Q, f.cur); text(E, `Gap ${fmt(r.gap, 3)} < 0.04, one move's cost`, pb.x, yEnd + 20, { size: 12, fill: MUTED }); }
  },
  label: () => 'The 4 by 3 grid with each cell split into four triangles coloured by the optimal action value Q star; the best action in each cell is outlined.',
};
export default mode;
