/**
 * loop (rl-basics step 1): one recorded episode of the uniformly random policy. The agent glides cell to cell; each
 * move appends a ticker row (state, action asked, move made, reward) and the running total counts.
 * Props: seed (default 34: 15 moves, one slip on move 1, reaches the goal, total +0.40), stepReward, slip.
 */
import { createRng } from '../../../lib/math/rng';
import { ACTIONS, ARROWS, DX, DY, GOAL, cellName, randomPolicy, rollout, xy, type Episode } from '../../../lib/rl/gridworld';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, arrow, ease, fmt, signFmt, table, text, type Env } from '../_rl/svg';
import { gridOf } from './sim';
import { cellXY, drawAgent, drawGrid, gridIn } from './draw';

const LEAD = 0.6, MOVE = 0.9, HOLD = 2.2;
interface Scene { ep: Episode; T: number; total: number[] }
interface F extends Frame { k: number; u: number; fx: number; fy: number; done: boolean }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const ep = rollout(gridOf(o), randomPolicy, createRng(int(o.seed, 34)), 100), total = [0];
    for (const r of ep.rewards) total.push(total[total.length - 1] + r);
    return { ep, T: ep.actions.length, total };
  },
  timeline(s) {
    const end = LEAD + s.T * MOVE, slip = s.ep.slipped.indexOf(true);
    const markers = [{ at: 0, label: slip === 0 ? 'Slip on move 1' : 'Moving' }];
    if (slip > 0) markers.push({ at: LEAD + slip * MOVE, label: `Slip on move ${slip + 1}` });
    if (slip >= 0 && slip + 4 < s.T) markers.push({ at: LEAD + (slip + 1) * MOVE, label: 'Moving on' });
    markers.push({ at: end, label: 'Episode over' });
    return { duration: end + HOLD, markers };
  },
  frameAt(t, s) {
    const { ep, T } = s, raw = (t - LEAD) / MOVE, k = Math.max(0, Math.min(T, Math.floor(raw))), done = raw >= T;
    const u = done || t < LEAD ? 0 : raw - k;
    // during move k (0-based) the agent glides from states[k] to states[k+1]; a bump nudges toward the wall and back
    const [x0, y0] = xy(ep.states[k]), [x1, y1] = xy(ep.states[Math.min(k + 1, T)]), m = ease(u / 0.6);
    let fx = x0 + (x1 - x0) * m, fy = y0 + (y1 - y0) * m;
    if (!done && t >= LEAD && ep.states[k] === ep.states[k + 1]) { const b = 0.18 * Math.sin(Math.PI * Math.min(1, u / 0.6)); fx += DX[ep.dirs[k]] * b; fy += DY[ep.dirs[k]] * b; }
    const n = done ? T : t < LEAD ? 0 : k + 1, total = s.total[n];
    let status: string;
    if (t < LEAD) status = `The agent starts at ${cellName(ep.states[0])}. It picks every action at random.`;
    else if (done) status = `The episode ends after ${T} moves in the ${ep.states[T] === GOAL ? '+1 goal' : '−1 pit'}. Total reward ${signFmt(total)}.`;
    else {
      const a = ep.actions[k], d = ep.dirs[k], moved = ep.states[k] !== ep.states[k + 1];
      status = `Move ${k + 1}: at ${cellName(ep.states[k])} the agent asks for ${ACTIONS[a]}` +
        (ep.slipped[k] ? `, but it slips ${ACTIONS[d]}` : '') + (moved ? '' : ' and bumps into a wall, so it stays put') + `. Reward ${fmt(ep.rewards[k])}.`;
    }
    return {
      k, u, fx, fy, done, status,
      readouts: [['Move', `${n} of ${T}`], ['Slips so far', String(ep.slipped.slice(0, n).filter(Boolean).length)], ['Total reward', signFmt(total)]],
    };
  },
  draw(E: Env, f, s) {
    const { ep, T } = s, narrow = E.narrow;
    const gb = narrow ? { x: 8, y: 8, w: E.W - 16, h: E.H * 0.5 } : { x: 12, y: 12, w: E.W * 0.56, h: E.H - 24 };
    const G = gridIn(gb);
    drawGrid(E, G, { highlight: f.done ? null : ep.states[f.k] });
    if (!f.done && f.u > 0) {
      // asked-for direction (muted) and the move actually made (forward colour)
      const c = cellXY(G, ep.states[f.k]), a = ep.actions[f.k], d = ep.dirs[f.k], L = G.cs * 0.42;
      if (ep.slipped[f.k]) arrow(E, c.cx, c.cy, c.cx + DX[a] * L, c.cy - DY[a] * L, { stroke: MUTED, width: 2, dash: '4 3' });
      arrow(E, c.cx, c.cy, c.cx + DX[d] * L, c.cy - DY[d] * L, { stroke: 'var(--viz-forward)', width: 2.5 });
    }
    drawAgent(E, G, f.fx, f.fy);
    // ticker of the last moves
    const n = f.done ? T : f.u > 0 || f.k > 0 ? Math.min(T, f.k + 1) : 0, rowsN = narrow ? 7 : 12, from = Math.max(0, n - rowsN);
    const tx = narrow ? 12 : gb.x + gb.w + 28, ty = narrow ? gb.y + gb.h + 26 : 28, cols = narrow ? [0, 34, 94, 168, 252] : [0, 34, 92, 162, 236];
    const rows = [];
    for (let i = from; i < n; i++) rows.push([String(i + 1), cellName(ep.states[i]), `${ARROWS[ep.actions[i]]} ${ACTIONS[ep.actions[i]]}`, ep.slipped[i] ? `${ARROWS[ep.dirs[i]]} slip` : ARROWS[ep.dirs[i]], fmt(ep.rewards[i])]);
    table(E, tx, ty, cols, rows, { header: ['#', 'State', 'Asked', 'Moved', 'Reward'], rh: 19, fill: (r) => (ep.slipped[from + r] ? 'var(--viz-neg)' : 'var(--text)') });
    if (n === 0) text(E, 'Moves appear here.', tx, ty + 22, { fill: MUTED });
  },
  label: (s) => `A 4 by 3 grid. A random agent starts bottom left and moves until it reaches the +1 goal or the −1 pit; this episode takes ${s.T} moves.`,
};
export default mode;
