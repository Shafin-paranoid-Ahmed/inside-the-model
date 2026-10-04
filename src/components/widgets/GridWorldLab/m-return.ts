/**
 * return (rl-basics step 3): a fixed policy (the optimal one for the book's world) plays one seeded episode; then the
 * discounted return G_t of every move is filled in backward with G_t = r + gamma * G_(t+1).
 * Props: seed (3), gamma (0.9), stepReward, slip.
 * Measured (seed 3, gamma 0.9): 5 moves, no slip, G_0 = 0.492, plain sum 0.800.
 */
import { createRng } from '../../../lib/math/rng';
import { ACTIONS, DX, DY, cellName, fixedPolicy, makeGrid, returns, rollout, xy, type Episode } from '../../../lib/rl/gridworld';
import { optimal } from '../../../lib/rl/tabular';
import { int, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, ease, fmt, line, rect, signFmt, text, type Env } from '../_rl/svg';
import { at, drawAgent, drawGrid, drawPolicy, gridIn } from './draw';

const LEAD = 0.5, MOVE = 0.85, PAUSE = 0.8, BACK = 1.3, HOLD = 3;
interface Scene { ep: Episode; T: number; G: Float64Array; gamma: number; sum: number; policy: Int8Array; wEnd: number; bStart: number; bEnd: number }
interface F extends Frame { n: number; fx: number; fy: number; phase: 'walk' | 'back' | 'done'; row: number; known: number; fade: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const gamma = num(o.gamma, 0.9), sr = num(o.stepReward, -0.04), slip = num(o.slip, 0.2);
    const policy = optimal(makeGrid({ stepReward: sr, slip }), 1).policy;
    const ep = rollout(makeGrid({ stepReward: sr, slip, gamma }), fixedPolicy(policy), createRng(int(o.seed, 3)), 40);
    const T = ep.actions.length, G = returns(ep.rewards, gamma), wEnd = LEAD + T * MOVE, bStart = wEnd + PAUSE;
    return { ep, T, G, gamma, sum: ep.rewards.reduce((a, b) => a + b, 0), policy, wEnd, bStart, bEnd: bStart + T * BACK };
  },
  timeline: (s) => ({ duration: s.bEnd + HOLD, markers: [{ at: 0, label: 'Walk' }, { at: s.wEnd, label: 'Work backward' }, { at: s.bEnd, label: 'Compare' }] }),
  frameAt(t, s) {
    const { ep, T, G, gamma } = s;
    const phase: F['phase'] = t >= s.bEnd ? 'done' : t >= s.wEnd ? 'back' : 'walk';
    let n = 0, fx: number, fy: number, row = -1, known = 0, fade = 1, status: string;
    if (phase === 'walk') {
      const raw = (t - LEAD) / MOVE, k = Math.max(0, Math.min(T - 1, Math.floor(raw))), u = t < LEAD ? 0 : raw - k, m = ease(u / 0.6);
      const [x0, y0] = xy(ep.states[k]), [x1, y1] = xy(ep.states[k + 1]);
      fx = x0 + (x1 - x0) * m; fy = y0 + (y1 - y0) * m;
      if (t >= LEAD && ep.states[k] === ep.states[k + 1]) { const b = 0.18 * Math.sin(Math.PI * Math.min(1, u / 0.6)); fx += DX[ep.dirs[k]] * b; fy += DY[ep.dirs[k]] * b; }
      n = t < LEAD ? 0 : k + 1;
      status = n === 0 ? 'The agent follows the arrow in each cell. Every move earns a reward.'
        : `Move ${n}: from ${cellName(ep.states[n - 1])} ${ep.slipped[n - 1] ? `it slips and goes ${ACTIONS[ep.dirs[n - 1]]}` : `it goes ${ACTIONS[ep.actions[n - 1]]}`}. Reward ${fmt(ep.rewards[n - 1])}.`;
    } else {
      n = T;
      const j = t < s.bStart ? -1 : Math.min(T - 1, Math.floor((t - s.bStart) / BACK));
      row = j < 0 ? T : T - 1 - j; known = T - row;
      [fx, fy] = xy(ep.states[Math.min(row, T)]);
      if (phase === 'back' && j >= 0) fade = ease(((t - s.bStart) - j * BACK) / 0.3);
      if (j < 0) status = `The episode is over after ${T} moves. Now work backward from the end to find the return of each move.`;
      else if (phase === 'done') status = `The return from the start is ${fmt(G[0], 3)}; the plain sum of the same rewards is ${fmt(s.sum, 2)}. The big reward at the end counts for less from far away.`;
      else status = row === T - 1
        ? `Last move: nothing comes after it, so its return is just its reward, ${fmt(ep.rewards[row], 2)}.`
        : `Move ${row + 1}: return = reward + ${fmt(gamma, 1)} × (return of the next move) = ${fmt(ep.rewards[row], 2)} + ${fmt(gamma, 1)} × ${fmt(G[row + 1], 3)} = ${fmt(G[row], 3)}.`;
    }
    const readouts: [string, string][] = phase === 'walk' ? [['Move', `${n} of ${T}`]]
      : phase === 'back' ? [['Returns found', `${known} of ${T}`]] : [['Return from start', fmt(G[0], 3)], ['Plain sum', fmt(s.sum, 2)]];
    return { n, fx: fx!, fy: fy!, phase, row, known, fade, status, readouts };
  },
  draw(E: Env, f, s) {
    const { ep, T } = s, nar = E.narrow;
    const gb = nar ? { x: 8, y: 8, w: E.W - 16, h: E.H * 0.46 } : { x: 8, y: 8, w: E.W * 0.5, h: E.H - 16 };
    const G = gridIn(gb);
    drawGrid(E, G, { highlight: f.phase === 'back' && f.row < T ? ep.states[f.row] : null });
    drawPolicy(E, G, { actions: s.policy, color: 'var(--text-muted)' });
    for (let i = 0; i < f.n; i++) {
      const [x1, y1] = at(G, ...xy(ep.states[i])), [x2, y2] = at(G, ...xy(ep.states[i + 1]));
      if (x1 !== x2 || y1 !== y2) line(E, x1, y1, x2, y2, { stroke: 'var(--viz-forward)', sw: 3, op: 0.8 });
    }
    drawAgent(E, G, f.fx, f.fy);
    // table: one row per move
    const tx = nar ? 12 : G.x0 + G.w + 28, ty = nar ? gb.y + gb.h + 26 : 30, cols = nar ? [0, 40, 130, 210] : [0, 44, 130, 220], rh = nar ? 21 : 24, rows = Math.min(f.n, 14);
    ['Move', 'From', 'Reward', 'Return'].forEach((h, j) => text(E, h, tx + cols[j], ty, { size: 12, fill: MUTED }));
    for (let i = 0; i < rows; i++) {
      const y = ty + 22 + i * rh, active = f.phase === 'back' && f.row === i, has = i >= T - f.known;
      if (active) rect(E, tx - 6, y - 15, (nar ? E.W - 12 : E.W - tx - 4), rh - 2, { fill: 'none', stroke: ACCENT, sw: 1.8 });
      text(E, String(i + 1), tx + cols[0], y);
      text(E, cellName(ep.states[i]), tx + cols[1], y);
      text(E, signFmt(ep.rewards[i]), tx + cols[2], y, { mono: true });
      if (has) text(E, fmt(s.G[i], 3), tx + cols[3], y, { mono: true, weight: 650, fill: 'var(--viz-gradient)', op: active ? f.fade : 1 });
      else text(E, '·', tx + cols[3], y, { fill: MUTED });
    }
    if (f.n === 0) text(E, 'Moves appear here.', tx, ty + 22, { fill: MUTED });
  },
  label: (s) => `A fixed policy walks one episode of ${s.T} moves on the grid; a table then fills in the discounted return of each move, last move first.`,
};
export default mode;
