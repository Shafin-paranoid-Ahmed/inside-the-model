/**
 * dqn-network (deep-rl step 1): the untrained 2 -> 16 -> 4 network (116 weights, init seed 17) turns each cell's
 * coordinates into four Q-values. The tour visits (1,1), (4,1) and (1,3); click a cell to pin it.
 * Props: hidden (16), seed (17).
 */
import { ACTIONS, cellName, coords, idx } from '../../../lib/rl/gridworld';
import { allQ, initNet, paramCount } from '../../../lib/rl/dqn';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, ease, fmt, rect, seg, text, type Env } from '../_rl/svg';
import { cellXY, drawGrid, drawQ, gridIn } from './draw';
import { dqnGrid, netDiagram } from './dqn-common';

const SEG = 3.6, CELLS = [idx(1, 1), idx(4, 1), idx(1, 3)];
interface Scene { w: Float64Array; H: number; Q: Float64Array; states: number[] }
interface F extends Frame { s: number; rev: { i: number; h: number; o: number } }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const H = int(o.hidden, 16), w = initNet(H, int(o.seed, 17)), g = dqnGrid(o);
    return { w, H, Q: allQ(g, w, H), states: g.states };
  },
  timeline: () => ({ duration: CELLS.length * SEG + 0.8, markers: CELLS.map((s, i) => ({ at: i * SEG, label: `Cell ${cellName(s)}` })) }),
  frameAt(t, s, _o, ui) {
    const pin = ui.pick?.startsWith('c') ? Number(ui.pick.slice(1)) : null, { i, u } = seg(t, SEG, CELLS.length), cell = pin ?? CELLS[i];
    const rev = pin !== null || t >= CELLS.length * SEG ? { i: 1, h: 1, o: 1 } : { i: ease(u / 0.2), h: ease((u - 0.2) / 0.3), o: ease((u - 0.5) / 0.3) };
    const [x0, x1] = coords(cell), q = s.Q.subarray(cell * 4, cell * 4 + 4);
    let b = 0;
    for (let a = 1; a < 4; a++) if (q[a] > q[b]) b = a;
    return {
      s: cell, rev,
      status: `At ${cellName(cell)} the network is given two numbers, ${fmt(x0)} for the column and ${fmt(x1)} for the row. The same ${paramCount(s.H)} weights turn them into four Q-values; the largest is ${ACTIONS[b]} at ${fmt(q[b])}.`,
      detail: pin !== null ? 'Click the cell again to resume the tour.' : 'Click any cell to pin it.',
      readouts: [['Cell', cellName(cell)], ['Input', `${fmt(x0)}, ${fmt(x1)}`], ['Weights', String(paramCount(s.H))], ['Q (↑ → ↓ ←)', Array.from(q, (v) => fmt(v)).join('  ')]],
    };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, gb = nar ? { x: 8, y: 22, w: E.W - 16, h: E.H * 0.34 } : { x: 12, y: 26, w: E.W * 0.4, h: E.H - 36 };
    const nb = nar ? { x: 8, y: gb.y + gb.h + 14, w: E.W - 16, h: E.H - gb.y - gb.h - 22 } : { x: E.W * 0.45, y: 6, w: E.W * 0.55 - 8, h: E.H - 14 };
    const G = gridIn(gb);
    text(E, 'Q-values of every cell', gb.x, 14, { fill: MUTED });
    drawGrid(E, G);
    drawQ(E, G, s.Q, { vmax: 1.3, best: false, numbers: false });
    for (const k of s.states) { const p = cellXY(G, k); rect(E, p.x, p.y, G.cs, G.cs, { fill: 'transparent', pick: `c${k}`, tip: `Show ${cellName(k)}` }); }
    const c = cellXY(G, f.s);
    rect(E, c.x + 1.5, c.y + 1.5, G.cs - 3, G.cs - 3, { fill: 'none', stroke: ACCENT, sw: 2.5 });
    text(E, cellName(f.s), c.cx, c.cy + 4, { anchor: 'middle', size: 11, weight: 700, halo: true });
    const [x0, x1] = coords(f.s);
    netDiagram(E, nb, s.w, s.H, x0, x1, f.rev);
  },
  label: (s) => `A 4 by 3 grid coloured by the Q-values of a small neural network, beside a diagram of that network with ${s.H} hidden units turning one cell's coordinates into four Q-values.`,
};
export default mode;
