/**
 * optimal-policy (rl-basics step 7): the best policy as the reward for every move (step reward) sweeps from -2 to 0.1.
 * The scan (worker job rewardSwitches, 400 steps) finds every value where an arrow changes; the timeline spends the same
 * time in each region between changes, and the step reward moves linearly inside a region.
 * gamma is 1 for negative step rewards and 0.99 for positive ones (at gamma 1 their values grow without bound).
 * Props: slip (0.2), range ("-2,0.1").
 * Measured changes: -1.65, -1.56, -0.731, -0.453, -0.0850, -0.0448, -0.0274, -0.0221, +0.0101.
 */
import { ARROWS, GOAL, PIT, cellName, makeGrid, move } from '../../../lib/rl/gridworld';
import { nums, num, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, ease, fmt, line, rect, text, type Env } from '../_rl/svg';
import { drawGrid, drawPolicy, gridIn } from './draw';

const REG = 1.2, HOLD = 3, FLASH = 1.0;
interface Scene { lo: number; hi: number; cuts: number[]; pol: Int8Array[]; states: number[]; pit: number[][]; goal: number[][]; groups: { from: number; label: string }[] }
interface F extends Frame { r: number; R: number; flash: number }

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit: Kit) {
    const [lo, hi] = nums(o.range, [-2, 0.1]), slip = num(o.slip, 0.2), g = makeGrid({ slip });
    const sw = await kit.job<{ at: number; before: Int8Array; after: Int8Array }[]>('rewardSwitches', { slip, gamma: 1 }, lo, hi, 400);
    const pol = [sw[0].before, ...sw.map((x) => x.after)], cuts = sw.map((x) => x.at);
    const into = (term: number) => pol.map((p) => g.states.filter((c) => move(c, p[c]) === term));
    const pit = into(PIT), goal = into(GOAL);
    // group regions by how many cells step straight into the pit
    const groups: Scene['groups'] = [];
    pol.forEach((_, i) => { if (i === 0 || pit[i].length !== pit[i - 1].length || (pit[i].length + goal[i].length === 0) !== (pit[i - 1].length + goal[i - 1].length === 0)) groups.push({ from: i, label: '' }); });
    groups.forEach((gr, k) => {
      const a = gr.from === 0 ? null : cuts[gr.from - 1], b = k + 1 < groups.length ? cuts[groups[k + 1].from - 1] : null;
      gr.label = a === null ? `Below ${fmt(b!, 2)}` : b === null ? `Above ${fmt(a, 2)}` : `${fmt(a, 2)} to ${fmt(b, 2)}`;
    });
    return { lo, hi, cuts, pol, states: g.states, pit, goal, groups };
  },
  timeline: (s) => ({ duration: s.pol.length * REG + HOLD, markers: s.groups.map((g) => ({ at: g.from * REG, label: g.label })) }),
  frameAt(t, s) {
    const n = s.pol.length, r = Math.max(0, Math.min(n - 1, Math.floor(t / REG + 1e-9))), u = Math.max(0, Math.min(1, t / REG - r));
    const a = r === 0 ? s.lo : s.cuts[r - 1], b = r === n - 1 ? s.hi : s.cuts[r], R = a + (b - a) * u;
    const since = t - r * REG, flash = r > 0 && since < FLASH ? 1 - since / FLASH : 0;
    const names = (cs: number[]) => cs.map(cellName).join(' and ');
    const parts = [s.pit[r].length ? `${names(s.pit[r])} step${s.pit[r].length === 1 ? 's' : ''} straight into the pit` : 'no cell steps into the pit', s.goal[r].length ? `${names(s.goal[r])} step${s.goal[r].length === 1 ? 's' : ''} into the goal` : 'no cell steps into the goal'];
    let status = `Step reward ${fmt(R, 3)}: ${parts[0]}, and ${parts[1]}.`;
    if (r > 0) {
      const before = s.pol[r - 1], after = s.pol[r], ch = s.states.filter((c) => before[c] !== after[c]).map((c) => `${cellName(c)} from ${ARROWS[before[c]]} to ${ARROWS[after[c]]}`).join(' and ');
      status += ` At ${fmt(s.cuts[r - 1], 3)} the best move changes at ${ch}.`;
    }
    return { r, R, flash, status, readouts: [['Step reward', fmt(R, 3)], ['Into the pit', String(s.pit[r].length)], ['Into the goal', String(s.goal[r].length)], ['Region', `${r + 1} of ${n}`]] };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, stripH = nar ? 96 : 84, gb = nar ? { x: 8, y: 8, w: E.W - 16, h: 270 } : { x: 8, y: 8, w: E.W * 0.52, h: E.H - stripH - 16 };
    const G = gridIn(gb), cur = s.pol[f.r], prev = s.pol[Math.max(0, f.r - 1)];
    const changed = new Set(s.states.filter((c) => cur[c] !== prev[c]));
    drawGrid(E, G);
    drawPolicy(E, G, { actions: cur, flash: (c) => (changed.has(c) ? 0.65 + 0.35 * ease(f.flash) : 0) });
    // big number
    const tx = nar ? 12 : G.x0 + G.w + 28, ty = nar ? gb.y + gb.h + 30 : 44;
    text(E, 'Reward for every move', tx, ty - 16, { fill: MUTED });
    text(E, fmt(f.R, 3), tx, ty + 18, { size: 28, weight: 650, mono: true });
    text(E, `Cells that step into the pit: ${s.pit[f.r].length}`, tx, ty + 46);
    text(E, `Cells that step into the goal: ${s.goal[f.r].length}`, tx, ty + 66);
    text(E, f.R > 0 ? 'Discount 0.99, because with γ = 1 values would grow without bound.' : 'Discount γ = 1.', tx, ty + 90, { fill: MUTED, size: 11 });
    // region strip: equal-width boxes, one per region, with the step reward where each begins
    const n = s.pol.length, x0 = 16, x1 = E.W - 16, w = (x1 - x0) / n, y = E.H - stripH + 18;
    for (let i = 0; i < n; i++) rect(E, x0 + i * w, y, w - 2, 16, { fill: i === f.r ? ACCENT : 'none', stroke: i === f.r ? ACCENT : RULE, sw: 1, op: i === f.r ? 0.85 : 1 });
    for (let i = 1; i < n; i++) {
      const lab = fmt(s.cuts[i - 1], Math.abs(s.cuts[i - 1]) >= 0.1 ? 2 : 3);
      if (nar && i % 2 === 0) continue;
      line(E, x0 + i * w - 1, y + 16, x0 + i * w - 1, y + 22, { stroke: RULE });
      text(E, lab, x0 + i * w - 1, y + 35 + (!nar && i % 2 === 0 ? 14 : nar ? 0 : 0), { size: 11, anchor: 'middle', fill: MUTED });
    }
    if (!nar) text(E, fmt(s.lo, 0), x0, y + 35, { size: 11, anchor: 'start', fill: MUTED });
    text(E, nar ? 'Numbers: where the best moves change.' : 'Each box is one set of best moves; the numbers are where the best moves change.', x0, y + 62, { size: 11, fill: MUTED });
  },
  label: (s) => `The 4 by 3 grid with the best move in each cell as the reward for every move runs from ${s.lo} to ${s.hi}; the best moves change at ${s.cuts.map((c) => c.toFixed(3)).join(', ')}.`,
  restTime(s) {
    const r = s.cuts.filter((c) => c < -0.04).length, a = r === 0 ? s.lo : s.cuts[r - 1], b = r === s.pol.length - 1 ? s.hi : s.cuts[r];
    return r * REG + ((-0.04 - a) / (b - a)) * REG;
  },
};
export default mode;
