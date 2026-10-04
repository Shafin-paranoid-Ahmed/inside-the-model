/**
 * discount (rl-basics step 4): the optimal policy as the discount factor sweeps from gammas[0] to gammas[1].
 * Left: the policy at gammas[0] (fixed). Right: the policy at the current gamma, in 0.01 steps (each is its own
 * value-iteration solve, run in the worker); cells that differ from the left grid are drawn in the accent colour.
 * Props: gammas ("0.5,0.99"), stepReward, slip.
 * Measured: the policy changes at gamma 0.66005 ((4,1) down -> left) and 0.93781 ((2,1) right -> left).
 */
import { ARROWS, START, cellName, makeGrid } from '../../../lib/rl/gridworld';
import { nums, num, type Frame, type Kit, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, circle, ease, fmt, line, text, type Env } from '../_rl/svg';
import { drawGrid, drawPolicy, gridIn, twoBoxes } from './draw';

const PLAY = 12, HOLD = 2.5, FLASH = 0.9;
interface Opt { V: Float64Array; policy: Int8Array }
interface Scene { gs: number[]; pol: Int8Array[]; v0: number[]; changes: number[]; exact: number[]; n: number }
interface F extends Frame { i: number; region: number; flash: number }
const same = (a: Int8Array, b: Int8Array) => a.every((x, i) => x === b[i]);
const diff = (a: Int8Array, b: Int8Array) => { const d: number[] = []; for (let s = 0; s < a.length; s++) if (a[s] !== b[s]) d.push(s); return d; };

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit: Kit) {
    const [lo, hi] = nums(o.gammas, [0.5, 0.99]), sr = num(o.stepReward, -0.04), slip = num(o.slip, 0.2), n = Math.round((hi - lo) * 100);
    const gs = Array.from({ length: n + 1 }, (_, i) => Math.round((lo + i * 0.01) * 100) / 100);
    const [sw, opts] = await Promise.all([
      kit.job<{ at: number }[]>('gammaSwitches', { stepReward: sr, slip }, lo, hi, 100),
      Promise.all(gs.map((x) => kit.job<Opt>('optimal', makeGrid({ stepReward: sr, slip, gamma: x }), x, 1e-8))),
    ]);
    const pol = opts.map((x) => x.policy), changes: number[] = [];
    for (let i = 1; i <= n; i++) if (!same(pol[i], pol[i - 1])) changes.push(i);
    return { gs, pol, v0: opts.map((x) => x.V[START]), changes, exact: sw.map((x) => x.at), n };
  },
  timeline(s) {
    const at = (i: number) => (i / s.n) * PLAY, c = s.changes;
    const names = c.length === 2 ? ['Short-sighted', 'Middle ground', 'Patient'] : ['From the start', ...c.map((_, k) => `After change ${k + 1}`)];
    return { duration: PLAY + HOLD, markers: [{ at: 0, label: names[0] }, ...c.map((i, k) => ({ at: at(i), label: names[k + 1] }))] };
  },
  frameAt(t, s) {
    const i = Math.max(0, Math.min(s.n, Math.floor((t / PLAY) * s.n + 1e-9))), region = s.changes.filter((c) => c <= i).length, g = s.gs[i];
    const since = region > 0 ? t - (s.changes[region - 1] / s.n) * PLAY : Infinity, flash = since < FLASH ? 1 - since / FLASH : 0;
    const d = diff(s.pol[0], s.pol[i]);
    let status = `γ = ${fmt(g, 2)}: the arrows are the same as at γ = ${fmt(s.gs[0], 2)}. The start cell is worth ${fmt(s.v0[i], 2)}.`;
    if (region > 0) {
      const a = s.pol[s.changes[region - 1] - 1], b = s.pol[s.changes[region - 1]], ch = diff(a, b).map((c) => `${cellName(c)} from ${ARROWS[a[c]]} to ${ARROWS[b[c]]}`).join(' and ');
      const ex = s.exact.length === s.changes.length ? s.exact[region - 1] : s.gs[s.changes[region - 1]];
      status = `γ = ${fmt(g, 2)}: ${d.length} ${d.length === 1 ? 'arrow differs' : 'arrows differ'} from the left grid. At γ = ${fmt(ex, 3)} the best move at ${ch}. The start cell is worth ${fmt(s.v0[i], 2)}.`;
    }
    return { i, region, flash, status, readouts: [['γ', fmt(g, 2)], ['Start value', fmt(s.v0[i], 2)], ['Left start value', fmt(s.v0[0], 2)]] };
  },
  draw(E: Env, f, s) {
    const stripH = 62, area = { x: 8, y: 4, w: E.W - 16, h: E.H - stripH - 8 }, [bl, br] = twoBoxes(E, area, E.narrow ? 14 : 28, 22);
    const GL = gridIn(bl), GR = gridIn(br), cur = s.pol[f.i], ref = s.pol[0];
    const dif = new Set(diff(ref, cur));
    drawGrid(E, GL); drawPolicy(E, GL, { actions: ref });
    drawGrid(E, GR); drawPolicy(E, GR, { actions: cur, flash: (c) => (dif.has(c) ? 0.55 + 0.45 * ease(f.flash) : 0), color: 'var(--text)' });
    text(E, `γ = ${fmt(s.gs[0], 2)}, start value ${fmt(s.v0[0], 2)}`, GL.x0, bl.y - 7, { weight: 650 });
    text(E, `γ = ${fmt(s.gs[f.i], 2)}, start value ${fmt(s.v0[f.i], 2)}`, GR.x0, br.y - 7, { weight: 650 });
    // the discount-factor strip with the places where the best policy changes
    const x0 = 28, x1 = E.W - 28, y = E.H - 34, X = (g: number) => x0 + ((g - s.gs[0]) / (s.gs[s.n] - s.gs[0])) * (x1 - x0);
    line(E, x0, y, x1, y, { stroke: RULE, sw: 2 });
    line(E, x0, y, X(s.gs[f.i]), y, { stroke: 'var(--viz-forward)', sw: 3 });
    for (const g of [s.gs[0], s.gs[s.n]]) text(E, fmt(g, 2), X(g), y + 22, { anchor: 'middle', fill: MUTED, size: 11 });
    s.changes.forEach((c, k) => {
      const ex = s.exact.length === s.changes.length ? s.exact[k] : s.gs[c];
      line(E, X(ex), y - 8, X(ex), y + 8, { stroke: ACCENT, sw: 2 });
      text(E, fmt(ex, 3), X(ex), y - 14, { anchor: 'middle', size: 11, fill: MUTED });
    });
    circle(E, X(s.gs[f.i]), y, 6, { fill: ACCENT, stroke: 'var(--bg-raised)', sw: 2 });
  },
  label: (s) => `Two 4 by 3 grids of best moves: the left at discount factor ${s.gs[0]}, the right sweeping up to ${s.gs[s.n]}. The best moves change at ${s.exact.map((x) => fmt(x, 3)).join(' and ')}.`,
};
export default mode;
