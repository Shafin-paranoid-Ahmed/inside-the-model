/**
 * bellman (q-learning step 2): segment 1 assembles one Bellman backup at the focus cell and action term by term from
 * the converged values; segment 2 runs whole-grid sweeps of value iteration from zero (Q triangles, the focus cell's
 * three terms, and the largest change per sweep on a log axis). Props: focus ("3,1"), action ("up"), sweeps (20).
 */
import { ACTIONS, DX, DY, cellName } from '../../../lib/rl/gridworld';
import { backupTerms, valueIteration } from '../../../lib/rl/tabular';
import { int, key, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, arrow, ease, fmt, lineChart, text, type Env } from '../_rl/svg';
import { actionOf, focusOf, gridOf } from './sim';
import { cellXY, drawGrid, drawQ, drawValues, gridIn } from './draw';
import { qStar } from './q-common';

const LEAD = 0.8, TERM = 1.6, FADE = 0.5, SWEEP = 0.8;
type Terms = ReturnType<typeof backupTerms>;
interface Scene { s: number; a: number; sweeps: number; Vstar: Float64Array; Qstar: Float64Array; star: Terms; vi: ReturnType<typeof valueIteration>; per: Terms[] }
interface F extends Frame { seg1: boolean; k: number; vis: number[]; sumVis: number; terms: Terms }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const g = gridOf(o), s = focusOf(o, 2), a = actionOf(o, 0), sweeps = Math.max(2, int(o.sweeps, 20)), q = qStar(kit, g);
    const vi = kit.memo(key('vi', g.stepReward, g.slip, g.gamma, sweeps), () => valueIteration(g, sweeps));
    return { s, a, sweeps, Vstar: q.V, Qstar: q.Q, star: backupTerms(g, q.V, s, a), vi, per: vi.V.slice(0, sweeps).map((V) => backupTerms(g, V, s, a)) };
  },
  timeline(s) {
    const n = s.star.terms.length, t1 = LEAD + (n + 1) * TERM + 1.2;
    return { duration: t1 + s.sweeps * SWEEP + 1.5, markers: [{ at: 0, label: 'One backup' }, { at: t1, label: 'Sweeps of the grid' }] };
  },
  frameAt(t, s) {
    const n = s.star.terms.length, t1 = LEAD + (n + 1) * TERM + 1.2, seg1 = t < t1, name = cellName(s.s), act = ACTIONS[s.a];
    if (seg1) {
      const vis = s.star.terms.map((_, j) => ease((t - LEAD - j * TERM) / FADE)), sumVis = ease((t - LEAD - n * TERM) / FADE);
      const j = Math.min(n - 1, Math.max(0, Math.floor((t - LEAD) / TERM)));
      let status: string;
      if (t < LEAD) status = `Going ${act} from ${name}, the move can turn out ${n} ways. We add up what each outcome is worth, weighted by its chance.`;
      else if (t < LEAD + n * TERM) {
        const o = s.star.terms[j];
        status = `Outcome ${j + 1}: with chance ${fmt(o.p, 1)} the agent ends in ${cellName(o.s2)}. The move pays ${fmt(o.r)} and that cell is worth ${fmt(o.v, 3)}, so this outcome adds ${fmt(o.term, 3)}.`;
      } else status = `The ${n} terms add up to ${fmt(s.star.sum, 3)}, exactly Q*(${name}, ${act}). The best values satisfy the Bellman equation.`;
      return { seg1, k: 0, vis, sumVis, terms: s.star, status, readouts: [['Cell and action', `${name} ${act}`], ['Sum of terms', sumVis > 0.99 ? fmt(s.star.sum, 3) : '…']] };
    }
    const k = Math.min(s.sweeps, 1 + Math.floor((t - t1) / SWEEP)), terms = s.per[k - 1], d = s.vi.delta[k - 1];
    return {
      seg1, k, vis: terms.terms.map(() => 1), sumVis: 1, terms,
      status: `Sweep ${k}: every cell is backed up at once using the values from sweep ${k - 1}. Q(${name}, ${act}) becomes ${fmt(terms.sum, 3)}, and the biggest change anywhere in the grid is ${fmt(d, 3)}.`,
      readouts: [['Sweep', `${k} of ${s.sweeps}`], [`Q(${name}, ${act})`, fmt(terms.sum, 3)], ['Largest change', d < 0.001 ? d.toExponential(1).replace('-', '−') : fmt(d, 3)]],
    };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow;
    const gb = nar ? { x: 8, y: 6, w: E.W - 16, h: E.W * 0.74 } : { x: 10, y: 10, w: E.W * 0.5, h: E.H * 0.92 };
    const G = gridIn(gb);
    drawGrid(E, G, { highlight: s.s });
    if (f.seg1) {
      drawValues(E, G, s.Vstar, { digits: 3 });
      f.terms.terms.forEach((o, j) => {
        const c = cellXY(G, s.s), dx = DX[o.dir], dy = -DY[o.dir], L = G.cs, op = f.vis[j], w = 1.5 + 9 * o.p;
        const e = o.moved ? 0.66 : 0.5;
        arrow(E, c.cx + dx * L * 0.3, c.cy + dy * L * 0.3, c.cx + dx * L * e, c.cy + dy * L * e, { stroke: 'var(--viz-forward)', width: w, op });
        text(E, (o.moved ? '' : 'stay ') + fmt(o.p, 1), c.cx + dx * L * 0.48 + (dy ? 14 : 0), c.cy + dy * L * 0.48 + (dx ? -8 : 4), { size: 12, anchor: 'middle', weight: 650, halo: true, op });
      });
    } else drawQ(E, G, s.vi.Q[f.k - 1], { digits: 2, numbers: nar ? false : undefined, best: f.k >= 3, tip: (c, a) => `${cellName(c)} ${ACTIONS[a]}` });
    // the backup as a table of terms
    const x = nar ? 14 : G.x0 + G.w + 26, w = nar ? E.W - 28 : E.W - x - 12, y0 = nar ? G.y0 + G.h + 22 : 28;
    text(E, f.seg1 ? `Backup of Q(${cellName(s.s)}, ${ACTIONS[s.a]}) with the best values` : `Sweep ${f.k}: same backup, earlier values`, x, y0, { size: 12, weight: 650 });
    f.terms.terms.forEach((o, j) => {
      const y = y0 + 24 + j * 22, op = f.vis[j];
      text(E, `${o.moved ? '' : 'stay '}${cellName(o.s2)}`, x, y, { size: 12, fill: MUTED, op });
      text(E, `${fmt(o.p, 1)} × (${fmt(o.r)} + ${fmt(o.v, 3)}) = ${fmt(o.term, 3)}`, x + (nar ? 70 : 78), y, { size: 12, op, mono: true });
    });
    const ys = y0 + 24 + f.terms.terms.length * 22 + 4;
    text(E, `Sum = ${fmt(f.terms.sum, 3)}`, x + (nar ? 70 : 78), ys, { size: 13, weight: 650, op: f.sumVis, fill: ACCENT });
    if (f.seg1) return;
    const cb = nar ? { x: 44, y: ys + 44, w: E.W - 60, h: E.H - ys - 44 - 40 } : { x: x + 30, y: ys + 44, w: w - 38, h: E.H - ys - 44 - 44 };
    const ds = Array.from(s.vi.delta), lo = Math.pow(10, Math.floor(Math.log10(Math.min(...ds)))), xs = ds.map((_, i) => i + 1);
    const keep = Math.min(f.k, ds.length);
    lineChart(E, cb, { lo: 1, hi: s.sweeps, ticks: [1, 5, 10, 15, 20].filter((v) => v <= s.sweeps), label: 'Sweep' }, { lo, hi: 1, log: true, label: 'Largest change', fmt: (v: number) => String(Number(v.toPrecision(1))) }, [{ xs, ys: ds, color: 'var(--viz-gradient)', width: 2 }], keep);
  },
  label: (s) => `Value iteration on the 4 by 3 world: first one Bellman backup at ${cellName(s.s)} going ${ACTIONS[s.a]} assembled from ${s.star.terms.length} weighted terms, then ${s.sweeps} sweeps of the whole grid with the largest change per sweep on a log axis.`,
};
export default mode;
