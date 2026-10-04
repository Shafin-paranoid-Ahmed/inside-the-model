/**
 * qlearn (q-learning step 3): tabular Q-learning (src/lib/rl qLearning, recorded). Segment 1 follows episode 1 move by
 * move with the update card (target, error, new Q); segment 2 plays episodes 2 to 10 one per beat; segment 3 plays
 * the remaining episodes quickly. Props: alpha (0.1), epsilon (0.3), episodes (1000), seed (10), gamma (1).
 */
import { ACTIONS, GOAL, cellName, xy } from '../../../lib/rl/gridworld';
import { greedy, maxRow, policyAgreement } from '../../../lib/rl/tabular';
import { num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, TEXT, ease, fmt, lineChart, poly, text, type Env } from '../_rl/svg';
import { gridOf } from './sim';
import { drawAgent, drawGrid, drawQ, gridIn, triangle } from './draw';
import { qName, qRun, qStar } from './q-common';

const LEAD = 0.7, MOVE = 0.65, HOLD = 1.0, EP = 0.8, FAST = 9;
interface Scene {
  alpha: number; epsilon: number; gamma: number; E: number; T1: number; run: ReturnType<typeof qRun>;
  Qm: Float64Array[]; startQ: Float64Array; agree: Int8Array; startStar: number;
}
interface F extends Frame { seg: 0 | 1 | 2; ep: number; Q: Float64Array; tx: number; flash: number; fx: number; fy: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const g = gridOf(o), q = qStar(kit, g), run = qRun(kit, g, o, { alpha: 0.1, epsilon: 0.3, episodes: 1000, seed: 10 }), tr = run.trace!;
    const T1 = run.lengths[0], Qm = [run.snapshots[0].slice()];
    for (let k = 0; k < T1; k++) { const Q = Qm[k].slice(); for (let a = 0; a < 4; a++) Q[tr.s[k] * 4 + a] = tr.qAfter[k * 4 + a]; Qm.push(Q); }
    const E = run.lengths.length, startQ = new Float64Array(E + 1), agree = new Int8Array(E + 1);
    for (let e = 0; e <= E; e++) { startQ[e] = maxRow(run.snapshots[e], 0); agree[e] = policyAgreement(g, greedy(g, run.snapshots[e]), q.policy); }
    return { alpha: num(o.alpha, 0.1), epsilon: num(o.epsilon, 0.3), gamma: g.gamma, E, T1, run, Qm, startQ, agree, startStar: maxRow(q.Q, 0) };
  },
  timeline(s) {
    const t1 = LEAD + s.T1 * MOVE + HOLD, t2 = t1 + Math.min(9, s.E - 1) * EP;
    return { duration: t2 + (s.E > 10 ? FAST : 0) + 1.5, markers: [{ at: 0, label: 'Episode 1' }, { at: t1, label: 'Episodes 2–10' }, ...(s.E > 10 ? [{ at: t2, label: `Episodes 11–${s.E}` }] : [])] };
  },
  frameAt(t, s) {
    const { run, T1, E } = s, tr = run.trace!, t1 = LEAD + T1 * MOVE + HOLD, t2 = t1 + Math.min(9, E - 1) * EP;
    let seg: 0 | 1 | 2 = 0, ep = 1, Q = s.Qm[0], tx = 0, flash = 0;
    let fx = xy(tr.s[0])[0], fy = xy(tr.s[0])[1];
    if (t < t1) {
      const raw = (t - LEAD) / MOVE, m = Math.max(0, Math.min(T1 - 1, Math.floor(raw))), u = t < LEAD || raw >= T1 ? 0 : raw - m;
      const pastUpdate = raw >= T1 || (t >= LEAD && u >= 0.6);
      Q = s.Qm[t < LEAD ? 0 : pastUpdate ? m + 1 : m]; tx = m;
      const [x0, y0] = xy(tr.s[m]), [x1, y1] = xy(tr.s2[m]), e = t < LEAD ? 0 : raw >= T1 ? 1 : ease(u / 0.6);
      fx = x0 + (x1 - x0) * e; fy = y0 + (y1 - y0) * e;
      flash = t >= LEAD && raw < T1 && u >= 0.6 ? 1 - (u - 0.6) / 0.4 : 0;
    } else {
      seg = t < t2 ? 1 : 2;
      ep = seg === 1 ? 2 + Math.min(8, Math.floor((t - t1) / EP)) : 11 + Math.floor(Math.min(0.9999, (t - t2) / FAST) * (E - 10));
      ep = Math.min(ep, E); Q = run.snapshots[ep]; tx = run.start[ep - 1] + run.lengths[ep - 1] - 1;
    }
    const s0 = tr.s[tx], a0 = tr.a[tx], next = tr.s2[tx], done = tr.done[tx] === 1, r = tr.r[tx];
    const maxNext = done ? 0 : (tr.target[tx] - r) / s.gamma, newQ = tr.qBefore[tx] + s.alpha * tr.tdError[tx];
    let status: string;
    if (seg === 0) {
      const moved = `${cellName(s0)} ${ACTIONS[a0]}${tr.slipped[tx] ? ' (slipped)' : ''} → ${cellName(next)}`;
      if (t < LEAD) status = `Episode 1 starts at (1,1) with every Q at 0. The agent usually picks the best-looking action and, ${Math.round(s.epsilon * 100)}% of the time, a random one.`;
      else if (done) status = `Move ${tx + 1}: ${moved}, and the episode ends with reward ${fmt(r)}. Nothing comes after it, so the target is just the reward, and Q(${cellName(s0)}, ${ACTIONS[a0]}) moves from ${fmt(tr.qBefore[tx], 3)} to ${fmt(newQ, 3)}.`;
      else status = `Move ${tx + 1}: ${moved}. The move cost ${fmt(r)} and the best next value is ${fmt(maxNext, 3)}, so Q(${cellName(s0)}, ${ACTIONS[a0]}) moves from ${fmt(tr.qBefore[tx], 3)} to ${fmt(newQ, 3)}.`;
    } else {
      const end = tr.s2[tx] === GOAL ? 'the goal' : 'the pit';
      status = seg === 1
        ? `Episode ${ep}: ${run.lengths[ep - 1]} moves, ending in ${end}. Every move nudged one Q entry; the card shows the last update.`
        : `Episode ${ep} of ${E}. The start cell now looks worth ${fmt(s.startQ[ep], 3)}; its true value is ${fmt(s.startStar, 3)}. ${s.agree[ep]} of 9 cells pick the best action.`;
    }
    return {
      seg, ep, Q, tx, flash, fx, fy, status,
      readouts: [['Episode', `${ep} of ${E}`], ['Start-cell estimate', fmt(s.startQ[ep], 3)], ['True value Q*', fmt(s.startStar, 3)], ['Cells with the best action', `${s.agree[ep]} of 9`]],
    };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, tr = s.run.trace!;
    const gb = nar ? { x: 8, y: 6, w: E.W - 16, h: E.W * 0.74 } : { x: 10, y: 10, w: E.W * 0.5, h: E.H - 20 };
    const G = gridIn(gb), s0 = tr.s[f.tx], a0 = tr.a[f.tx];
    drawGrid(E, G, { highlight: f.seg === 0 ? s0 : null });
    drawQ(E, G, f.Q, { digits: 2, best: f.seg !== 0, tip: (c, a) => `${qName(c, a)}: Q = ${fmt(f.Q[c * 4 + a], 3)}` });
    if (f.flash > 0) poly(E, triangle(G, s0, a0), { fill: 'none', stroke: TEXT, sw: 3.5, op: f.flash });
    if (f.seg === 0) drawAgent(E, G, f.fx, f.fy);
    // the update card: real numbers of transition tx
    const x = nar ? 14 : G.x0 + G.w + 26, y0 = nar ? G.y0 + G.h + 22 : 28, w = nar ? E.W - 28 : E.W - x - 12;
    const done = tr.done[f.tx] === 1, r = tr.r[f.tx], before = tr.qBefore[f.tx], err = tr.tdError[f.tx], target = tr.target[f.tx], maxNext = done ? 0 : (target - r) / s.gamma;
    text(E, `${f.seg === 0 ? `Move ${f.tx + 1}` : `Last move of episode ${f.ep}`}: ${qName(s0, a0)}`, x, y0, { size: 13, weight: 650 });
    const L = (i: number, k: string, v: string) => { text(E, k, x, y0 + 22 + i * 21, { size: 12, fill: MUTED }); text(E, v, x + 58, y0 + 22 + i * 21, { size: 12, mono: true }); };
    L(0, 'target', done ? `${fmt(r)}   (episode over)` : `${fmt(r)} + ${s.gamma === 1 ? '' : `${fmt(s.gamma, 1)} × `}${fmt(maxNext, 3)} = ${fmt(target, 3)}`);
    L(1, 'error', `${fmt(target, 3)} − (${fmt(before, 3)}) = ${fmt(err, 3)}`);
    L(2, 'new Q', `${fmt(before, 3)} + ${fmt(s.alpha, 1)} × (${fmt(err, 3)}) = ${fmt(before + s.alpha * err, 3)}`);
    // the start-cell estimate over episodes, against Q*
    const top = y0 + 112, cb = nar ? { x: 44, y: top, w: E.W - 60, h: E.H - top - 40 } : { x: x + 30, y: top, w: w - 38, h: E.H - top - 44 };
    const xs = Array.from({ length: s.E + 1 }, (_, i) => i);
    lineChart(E, cb, { lo: 0, hi: s.E, label: 'Episode' }, { lo: -0.2, hi: 1, ticks: [0, 0.5, 1], label: 'Start-cell estimate' }, [
      { xs: [0, s.E], ys: [s.startStar, s.startStar], color: 'var(--text)', width: 1.25, dash: '5 4' },
      { xs, ys: s.startQ, color: 'var(--viz-forward)', width: 2 },
    ], f.seg === 0 ? 0 : f.ep);
  },
  label: (s) => `Tabular Q-learning on the 4 by 3 world: episode 1 move by move with each update's target and error, then ${s.E - 1} more episodes while the Q values and the start-cell estimate approach Q star.`,
};
export default mode;
