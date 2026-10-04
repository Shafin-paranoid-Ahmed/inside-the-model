/**
 * actor-critic (policy-gradients step 5): first one episode step by step (the critic's TD error delta = r + V(s') - V(s)
 * moves both the critic's value for the cell and the actor's action probabilities there), then the learning curves of
 * actor-critic against REINFORCE with and without a baseline (10 runs each).
 * Props: episode (7, as numbered on screen), episodeSeed (13), lrActor (0.1), lrCritic (0.2), episodes (1000), runs (10),
 * seed (34, runs use 34..43), threshold (0.5: expected return used for "episodes to reach").
 */
import { ARROWS, ACTIONS, GOAL, START, cellName, xy } from '../../../lib/rl/gridworld';
import { actorCritic, optimal, softmaxPolicy, type ACStep } from '../../../lib/rl/tabular';
import { int, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, ease, fmt, inkOn, lerp, lineChart, signFmt, signedFill, text, rect, table, type Env, type Shape } from '../_rl/svg';
import { cellXY, drawAgent, drawGrid, drawPolicy, gridIn } from './draw';
import { EVAL_EVERY, episodesAt, firstReach, pgRuns, timeAt, type PGRuns } from './pg-runs';
import { gridOf } from './sim';

const LEAD = 0.6, STEP = 1.2, GAP = 0.5, PLAY = 12, HOLD = 2.5;
interface Scene {
  steps: ACStep[]; T: number; ep: number; lrC: number; lrA: number; vs: Float64Array[]; ps: Float64Array[];
  none: PGRuns; sv: PGRuns; ac: PGRuns; vstar: number; episodes: number; nRuns: number; thr: number;
  tA: number; tB: number; reach: [number, number, number];
}
interface F extends Frame { part: 'walk' | 'curves'; k: number; u: number; fx: number; fy: number; e: number; kEval: number }

/** Multiply the opacity of the shapes pushed after index i0 (cross-fade between the two parts). */
const fadeFrom = (E: Env, i0: number, k: number) => { for (let i = i0; i < E.out.length; i++) { const s: Shape = E.out[i]; s.op = (s.op ?? 1) * k; } };
const tdLine = (st: ACStep, g: number) => `δ = r + γV(next) − V(here) = ${fmt(st.r)} + ${st.done ? '0' : fmt(g * (st.vAfter * 0 + (st.delta - st.r + st.v)), 2)} − ${fmt(st.v)}`;

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const g = gridOf(o), ep = int(o.episode, 7), lrA = num(o.lrActor, 0.1), lrC = num(o.lrCritic, 0.2), seed = int(o.episodeSeed, 13);
    const prior = actorCritic(g, { lrActor: lrA, lrCritic: lrC, episodes: ep - 1, seed, evalEvery: 0 });
    const one = actorCritic(g, { lrActor: lrA, lrCritic: lrC, episodes: ep, seed, evalEvery: 0, detail: [ep - 1] });
    const steps = one.steps[0], T = steps.length;
    const V = prior.V.slice(), th = prior.theta.slice(), vs = [V.slice()], ps = [softmaxPolicy(th)];
    for (const st of steps) {
      V[st.s] = st.vAfter;
      for (let a = 0; a < 4; a++) th[st.s * 4 + a] += lrA * st.delta * ((a === st.a ? 1 : 0) - st.probsBefore[a]);
      vs.push(V.slice()); ps.push(softmaxPolicy(th));
    }
    const [none, sv, ac] = await Promise.all([pgRuns(kit, g, 'none', o), pgRuns(kit, g, 'state-value', o), pgRuns(kit, g, 'actor-critic', o)]);
    const thr = num(o.threshold, 0.5), tA = LEAD + T * STEP, tB = tA + GAP;
    return {
      steps, T, ep, lrC, lrA, vs, ps, none, sv, ac, vstar: optimal(g).V[START], episodes: int(o.episodes, 1000), nRuns: int(o.runs, 10), thr, tA, tB,
      reach: [firstReach(none, thr), firstReach(sv, thr), firstReach(ac, thr)],
    };
  },
  timeline(s) {
    const last = s.steps[s.T - 1], mk = (at: number, label: string) => ({ at, label });
    return {
      duration: s.tB + PLAY + HOLD,
      markers: [mk(0, 'Walk one episode'), mk(LEAD + (s.T - 1) * STEP, last.s2 === GOAL ? 'The step into the goal' : 'The step into the pit'), mk(s.tB, 'First 100 episodes'), mk(s.tB + timeAt(Math.min(100, s.episodes), PLAY, s.episodes), `Episodes 100 to ${s.episodes}`), mk(s.tB + PLAY, 'Training done')],
    };
  },
  frameAt(t, s) {
    if (t >= s.tB) {
      const e = episodesAt(t - s.tB, PLAY, s.episodes), k = e / EVAL_EVERY, a = s.none.mean[k], b = s.sv.mean[k], c = s.ac.mean[k];
      const reach = (n: number) => (n < 0 ? 'never' : `${n} episodes`);
      const status = e >= s.episodes
        ? `The average expected return first reaches ${fmt(s.thr, 1)} after ${reach(s.reach[2])} for actor-critic, ${reach(s.reach[1])} for REINFORCE with a learned baseline and ${reach(s.reach[0])} without. At ${s.episodes} episodes the three are ${fmt(c, 2)}, ${fmt(b, 2)} and ${fmt(a, 2)}.`
        : `After ${e} episodes the average expected return is ${fmt(c, 2)} for actor-critic, ${fmt(b, 2)} for REINFORCE with a learned baseline and ${fmt(a, 2)} without.`;
      return { part: 'curves', k: s.T - 1, u: 1, fx: 0, fy: 0, e, kEval: k, status, detail: `Exact expected return of the current policy from the start cell, averaged over ${s.nRuns} runs of each method.`, readouts: [['Episodes', `${e} of ${s.episodes}`], ['Actor-critic', fmt(c, 3)], ['REINFORCE, baseline', fmt(b, 3)], ['REINFORCE, none', fmt(a, 3)]] };
    }
    const raw = (t - LEAD) / STEP, k = Math.max(0, Math.min(s.T - 1, Math.floor(raw))), u = t < LEAD ? 0 : Math.min(1, raw - k), st = s.steps[k];
    const [x0, y0] = xy(st.s), [x1, y1] = xy(st.s2), m = ease(u / 0.35), fx = x0 + (x1 - x0) * m, fy = y0 + (y1 - y0) * m;
    const d = fmt(st.delta), updated = u >= 0.6, cell = cellName(st.s), nxt = cellName(st.s2);
    let status: string;
    if (t < LEAD) status = `The critic starts the episode with its current values. The actor will move after every step, using the critic's one-step error.`;
    else if (u < 0.35) status = `Step ${k + 1}: at ${cell} the actor picks ${ACTIONS[st.a]} and the agent ${st.s === st.s2 ? 'bumps into a wall' : `moves to ${nxt}`}. Reward ${fmt(st.r)}.`;
    else if (u < 0.6) status = `The critic expected ${fmt(st.v)} from ${cell}. It got ${fmt(st.r)} plus ${st.done ? 'nothing more, because the episode ended' : `the value ${fmt(st.delta - st.r + st.v)} it expects from ${nxt}`}. The TD error is δ = ${signFmt(st.delta)}: ${st.delta >= 0 ? 'better' : 'worse'} than expected.`;
    else status = `The critic moves V(${cell}) from ${fmt(st.v)} to ${fmt(st.vAfter)}. The actor ${st.delta >= 0 ? 'raises' : 'lowers'} the chance of ${ACTIONS[st.a]} at ${cell} from ${fmt(st.probsBefore[st.a])} to ${fmt(st.probsAfter[st.a])}, because δ is ${st.delta >= 0 ? 'positive' : 'negative'}.`;
    const done = u >= 0.35 || t < LEAD ? (u >= 0.35 ? k + 1 : k) : k;
    return {
      part: 'walk', k, u, fx, fy, e: 0, kEval: 0, status,
      detail: `Episode ${s.ep}. The cell colour and number are the critic's value V; the arrows are the actor's action probabilities.`,
      readouts: [['Step', `${t < LEAD ? 0 : k + 1} of ${s.T}`], ['TD error δ', t < LEAD || u < 0.35 ? '–' : signFmt(st.delta)], [`V(${cell})`, t < LEAD ? '–' : fmt(updated ? st.vAfter : st.v)], ['Steps scored', String(done)]],
    };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow;
    if (f.part === 'curves') {
      const cb = nar ? { x: 46, y: 36, w: E.W - 62, h: E.H - 100 } : { x: 56, y: 34, w: E.W - 76, h: E.H - 100 };
      const series = [] as { ys: ArrayLike<number>; xs: ArrayLike<number>; color: string; width?: number; op?: number; dash?: string }[];
      for (const [r, color] of [[s.none, 'var(--viz-cat-1)'], [s.sv, 'var(--viz-cat-2)'], [s.ac, 'var(--viz-cat-3)']] as [PGRuns, string][]) {
        for (const y of r.runs) series.push({ ys: y, xs: r.at, color, width: 1, op: 0.18 });
        series.push({ ys: r.mean, xs: r.at, color, width: 2.75 });
      }
      series.push({ ys: [s.vstar, s.vstar], xs: [0, s.episodes], color: 'var(--text-muted)', width: 1.25, dash: '5 4' });
      series.push({ ys: [s.thr, s.thr], xs: [0, s.episodes], color: 'var(--text-muted)', width: 1, dash: '1 4' });
      lineChart(E, cb, { lo: 0, hi: s.episodes, ticks: [0, 250, 500, 750, 1000].filter((v) => v <= s.episodes), label: 'Episodes of training' }, { lo: -1.75, hi: 0.85, ticks: [-1.5, -1, -0.5, 0, 0.5], label: 'Expected return of the current policy (average of runs; faint: single runs)' }, series, f.e);
      text(E, `Best possible ${fmt(s.vstar, 2)}`, cb.x + cb.w - 4, cb.y + 12, { anchor: 'end', fill: MUTED, size: 11 });
      text(E, `Threshold ${fmt(s.thr, 1)}`, cb.x + cb.w - 4, lineY(cb, s.thr) - 4, { anchor: 'end', fill: MUTED, size: 11 });
      fadeFrom(E, 0, ease((f.e * 0 + 1)));
      return;
    }
    const st = s.steps[f.k], vS = f.u >= 0.6 ? s.vs[f.k + 1] : s.vs[f.k], tw = ease((f.u - 0.6) / 0.4);
    const probs = Float64Array.from(s.ps[f.k], (a, i) => lerp(a, s.ps[f.k + 1][i], tw));
    const gb = nar ? { x: 8, y: 8, w: E.W - 16, h: E.H * 0.5 } : { x: 8, y: 8, w: E.W * 0.52, h: E.H - 16 };
    const G = gridIn(gb), flash = f.u >= 0.35 && f.u < 0.85;
    drawGrid(E, G, { fill: (c) => signedFill(vS[c], 1) });
    for (let c = 0; c < 12; c++) {
      if (c === 5 || c === 7 || c === GOAL) continue;
      const p = cellXY(G, c);
      text(E, fmt(vS[c], 2), p.x + 5, p.y + 14, { size: 11, fill: inkOn(Math.min(1, Math.abs(vS[c]))), weight: 600 });
    }
    drawPolicy(E, G, { probs });
    if (flash) { const p = cellXY(G, st.s); rect(E, p.x + 1.5, p.y + 1.5, G.cs - 3, G.cs - 3, { fill: 'none', stroke: st.delta >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', sw: 3.5 }); }
    drawAgent(E, G, f.fx, f.fy);
    const rows: string[][] = [], sgn: number[] = [], shown = f.u >= 0.35 ? f.k + 1 : f.k, from = Math.max(0, shown - (nar ? 6 : 12));
    for (let i = from; i < shown; i++) { const q = s.steps[i]; rows.push([String(i + 1), cellName(q.s), `${ARROWS[q.a]}`, fmt(q.r), signFmt(q.delta)]); sgn.push(q.delta); }
    const tx = nar ? 12 : gb.x + gb.w + 22, ty = nar ? gb.y + gb.h + 26 : 28, cols = nar ? [0, 34, 94, 140, 200] : [0, 34, 92, 140, 196];
    const yEnd = table(E, tx, ty, cols, rows, { header: ['#', 'Cell', 'Move', 'Reward', 'δ'], rh: 19 });
    sgn.forEach((d, i) => text(E, signFmt(d), tx + cols[4], ty + 19 * (i + 1), { fill: d >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', weight: 650 }));
    if (!rows.length) text(E, 'Each scored step appears here.', tx, ty + 22, { fill: MUTED });
    void yEnd; void tdLine;
  },
  label: (s) => `Actor-critic on the 4 by 3 grid: first episode ${s.ep}, step by step, with the critic's values as cell colours, the actor's policy as arrows and the TD error of each step; then the learning curves of actor-critic and REINFORCE over ${s.episodes} episodes.`,
};
export default mode;

function lineY(cb: { y: number; h: number }, v: number): number { return cb.y + ((0.85 - v) / (0.85 + 1.75)) * cb.h; }
