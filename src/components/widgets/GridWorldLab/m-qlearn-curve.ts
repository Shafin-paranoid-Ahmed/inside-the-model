/**
 * qlearn-curve (q-learning step 4): the same Q-learning with several exploration rates, `runs` seeds each. Left: the
 * greedy policy of one run (arrows turn red where they differ from the optimal policy). Right: how many of the 9
 * action cells have the optimal greedy action, and the largest |Q - Q*|, averaged over runs.
 * Props: epsilons ("0,0.1,0.3"), alpha (0.1), episodes (500), runs (20), seed (21), gamma (1).
 */

import { greedy, qLearning, qError, policyAgreement } from '../../../lib/rl/tabular';
import { N } from '../../../lib/rl/gridworld';
import { int, key, nums, num, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { TEXT, fmt, lineChart, text, type Env } from '../_rl/svg';
import { gridOf } from './sim';
import { drawGrid, drawPolicy, gridIn } from './draw';
import { qStar } from './q-common';

const DUR = 14, HOLD = 3, COLORS = ['var(--viz-cat-1)', 'var(--viz-cat-2)', 'var(--viz-cat-3)'];
interface Agg { epsilon: number; agree: Float64Array; err: Float64Array; first: number[]; pi0: Int8Array }
interface Scene { aggs: Agg[]; E: number; runs: number; shown: number; opt: Int8Array }
interface F extends Frame { ep: number }

const tick = () => new Promise<void>((r) => setTimeout(r, 0));
const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const g = gridOf(o), q = qStar(kit, g), eps = nums(o.epsilons, [0, 0.1, 0.3]).slice(0, 3), alpha = num(o.alpha, 0.1);
    const E = int(o.episodes, 500), runs = int(o.runs, 20), seed = int(o.seed, 21), aggs: Agg[] = [];
    for (const epsilon of eps) {
      await tick(); // one epsilon per task keeps each main-thread task short
      aggs.push(kit.memo(key('qcurve', g.stepReward, g.slip, g.gamma, epsilon, alpha, E, runs, seed), () => {
        const agree = new Float64Array(E + 1), err = new Float64Array(E + 1), first: number[] = [], pi0 = new Int8Array((E + 1) * N);
        for (let r = 0; r < runs; r++) {
          const res = qLearning(g, { alpha, epsilon, episodes: E, seed: seed * 1000 + r + 1, record: false });
          let hit = -1;
          for (let e = 0; e <= E; e++) {
            const pi = greedy(g, res.snapshots[e]), a = policyAgreement(g, pi, q.policy);
            agree[e] += a / runs; err[e] += qError(g, res.snapshots[e], q.Q) / runs;
            if (a === 9 && hit < 0) hit = e;
            if (r === 0) pi0.set(pi, e * N);
          }
          first.push(hit);
        }
        return { epsilon, agree, err, first, pi0 };
      }));
    }
    const shown = Math.max(0, eps.indexOf(0.1)) || (eps.length > 1 ? 1 : 0);
    return { aggs, E, runs, shown, opt: q.policy };
  },
  timeline(s) {
    return { duration: DUR + HOLD, markers: [{ at: 0, label: 'Learning' }, { at: DUR, label: `All ${s.E} episodes` }] };
  },
  frameAt(t, s) {
    const ep = Math.min(s.E, Math.round((Math.min(1, t / DUR)) * s.E)), a = s.aggs;
    const ro: [string, string][] = [['Episode', `${ep} of ${s.E}`]];
    for (const x of a) ro.push([`ε = ${x.epsilon}`, `${fmt(x.agree[ep], 1)} of 9 cells · error ${fmt(x.err[ep], 2)}`]);
    let status: string;
    if (ep < s.E) status = `Episode ${ep}: averaged over ${s.runs} runs, ` + a.map((x) => `ε = ${x.epsilon} has ${fmt(x.agree[ep], 1)} of 9 cells right`).join(', ') + '.';
    else status = `After ${s.E} episodes, runs that ever had all 9 cells right: ` + a.map((x) => `ε = ${x.epsilon}: ${x.first.filter((v) => v >= 0).length} of ${s.runs}`).join(', ') + '.';
    return { ep, status, readouts: ro };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow;
    const gb = nar ? { x: 8, y: 24, w: E.W - 16, h: E.W * 0.55 } : { x: 10, y: 30, w: E.W * 0.3, h: E.H * 0.5 };
    const G = gridIn(gb), sh = s.aggs[s.shown];
    text(E, `Greedy policy of one run, ε = ${sh.epsilon}`, gb.x + 2, 16, { size: 12, weight: 650 });
    drawGrid(E, G, { labels: true });
    const pi = sh.pi0.subarray(f.ep * N, f.ep * N + N);
    drawPolicy(E, G, { actions: pi, color: TEXT, show: (c) => (pi[c] === s.opt[c] ? 1 : 0) });
    drawPolicy(E, G, { actions: pi, color: 'var(--viz-neg)', show: (c) => (pi[c] === s.opt[c] ? 0 : 1) });
    const x0 = nar ? 44 : G.x0 + G.w + 64, w = E.W - x0 - 56, xs = Array.from({ length: s.E + 1 }, (_, i) => i);
    const top = nar ? G.y0 + G.h + 40 : 26, avail = E.H - top - 44, h = (avail - 46) / 2;
    const b1 = { x: x0, y: top, w, h }, b2 = { x: x0, y: top + h + 54, w, h };
    const xa = { lo: 0, hi: s.E, label: 'Episode' };
    const mk = (pick: (a: Agg) => Float64Array) => s.aggs.map((a, i) => ({ xs, ys: pick(a), color: COLORS[i], width: 2 }));
    lineChart(E, b1, { ...xa, label: undefined }, { lo: 0, hi: 9, ticks: [0, 3, 6, 9], label: 'Cells with the optimal action (of 9)' }, mk((a) => a.agree), f.ep);
    lineChart(E, b2, xa, { lo: 0, hi: 1, ticks: [0, 0.5, 1], label: 'Largest error |Q − Q*|' }, mk((a) => a.err), f.ep);
    // direct labels at the right edge of the chart, at the current value
    s.aggs.forEach((a, i) => {
      const y1 = b1.y + b1.h * (1 - a.agree[f.ep] / 9), y2 = b2.y + b2.h * (1 - Math.min(1, a.err[f.ep]));
      text(E, `ε ${a.epsilon}`, b1.x + b1.w + 6, y1 + 4, { size: 11, fill: COLORS[i] });
      text(E, `ε ${a.epsilon}`, b2.x + b2.w + 6, y2 + 4, { size: 11, fill: COLORS[i] });
    });
  },
  label: (s) => `Learning curves of tabular Q-learning with exploration rates ${s.aggs.map((a) => a.epsilon).join(', ')}, averaged over ${s.runs} runs: the number of cells with the optimal greedy action and the largest error against Q star, over ${s.E} episodes.`,
};
export default mode;
