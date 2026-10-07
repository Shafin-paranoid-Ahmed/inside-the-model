/**
 * edit (constitutional-ai step 6): switch principles on and off and retrain. For the chosen constitution the AI judge
 * labels 200 pairs (seed 4), a reward model is fitted, and PPO (beta 0.1, 200 iterations, seed 6) trains the policy
 * against it, like the rlhf lesson but with the rule checker as the labeler. One worker job per toggle set (about 45 ms);
 * the default constitution's run is drawn as a reference (ticks on the bars, faint gold line).
 * Toggles (the lesson passes the initial state as props): answer, accurate, flattery, refuse, cautious.
 * Measured (P(refuse) + P(hedge) / gold at the end): default 0.358 / 2.611, without "answer" 0.127 / 2.866, without
 * "accurate" 0.225 / 2.602, without "no flattery" 0.000 / 1.153 (padded answers win), without "no refusal" 0.016 / 2.858,
 * with "cautious" 0.999 / 2.169. SFT starts at 0.217.
 */
import { TRAIN, Y_HEDGED, Y_REFUSAL } from '../../../lib/align/bank';
import { DEFAULT_PRINCIPLES } from '../../../lib/align/constitution';
import { GOLD, expected } from '../../../lib/align/policy';
import type { AlignPPOResult } from '../../../lib/align/ppo';
import { fitRM, samplePairs } from '../../../lib/align/reward';
import { sftTheta } from '../../../lib/align/sft';
import { bool, key, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, clamp, fmt, text, type Env } from '../_rl/svg';
import { candidateBars, rewardChart } from '../_align/draw';
import { avgProbs } from './cr-common';

const LEAD = 0.6, RUN = 7, HOLD = 2.6, IDS: [string, string][] = [['answer', 'answer'], ['accurate', 'accurate'], ['flattery', 'no-flattery'], ['refuse', 'no-refuse'], ['cautious', 'cautious']];
interface Run { probs: Float64Array[]; hr: Float64Array; gold: Float64Array; iters: number }
interface Scene { cur: Run; def: Run; act: string[]; isDefault: boolean }
interface F extends Frame { k: number }

const activeOf = (o: Opts): string[] => IDS.filter(([k, id]) => bool(o[k], DEFAULT_PRINCIPLES.includes(id))).map(([, id]) => id);
const hedgeRefuse = (p: Float64Array) => p[Y_REFUSAL] + p[Y_HEDGED];
async function run(act: string[], kit: Parameters<ModeImpl<Scene, F>['setup']>[1]): Promise<Run> {
  const rm = kit.memo(key('cr-edit-rm', act), () => Array.from(fitRM(samplePairs(sftTheta(), 200, 4, 'ai', { active: act })).w));
  const res = await kit.job<AlignPPOResult>('trainAlignPPO', { rm, beta: 0.1, iters: 200, seed: 6 });
  const probs = res.thetas.map(avgProbs);
  return { probs, hr: Float64Array.from(probs, hedgeRefuse), gold: Float64Array.from(res.thetas, (th) => expected(th, GOLD, TRAIN)), iters: res.iters };
}

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const act = activeOf(o), isDefault = act.join() === DEFAULT_PRINCIPLES.join();
    const [def, cur] = await Promise.all([run(DEFAULT_PRINCIPLES, kit), isDefault ? null : run(act, kit)]);
    return { def, cur: cur ?? def, act, isDefault };
  },
  timeline: () => ({ duration: LEAD + RUN + HOLD, markers: [{ at: 0, label: 'Before training' }, { at: LEAD, label: 'Training on AI labels' }, { at: LEAD + RUN, label: 'Trained' }] }),
  frameAt(t, s) {
    const c = s.cur, k = Math.floor(clamp((t - LEAD) / RUN, 0, 1) * c.iters + 1e-9), hr = c.hr[k], last = k === c.iters;
    const status = k === 0 ? `Before training: the policy refuses or hedges with probability ${fmt(c.hr[0], 3)}, gold reward ${fmt(c.gold[0], 2)}.`
      : !last ? `Iteration ${k} of ${c.iters}: refusing or hedging has probability ${fmt(hr, 3)}, gold reward ${fmt(c.gold[k], 2)}.`
        : `Trained: refusing or hedging has probability ${fmt(hr, 3)} and gold reward is ${fmt(c.gold[k], 2)}` + (s.isDefault ? '.' : `; the default constitution ends at ${fmt(s.def.hr[s.def.iters], 3)} and ${fmt(s.def.gold[s.def.iters], 2)}.`);
    return {
      k, status,
      readouts: [['Iteration', `${k} of ${c.iters}`], ['Chance of refusing or hedging', fmt(hr, 3)], ['Gold reward', fmt(c.gold[k], 2)], ['Principles on', `${s.act.length} of 5`]],
      detail: s.isDefault ? 'This is the default constitution: the four principles on, the cautious one off.' : `Default constitution, trained: refusing or hedging ${fmt(s.def.hr[s.def.iters], 3)}, gold ${fmt(s.def.gold[s.def.iters], 2)}.`,
    };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, c = s.cur;
    text(E, 'Chance of each response', pad, 14, { size: 11, fill: MUTED });
    const barsW = nw ? W - 2 * pad : Math.round((W - 2 * pad) * 0.5), barsH = nw ? 230 : H - 34;
    candidateBars(E, { x: pad + 4, y: 22, w: barsW - 8, h: barsH }, { p: c.probs[f.k], ref: c.probs[0], ticks: s.isDefault ? undefined : s.def.probs[s.def.iters], highlight: Y_HEDGED });
    const lo = Math.min(0, Math.floor(Math.min(...c.gold, ...s.def.gold))), hi = Math.ceil(Math.max(...c.gold, ...s.def.gold) + 0.3);
    const cb = nw ? { x: pad + 36, y: 22 + barsH + 44, w: W - 2 * pad - 44, h: H - barsH - 22 - 44 - 50 } : { x: pad + barsW + 24 + 36, y: 40, w: W - (pad + barsW + 24) - pad - 44, h: H - 40 - 56 };
    rewardChart(E, cb, { gold: c.gold, goldCompare: s.isDefault ? undefined : s.def.gold, upto: f.k, x: { lo: 0, hi: c.iters, ticks: [0, 100, 200], label: 'PPO iteration' }, y: { lo, hi, label: 'Gold reward' } });
  },
  label: (s) => `Policy trained against an AI judge using ${s.act.length} of the 5 principles: the chance of refusing or hedging ends at ${fmt(s.cur.hr[s.cur.iters], 3)} and the gold reward at ${fmt(s.cur.gold[s.cur.iters], 2)}.`,
};
export default mode;
