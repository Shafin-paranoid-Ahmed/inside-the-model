/**
 * cap (sft step 6): three SFT runs on demonstration sets of different accuracy; each curve is the expected correctness of the
 * fine-tuned policy (averaged over the 8 training prompts) over training, with a dashed line at the demonstrators' own rate.
 * Props: demos (40, so the sets are exactly 0.6 / 0.8 / 0.95 correct), demoAccuracies ("0.6,0.8,0.95"), steps (300), lr
 * (0.05), seed (2). Measured (demos 40): the curves overshoot their lines early (peaks 0.716, 0.830, none) and end at
 * 0.602, 0.804, 0.955 after 300 steps (moment matching: at the optimum E_pi[correct] equals the demonstrations' mean).
 */
import { F_CORRECT } from '../../../lib/align/bank';
import { expectedFeatures } from '../../../lib/align/policy';
import { demoFeatures, makeDemos, trainSFT } from '../../../lib/align/sft';
import { int, key, num, nums, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, TEXT, circle, fmt, lineChart, path, text, type Env } from '../_rl/svg';

const SP = 0.04, FAST = 50, HOLD = 2.5;
interface Run { rate: number; corr: Float64Array }
interface Scene { runs: Run[]; steps: number }
interface F extends Frame { k: number }

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts, kit) {
    const accs = nums(o.demoAccuracies, [0.6, 0.8, 0.95]), n = int(o.demos, 40), steps = int(o.steps, 300), lr = num(o.lr, 0.05), seed = int(o.seed, 2);
    return kit.memo(key('sft-cap', accs, n, steps, lr, seed), () => ({
      steps,
      runs: accs.map((accuracy) => {
        const demos = makeDemos({ n, accuracy, seed }), run = trainSFT(demos, { steps, lr });
        return { rate: demoFeatures(demos)[F_CORRECT], corr: Float64Array.from(run.thetas, (th) => expectedFeatures(th)[F_CORRECT]) };
      }),
    }));
  },
  timeline: (s) => ({ duration: s.steps * SP + HOLD, markers: [{ at: 0, label: 'Rise' }, { at: Math.min(FAST, s.steps) * SP, label: 'Settle' }] }),
  frameAt(t, s) {
    const k = Math.min(s.steps, Math.floor(t / SP + 1e-9));
    const status = k === s.steps
      ? `After ${s.steps} steps each model answers correctly almost exactly as often as its demonstrators did: ${s.runs.map((r) => `${fmt(r.corr[k], 3)} against ${fmt(r.rate, 3)}`).join(', ')}.`
      : `Step ${k} of ${s.steps}: the three models answer correctly ${s.runs.map((r) => fmt(r.corr[k], 2)).join(', ')} of the time; their demonstrators were right ${s.runs.map((r) => fmt(r.rate, 2)).join(', ')} of the time.`;
    return { k, status, readouts: [['Training step', `${k} of ${s.steps}`], ...s.runs.map((r): [string, string] => [`Demonstrators ${fmt(100 * r.rate, 0)}% correct`, `model ${fmt(r.corr[k], 3)}`])] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, b = { x: pad + 34, y: 26, w: W - 2 * pad - 40, h: H - 26 - 46 };
    text(E, 'Chance of a correct answer', pad, 14, { size: 11, fill: MUTED });
    const { sx, sy } = lineChart(E, b, { lo: 0, hi: s.steps, ticks: [0, 100, 200, 300].filter((v) => v <= s.steps), label: 'Training step' }, { lo: 0, hi: 1, ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] },
      [
        ...s.runs.map((r) => ({ ys: r.corr, color: 'var(--viz-forward)', width: 2.75 })),
      ], f.k);
    s.runs.forEach((r) => path(E, `M${sx(0)} ${sy(r.rate)}L${sx(s.steps)} ${sy(r.rate)}`, { stroke: TEXT, sw: 1.5, dash: '6 4' }));
    s.runs.forEach((r) => {
      text(E, `demonstrators ${fmt(100 * r.rate, 0)}% correct`, b.x + b.w - 4, sy(r.rate) - 6, { size: nw ? 11 : 12, anchor: 'end', halo: true });
      circle(E, sx(f.k), sy(r.corr[f.k]), 4, { fill: 'var(--accent)' });
    });
  },
  label: (s) => `Three fine-tuning runs on demonstrations that are ${s.runs.map((r) => fmt(100 * r.rate, 0)).join('%, ')}% correct: each model's chance of a correct answer settles at ${s.runs.map((r) => fmt(r.corr[s.steps], 3)).join(', ')}, on its demonstrators' rate.`,
};
export default mode;
