/**
 * ppo (rlhf steps 4 and 7; constitutional-ai step 5): PPO against the reward model of prefData(o), with the KL penalty
 * to the SFT reference (src/lib/align trainAlignPPO, a kit.job). Left: the policy's probabilities for one prompt (SFT
 * model as outlined ghost bars) with the sampled responses of the current batch as dots (blue: above the prompt's
 * average reward, orange: below), and KL to the SFT model. Right: proxy (reward model) and gold reward per update.
 * Axes are fixed by rlhf-common.ts (reward -1..5) so every PPO chart reads the same.
 * Props: labels ("human" | "ai"), beta (0.1), iters (200), batch (64), seed (6 = PPO seed; pairs use pairSeed 4),
 *   compare ("human": also train on the human-label reward model and draw its gold curve faint), prompt (0), peak
 *   ("true" | "false": mark the gold peak; default on when beta < 0.05).
 * Measured (plans/day-11.md §7, human labels): beta 0.1 proxy / gold / KL 3.345 / 2.273 / 0 -> 4.076 / 2.808 / 0.802;
 *   beta 0: gold peaks at iteration 19 (2.819), final proxy 4.719, gold -0.439, KL 13.01.
 */
import { K, PROMPTS, Y_RAMBLING } from '../../../lib/align/bank';
import { meanProb } from '../../../lib/align/policy';
import type { AlignPPOResult } from '../../../lib/align/ppo';
import { bool, int, num, str, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { BG, MUTED, axes, circle, clamp, ease, fmt, lerp, path, text, type Env } from '../_rl/svg';
import { GOLD_INK, PROXY, candidateBars, charsFor, clip, rewardChart } from '../_align/draw';
import { labelsOf, ppoRun } from './sim';
import { REWARD_Y, argmax, candName, iterAxis, niceCeil, peakIndex, quoted, rowOf } from './rlhf-common';

const SCORE_AT = 1.6, INTRO = 3.2, PACE = 0.1, HOLD = 3;
interface Scene {
  res: AlignPPOResult; cmp?: Float64Array; iters: number; beta: number; x: number; labels: string; peak: number; showPeak: boolean;
  klHi: number; ramble: Float64Array; refRow: Float64Array; B: number;
}
interface F extends Frame { n: number; k: number; w: number; phase: 'sample' | 'score' | 'run'; reveal: number }

function at(t: number, iters: number): Pick<F, 'n' | 'k' | 'w' | 'phase' | 'reveal'> {
  if (t < INTRO) return { n: 0, k: 0, w: 0, phase: t < SCORE_AT ? 'sample' : 'score', reveal: clamp(t / (SCORE_AT - 0.3), 0, 1) };
  const u = Math.min(iters, (t - INTRO) / PACE);
  if (u >= iters) return { n: iters, k: iters, w: 0, phase: 'run', reveal: 1 };
  const k = Math.floor(u), fr = u - k;
  return { n: k + (fr >= 0.5 ? 1 : 0), k, w: ease(fr), phase: 'run', reveal: 1 };
}
/** The batch of update n restricted to the shown prompt: [response, advantage] per sample, sorted by response. */
function batchOf(s: Scene, n: number): [number, number][] {
  const out: [number, number][] = [];
  if (n >= s.iters) return out;
  for (let i = 0; i < s.B; i++) if (s.res.bx[n * s.B + i] === s.x) out.push([s.res.by[n * s.B + i], s.res.badv[n * s.B + i]]);
  return out.sort((a, b) => a[0] - b[0]);
}

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const labels = labelsOf(o), beta = num(o.beta, 0.1), iters = int(o.iters, 200), x = int(o.prompt, 0), B = int(o.batch, 64);
    const res = await ppoRun(o, kit);
    let cmp: Float64Array | undefined;
    if (str(o.compare, ['human', 'none'], 'none') === 'human' && labels !== 'human') cmp = (await ppoRun({ ...o, labels: 'human' } as Opts, kit)).gold;
    const ramble = Float64Array.from(res.thetas, (th) => meanProb(th, Y_RAMBLING));
    return { res, cmp, iters, beta, x, labels, peak: peakIndex(res.gold), showPeak: bool(o.peak, beta < 0.05), klHi: niceCeil(Math.max(...res.kl) * 1.08), ramble, refRow: rowOf(res.thetas[0], x), B };
  },
  timeline(s) {
    const mk = [{ at: 0, label: 'Sample' }, { at: SCORE_AT, label: 'Score' }];
    for (let i = 0; i < 4; i++) mk.push({ at: INTRO + Math.round((s.iters * i) / 4) * PACE + (i ? 0 : 1e-4), label: `${Math.round((s.iters * i) / 4) + 1}–${Math.round((s.iters * (i + 1)) / 4)}` });
    return { duration: INTRO + s.iters * PACE + HOLD, markers: mk };
  },
  frameAt(t, s) {
    const a = at(t, s.iters), n = a.n, r = s.res, row = rowOf(r.thetas[n], s.x), top = argmax(row), dots = batchOf(s, n);
    const pos = dots.filter((d) => d[1] > 0).length, name = PROMPTS[s.x].q;
    const readouts: [string, string][] = [['Update', `${n} of ${s.iters}`], ['Proxy reward', fmt(r.proxy[n], 2)], ['Gold reward', fmt(r.gold[n], 2)], ['KL to SFT', fmt(r.kl[n], n && r.kl[n] < 1 ? 3 : 2)]];
    if (s.cmp) readouts.push(['Gold, human labels', fmt(s.cmp[n], 2)]);
    if (s.beta < 0.05) readouts.push(['Rambling answer', fmt(s.ramble[n], 2)]);
    let status: string;
    if (a.phase === 'sample') status = `Update 0: the policy writes ${s.B} responses to the training prompts, ${dots.length} of them to “${name}” (the dots). It has not changed yet, so the bars match the SFT model.`;
    else if (a.phase === 'score') status = `The reward model scores every response${s.beta ? ', minus a KL charge for drifting from the SFT model' : ''}. For this prompt ${pos} of the ${dots.length} responses score above their prompt's average (blue) and ${dots.length - pos} below (orange). PPO raises the above-average ones.`;
    else {
      status = `After ${n} update${n === 1 ? '' : 's'}: proxy ${fmt(r.proxy[n], 2)}, gold ${fmt(r.gold[n], 2)}, KL ${fmt(r.kl[n], 2)}.`;
      if (s.showPeak && n >= s.peak) status += ` Gold peaked at update ${s.peak} (${fmt(r.gold[s.peak], 2)}).`;
      status += ` For “${name}” the policy now writes ${candName(top)} (${fmt(row[top], 2)}): ${quoted(s.x, top)}`;
    }
    return { ...a, status, readouts, detail: s.cmp ? 'Faint dashed line: gold reward of the run trained on human labels.' : undefined };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, r = s.res, n = f.n, moving = f.phase === 'run' && f.k < s.iters && f.w > 0;
    const row = rowOf(r.thetas[n], s.x);
    const from = moving ? rowOf(r.thetas[f.k], s.x) : row, to = moving ? rowOf(r.thetas[f.k + 1], s.x) : row;
    const bars = Float64Array.from(from, (v, i) => lerp(v, to[i], f.w));
    const leftW = nw ? W - 20 : Math.min(360, W * 0.5), dotsW = nw ? 50 : 66, bx = nw ? 10 : 12, rh = nw ? 21 : 24;
    text(E, clip(`Prompt: ${PROMPTS[s.x].q}`, charsFor(leftW, 12)), bx, 16, { size: 12, weight: 600 });
    const box = { x: bx, y: 26, w: leftW - dotsW - 10, h: K * rh };
    candidateBars(E, box, { p: bars, values: row, ref: s.refRow, highlight: argmax(row), max: 1 });
    // the sampled responses of this batch as dots beside their rows
    const dots = batchOf(s, n), dx0 = bx + leftW - dotsW + 4, per = nw ? 7 : 9, seen: number[] = new Array(K).fill(0);
    const shown = f.phase === 'sample' ? Math.ceil(dots.length * f.reveal) : dots.length;
    for (const [y, adv] of dots.slice(0, shown)) {
      const j = seen[y]++, cy = box.y + y * rh + rh / 2 + (j >= per ? 5 : -2);
      circle(E, dx0 + (j % per) * 6.2, cy, 2.6, { fill: f.phase === 'sample' ? MUTED : adv > 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', stroke: BG, sw: 0.75 });
    }
    const cy0 = box.y + K * rh + 16;
    ['Outline: the SFT model. Dots: this batch’s responses to the prompt', '(blue above the prompt’s average reward, orange below).'].forEach((l, i) => text(E, clip(l, charsFor(leftW, 11)), bx, cy0 + i * 14, { size: 11, fill: MUTED }));
    const rb = nw ? { x: 44, y: cy0 + 50, w: W - 56, h: 150 } : { x: leftW + bx + 52, y: 34, w: W - (leftW + bx + 52) - 14, h: H - 34 - 40 };
    const kb = nw ? { x: 44, y: rb.y + rb.h + 44, w: W - 56, h: 76 } : { x: 54, y: cy0 + 40, w: leftW - 54 + bx - 8, h: Math.max(70, H - (cy0 + 40) - 44) };
    const { sx, sy } = rewardChart(E, rb, { proxy: r.proxy, gold: r.gold, goldCompare: s.cmp, upto: n, x: iterAxis(s.iters, nw ? undefined : 'PPO update'), y: REWARD_Y, mark: s.showPeak ? { i: s.peak, label: `Gold peaks at update ${s.peak}` } : undefined });
    circle(E, sx(n), clamp(sy(r.proxy[n]), rb.y, rb.y + rb.h), 4, { fill: PROXY, stroke: BG, sw: 2 });
    circle(E, sx(n), clamp(sy(r.gold[n]), rb.y, rb.y + rb.h), 4, { fill: GOLD_INK, stroke: BG, sw: 2 });
    const { sx: kx, sy: ky } = axes(E, kb, iterAxis(s.iters, 'PPO update'), { lo: 0, hi: s.klHi, label: 'KL to the SFT model' });
    let d = '';
    for (let i = 0; i <= n; i++) d += `${d ? 'L' : 'M'}${kx(i).toFixed(1)} ${clamp(ky(r.kl[i]), kb.y, kb.y + kb.h).toFixed(1)}`;
    path(E, d, { stroke: MUTED, sw: 2.25 });
    circle(E, kx(n), clamp(ky(r.kl[n]), kb.y, kb.y + kb.h), 4, { fill: MUTED, stroke: BG, sw: 2 });
  },
  label: (s) => `PPO training against a ${s.labels === 'ai' ? 'AI-label' : 'human-label'} reward model, ${s.iters} updates with KL penalty ${s.beta}: probabilities of eight responses to one prompt, and the proxy reward, gold reward and KL to the SFT model per update. Gold ends at ${fmt(s.res.gold[s.iters], 2)}.`,
};
export default mode;
