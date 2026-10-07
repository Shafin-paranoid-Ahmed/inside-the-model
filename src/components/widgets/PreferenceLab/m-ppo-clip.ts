/**
 * ppo-clip (rlhf step 5): one PPO update on a batch of responses, with Day 11's clipped objective. The 64 samples of the
 * batch collected at update `iter` (default 0; by update 20 the policy has settled and nothing leaves the band) are
 * shown as dots at their probability ratio pi_new(y|x) / pi_old(y|x) after each of the 4 epochs. Colour = sign of the
 * advantage; a hollow dot is on the flat side of the clip (zero gradient); a filled dot outside the band is on the
 * penalised side and still pushes. Below, the clipped objective of one picked sample (click a dot) against its ratio.
 * Props: iter (0), epochs (4), eps (0.2), beta (0.1), seed (6), labels ("human").
 * Measured (human labels, beta 0.1, seed 6, iter 0), outside the band / flat after epochs 1..4: 0.078/0.078, 0.391/0.359,
 *   0.422/0.359, 0.359/0.328; ratios 0.43 to 1.29 at epoch 4; 45 of 64 advantages positive (plan §7).
 */
import { PROMPTS, CANDIDATES } from '../../../lib/align/bank';
import type { AlignPPOResult } from '../../../lib/align/ppo';
import { clippedSurrogate } from '../../../lib/rl/surrogate';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, BG, MUTED, circle, clamp, ease, fmt, lerp, line, mix, rect, text, type Env } from '../_rl/svg';
import { surrogatePlot } from '../PPOLab/draw';
import { ppoRun } from './sim';

const LEAD = 1.8, EP = 3.4, MOVE = 0.62, HOLD = 2.2, LO = 0.4, HI = 1.6;
interface Scene {
  iter: number; E: number; B: number; eps: number; ratios: Float64Array[]; adv: Float64Array; x: Uint8Array; y: Uint8Array;
  oob: number[]; flat: number[]; start: number;
}
interface F extends Frame { n: number; k: number; w: number; pick: number; moving: boolean }

const flatAt = (s: Scene, e: number, i: number) => s.adv[i] !== 0 && clippedSurrogate(s.ratios[e][i], s.adv[i], s.eps).dRho === 0;

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const iter = int(o.iter, 0), res: AlignPPOResult = await ppoRun(o, kit, { iters: iter + 1, record: [iter] });
    const E = res.epochs, B = res.batch, det = res.detail[0].ratios, ratios = [new Float64Array(B).fill(1)];
    for (let e = 0; e < E; e++) ratios.push(det.slice(e * B, (e + 1) * B));
    const adv = Float64Array.from(res.badv.subarray(iter * B, (iter + 1) * B)), x = res.bx.slice(iter * B, (iter + 1) * B), y = res.by.slice(iter * B, (iter + 1) * B);
    const s: Scene = { iter, E, B, eps: res.eps, ratios, adv, x, y, oob: [0, ...Array.from(res.outOfBand.subarray(iter * E, (iter + 1) * E))], flat: [0, ...Array.from(res.plateau.subarray(iter * E, (iter + 1) * E))], start: 0 };
    // default pick: a sample outside the band on the side that still pushes (positive advantage and ratio below the band, or negative and above), else the largest |advantage|
    const last = ratios[E];
    let pick = -1;
    for (let i = 0; i < B && pick < 0; i++) if ((adv[i] > 0 && last[i] < 1 - s.eps) || (adv[i] < 0 && last[i] > 1 + s.eps)) pick = i;
    if (pick < 0) pick = adv.reduce((a, v, i) => (Math.abs(v) > Math.abs(adv[a]) ? i : a), 0);
    s.start = pick;
    return s;
  },
  timeline(s) {
    const mk = [{ at: 0, label: 'Before' }];
    for (let e = 0; e < s.E; e++) mk.push({ at: LEAD + e * EP, label: `Epoch ${e + 1}` });
    return { duration: LEAD + s.E * EP + HOLD, markers: mk };
  },
  frameAt(t, s, _o, ui) {
    const u = (t - LEAD) / EP, ended = u >= s.E, k = ended ? s.E : clamp(Math.floor(u), 0, s.E - 1), fr = ended ? 1 : clamp(u - k, 0, 1), moving = t >= LEAD && !ended && fr < MOVE;
    const n = t < LEAD ? 0 : ended ? s.E : fr >= MOVE ? k + 1 : k, w = t < LEAD ? 0 : ended ? 0 : ease(fr / MOVE);
    const pk = Number(ui.pick), pick = ui.pick !== null && Number.isInteger(pk) && pk >= 0 && pk < s.B ? pk : s.start;
    const r = s.ratios[n], lo = Math.min(...r), hi = Math.max(...r), nOut = Math.round(s.oob[n] * s.B), nFlat = Math.round(s.flat[n] * s.B), pos = s.adv.filter((a) => a > 0).length;
    let status: string;
    if (t < LEAD) status = `One batch of ${s.B} responses (${pos} with positive advantage, ${s.B - pos} with negative) was collected at update ${s.iter}. Before any update every probability ratio is exactly 1.`;
    else if (moving) status = `Epoch ${k + 1}: the policy takes ${s.B / 16} mini-batch steps on the same batch, so the dots move.`;
    else status = `After epoch ${n}: ratios run from ${fmt(lo, 2)} to ${fmt(hi, 2)}. ${nOut} of ${s.B} samples are outside the band; ${nFlat} of them are on the flat side (hollow, no gradient) and ${nOut - nFlat} are on the side where the objective still pushes.`;
    const c = clippedSurrogate(r[pick], s.adv[pick], s.eps), x = s.x[pick], y = s.y[pick];
    const detail = `Sample ${pick + 1}: ${CANDIDATES[y].label.toLowerCase()} for “${PROMPTS[x].q}”, advantage ${fmt(s.adv[pick], 2)}, ratio ${fmt(r[pick], 2)}, objective ${fmt(c.value, 2)}, slope ${fmt(c.dRho, 2)}${c.dRho === 0 ? ' (flat)' : ''}. Click a dot to follow another sample.`;
    return { n, k, w, pick, moving, status, detail, readouts: [['Epochs done', `${n} of ${s.E}`], ['Outside the band', `${nOut} of ${s.B}`], ['Flat, no gradient', `${nFlat} of ${s.B}`], ['Outside but still pushing', `${nOut - nFlat} of ${s.B}`]] };
  },
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, sb = { x: nw ? 12 : 18, y: 40, w: W - (nw ? 24 : 36), h: nw ? 130 : 140 };
    const sx = (v: number) => sb.x + ((clamp(v, LO, HI) - LO) / (HI - LO)) * sb.w;
    text(E, `Probability ratio of each of the ${s.B} sampled responses`, sb.x, sb.y - 14, { size: 12, fill: MUTED });
    rect(E, sx(1 - s.eps), sb.y, sx(1 + s.eps) - sx(1 - s.eps), sb.h, { fill: mix(ACCENT, BG, 16) });
    line(E, sx(1), sb.y, sx(1), sb.y + sb.h, { stroke: MUTED, sw: 1, dash: '3 3' });
    line(E, sb.x, sb.y + sb.h, sb.x + sb.w, sb.y + sb.h, { stroke: 'var(--rule)', sw: 1 });
    for (const v of [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]) text(E, fmt(v, 1), sx(v), sb.y + sb.h + 14, { size: 11, anchor: 'middle', fill: MUTED });
    const a0 = s.ratios[f.k], a1 = s.ratios[Math.min(s.E, f.k + 1)], next = f.moving ? f.w : 0, from = f.moving ? a0 : s.ratios[f.n], to = f.moving ? a1 : s.ratios[f.n];
    const rows: number[] = [];
    for (let i = 0; i < s.B; i++) rows.push(i);
    for (const i of rows.sort((a, b) => (a === f.pick ? 1 : 0) - (b === f.pick ? 1 : 0))) {
      const cx = sx(lerp(from[i], to[i], next)), cy = sb.y + 10 + (((i * 0.6180339) % 1) * (sb.h - 20)), col = s.adv[i] >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)', flat = flatAt(s, f.n, i);
      circle(E, cx, cy, i === f.pick ? 5.5 : 4, { fill: flat ? BG : col, stroke: i === f.pick ? ACCENT : flat ? col : BG, sw: i === f.pick ? 2.5 : flat ? 1.5 : 0.75, op: 0.92, pick: String(i), tip: `Sample ${i + 1}: advantage ${fmt(s.adv[i], 2)}, ratio ${fmt(to[i], 2)}` });
    }
    const pb = { x: nw ? 44 : 60, y: sb.y + sb.h + (nw ? 62 : 56), w: W - (nw ? 44 : 60) - 16, h: H - (sb.y + sb.h + (nw ? 62 : 56)) - 46 };
    surrogatePlot(E, pb, { A: s.adv[f.pick], eps: s.eps, lo: LO, hi: HI, rho: lerp(from[f.pick], to[f.pick], next) });
  },
  label: (s) => `One PPO update on ${s.B} sampled responses over ${s.E} epochs: each response's probability ratio and the clipped objective of one picked sample. After the last epoch ${Math.round(s.oob[s.E] * s.B)} samples are outside the clip band.`,
};
export default mode;
