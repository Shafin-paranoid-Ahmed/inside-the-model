/**
 * compare (dpo step 5): PPO (trainAlignPPO, seed 6, 200 iterations, batch 64) and DPO (trainDPO, 300 steps) at each beta
 * of `betas`, traced together by progress through training. Panels (one per beta, same axes): gold reward against the
 * distance moved from the reference, sqrt of the KL; last panel: the mean log-probability the DPO policy gives the chosen
 * and the rejected responses of the 200 pairs at the last beta. Props: betas ("0.1,0.02"), seed (6, PPO), pair seed 4.
 * Measured: beta 0.1 PPO gold / KL 2.808 / 0.802, DPO 2.819 / 0.968; beta 0.02 both -0.440 / 13.0; mean log pi(chosen)
 * -1.402 -> -3.862 (beta 0.1), -> -31.17 (beta 0.02); rejected -1.762 -> -15.30 / -86.92.
 */
import type { AlignPPOResult } from '../../../lib/align/ppo';
import { int, nums, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { BG, TEXT as INK, axes, circle, clamp, fmt, path, text, type Env } from '../_rl/svg';
import { GOLD_DASH } from '../_align/draw';
import { dpoRun } from './dpoKit';
import { ppoRun, prefData } from './sim';

const LEAD = 1, RUN = 12, TAIL = 2.5, GOLD_Y = { lo: -1, hi: 3.5, ticks: [-1, 0, 1, 2, 3], label: 'Gold reward' };
interface Curve { gold: Float64Array; x: Float64Array; kl: Float64Array }
interface One { beta: number; ppo: Curve; dpo: Curve; chosen: Float64Array; rejected: Float64Array }
interface Scene { runs: One[]; xmax: number; lpLo: number; steps: number; iters: number }
interface F extends Frame { u: number }
const curve = (gold: Float64Array, kl: Float64Array): Curve => ({ gold, kl, x: kl.map(Math.sqrt) });
const at = (c: Curve, u: number) => Math.round(u * (c.gold.length - 1));

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const d = prefData(o, kit, int(o.pairSeed, 4)), betas = nums(o.betas, [0.1, 0.02]), steps = 300;
    const runs = await Promise.all(betas.map(async (beta): Promise<One> => {
      const [p, q] = await Promise.all([ppoRun(o, kit, { beta }) as Promise<AlignPPOResult>, dpoRun(kit, d, beta, steps)]);
      return { beta, ppo: curve(p.gold, p.kl), dpo: curve(q.gold, q.kl), chosen: q.chosen, rejected: q.rejected };
    }));
    const xmax = Math.ceil(Math.max(...runs.flatMap((r) => [...r.ppo.x, ...r.dpo.x]))), last = runs[runs.length - 1];
    return { runs, xmax, lpLo: -Math.ceil(Math.max(...last.rejected.map(Math.abs)) / 10) * 10, steps, iters: runs[0].ppo.gold.length - 1 };
  },
  timeline: () => ({ duration: LEAD + RUN + TAIL, markers: [{ at: 0, label: 'Same start' }, { at: LEAD, label: 'Training both' }, { at: LEAD + RUN, label: 'Trained' }] }),
  frameAt(t, s) {
    const u = clamp((t - LEAD) / RUN, 0, 1), last = s.runs[s.runs.length - 1], di = at(last.dpo, u);
    const readouts: [string, string][] = s.runs.flatMap((r) => [
      [`β ${r.beta}: PPO gold / KL`, `${fmt(r.ppo.gold[at(r.ppo, u)], 2)} / ${fmt(r.ppo.kl[at(r.ppo, u)], 2)}`],
      [`β ${r.beta}: DPO gold / KL`, `${fmt(r.dpo.gold[at(r.dpo, u)], 2)} / ${fmt(r.dpo.kl[at(r.dpo, u)], 2)}`],
      [`β ${r.beta}: mean log π(chosen)`, fmt(r.chosen[at(r.dpo, u)], 2)],
    ] as [string, string][]);
    const status = u === 0
      ? 'PPO and DPO start from the same SFT policy and the same 200 preference pairs.'
      : `${Math.round(u * 100)}% through training: ` + s.runs.map((r) => `at β ${r.beta} PPO has gold ${fmt(r.ppo.gold[at(r.ppo, u)], 2)} and DPO ${fmt(r.dpo.gold[at(r.dpo, u)], 2)}`).join('; ') + `. The mean log-probability of the chosen responses at β ${last.beta} is ${fmt(last.chosen[di], 2)}, from ${fmt(last.chosen[0], 2)}.`;
    return { u, status, readouts };
  },
  restTime: () => LEAD + RUN + 0.5,
  draw(E: Env, f, s) {
    const nw = E.narrow, W = E.W, H = E.H, pad = nw ? 8 : 12, R = s.runs.length + 1, gap = nw ? 0 : 14, last = s.runs[s.runs.length - 1];
    const cell = (i: number) => nw
      ? { x: 46, y: 36 + i * ((H - 36) / R), w: W - 46 - 16, h: (H - 36) / R - 78 }
      : { x: pad + 40 + i * ((W - 2 * pad - 2 * gap) / R + gap), y: 54, w: (W - 2 * pad - gap * (R - 1)) / R - 48, h: H - 54 - 52 };
    s.runs.forEach((r, i) => {
      const b = cell(i), { sx, sy } = axes(E, b, { lo: 0, hi: s.xmax, label: 'Distance moved, √KL' }, GOLD_Y);
      text(E, `β = ${r.beta}`, b.x, b.y - 22, { size: 12, weight: 650 });
      for (const [c, dash, name, up] of [[r.ppo, undefined, 'PPO', true], [r.dpo, GOLD_DASH, 'DPO', false]] as const) {
        const n = at(c, f.u);
        let d = '';
        for (let k = 0; k <= n; k++) d += `${k ? 'L' : 'M'}${sx(c.x[k]).toFixed(1)} ${clamp(sy(c.gold[k]), b.y, b.y + b.h).toFixed(1)}`;
        path(E, d, { stroke: INK, sw: dash ? 2 : 2.5, dash });
        const ex = sx(c.x[n]), ey = clamp(sy(c.gold[n]), b.y, b.y + b.h);
        circle(E, ex, ey, 4, { fill: dash ? BG : INK, stroke: INK, sw: 2 });
        text(E, name, clamp(ex + (ex > b.x + b.w * 0.7 ? -8 : 8), b.x + 4, b.x + b.w - 4), ey + (up ? -9 : 15), { size: 11, weight: 650, anchor: ex > b.x + b.w * 0.7 ? 'end' : 'start', halo: true });
      }
    });
    const b = cell(R - 1), n = at(last.dpo, f.u);
    const { sx, sy } = axes(E, b, { lo: 0, hi: s.steps, ticks: [0, 100, 200, 300], label: 'DPO step' }, { lo: s.lpLo, hi: 0, label: 'Mean log-probability' });
    text(E, `DPO at β = ${last.beta}: chosen and rejected`, b.x, b.y - 22, { size: 12, weight: 650 });
    for (const [ys, col, nm] of [[last.chosen, 'var(--viz-pos)', 'chosen'], [last.rejected, 'var(--viz-neg)', 'rejected']] as const) {
      let d = '';
      for (let k = 0; k <= n; k++) d += `${k ? 'L' : 'M'}${sx(k).toFixed(1)} ${clamp(sy(ys[k]), b.y, b.y + b.h).toFixed(1)}`;
      path(E, d, { stroke: col, sw: 2.5 });
      const ey = clamp(sy(ys[n]), b.y, b.y + b.h);
      circle(E, sx(n), ey, 4, { fill: col, stroke: BG, sw: 1.5 });
      text(E, nm, clamp(sx(n) + 8, b.x + 4, b.x + b.w - 40), ey + (nm === 'chosen' ? -9 : 15), { size: 11, weight: 650, fill: INK, halo: true });
    }
   
  },
  label: (s) => `PPO and DPO compared at beta ${s.runs.map((r) => r.beta).join(' and ')}: gold reward against distance moved from the reference, and the mean log-probability of chosen and rejected responses during DPO.`,
};
export default mode;
