/**
 * Helpers shared by the rlhf modes (ppo, kl-sweep, overopt, reward-model, rm-bias, ppo-clip). Pure, no DOM.
 * The reward axis is FIXED for every PPO chart (proxy and gold from -1 to 5) so a beta 0.1 run, a beta 0 run and a
 * constitutional-ai run read on the same scale; the KL axis fits the run it shows.
 */
import { TEXT, CANDIDATES, K } from '../../../lib/align/bank';
import { probsInto } from '../../../lib/align/policy';
import type { AlignPPOResult } from '../../../lib/align/ppo';
import type { Kit, Opts } from '../_rl/kit';
import type { Axis } from '../_rl/svg';
import { ppoRun } from './sim';

export const REWARD_Y: Axis = { lo: -1, hi: 5, ticks: [-1, 0, 1, 2, 3, 4, 5], label: 'Reward (average over prompts)' };
export const iterAxis = (iters: number, label?: string): Axis => ({ lo: 0, hi: iters, ticks: Array.from({ length: 5 }, (_, i) => Math.round((iters * i) / 4)), label });
/** Smallest 1-2-5 number at or above x (an axis top that fits the data). */
export function niceCeil(x: number): number {
  const m = 10 ** Math.floor(Math.log10(x || 1));
  return ([1, 2, 5, 10].map((k) => k * m).find((v) => v >= x * 0.999) ?? x);
}
export const peakIndex = (a: ArrayLike<number>): number => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };
/** Spearman rank correlation (ties get average ranks). */
export function spearman(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const rank = (v: ArrayLike<number>) => {
    const idx = Array.from({ length: v.length }, (_, i) => i).sort((i, j) => v[i] - v[j]), r = new Float64Array(v.length);
    for (let i = 0; i < idx.length;) { let j = i; while (j + 1 < idx.length && v[idx[j + 1]] === v[idx[i]]) j++; for (let k = i; k <= j; k++) r[idx[k]] = (i + j) / 2; i = j + 1; }
    return r;
  };
  const ra = rank(a), rb = rank(b), n = ra.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += ra[i]; mb += rb[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (ra[i] - ma) * (rb[i] - mb); saa += (ra[i] - ma) ** 2; sbb += (rb[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
}
/** pi_theta(. | prompt) as 8 numbers. */
export const rowOf = (theta: ArrayLike<number>, x: number): Float64Array => probsInto(theta, x, new Float64Array(K));
export const argmax = (p: ArrayLike<number>): number => { let k = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[k]) k = i; return k; };
export const candName = (y: number): string => CANDIDATES[y].label.toLowerCase();
/** A response in quotes, cut to n characters. */
export const quoted = (x: number, y: number, n = 40): string => { const s = TEXT[x][y]; return `“${s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s}”`; };

/** PPO runs for several betas at the brief's other settings (the jobs are memoised, so modes of a page share them). */
export const runsFor = (o: Opts, kit: Kit, betas: number[], iters: number): Promise<AlignPPOResult[]> => Promise.all(betas.map((beta) => ppoRun(o, kit, { beta, iters })));
/** Timeline markers naming four segments of `iters` updates that start at `lead` seconds, `pace` seconds per update. */
export function quarterMarkers(lead: number, pace: number, iters: number): { at: number; label: string }[] {
  return [0, 1, 2, 3].map((i) => ({ at: lead + Math.round((iters * i) / 4) * pace + (i ? 0 : 1e-4), label: `Updates ${Math.round((iters * i) / 4) + 1}–${Math.round((iters * (i + 1)) / 4)}` }));
}
