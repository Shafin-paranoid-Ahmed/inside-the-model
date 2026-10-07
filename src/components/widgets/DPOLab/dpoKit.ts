/** Helpers shared by the DPO lesson's PreferenceLab modes (not a mode: the file name does not start with m-). No DOM. */
import { CANDIDATES, K, P } from '../../../lib/align/bank';
import type { DPORun } from '../../../lib/align/dpo';
import { num, type Kit, type Opts } from '../_rl/kit';
import type { PrefData } from './sim';

export const name = (y: number): string => CANDIDATES[y].label.toLowerCase();
/** DPO from the SFT reference on the 200 pairs of `d`, in the shared worker (45 to 55 ms for 300 steps). */
export const dpoRun = (kit: Kit, d: PrefData, beta: number, steps: number): Promise<DPORun> => kit.job<DPORun>('trainDPO', d.pairs, d.sft, { beta, steps });
export const betaOf = (o: Opts, d = 0.1): number => num(o.beta, d);
/** Subtract each prompt's mean from a flat P x K table (scores are only meaningful within a prompt). */
export function centred(v: ArrayLike<number>): Float64Array {
  const out = new Float64Array(P * K);
  for (let x = 0; x < P; x++) {
    let m = 0;
    for (let y = 0; y < K; y++) m += v[x * K + y] / K;
    for (let y = 0; y < K; y++) out[x * K + y] = v[x * K + y] - m;
  }
  return out;
}
export function pearson(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i] / n; mb += b[i] / n; }
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  return sab / Math.sqrt(saa * sbb);
}
