/** Shared by the policy-gradients modes (reinforce, actor-critic): the 10-run learning-curve jobs, memoised per page by the worker kit. */
import type { Grid } from '../../../lib/rl/gridworld';
import { int, num, type Kit, type Opts } from '../_rl/kit';

export type PGKind = 'none' | 'state-value' | 'actor-critic';
export const EVAL_EVERY = 10;
export interface PGRuns {
  /** Episodes completed at each evaluation point (0, 10, ...). */
  at: Int32Array;
  /** Exact expected return from the start cell, per run per evaluation point, and their average over the runs. */
  runs: Float64Array[]; mean: Float64Array;
  /** Logits of run 1 every EVAL_EVERY episodes (N x 4 each). */
  snaps: Float64Array[];
}
interface Raw { evalAt: Int32Array; evalValue: Float64Array; snapshots: Float64Array[] }

/** `runs` independent trainings (seeds seed .. seed + runs - 1) of one algorithm; the result is shared by every figure that asks for the same arguments. */
export async function pgRuns(kit: Kit, g: Grid, kind: PGKind, o: Opts): Promise<PGRuns> {
  const runs = int(o.runs, 10), seed = int(o.seed, 34), episodes = int(o.episodes, 1000);
  const base = { episodes, evalEvery: EVAL_EVERY };
  const all = await Promise.all(Array.from({ length: runs }, (_, r) => {
    const snapshotEvery = r === 0 ? EVAL_EVERY : 0;
    return kind === 'actor-critic'
      ? kit.job<Raw>('actorCritic', g, { ...base, lrActor: num(o.lrActor, 0.1), lrCritic: num(o.lrCritic, 0.2), seed: seed + r, snapshotEvery })
      : kit.job<Raw>('gridReinforce', g, { ...base, lr: num(o.lr, 0.1), baseline: kind, lrCritic: 0.1, seed: seed + r, snapshotEvery });
  }));
  const at = all[0].evalAt, mean = new Float64Array(at.length);
  for (const r of all) for (let i = 0; i < at.length; i++) mean[i] += r.evalValue[i] / runs;
  return { at, runs: all.map((r) => r.evalValue), mean, snaps: all[0].snapshots };
}
/** First evaluation episode at which the average reaches `thr` (or -1). */
export const firstReach = (r: PGRuns, thr: number): number => { for (let i = 0; i < r.at.length; i++) if (r.mean[i] >= thr) return r.at[i]; return -1; };
/** Episodes completed at time t when the playhead sweeps 0..episodes over `play` seconds, slow at first (quadratic), snapped to the evaluation grid. */
export const episodesAt = (t: number, play: number, episodes: number): number =>
  Math.floor((episodes * Math.min(1, Math.max(0, t / play)) ** 2) / EVAL_EVERY + 1e-9) * EVAL_EVERY;
/** The time at which `episodesAt` first reaches `e`. */
export const timeAt = (e: number, play: number, episodes: number): number => play * Math.sqrt(e / episodes);
