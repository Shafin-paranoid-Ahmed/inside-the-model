/**
 * reinforce (policy-gradients step 3): REINFORCE on the 4x3 grid, 10 runs without and 10 with a learned state-value
 * baseline. Left: the policy (arrow opacity = probability) of run 1 with the baseline sharpening as episodes pass.
 * Right: the exact expected return of the current policy from the start cell, averaged over the runs (faint lines: single runs).
 * Props: lr (0.1), episodes (1000), runs (10), seed (34, runs use seeds 34..43), baseline ("both" | "none" | "state-value").
 */
import { START } from '../../../lib/rl/gridworld';
import { optimal, softmaxPolicy } from '../../../lib/rl/tabular';
import { int, str, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { MUTED, fmt, lineChart, text, type Env } from '../_rl/svg';
import { drawGrid, drawPolicy, gridIn } from './draw';
import { EVAL_EVERY, episodesAt, pgRuns, timeAt, type PGRuns } from './pg-runs';
import { gridOf } from './sim';

const PLAY = 14, HOLD = 2.5;
interface Scene { none: PGRuns | null; sv: PGRuns | null; vstar: number; episodes: number; nRuns: number }
interface F extends Frame { e: number; k: number }

const mode: ModeImpl<Scene, F> = {
  async setup(o: Opts, kit) {
    const g = gridOf(o), which = str(o.baseline, ['both', 'none', 'state-value'] as const, 'both');
    const [none, sv] = await Promise.all([which !== 'state-value' ? pgRuns(kit, g, 'none', o) : null, which !== 'none' ? pgRuns(kit, g, 'state-value', o) : null]);
    return { none, sv, vstar: optimal(g).V[START], episodes: int(o.episodes, 1000), nRuns: int(o.runs, 10) };
  },
  timeline(s) {
    const at = (e: number) => timeAt(Math.min(e, s.episodes), PLAY, s.episodes);
    return { duration: PLAY + HOLD, markers: [{ at: 0, label: 'First 100 episodes' }, { at: at(100), label: '100 to 500' }, { at: at(500), label: `500 to ${s.episodes}` }, { at: PLAY, label: 'Done' }] };
  },
  frameAt(t, s) {
    const e = episodesAt(t, PLAY, s.episodes), k = e / EVAL_EVERY, a = s.none?.mean[k], b = s.sv?.mean[k];
    const parts = [a !== undefined && `${fmt(a, 2)} without a baseline`, b !== undefined && `${fmt(b, 2)} with one`].filter(Boolean).join(' and ');
    const status = e === 0
      ? `Before any training every action is equally likely, so the policy's expected return from the start cell is ${fmt((a ?? b)!, 2)}.`
      : `After ${e} episodes the policy's expected return from the start cell, averaged over ${s.nRuns} runs, is ${parts}. The best possible is ${fmt(s.vstar, 2)}.`;
    const ro: [string, string][] = [['Episodes', `${e} of ${s.episodes}`]];
    if (a !== undefined) ro.push(['No baseline', fmt(a, 3)]);
    if (b !== undefined) ro.push(['With baseline', fmt(b, 3)]);
    return { e, k, status, detail: `Expected return is computed exactly for the current policy, then averaged over ${s.nRuns} runs. Arrows: run 1.`, readouts: ro };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, show = s.sv ?? s.none!;
    const gb = nar ? { x: 8, y: 22, w: E.W - 16, h: E.H * 0.34 } : { x: 8, y: 24, w: E.W * 0.38, h: E.H - 90 };
    text(E, `Run 1 policy after ${f.e} episodes`, gb.x + 4, 14, { fill: MUTED });
    const G = gridIn(gb), th = show.snaps[Math.min(show.snaps.length - 1, f.k)];
    drawGrid(E, G);
    drawPolicy(E, G, { probs: softmaxPolicy(th) });
    const cb = nar ? { x: 46, y: gb.y + gb.h + 40, w: E.W - 62, h: E.H - gb.y - gb.h - 100 } : { x: E.W * 0.38 + 62, y: 30, w: E.W * 0.62 - 80, h: E.H - 96 };
    const series = [] as { ys: ArrayLike<number>; xs: ArrayLike<number>; color: string; width?: number; op?: number; dash?: string }[];
    const add = (r: PGRuns | null, color: string) => { if (!r) return; for (const y of r.runs) series.push({ ys: y, xs: r.at, color, width: 1, op: 0.22 }); series.push({ ys: r.mean, xs: r.at, color, width: 2.75 }); };
    add(s.none, 'var(--viz-cat-1)'); add(s.sv, 'var(--viz-cat-2)');
    series.push({ ys: [s.vstar, s.vstar], xs: [0, s.episodes], color: 'var(--text-muted)', width: 1.25, dash: '5 4' });
    lineChart(E, cb, { lo: 0, hi: s.episodes, ticks: [0, 250, 500, 750, 1000].filter((v) => v <= s.episodes), label: 'Episodes of training' }, { lo: -1.75, hi: 0.85, ticks: [-1.5, -1, -0.5, 0, 0.5], label: 'Expected return of the current policy' }, series, f.e);
  },
  label: (s) => `Left: the REINFORCE policy of one run on the 4 by 3 grid sharpening over ${s.episodes} episodes. Right: expected return of the current policy, averaged over ${s.nRuns} runs, without and with a learned baseline; the two curves are close.`,
};
export default mode;
