/**
 * Helpers shared by the rlvr modes (not a mode: the shell only loads m-*.ts). Style bars for the ten response styles,
 * the training-run job, the iteration clock and the sample-response card.
 */
import { createRng } from '../../../lib/math/rng';
import { NACT, isList, kOf, respond, sampleGroups, startLogits, truth, verify, type GRPORun, type Groups, type Problem, type VerifierKind } from '../../../lib/align/verifier';
import { int, type Kit, type Opts } from '../_rl/kit';
import { MUTED, RULE, clamp, fmt, line, rect, text, type Box, type Env } from '../_rl/svg';
import { charsFor, clip, responseCard } from '../_align/draw';
import { cross, tick } from './sim';

export const sum = (a: ArrayLike<number>): number => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
/** The attempts of the "sample" and "grpo" steps: G attempts at each of `problems` problems from the start policy. */
export function attempts(o: Opts): Groups {
  return sampleGroups(startLogits(), { problems: int(o.problems, 6), G: int(o.G, 8), rng: createRng(int(o.seed, 0)), verifier: 'strict' });
}

/** Rows of the style table for a d-digit problem: writing all d columns is one style (k above d writes the same lines). */
export const styleRows = (d: number) => [0, 1].flatMap((f) => Array.from({ length: d + 1 }, (_, k) => ({ f, k })));
export const nameOf = (r: { f: number; k: number }, d: number) => (r.k === 0 ? 'no working' : r.k === d && d > 1 ? `all ${d} columns` : `${r.k} column${r.k === 1 ? '' : 's'}`) + (r.f ? ', list' : '');
/** Probability of each style row for a d-digit problem, from the 10 action values (summing k above d into k = d). */
export function styleValues(v: ArrayLike<number>, d: number): number[] {
  return styleRows(d).map(({ f, k }) => { let s = 0; for (let a = 0; a < NACT; a++) if ((isList(a) ? 1 : 0) === f && Math.min(kOf(a), d) === k) s += v[a]; return s; });
}
export interface StyleBarsOpts {
  /** Start policy drawn as dashed ghost bars. */
  ref?: ArrayLike<number>;
  /** Signed push per style (e.g. total advantage): an arrow column to the right of the bars. */
  push?: number[]; pushMax?: number; pushGrow?: number;
  title: string; faded?: boolean;
  /** Real state whose numbers are printed while `p` tweens. */
  values?: ArrayLike<number>;
}
/** One row per response style: name, probability bar (with optional ghost) and value, and an optional push column. */
export function styleBars(E: Env, b: Box, p: ArrayLike<number>, d: number, o: StyleBarsOpts) {
  const rows = styleRows(d), sz = E.narrow ? 11 : 12, top = 18, rh = (b.h - top) / rows.length, pv = styleValues(p, d), rv = o.ref ? styleValues(o.ref, d) : null, lv = o.values ? styleValues(o.values, d) : pv;
  const pushW = o.push ? (E.narrow ? 70 : 96) : 0, labW = E.narrow ? 126 : 142, valW = 32, bx = b.x + labW, bw = Math.max(30, b.w - labW - valW - pushW - 6), bh = Math.min(14, rh * 0.6), op = o.faded ? 0.5 : 1;
  text(E, o.title, b.x, b.y + 11, { size: sz, fill: MUTED });
  if (o.push) text(E, 'Update', b.x + b.w - pushW / 2, b.y + 11, { size: 11, anchor: 'middle', fill: MUTED });
  rows.forEach((r, i) => {
    const cy = b.y + top + i * rh + rh / 2, x1 = bx + clamp(pv[i], 0, 1) * bw;
    text(E, clip(nameOf(r, d), charsFor(labW - 6, sz)), b.x, cy + 4, { size: sz, op });
    rect(E, bx, cy - bh / 2, bw, bh, { fill: 'none', stroke: RULE, sw: 1 });
    rect(E, bx, cy - bh / 2, x1 - bx, bh, { fill: 'var(--viz-forward)', op: 0.9 * op });
    if (rv) rect(E, bx, cy - bh / 2 - 2, clamp(rv[i], 0, 1) * bw, bh + 4, { fill: 'none', stroke: MUTED, sw: 1.25, dash: '3 2' });
    text(E, fmt(lv[i], 2), bx + bw + 4, cy + 4, { size: sz, op });
    if (o.push) {
      const c = b.x + b.w - pushW / 2, half = pushW / 2 - 14, v = o.push[i], grow = o.pushGrow ?? 1, len = clamp(Math.abs(v) / (o.pushMax || 1), 0, 1) * half * grow;
      line(E, c, cy - rh / 2 + 2, c, cy + rh / 2 - 2, { stroke: RULE, sw: 1 });
      if (Math.abs(v) > 1e-9) {
        const col = v > 0 ? 'var(--viz-pos)' : 'var(--viz-neg)';
        const stacked = E.narrow || b.w < 360;
        rect(E, v > 0 ? c : c - len, cy - bh / 2 - (stacked ? 5 : 0), len, bh, { fill: col, op: 0.9 });
        if (grow > 0.98) text(E, (v > 0 ? '+' : '−') + fmt(Math.abs(v), 1), stacked ? c : v > 0 ? c + len + 3 : c - len - 3, cy + (stacked ? 13 : 4), { size: 11, anchor: stacked ? 'middle' : v > 0 ? 'start' : 'end' });
      }
    }
  });
}

/** Training-run time: the whole run plays in RUN_SECONDS; iteration is real-valued, `k` its whole part. */
export const RUN_SECONDS = 14, HOLD = 2.5;
export const iterAt = (t: number, iters: number) => clamp((t / RUN_SECONDS) * iters, 0, iters);
export const runMarkers = (iters: number) => [0, 1, 2].map((i) => ({ at: (i * RUN_SECONDS) / 3, label: `${Math.round((i * iters) / 3)} to ${Math.round(((i + 1) * iters) / 3)}` }));
export const runTimeline = (iters: number) => ({ duration: RUN_SECONDS + HOLD, markers: runMarkers(iters) });

export interface RunProps { verifier: VerifierKind; iters: number; G: number; problems: number; seed: number; domains?: boolean }
export const runProps = (o: Opts, verifier: VerifierKind, domains = false): RunProps => ({ verifier, iters: int(o.iters, 150), G: int(o.G, 8), problems: int(o.problems, 8), seed: int(o.seed, 5), ...(domains ? { domains: true } : {}) });
/** GRPO training in the worker (44 to 90 ms); memoised per page by name + args. */
export const trainRun = (p: RunProps, kit: Kit) => kit.job<GRPORun>('trainGRPO', p);

/** The first attempt at a problem with `digits` digits in the groups sampled at iteration it (rounded down to a multiple of 10). */
export function liveSample(run: GRPORun, it: number, digits: number, kind: VerifierKind) {
  const k = Math.min(run.iters - 1, Math.floor(it / 10) * 10), row = run.groups[k].rows.find((r) => r.row === digits - 1)!, p = row.p as Problem, a = row.actions[0], t = respond(p, a);
  return { k, p, text: t, pass: verify(t, truth(p), kind), strictPass: verify(t, truth(p), 'strict'), lenientPass: verify(t, truth(p), 'lenient') };
}
/** Draw a sample response card with its sum and verdict tick or cross. */
export function sampleCard(E: Env, b: Box, s: { k: number; p: Problem; text: string; pass: boolean }, title: string) {
  const h = responseCard(E, b, { title: `${title} · ${s.p.a} + ${s.p.b}, iteration ${s.k}`, text: s.text, mono: true, size: E.narrow ? 11 : 12 });
  (s.pass ? tick : cross)(E, b.x + b.w - 16, b.y + 14, 7);
  return h;
}
