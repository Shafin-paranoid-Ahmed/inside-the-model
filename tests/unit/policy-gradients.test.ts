import { describe, expect, it } from 'vitest';
import * as rl from '../../src/lib/rl';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import softmaxPg from '../../src/components/widgets/BanditLab/m-softmax-pg';
import baseline from '../../src/components/widgets/BanditLab/m-baseline';
import reinforce from '../../src/components/widgets/GridWorldLab/m-reinforce';
import update from '../../src/components/widgets/GridWorldLab/m-reinforce-update';
import actorCritic from '../../src/components/widgets/GridWorldLab/m-actor-critic';

const fns = rl as unknown as Record<string, (...a: unknown[]) => unknown>;
const kit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: <T,>(fn: string, ...args: unknown[]) => Promise.resolve(fns[fn](...args) as T),
  };
};
const ui = { pick: null };
const MODES: [string, ModeImpl<unknown, Frame>, Record<string, unknown>][] = [
  ['softmax-pg', softmaxPg as ModeImpl<unknown, Frame>, {}],
  ['baseline', baseline as ModeImpl<unknown, Frame>, {}],
  ['reinforce', reinforce as ModeImpl<unknown, Frame>, {}],
  ['reinforce-update', update as ModeImpl<unknown, Frame>, { seed: 27, focus: '3,2' }],
  ['actor-critic', actorCritic as ModeImpl<unknown, Frame>, {}],
];

describe.each(MODES)('mode %s', (name, mode, props) => {
  const o = { mode: name, ...props } as Opts;
  it('markers increase; frames narrate and draw deterministically at both widths', async () => {
    const [a, b] = [await mode.setup(o, kit()), await mode.setup(o, kit())];
    const { duration, markers } = mode.timeline(a, o);
    markers.forEach((m, i) => { expect(m.at).toBeLessThan(duration); if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at); });
    for (const t of [0, duration * 0.25, duration * 0.5, duration * 0.8, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(0);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 700 : 460, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
  });
});

describe('numbers quoted in the lesson', () => {
  it('softmax-pg: best-arm probability 0.333, 0.835, 0.964 at pulls 0, 50, 200', async () => {
    const o = { mode: 'softmax-pg', lr: 0.1 } as Opts, s = await softmaxPg.setup(o, kit());
    expect([0, 50, 200].map((n) => +s.series[2][n].toFixed(3))).toEqual([0.333, 0.835, 0.964]);
    expect(s.run.arms[7]).toBe(0);                                  // pull 8 picks the worst arm...
    expect(s.run.rewards[7]).toBeCloseTo(1.46, 2);
    expect(s.series[0][8]).toBeGreaterThan(s.series[0][7]);       // ...and still raises its probability
    expect(s.run.rewards.every((r) => r > 0)).toBe(true);
  });
  it('baseline: variances 0.727/0.232, 1.268/0.274, 1.788/0.305 (seed 9)', async () => {
    const got = [];
    for (const arm of [1, 2, 3]) {
      const s = await baseline.setup({ mode: 'baseline', arm } as Opts, kit());
      got.push([s.stats[0].vari[400], s.stats[1].vari[400], s.stats[0].vari[400] / s.stats[1].vari[400]].map((x) => +x.toFixed(3)));
    }
    expect(got.map((g) => g.slice(0, 2))).toEqual([[0.727, 0.232], [1.268, 0.274], [1.788, 0.305]]);
    expect(got.map((g) => +g[2].toFixed(1))).toEqual([3.1, 4.6, 5.9]);
  });
  it('grid REINFORCE gradient variance at the uniform policy halves with the exact baseline', () => {
    const g = rl.makeGrid({ gamma: 1 });
    const probs = new Float64Array(48).fill(0.25), V = rl.policyValue(g, probs);
    const sum = [new Float64Array(48), new Float64Array(48)], sq = [new Float64Array(48), new Float64Array(48)], n = 3000;
    for (let e = 0; e < n; e++) {
      const r = rl.gridReinforce(g, { lr: 1, episodes: 1, seed: 1000 + e, baseline: 'none', evalEvery: 0, detail: [0] }).details[0];
      for (const k of [0, 1]) {
        const grad = new Float64Array(48);
        for (const st of r.steps) for (let a = 0; a < 4; a++) grad[st.s * 4 + a] += (st.G - (k ? V[st.s] : 0)) * ((a === st.a ? 1 : 0) - 0.25);
        for (let i = 0; i < 48; i++) { sum[k][i] += grad[i]; sq[k][i] += grad[i] ** 2; }
      }
    }
    const tot = sum.map((s_, k) => s_.reduce((acc, x, i) => acc + sq[k][i] / n - (x / n) ** 2, 0));
    expect(tot[0] / tot[1]).toBeGreaterThan(1.8);
  });
  it('reinforce-update: cell (3,2) is visited once, G -1.04, b -0.49, advantage -0.55, and its chance of right falls', async () => {
    const o = { mode: 'reinforce-update', seed: 27, focus: '3,2' } as Opts, s = await update.setup(o, kit());
    expect(s.T).toBe(8);
    const st = s.d.steps[7];
    expect([st.s, st.a, +st.G.toFixed(2), +st.b.toFixed(2), +st.adv.toFixed(2)]).toEqual([6, 1, -1.04, -0.49, -0.55]);
    expect(s.after[1]).toBeLessThan(s.before[1]);
    expect([0, 3].map((j) => [s.d.steps[j].s, s.d.steps[j].a, +s.d.steps[j].G.toFixed(2), +s.d.steps[j].b.toFixed(2), +s.d.steps[j].adv.toFixed(2)])).toEqual([[0, 1, -1.32, -2.41, 1.09], [0, 1, -1.2, -2.41, 1.21]]);
    expect([+s.before[1].toFixed(2), +s.after[1].toFixed(2), +s.push[1].toFixed(3)]).toEqual([0.21, 0.2, -0.044]);
  });
  it('reinforce / actor-critic curves match the prose (episodes 0, 100, 500, 1000; actor-critic episode 7)', async () => {
    const s = await reinforce.setup({ mode: 'reinforce' } as Opts, kit()), f = (k: number) => [s.none!.mean[k], s.sv!.mean[k]].map((x) => +x.toFixed(2));
    expect([0, 10, 50, 100].map(f)).toEqual([[-1.59, -1.59], [0.26, 0.29], [0.59, 0.6], [0.65, 0.66]]);
    const a = await actorCritic.setup({ mode: 'actor-critic' } as Opts, kit()), last = a.steps[a.steps.length - 1];
    expect(+a.ac.mean[10].toFixed(2)).toBe(0.38);
    expect([a.steps.length, +last.r.toFixed(2), +last.v.toFixed(2), +last.delta.toFixed(2), +last.probsBefore[last.a].toFixed(3), +last.probsAfter[last.a].toFixed(3)]).toEqual([11, 0.96, 0.15, 0.81, 0.241, 0.256]);
  });
});
