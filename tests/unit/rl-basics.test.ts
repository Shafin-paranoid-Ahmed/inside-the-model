import { describe, expect, it } from 'vitest';
import * as rl from '../../src/lib/rl';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import loop from '../../src/components/widgets/GridWorldLab/m-loop';
import mdp from '../../src/components/widgets/GridWorldLab/m-mdp';
import ret from '../../src/components/widgets/GridWorldLab/m-return';
import discount from '../../src/components/widgets/GridWorldLab/m-discount';
import value from '../../src/components/widgets/GridWorldLab/m-value';
import optimalPolicy from '../../src/components/widgets/GridWorldLab/m-optimal-policy';
import explore from '../../src/components/widgets/BanditLab/m-explore';

// Node stand-in for the page kit: memo works and jobs run in-process.
const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: async <T,>(fn: string, ...args: unknown[]) => (rl as unknown as Record<string, (...a: unknown[]) => T>)[fn](...args),
  };
};
const ui = { pick: null };
const MODES: [string, ModeImpl<unknown, Frame>, Partial<Opts>][] = [
  ['loop', loop, {}], ['mdp', mdp, {}], ['return', ret, {}], ['discount', discount, { gammas: '0.5,0.99' }], ['value', value, {}],
  ['optimal-policy', optimalPolicy, {}], ['explore', explore, {}],
].map(([n, m, o]) => [n, m, o] as [string, ModeImpl<unknown, Frame>, Partial<Opts>]);

describe.each(MODES)('rl-basics mode %s', (name, mode, extra) => {
  const o = { mode: name, ...extra } as Opts;
  it('has increasing markers, deterministic frames and draws at both widths', async () => {
    const [a, b] = [await mode.setup(o, nodeKit()), await mode.setup(o, nodeKit())];
    const { duration, markers } = mode.timeline(a, o);
    markers.forEach((m, i) => { expect(m.at).toBeLessThanOrEqual(duration); if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at); });
    for (const t of [0, duration * 0.3, duration * 0.6, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(10);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 600 : 400, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0);
        expect(E.out.length).toBeLessThan(260);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
    expect(mode.label(a, o).length).toBeGreaterThan(20);
  }, 20000);
});

const end = async (mode: ModeImpl<unknown, Frame>, o: Opts) => {
  const s = await mode.setup(o, nodeKit());
  return { s, f: mode.frameAt(mode.timeline(s, o).duration, s, o, ui) };
};

describe('numbers quoted in the lesson', () => {
  it('loop: seed 34 is 15 moves, one slip on move 1, goal, total +0.40', async () => {
    const o = { mode: 'loop' } as Opts, { f } = await end(loop as ModeImpl<unknown, Frame>, o);
    expect(f.status).toContain('15 moves');
    expect(f.status).toContain('goal');
    expect(f.readouts).toEqual([['Move', '15 of 15'], ['Slips so far', '1'], ['Total reward', '+0.40']]);
  });

  it('mdp: every action tallies 20 tries and the exact chances are 0.8, 0.1, 0.1', async () => {
    const o = { mode: 'mdp' } as Opts, s = await mdp.setup(o, nodeKit());
    for (let a = 0; a < 4; a++) {
      const f = mdp.frameAt(a * 5.4 + 5.3, s, o, ui);
      const total = [...f.tally.values()].reduce((x, y) => x + y, 0);
      expect(total).toBe(20);
      expect(s.groups[a].reduce((x, g) => x + g.p, 0)).toBeCloseTo(1, 12);
    }
    expect(s.groups[0].map((g) => +g.p.toFixed(1))).toEqual([0.8, 0.1, 0.1]);
    expect(s.groups[2].map((g) => +g.p.toFixed(1))).toEqual([0.8, 0.1, 0.1]); // down at (3,1) bumps the edge: 0.8 stay
    expect(s.groups[2][0].s2).toBe(s.focus);
  });

  it('return: seed 3 has G0 0.492 and plain sum 0.800, and G follows the recursion', async () => {
    const o = { mode: 'return' } as Opts, { s, f } = await end(ret as ModeImpl<unknown, Frame>, o);
    const sc = s as { T: number; G: Float64Array; ep: { rewards: number[] }; gamma: number };
    expect(sc.T).toBe(5);
    expect(sc.G[0]).toBeCloseTo(0.492, 3);
    expect(f.readouts).toEqual([['Return from start', '0.492'], ['Plain sum', '0.80']]);
    for (let i = 0; i < sc.T - 1; i++) expect(sc.G[i]).toBeCloseTo(sc.ep.rewards[i] + sc.gamma * sc.G[i + 1], 12);
  });

  it('discount: the policy changes at gamma 0.660 and 0.938; start values -0.045 and 0.660', async () => {
    const o = { mode: 'discount', gammas: '0.5,0.99' } as Opts, s = await discount.setup(o, nodeKit());
    expect(s.exact.map((x) => +x.toFixed(3))).toEqual([0.66, 0.938]);
    expect(s.changes.map((i) => s.gs[i])).toEqual([0.67, 0.94]);
    expect(s.v0[0]).toBeCloseTo(-0.045, 3);
    expect(s.v0[s.n]).toBeCloseTo(0.66, 3);
  }, 20000);

  it('value: V(start) by sweep matches the plan, settles within 0.005 at sweep 13 of 0.705', async () => {
    const o = { mode: 'value' } as Opts, s = await value.setup(o, nodeKit());
    expect(Array.from({ length: 12 }, (_, k) => +s.V[k + 1][0].toFixed(3))).toEqual([-0.04, -0.08, -0.12, -0.16, 0.128, 0.326, 0.482, 0.579, 0.636, 0.669, 0.686, 0.696]);
    expect(s.final).toBeCloseTo(0.705, 3);
    expect(s.n1).toBe(4);
    expect(Math.abs(s.V[12][0] - s.final)).toBeGreaterThan(0.005);
    expect(Math.abs(s.V[13][0] - s.final)).toBeLessThan(0.005);
  });

  it('optimal-policy: nine changes of the best moves, from -1.65 to +0.0101', async () => {
    const o = { mode: 'optimal-policy' } as Opts, s = await optimalPolicy.setup(o, nodeKit());
    expect(s.cuts.map((c) => +c.toFixed(3))).toEqual([-1.65, -1.564, -0.731, -0.453, -0.085, -0.045, -0.027, -0.022, 0.01]);
    expect(s.pit.map((p) => p.length)).toEqual([2, 1, 1, 1, 0, 0, 0, 0, 0, 0]);
    expect(s.goal.map((p) => p.length)).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
  }, 20000);
});
