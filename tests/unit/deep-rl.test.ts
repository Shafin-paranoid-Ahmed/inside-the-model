import { describe, expect, it } from 'vitest';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import network from '../../src/components/widgets/GridWorldLab/m-dqn-network';
import update from '../../src/components/widgets/GridWorldLab/m-dqn-update';
import instability from '../../src/components/widgets/GridWorldLab/m-dqn-instability';
import replay from '../../src/components/widgets/GridWorldLab/m-dqn-replay';
import target from '../../src/components/widgets/GridWorldLab/m-dqn-target';
import compare from '../../src/components/widgets/GridWorldLab/m-dqn-compare';
import * as rl from '../../src/lib/rl';

// Node kit: jobs run the src/lib/rl function directly (the page runs them in a worker).
const kit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: <T,>(fn: string, ...a: unknown[]) => Promise.resolve((rl as unknown as Record<string, (...x: unknown[]) => T>)[fn](...a)),
  };
};
const ui = { pick: null };
const MODES: [string, ModeImpl<unknown, Frame>][] = [['dqn-network', network], ['dqn-update', update], ['dqn-instability', instability], ['dqn-replay', replay], ['dqn-target', target]] as never;

describe.each(MODES)('%s', (name, mode) => {
  const o = { mode: name } as Opts;
  it('frames at start, middle and end are deterministic, drawn and narrated', async () => {
    const [a, b] = [await mode.setup(o, kit()), await mode.setup(o, kit())];
    const { duration, markers } = mode.timeline(a, o);
    markers.forEach((m, i) => { expect(m.at).toBeLessThanOrEqual(duration); if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at); });
    for (const t of [0, duration / 2, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(10);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 640 : 400, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(5);
        expect(E.out.length).toBeLessThan(330);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
  });
});

describe('numbers quoted in the lesson', () => {
  it('network: 116 weights and the (1,1) outputs', async () => {
    const s = await network.setup({ mode: 'dqn-network' } as Opts, kit()), f = network.frameAt(0, s, { mode: 'x' } as Opts, ui);
    expect(f.readouts?.find(([k]) => k === 'Weights')?.[1]).toBe('116');
    expect(f.readouts?.find(([k]) => k.startsWith('Q'))?.[1]).toBe('−0.47  0.16  0.79  −0.92');
  });
  it('target: y is 0.32 after update 49 and 0.13 after update 50, distance 0 at the copy', async () => {
    const o = { mode: 'dqn-target' } as Opts, s = (await target.setup(o, kit())) as { r: { diagY: Float64Array; weightDist: Float64Array } };
    expect(s.r.diagY[48]).toBeCloseTo(0.3209, 3); expect(s.r.diagY[49]).toBeCloseTo(0.1312, 3);
    expect(s.r.weightDist[49]).toBe(0); expect(s.r.weightDist[48]).toBeCloseTo(1.488, 2); expect(s.r.weightDist[50]).toBeCloseTo(0.024, 3);
  });
  it('instability: Adam 0.1 runs away, the fixed fit does not', async () => {
    const o = { mode: 'dqn-instability', lr: 0.1 } as Opts, s = (await instability.setup(o, kit())) as { moving: { maxAbsQ: Float64Array }; fixed: { maxAbsQ: Float64Array } };
    expect(Math.max(...s.moving.maxAbsQ)).toBeGreaterThan(200); expect(Math.max(...s.fixed.maxAbsQ)).toBeLessThan(6);
  });
  it('replay: sampled batches touch more cells than the newest 16 and break the step-to-step correlation', async () => {
    const s = (await replay.setup({ mode: 'dqn-replay' } as Opts, kit())) as { beats: { cells: number[]; seqCells: number; ages: number[] }[]; corr: number[] };
    expect(s.beats.map((b) => b.cells.filter((x) => x > 0).length)).toEqual([6, 6, 7]);
    expect(s.beats.map((b) => b.seqCells)).toEqual([4, 4, 4]);
    expect(s.corr[0]).toBeCloseTo(0.71, 2); expect(s.corr[1]).toBeCloseTo(0.035, 2);
  });
  it('compare: lr 0.001 ordering (slow, 4 variants x 10 runs x 500 episodes)', async () => {
    const o = { mode: "dqn-compare" } as Opts, s = (await compare.setup(o, kit())) as unknown as { stats: { return: Record<string, { mean: Float64Array; sd: Float64Array }> } };
    const last = (v: string) => [s.stats.return[v].mean.at(-1)!, s.stats.return[v].sd.at(-1)!];
    expect(last('naive')[0]).toBeCloseTo(0.26, 2); expect(last('both')[0]).toBeCloseTo(0.331, 2); expect(last('both')[1]).toBeLessThan(0.03);
    const f = compare.frameAt(compare.timeline(s as never, o).duration, s as never, o, ui);
    expect(f.status).toContain('replay + target 0.33');
  }, 120000);
});

it('target: the diagnostic move is in 5 of the 49 batches between updates 50 and 99, and the other batches carry most of its rise', async () => {
  const { makeGrid, idx } = rl, g = makeGrid({ gamma: 0.9 }), diag = { s: idx(3, 1), a: 0, r: -0.04, s2: idx(3, 2), done: false };
  const r = rl.trainDQN(g, { optimizer: 'adam', lr: 0.01, gamma: 0.9, hidden: 16, replay: true, target: true, capacity: 256, batch: 16, copyEvery: 50, seed: 19, diag, updates: 300, evalEvery: 100000, keepSteps: 100000 });
  const buf = new rl.ReplayBuffer(256), inB: boolean[] = [];
  for (const st of r.steps) { buf.push(st.t, 0); if (st.batch.length) inB.push(Array.from(st.batch).some((i) => buf.s[i] === diag.s && buf.a[i] === diag.a)); }
  let n = 0, out = 0;
  for (let k = 50; k < 99; k++) { if (inB[k]) n++; else out += r.diagQ[k] - r.diagQ[k - 1]; }
  expect(n).toBe(5); expect(out).toBeCloseTo(0.30, 2); expect(r.diagQ[98] - r.diagQ[49]).toBeCloseTo(0.36, 2);
});
