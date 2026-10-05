import { describe, expect, it } from 'vitest';
import * as rl from '../../src/lib/rl';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import advantage from '../../src/components/widgets/GridWorldLab/m-ppo-advantage';
import ratio from '../../src/components/widgets/GridWorldLab/m-ppo-ratio';
import epochs from '../../src/components/widgets/GridWorldLab/m-ppo-epochs';
import train from '../../src/components/widgets/GridWorldLab/m-ppo-train';
import clipPos from '../../src/components/widgets/PPOLab/m-clip-positive';
import clipNeg from '../../src/components/widgets/PPOLab/m-clip-negative';
import drift from '../../src/components/widgets/PPOLab/m-drift';

// Node stand-in for the page kit: memo works and "worker" jobs call src/lib/rl directly.
const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: async <T,>(fn: string, ...args: unknown[]) => (rl as unknown as Record<string, (...a: unknown[]) => T>)[fn](...args),
  };
};
const ui = { pick: null };
const MODES: [string, ModeImpl<unknown, Frame>][] = [
  ['ppo-advantage', advantage], ['ppo-ratio', ratio], ['ppo-epochs', epochs], ['ppo-train', train],
  ['clip-positive', clipPos], ['clip-negative', clipNeg], ['drift', drift],
] as never;

describe.each(MODES)('%s', (name, mode) => {
  const o = { mode: name } as Opts;
  it('markers are increasing; frames at start, middle and end are deterministic and draw legibly', async () => {
    const [a, b] = [await mode.setup(o, nodeKit()), await mode.setup(o, nodeKit())];
    const { duration, markers } = mode.timeline(a, o);
    markers.forEach((mk, i) => { expect(mk.at).toBeLessThanOrEqual(duration); if (i) expect(mk.at).toBeGreaterThan(markers[i - 1].at); });
    for (const t of [0, duration / 3, duration / 2, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(0);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 640 : 420, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0);
        expect(E.out.length).toBeLessThan(260);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
    expect(mode.label(a, o).length).toBeGreaterThan(20);
  });
});

const end = async (mode: ModeImpl<unknown, Frame>, o: Opts, pick: string | null = null) => {
  const s = await mode.setup(o, nodeKit());
  return { s, f: mode.frameAt(mode.timeline(s, o).duration, s, o, { pick }) };
};
const val = (f: Frame, k: string) => f.readouts?.find(([l]) => l === k)?.[1];

describe('numbers quoted in the lesson', () => {
  it('advantage: step 1 error 0.043, last step adds 0.400, A 0.493 over 6 steps', async () => {
    const { s } = await end(advantage as never, { mode: 'ppo-advantage' } as Opts);
    const sc = s as { T: number; d: number[]; w: number[]; A: number[] };
    expect([sc.T, +sc.d[0].toFixed(3), +sc.w[5].toFixed(3), +sc.A[0].toFixed(3), +sc.d[5].toFixed(3)]).toEqual([6, 0.043, 0.4, 0.493, 0.875]);
  });
  it('ratio: up 0.325 -> 0.355 (1.092); chip 10 falls to 0.746; 17 visits', async () => {
    const o = { mode: 'ppo-ratio' } as Opts;
    const a = await end(ratio as never, o), b = await end(ratio as never, o, 's9');
    expect(val(a.f, 'Ratio')).toBe('1.092');
    expect(val(b.f, 'Ratio')).toBe('0.746');
    expect((a.s as { idx: number[] }).idx.length).toBe(17);
  });
  it('epochs: 31% outside, 27% flat, 3 pushing; objective 0.124 -> 0.145', async () => {
    const { f } = await end(epochs as never, { mode: 'ppo-epochs' } as Opts);
    expect([val(f, 'Outside band'), val(f, 'Flat'), val(f, 'Outside but pushing'), val(f, 'Mean objective')]).toEqual(['31%', '27%', '3', '0.1446']);
  });
  it('drift: ratio ranges and KL per epsilon', async () => {
    const { f } = await end(drift as never, { mode: 'drift' } as Opts);
    expect(f.readouts?.map(([, v]) => v)).toEqual(['ratios 0.73 to 1.32, KL 0.0114', 'ratios 0.72 to 1.44, KL 0.0176', 'ratios 0.71 to 1.58, KL 0.0237']);
    const s = (await end(drift as never, { mode: 'drift' } as Opts)).s as { runs: { oob: number[] }[]; ups: number };
    expect(s.runs.map((r) => Math.round(r.oob[s.ups] * 100))).toEqual([59, 31, 14]);
  });
  it('train: start-cell value -0.410, 0.264, 0.330, 0.341; critic error 0.443, 0.227, 0.157', async () => {
    const o = { mode: 'ppo-train' } as Opts, s = (await end(train as never, o)).s as { res: { evalReturn: Float64Array; criticErr: Float64Array } };
    expect([0, 20, 50, 100].map((k) => +s.res.evalReturn[k].toFixed(3))).toEqual([-0.41, 0.264, 0.33, 0.341]);
    expect([0, 20, 99].map((k) => +s.res.criticErr[k].toFixed(3))).toEqual([0.443, 0.227, 0.157]);
  });
  it('clipping: A 1 gives 1.2 at 1.2 and 1.5; A -1 gives -0.8 at 0.6 and -1.4 at 1.4 with slope -1', () => {
    expect([1.2, 1.5].map((r) => rl.clippedSurrogate(r, 1, 0.2).value)).toEqual([1.2, 1.2]);
    expect(rl.clippedSurrogate(0.6, -1, 0.2)).toMatchObject({ value: -0.8, dRho: 0 });
    expect(rl.clippedSurrogate(1.4, -1, 0.2)).toMatchObject({ value: -1.4, dRho: -1 });
  });
});
