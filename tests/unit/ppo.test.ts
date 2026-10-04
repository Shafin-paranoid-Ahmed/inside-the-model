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
