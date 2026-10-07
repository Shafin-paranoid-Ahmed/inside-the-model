import { describe, expect, it } from 'vitest';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import * as align from '../../src/lib/align';
import base from '../../src/components/widgets/SFTLab/m-base';
import data from '../../src/components/widgets/SFTLab/m-data';
import train from '../../src/components/widgets/SFTLab/m-train';
import mask from '../../src/components/widgets/SFTLab/m-mask';
import heldout from '../../src/components/widgets/SFTLab/m-heldout';
import cap from '../../src/components/widgets/SFTLab/m-cap';

// Node stand-in for the page kit: memo works and worker jobs run in-process.
const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: <T,>(fn: string, ...args: unknown[]) => Promise.resolve((align as unknown as Record<string, (...a: unknown[]) => unknown>)[fn](...args) as T),
  };
};
const any = (m: unknown) => m as ModeImpl<unknown, Frame>;
const MODES: [string, ModeImpl<unknown, Frame>, Record<string, unknown>][] = [
  ['data', any(data), { demos: 24, demoAccuracy: 0.8, seed: 2 }],
  ['train', any(train), { demos: 24, demoAccuracy: 0.8, seed: 2, steps: 150, lr: 0.05 }],
  ['mask', any(mask), {}],
  ['heldout', any(heldout), { demos: 24, demoAccuracy: 0.8, seed: 2, steps: 150, lr: 0.05 }],
  ['cap', any(cap), { demos: 40, demoAccuracies: '0.6,0.8,0.95', steps: 300 }],
];

describe.each(MODES)('SFTLab mode %s', (name, mode, props) => {
  const o = { mode: name, ...props } as Opts;
  const ui = { pick: null };

  it('markers are strictly increasing inside the timeline', async () => {
    const s = await mode.setup(o, nodeKit());
    const { duration, markers } = mode.timeline(s, o);
    expect(markers.length).toBeGreaterThan(1);
    markers.forEach((m, i) => {
      expect(m.at).toBeGreaterThanOrEqual(0);
      expect(m.at).toBeLessThan(duration);
      if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at);
    });
  });

  it('frames at the endpoints and in between narrate and draw, deterministically, with text >= 11 px', async () => {
    const [a, b] = [await mode.setup(o, nodeKit()), await mode.setup(o, nodeKit())];
    const { duration } = mode.timeline(a, o);
    for (const t of [0, duration * 0.2, duration / 2, duration * 0.8, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(0);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 340 : 720, H: narrow ? 700 : 460, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0);
        expect(E.out.length).toBeLessThan(500);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
    expect(mode.label(a, o).length).toBeGreaterThan(20);
  });
});

const frame = async (mode: ModeImpl<unknown, Frame>, props: Record<string, unknown>, at: 'start' | 'end' | number) => {
  const o = props as Opts, s = await mode.setup(o, nodeKit()), d = mode.timeline(s, o).duration;
  return mode.frameAt(at === 'start' ? 0 : at === 'end' ? d : at, s, o, { pick: null });
};
const props = (name: string) => ({ mode: name, ...MODES.find((m) => m[0] === name)![2] });
const ro = (f: Frame, label: string) => f.readouts!.find((r) => r[0] === label)![1];

it('base (the core mode) is unchanged: 0 of 8 prompts answer on top', async () => {
  const f = await frame(any(base), { mode: 'base' }, 'end');
  expect(f.readouts?.[3][1]).toBe('0 of 8');
});

it('data: 19 of 24 correct, mean length 5.4 words, 5 wrong', async () => {
  const f = await frame(any(data), props('data'), 'end');
  expect(ro(f, 'Correct so far')).toBe('19 of 24 (79%)');
  expect(ro(f, 'Mean length')).toBe('5.4 words');
  expect(f.status).toContain('19 correct and 5 wrong');
});

it('train: loss 2.973 -> 1.431 and P(continues the list) 0.575 -> 0.057 over 150 steps', async () => {
  const a = await frame(any(train), props('train'), 'start'), b = await frame(any(train), props('train'), 'end');
  expect(ro(a, 'Loss')).toBe('2.973');
  expect(ro(b, 'Loss')).toBe('1.431');
  expect(ro(a, 'P(continues the list)')).toBe('0.575');
  expect(ro(b, 'P(continues the list)')).toBe('0.057');
  expect(ro(a, 'P(a correct answer)')).toBe('0.185');
  expect(ro(b, 'P(a correct answer)')).toBe('0.801');
});

it('mask: 12.6 vs 5.9 counted tokens per example; held-out loss 2.146 vs 2.150 at step 100 and 2.256 vs 2.317 at step 300', async () => {
  const o = props('mask') as Opts, s = await mask.setup(o, nodeKit()), d = mask.timeline(s, o).duration, sweep = 13 * 0.55;
  const f0 = mask.frameAt(1, s, o, { pick: null });
  expect(ro(f0, 'Counted in this example')).toBe('12 vs 6 tokens');
  expect(ro(f0, 'Average per training example')).toBe('12.6 vs 5.9 tokens');
  const at = (step: number) => mask.frameAt(sweep + ((step + 0.5) / 300) * 6.5, s, o, { pick: null });
  expect([ro(at(100), 'Held-out loss, every token'), ro(at(100), 'Held-out loss, response only')]).toEqual(['2.146', '2.150']);
  const end = mask.frameAt(d, s, o, { pick: null });
  expect([ro(end, 'Held-out loss, every token'), ro(end, 'Held-out loss, response only')]).toEqual(['2.256', '2.317']);
  // masked tokens are the prompt tokens: the response-only run counts only the response
  expect(s.cResp.every((i) => i >= s.respStart)).toBe(true);
});

it('heldout: P(correct) 0.22 -> 0.81 and 0.16 -> 0.68 on the two held-out prompts', async () => {
  const f = await frame(any(heldout), props('heldout'), 'end');
  expect(f.readouts![1][1]).toBe('0.22 → 0.81');
  expect(f.readouts![2][1]).toBe('0.16 → 0.68');
});

it('cap (40 demos): the demo sets are 0.600 / 0.800 / 0.950 correct and the models end at 0.602 / 0.804 / 0.955', async () => {
  const f = await frame(any(cap), props('cap'), 'end');
  expect(f.readouts!.slice(1).map((r) => r[0])).toEqual(['Demonstrators 60% correct', 'Demonstrators 80% correct', 'Demonstrators 95% correct']);
  expect(f.readouts!.slice(1).map((r) => r[1])).toEqual(['model 0.602', 'model 0.804', 'model 0.955']);
  expect(f.status).toContain('0.602 against 0.600');
});
