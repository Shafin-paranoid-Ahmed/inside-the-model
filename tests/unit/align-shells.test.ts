import { describe, expect, it } from 'vitest';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import base from '../../src/components/widgets/SFTLab/m-base';
import pairs from '../../src/components/widgets/PreferenceLab/m-pairs';
import rules from '../../src/components/widgets/CritiqueRevise/m-rules';
import verifyMode from '../../src/components/widgets/VerifierRL/m-verify';

// Node stand-in for the page kit: memo works, worker jobs are not available here.
const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  return {
    memo: <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T,
    job: () => Promise.reject(new Error('no worker in tests')),
  };
};
const MODES: [string, ModeImpl<unknown, Frame>, Record<string, unknown>][] = [
  ['base', base as ModeImpl<unknown, Frame>, {}],
  ['pairs', pairs as ModeImpl<unknown, Frame>, {}],
  ['pairs', pairs as ModeImpl<unknown, Frame>, { labels: 'ai' }],
  ['rules', rules as ModeImpl<unknown, Frame>, {}],
  ['verify', verifyMode as ModeImpl<unknown, Frame>, {}],
  ['verify', verifyMode as ModeImpl<unknown, Frame>, { verifier: 'lenient' }],
];

describe.each(MODES)('reference mode %s %o', (name, mode, props) => {
  const o = { mode: name, ...props } as Opts;
  const ui = { pick: null };

  it('markers are strictly increasing inside the timeline', async () => {
    const s = await mode.setup(o, nodeKit());
    const { duration, markers } = mode.timeline(s, o);
    expect(markers.length).toBeGreaterThan(1);
    markers.forEach((m, i) => {
      expect(m.at).toBeGreaterThanOrEqual(0);
      expect(m.at).toBeLessThanOrEqual(duration);
      if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at);
    });
  });

  it('frames at start, middle and end narrate and draw, deterministically, with text >= 11 px', async () => {
    const [a, b] = [await mode.setup(o, nodeKit()), await mode.setup(o, nodeKit())];
    const { duration } = mode.timeline(a, o);
    for (const t of [0, duration * 0.3, duration / 2, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(0);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 600 : 440, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(0);
        expect(E.out.length).toBeLessThan(500);
        for (const sh of E.out) if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
      }
    }
    expect(mode.label(a, o).length).toBeGreaterThan(20);
  });
});

const end = async (mode: ModeImpl<unknown, Frame>, props: Record<string, unknown>) => {
  const o = props as Opts, s = await mode.setup(o, nodeKit());
  return mode.frameAt(mode.timeline(s, o).duration, s, o, { pick: null });
};

it('base: every train prompt has "continues the list" on top; the narration names the averages', async () => {
  const f = await end(base as ModeImpl<unknown, Frame>, { mode: 'base' });
  expect(f.readouts?.[3][1]).toBe('0 of 8');
  expect(f.detail).toContain('0.575');
  expect(f.detail).toContain('0.185');
});

it('pairs: the plan §7 tallies (human 68.5% gold, 48.5% longer; AI 71% with human, 70.5% with gold)', async () => {
  const h = await end(pairs as ModeImpl<unknown, Frame>, { mode: 'pairs' });
  expect(h.readouts?.map(([, v]) => v)).toEqual(['200 of 200', '69%', '49%', '58']);
  const a = await end(pairs as ModeImpl<unknown, Frame>, { mode: 'pairs', labels: 'ai' });
  expect(a.readouts?.map(([, v]) => v)).toEqual(['200 of 200', '71%', '71%', '69%']);
});

it('rules: only the three clean answers break no principle', async () => {
  const f = await end(rules as ModeImpl<unknown, Frame>, { mode: 'rules' });
  expect(f.status).toBe('Only terse correct, clear correct and hedged correct break no principle.');
});

it('verify: three of six pass strictly, four leniently (the guess list)', async () => {
  expect((await end(verifyMode as ModeImpl<unknown, Frame>, { mode: 'verify' })).readouts?.[2][1]).toBe('3');
  expect((await end(verifyMode as ModeImpl<unknown, Frame>, { mode: 'verify', verifier: 'lenient' })).readouts?.[2][1]).toBe('4');
});
