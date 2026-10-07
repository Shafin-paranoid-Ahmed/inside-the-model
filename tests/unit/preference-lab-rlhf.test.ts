import { describe, expect, it } from 'vitest';
import * as align from '../../src/lib/align';
import type { Frame, Kit, ModeImpl, Opts } from '../../src/components/widgets/_rl/kit';
import type { Env } from '../../src/components/widgets/_rl/svg';
import ppo from '../../src/components/widgets/PreferenceLab/m-ppo';
import rewardModel from '../../src/components/widgets/PreferenceLab/m-reward-model';
import rmBias from '../../src/components/widgets/PreferenceLab/m-rm-bias';
import ppoClip from '../../src/components/widgets/PreferenceLab/m-ppo-clip';
import klSweep from '../../src/components/widgets/PreferenceLab/m-kl-sweep';
import overopt from '../../src/components/widgets/PreferenceLab/m-overopt';

// Node stand-in for the page kit: memo as in the page, worker jobs call src/lib/align directly.
const nodeKit = (): Kit => {
  const m = new Map<string, unknown>();
  const memo = <T,>(k: string, f: () => T) => (m.has(k) ? m.get(k) : (m.set(k, f()), m.get(k))) as T;
  return { memo, job: <T,>(fn: string, ...args: unknown[]) => memo(JSON.stringify(['job', fn, args]), async () => (align as unknown as Record<string, (...a: unknown[]) => T>)[fn](...args)) };
};
const kit = nodeKit();
type M = ModeImpl<unknown, Frame>;
const ui = { pick: null };
const MODES: [string, M, Record<string, unknown>][] = [
  ['ppo', ppo as M, {}],
  ['ppo', ppo as M, { beta: 0 }],
  ['ppo', ppo as M, { labels: 'ai', beta: 0, compare: 'human' }],
  ['reward-model', rewardModel as M, {}],
  ['rm-bias', rmBias as M, {}],
  ['ppo-clip', ppoClip as M, { iter: 0 }],
  ['kl-sweep', klSweep as M, {}],
  ['overopt', overopt as M, {}],
];
const at = async (mode: M, o: Opts, f = 1) => {
  const s = await mode.setup(o, kit);
  return mode.frameAt(mode.timeline(s, o).duration * f, s, o, ui);
};

describe.each(MODES)('rlhf mode %s %o', (name, mode, props) => {
  const o = { mode: name, ...props } as Opts;
  it('markers increase inside the timeline', async () => {
    const s = await mode.setup(o, kit), { duration, markers } = mode.timeline(s, o);
    expect(markers.length).toBeGreaterThan(1);
    markers.forEach((m, i) => {
      expect(m.at).toBeGreaterThanOrEqual(0);
      expect(m.at).toBeLessThanOrEqual(duration);
      if (i) expect(m.at).toBeGreaterThan(markers[i - 1].at);
    });
  });
  it('frames narrate and draw at several times, deterministically, with text >= 11 px', async () => {
    const [a, b] = [await mode.setup(o, kit), await mode.setup(o, nodeKit())], { duration } = mode.timeline(a, o);
    for (const t of [0, duration * 0.05, duration * 0.3, duration / 2, duration]) {
      const f = mode.frameAt(t, a, o, ui);
      expect(f.status.length).toBeGreaterThan(0);
      expect(mode.frameAt(t, b, o, ui)).toEqual(f);
      for (const narrow of [false, true]) {
        const E: Env = { W: narrow ? 360 : 720, H: narrow ? 560 : 440, narrow, out: [] };
        mode.draw(E, f, a, o);
        expect(E.out.length).toBeGreaterThan(5);
        expect(E.out.length).toBeLessThan(500);
        for (const sh of E.out) {
          if (sh.k === 'text') expect(sh.size).toBeGreaterThanOrEqual(11);
          for (const v of Object.values(sh)) if (typeof v === 'number') expect(Number.isFinite(v)).toBe(true);
        }
      }
    }
    expect(mode.label(a, o).length).toBeGreaterThan(30);
  });
});

const ro = (f: Frame, k: string) => f.readouts!.find((r) => r[0] === k)![1];

describe('numbers quoted in the lesson', () => {
  it('reward-model: loss 0.693 -> 0.487, held-out accuracy 68%', async () => {
    const o = { mode: 'reward-model' } as Opts, f0 = await at(rewardModel as M, o, 0), f1 = await at(rewardModel as M, o);
    expect(ro(f0, 'Loss')).toBe('0.693');
    expect(ro(f1, 'Loss')).toBe('0.487');
    expect(ro(f1, 'Held-out accuracy')).toBe('68%');
    expect(ro(f1, 'Training accuracy')).toBe('72%');
    expect(f1.status).toContain('weight on length is 0.76 (gold −0.30) and on correct 2.64 (gold 2.00)');
  });
  it('rm-bias: the rambling answer has the top score; rank correlation about 0.87 without it, 0.47 with it', async () => {
    const f = await at(rmBias as M, { mode: 'rm-bias' } as Opts);
    expect(f.status).toContain('rambling wrong answer (48 words, the longest response in the pairs was 28)');
    expect(f.status).toContain('scores 4.72');
    expect(ro(f, 'Highest reward model score')).toBe('4.72 (gold −0.44)');
    expect(ro(f, 'Rank correlation, all eight')).toBe('0.47');
    expect(ro(f, 'Rank correlation, other seven types')).toBe('0.88');
  });
  it('ppo beta 0.1: proxy, gold and KL 3.35, 2.27, 0 at the start and 4.08, 2.81, 0.802 at the end', async () => {
    const o = { mode: 'ppo' } as Opts, f0 = await at(ppo as M, o, 0), f1 = await at(ppo as M, o);
    expect([ro(f0, 'Proxy reward'), ro(f0, 'Gold reward'), ro(f0, 'KL to SFT')]).toEqual(['3.35', '2.27', '0.00']);
    expect([ro(f1, 'Proxy reward'), ro(f1, 'Gold reward'), ro(f1, 'KL to SFT')]).toEqual(['4.08', '2.81', '0.802']);
  });
  it('ppo beta 0: gold peaks at update 19 (2.82), ends at proxy 4.72, gold -0.44, KL 13.01, rambling answer 1.00', async () => {
    const f = await at(ppo as M, { mode: 'ppo', beta: 0 } as Opts);
    expect(f.status).toContain('Gold peaked at update 19 (2.82)');
    expect([ro(f, 'Proxy reward'), ro(f, 'Gold reward'), ro(f, 'KL to SFT'), ro(f, 'Rambling answer')]).toEqual(['4.72', '−0.44', '13.01', '1.00']);
  });
  it('ppo labels ai with compare: the faint human-label gold run ends at -0.44 while AI-label gold ends at 2.17', async () => {
    const f = await at(ppo as M, { mode: 'ppo', labels: 'ai', beta: 0, compare: 'human' } as Opts);
    expect(ro(f, 'Gold reward')).toBe('2.17');
    expect(ro(f, 'Gold, human labels')).toBe('−0.44');
  });
  it('ppo-clip, update 0: outside the band and flat counts per epoch', async () => {
    const o = { mode: 'ppo-clip', iter: 0 } as Opts, s = (await ppoClip.setup(o, kit)) as { oob: number[]; flat: number[]; B: number };
    expect(s.oob.slice(1).map((v) => Math.round(v * s.B))).toEqual([5, 25, 27, 23]);
    expect(s.flat.slice(1).map((v) => Math.round(v * s.B))).toEqual([5, 23, 23, 21]);
    const f = await at(ppoClip as M, o);
    expect(ro(f, 'Outside the band')).toBe('23 of 64');
    expect(ro(f, 'Outside but still pushing')).toBe('2 of 64');
  });
  it('kl-sweep: final gold 2.76, 2.81, -0.44, -0.44; the best is beta 0.1', async () => {
    const f = await at(klSweep as M, { mode: 'kl-sweep' } as Opts);
    expect(f.readouts!.map((r) => r[1])).toEqual(['KL 0.29, proxy 3.97, gold 2.76', 'KL 0.80, proxy 4.08, gold 2.81', 'KL 13.01, proxy 4.72, gold −0.44', 'KL 13.01, proxy 4.72, gold −0.44']);
    expect(f.status).toContain('The highest so far is β = 0.1');
  });
  it('overopt: gold peaks at distance 0.65, the run ends at distance 3.61 with proxy minus gold 5.16', async () => {
    const f = await at(overopt as M, { mode: 'overopt' } as Opts, 0.5);
    expect(f.status).toContain('Gold peaked at distance 0.65 (update 19)');
    const e = await at(overopt as M, { mode: 'overopt' } as Opts, 1);
    expect(e.status).toContain('ended at distance 3.61 with gold −0.44');
    const g = await at(overopt as M, { mode: 'overopt' } as Opts, 0.8);
    expect(ro(g, 'Proxy minus gold')).toBe('5.16');
  });
});
