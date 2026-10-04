import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { encode } from '../../src/lib/lm/charset';
import { parseModel, type Manifest } from '../../src/lib/transformer';
import { entropy, generate, softmaxT } from '../../src/lib/lm/sample';
import {
  BEAT, GREEDY_N, K_REST, LAB_DEFAULTS, LOGITS_T, PROMPTS, TEMPS, durationOf, findRepeat, greedyFrame, kAt, labFrame, logitsFacts, logitsFrame,
  labText, makeGreedy, makeLab, makePair, makeTemp, meanLogp, nextOf, pAt, tempAt, tempFrame, topkFrame, toppFrame, uniforms,
} from '../../src/components/widgets/SamplingLab/sim';

const manifest: Manifest = JSON.parse(readFileSync('public/models/house.json', 'utf8'));
const w = parseModel(manifest, new Uint8Array(readFileSync('public/models/house.bin')));
const nx = nextOf(w, PROMPTS.main);

describe('findRepeat', () => {
  test('finds the earliest copied block', () => {
    expect(findRepeat('abcdefghijklmnop abcdefghijkl x', 12)).toEqual({ start: 17, from: 0, seenAt: 29 });
    expect(findRepeat('no repeated blocks here at all', 12)).toBeNull();
  });
});

describe('logits mode', () => {
  test('phases end at probabilities that add up to 1, and a shift changes nothing', () => {
    const end = logitsFrame(LOGITS_T, nx), mid = logitsFrame(5.5, nx);
    expect(end.sum).toBeCloseTo(1, 12);
    expect(end.vals[0]).toBeCloseTo(nx.probs[nx.order[0]], 12);
    expect(mid.sum).toBeGreaterThan(1000);
    const shifted = logitsFrame(LOGITS_T, nx, 3);
    for (let i = 0; i < 38; i++) expect(shifted.vals[i]).toBeCloseTo(end.vals[i], 12);
    expect(logitsFrame(5.5, nx, 3).sum).toBeCloseTo(mid.sum * Math.exp(3), 3);
    // bar geometry stays on the plot at every time and shift
    for (const sh of [-4, 0, 4]) for (let t = 0; t <= LOGITS_T; t += 0.25) {
      const f = logitsFrame(t, nx, sh);
      for (const e of f.ends) { expect(e).toBeGreaterThanOrEqual(-1e-9); expect(e).toBeLessThanOrEqual(1 + 1e-9); }
    }
  });
  test('quoted numbers', () => {
    const f = logitsFacts(nx);
    expect(f.topScore).toBeCloseTo(8.41, 2); expect(f.topP).toBeCloseTo(0.4817, 4); expect(f.above1).toBe(5);
    expect(logitsFrame(5.5, nx).sum).toBeCloseTo(9298, 0);
  });
});

describe('greedy mode', () => {
  const d = makeGreedy(w);
  test('greedy copies itself, sampled does not, and the lanes use the shipped seed', () => {
    expect(d.greedy.text).toBe('ead the was the came of the said the came of the say and the was the came of the');
    expect(d.loop).toEqual({ start: 32, from: 11, seenAt: 44 });
    expect(d.sampled.text).toBe('eal: \'whe might last can\'t do the swards denother. "she would be grows rud, here');
    expect(meanLogp(d.greedy.steps, GREEDY_N)).toBeCloseTo(-0.899, 3);
    expect(meanLogp(d.sampled.steps, GREEDY_N)).toBeCloseTo(-1.564, 3);
    expect(d).toEqual(makeGreedy(w));
  });
  test('frames: characters appear 60% into their beat and agree with the lanes', () => {
    expect(greedyFrame(0.5 * BEAT, d).shown).toBe(0);
    expect(greedyFrame(0.7 * BEAT, d).shown).toBe(1);
    const end = greedyFrame(durationOf('greedy'), d);
    expect(end.shown).toBe(GREEDY_N); expect(end.greedyText).toBe(d.greedy.text); expect(end.loopShown).toBe(true);
    expect(greedyFrame(43 * BEAT, d).loopShown).toBe(false);
  });
});

describe('temperature mode', () => {
  const d = makeTemp(w);
  test('entropy and top probability at T = 0.5, 1, 2', () => {
    const at = (T: number) => tempFrame(0, d, T);
    expect(at(0.5).entropy).toBeCloseTo(1.19, 2); expect(at(1).entropy).toBeCloseTo(1.95, 2); expect(at(2).entropy).toBeCloseTo(2.87, 2);
    expect(at(0.5).topP).toBeCloseTo(0.718, 3); expect(at(1).topP).toBeCloseTo(0.482, 3); expect(at(2).topP).toBeCloseTo(0.303, 3);
    expect(at(1).entropy).toBeCloseTo(entropy(nx.probs) / Math.LN2, 12);
  });
  test('sweep is a log scale from 0.2 to 2 and back; text snaps to whole presets', () => {
    expect(tempAt(0)).toBeCloseTo(0.2, 12); expect(tempAt(8)).toBeCloseTo(2, 12); expect(tempAt(16)).toBeCloseTo(0.2, 12);
    expect(d.texts.length).toBe(TEMPS.length);
    for (const s of d.texts) expect(s?.length).toBe(60);
    expect(tempFrame(8, d).text).toBe(d.texts[5]);
    expect(d.texts[1]).toBe('ead, and when and something to her way in a little when with');
    expect(Math.min(...softmaxT(nx.logits, 0.2))).toBeGreaterThanOrEqual(0);
  });
});

describe('top-k and top-p', () => {
  const pair = makePair(w);
  test('mass removed and nucleus sizes', () => {
    const k5 = topkFrame(0, pair, 5);
    expect(k5.sure.removed).toBeCloseTo(0.0013, 4); expect(k5.unsure.removed).toBeCloseTo(0.6472, 4);
    expect(topkFrame(0, pair, 1).unsure.removed).toBeCloseTo(0.9242, 4);
    expect(pair.sure.probs[pair.sure.order[0]]).toBeCloseTo(0.9953, 4);
    const kept = (p: number) => [toppFrame(0, pair, p).sure.keep, toppFrame(0, pair, p).unsure.keep];
    expect(kept(0.5)).toEqual([1, 8]); expect(kept(0.9)).toEqual([1, 16]); expect(kept(0.95)).toEqual([1, 18]); expect(kept(0.99)).toEqual([1, 23]);
  });
  test('sweeps rest on k = 5 and p = 0.9, and k renormalises', () => {
    expect(kAt(K_REST).k).toBe(5); expect(kAt(K_REST).tween).toBe(1);
    expect(pAt(5)).toBeCloseTo(0.9, 12);
    const f = topkFrame(0, pair, 5);
    expect(f.unsure.kept.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(f.unsure.kept.slice(5).every((v) => v === 0)).toBe(true);
    expect(topkFrame(K_REST, pair).k).toBe(5);
  });
});

describe('lab', () => {
  test('default settings reproduce the quoted text and are deterministic', () => {
    const d = makeLab(w, { prompt: PROMPTS.main, ...LAB_DEFAULTS });
    expect(labText(d)).toBe('ead-" she said alice. "i think it would in the blow!" he were added the way in a little from office, and this alice was ');
    expect(meanLogp(d.steps, 120)).toBeCloseTo(-0.943, 3);
    expect(d.steps.length).toBe(120);
    // the character-at-a-time stepper makes exactly what lm/sample.ts `generate` makes
    const g = generate(w, encode(PROMPTS.main), 120, { temperature: 0.8, topP: 0.95 }, uniforms(1, 120));
    expect(d.steps.map((x) => x.id)).toEqual(g.steps.map((x) => x.id));
    expect(d.steps[119].logp).toBe(g.steps[119].logp);
    const f = labFrame(durationOf('lab'), d);
    expect(f.shown).toBe(120); expect(f.repeat).toBeNull();
  });
  test('across seeds 1 to 10 the 120-character text repeats itself less as temperature rises (top-p 0.95)', () => {
    const run = (T: number, seed: number) => makeLab(w, { prompt: PROMPTS.main, temperature: T, topK: 38, topP: 0.95, seed });
    const count = (T: number) => Array.from({ length: 10 }, (_, i) => run(T, i + 1)).filter((d) => findRepeat(labText(d))).length;
    // T = 0 ignores the random numbers, so one run stands for all ten seeds
    expect(findRepeat(labText(run(0, 1)))).not.toBeNull();
    expect([0.2, 0.5].map(count)).toEqual([5, 0]);
  }, 90_000);
  test('uniforms are the seeded stream', () => { expect(uniforms(1, 3)[0]).toBeCloseTo(0.627, 3); });
});
