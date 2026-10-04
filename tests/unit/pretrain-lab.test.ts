import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseModel } from '../../src/lib/transformer';
import { encode } from '../../src/lib/lm/charset';
import { generate } from '../../src/lib/lm/sample';
import * as S from '../../src/components/widgets/PretrainLab/sim';

const w = parseModel(JSON.parse(readFileSync('public/models/house.json', 'utf8')), new Uint8Array(readFileSync('public/models/house.bin')));
const log: S.Log = JSON.parse(readFileSync('public/models/house-log.json', 'utf8'));
const scored = S.scorePassage(w);

describe('objective and loss', () => {
  test('the passage is a held-out sentence of the corpus', () => {
    const txt = readFileSync('scripts/data/alice.txt', 'utf8').toLowerCase(), size = Math.floor(encode(txt).length / 50), at = txt.indexOf(S.PASSAGE);
    expect(at).toBeGreaterThan(-1);
    const chunk = Math.floor(at / size);
    expect(chunk % 10).toBe(9);
    expect(Math.floor((at + S.PASSAGE.length - 1) / size)).toBe(chunk);
  });
  test('distributions are normalised and the loss is the mean of -ln p', () => {
    for (const d of scored.probs) expect(d.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    expect(scored.mean).toBeCloseTo(scored.loss.reduce((a, b) => a + b, 0) / scored.loss.length, 12);
    expect(scored.ppl).toBeCloseTo(Math.exp(scored.mean), 12);
    expect(scored.mean).toBeCloseTo(1.3031, 3);
    expect(scored.p[3]).toBeGreaterThan(0.99); // "alic" -> e
  });
  test('objective frames: eight bars, the true one always present, determinism', () => {
    for (let t = 0; t < S.objectiveDuration(scored); t += 0.37) {
      const f = S.objectiveAt(t, scored);
      expect(f.bars).toHaveLength(8);
      expect(f.bars.filter((b) => b.isTrue)).toHaveLength(1);
    }
    expect(S.objectiveAt(2.2, scored)).toEqual(S.objectiveAt(2.2, scored));
    const end = S.objectiveAt(S.objectiveDuration(scored), scored);
    expect(end.pTrue).toBeCloseTo(scored.p.at(-1)!, 12);
  });
  test('loss frames end at the passage mean', () => {
    expect(S.lossAt(0, scored).heights.every((h) => h === 0 || h > 0)).toBe(true);
    const end = S.lossAt(S.lossDuration(scored), scored);
    expect(end.mean).toBeCloseTo(scored.mean, 12);
    expect(end.heights).toEqual(scored.loss);
    expect(scored.loss[end.tallest]).toBe(Math.max(...scored.loss));
  });
  test('house validation loss beats the baselines and ln 38 matches the log', () => {
    expect(log.validLossQuantised).toBeCloseTo(1.5044, 3);
    expect(log.baselines.uniformLnV).toBeCloseTo(Math.log(38), 4);
    expect(log.validLossQuantised).toBeLessThan(log.baselines.bigramValidCrossEntropy);
  });
});

describe('data', () => {
  test('sample documents fit in 3 KB; the figure has twelve', () => {
    expect(S.DOCS).toHaveLength(12);
    expect(S.DOC_BYTES).toBeLessThanOrEqual(3072);
  });
  test('MinHash estimate tracks the exact Jaccard of the near copies', () => {
    const r = S.analyse(0.7);
    expect(Math.abs(r.pair.est - r.pair.exact)).toBeLessThan(0.1);
    expect(Math.abs(r.pair2.est - r.pair2.exact)).toBeLessThan(0.1);
    expect(S.jaccard(S.shingles('a b c d e f'), S.shingles('a b c d e f'))).toBe(1);
    const s = S.shingles(S.DOCS[0].text), sig = S.minhash(s);
    expect(S.estimate(sig, sig)).toBe(1);
  });
  test('each gate removes what it should, and the threshold moves the near copy', () => {
    const r = S.analyse(0.7);
    expect(r.docs.map((d) => d.out)).toEqual([3, 3, 0, 2, 1, 2, 0, 3, 2, 2, 2, 3]);
    expect(r.docs[3].cells[2].note).toMatch(/words/);
    expect(r.docs[5].cells[2].note).toMatch(/top phrase/);
    expect(r.docs[8].cells[2].note).toMatch(/mean word/);
    expect(r.docs[9].cells[2].note).toMatch(/lines end/);
    expect(r.docs[10].cells[2].note).toMatch(/# marks/);
    expect(S.analyse(0.8).docs[4].out).toBe(3); // J about 0.73 is below 0.8
    expect(S.analyse(0.55).docs[7].out).toBe(1); // J about 0.59
    expect(S.analyse(0.9).kept).toBeGreaterThan(r.kept);
  });
  test('frames: counters fall, end state matches the run', () => {
    const r = S.analyse(0.7);
    expect(S.dataAt(0, r).chars).toBe(r.totalChars);
    const end = S.dataAt(r.duration, r);
    expect(end.chars).toBeCloseTo(r.keptChars, 6);
    expect(end.kept).toBe(r.kept);
    let prev = Infinity;
    for (let t = 0; t < r.duration; t += 0.2) { const c = S.dataAt(t, r).chars; expect(c).toBeLessThanOrEqual(prev + 1e-9); prev = c; }
  });
});

describe('compute', () => {
  test('house N, D, C come from the log and C = 6ND', () => {
    expect(log.N).toBe(35822); expect(log.D).toBe(6144000);
    expect(S.flops(log.N, log.D)).toBe(log.C);
    expect(log.tokensPerStep).toBe(1024);
  });
  test('published rungs: 6ND', () => {
    const [gpt3, chin, llama] = S.PUBLISHED;
    expect(gpt3.C / 1e23).toBeCloseTo(3.15, 2);
    expect(chin.C / 1e23).toBeCloseTo(5.88, 2);
    expect(llama.C / 1e25).toBeCloseTo(3.79, 2);
  });
  test('frames reveal rungs in order and zoom out', () => {
    const rs = S.ladder(log, 1e9, 2e10);
    expect(rs[1].C).toBeCloseTo(1.2e20, -8);
    let prev = 0;
    for (let t = 0; t < S.computeDuration(rs); t += 0.3) { const f = S.computeAt(t, rs); expect(f.maxExp).toBeGreaterThanOrEqual(prev - 1e-9); prev = f.maxExp; f.rows.forEach((r) => expect(r.x).toBeLessThanOrEqual(1 + 1e-9)); }
    expect(S.computeAt(S.computeDuration(rs), rs).rows.every((r) => r.grow === 1)).toBe(true);
    expect(S.eng(35822)).toBe('35.8K'); expect(S.eng(175e9)).toBe('175B'); expect(S.eng(1.4e12)).toBe('1.4T');
  });
});

describe('learning', () => {
  test('baselines are crossed at logged steps and the displayed step snaps to the log', () => {
    expect(S.crossing(log, log.baselines.unigramValidCrossEntropy)).toBe(100);
    expect(S.crossing(log, log.baselines.bigramValidCrossEntropy)).toBe(400);
    const steps = new Set(log.curve.map(([s]) => s));
    for (let t = 0; t <= S.LEARN_T; t += 0.25) { const f = S.learningAt(t, log); expect(steps.has(f.step)).toBe(true); expect(f.loss).toBe(log.curve[f.idx][1]); }
    const end = S.learningAt(S.LEARN_T, log);
    expect(end.step).toBe(6000);
    expect(end.probes[0]).toBeCloseTo(0.9826, 4);
    expect(S.learningMarkers(log)).toHaveLength(3);
  });
});

describe('base model', () => {
  test('three lanes of 48 characters, greedy is deterministic and starts from the top choice', () => {
    const a = S.baseRun(w), b = S.baseRun(w);
    expect(a).toEqual(b);
    expect(a).toHaveLength(3);
    a.forEach((l) => { expect(l.ids).toHaveLength(S.BASE_N); expect(l.alts).toHaveLength(S.BASE_N); });
    a[0].ids.forEach((id, i) => expect(a[0].alts[i][0].id).toBe(id));
    expect(a[1].ids).not.toEqual(a[2].ids);
    const ref = generate(w, encode(S.BASE_PROMPT), S.BASE_N, { temperature: 0 }, []).ids.slice(encode(S.BASE_PROMPT).length);
    expect(a[0].ids).toEqual(ref);
    const job = S.baseJob(w); let chunks = 0; while (!S.baseWork(job, 5)) chunks++;
    expect(job.lanes).toEqual(a); expect(chunks).toBeGreaterThan(0);
    expect(S.baseAt(0).typed).toBe(1);
    expect(S.baseAt(S.baseDuration()).typed).toBe(S.BASE_N);
  });
});
