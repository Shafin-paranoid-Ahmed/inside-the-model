import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseModel, type Manifest } from '../../src/lib/transformer/load';
import {
  CHANCE, COUNTS, DEFAULT_PROMPT, DEMO, FORMATS, HEADS, LETTERS, ablations, accuracy, attentionFrame, beatMarkers, frameAt,
  makeTask, norm, parsePairs, phaseFrame, predict, rowAt, shotRows, stepAt, timeOfStep, triangle, type TrainingLog,
} from '../../src/components/widgets/ICLLab/sim';
import { createRng } from '../../src/lib/math/rng';

const w = parseModel(JSON.parse(readFileSync('public/models/icl.json', 'utf8')) as Manifest, new Uint8Array(readFileSync('public/models/icl.bin')));
const log = JSON.parse(readFileSync('public/models/icl-log.json', 'utf8')) as TrainingLog;
const sweep = JSON.parse(readFileSync('public/models/icl-sweep.json', 'utf8'));

describe('tasks', () => {
  test('deterministic, and the answer follows the shift', () => {
    const a = makeTask(createRng(5), 4), b = makeTask(createRng(5), 4);
    expect(a).toEqual(b);
    const last = a.text.trim().split(' ').at(-1)!;
    expect(LETTERS[(LETTERS.indexOf(last[0]) + a.shift) % 8]).toBe(LETTERS[a.answer]);
  });
  test('formats change the text', () => {
    expect(makeTask(createRng(1), 2, { format: 'equals' }).text).toContain('=');
    expect(makeTask(createRng(1), 2, { format: 'newline' }).text).toContain('\n');
    expect(makeTask(createRng(1), 2, { format: 'packed' }).text).not.toContain(' ');
  });
  test('parsePairs', () => {
    expect(parsePairs('c>f a>d')).toEqual(['c>f', 'a>d']);
    expect(parsePairs('c>z')).toBeNull();
    expect(parsePairs('')).toBeNull();
  });
});

describe('shipped model', () => {
  test('shots demo: no examples near chance, one example is enough', () => {
    const rows = shotRows(w, DEMO.pairs);
    expect(rows).toHaveLength(7);
    expect(rows[0].probs[1]).toBeCloseTo(0.1, 1);
    expect(rows[1].best).toBe(1);
    expect(rows[1].probs[1]).toBeGreaterThan(0.99);
    expect(rows[3].probs[1]).toBeGreaterThan(0.99);
  });
  test('accuracy over 500 tasks matches the numbers quoted in the lesson', () => {
    expect(COUNTS.map((k) => accuracy(w, k))).toEqual([0.124, 1, 0.994, 1, 1, 1]);
    expect(accuracy(w, 4, { random: true })).toBe(0.14);
    expect(FORMATS.map((f) => accuracy(w, 4, { format: f.id }))).toEqual([1, 0.302, 0.262, 0.168, 0.232]);
  });
  test('attention rows are distributions; layer 2 looks at the query letter', () => {
    const p = predict(w, DEFAULT_PROMPT);
    expect(p.weights).toHaveLength(4);
    p.weights.forEach((r) => expect(r.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5));
    expect(p.weights[2][16]).toBeGreaterThan(0.99);
    expect(p.weights[3][16]).toBeCloseTo(0.88, 2);
    expect(p.weights[3][6]).toBeCloseTo(0.037, 2);
  });
  test('switching off any head breaks the answer', () => {
    const a = ablations(w);
    expect(a.probOn).toBeGreaterThan(0.99);
    a.probOff.forEach((p) => expect(p).toBeLessThan(0.01));
    expect(a.accuracyOff).toEqual([0.124, 0.156, 0.188, 0.186]);
    expect(HEADS).toHaveLength(4);
  });
});

describe('frames and logs', () => {
  test('frameAt endpoints and beats', () => {
    expect(frameAt(0, 6)).toMatchObject({ index: 0, local: 0, settle: 0 });
    expect(frameAt(14, 6).index).toBe(5);
    expect(frameAt(NaN, 6).index).toBe(0);
    expect(beatMarkers(['a', 'b']).map((m) => m.at)).toEqual([0, 7]);
    expect(attentionFrame(0).layer1).toBe(0);
    expect(attentionFrame(14).layer2).toBe(1);
  });
  test('phase frame shows a real logged row', () => {
    expect(timeOfStep(stepAt(7))).toBeCloseTo(7);
    const f = phaseFrame(timeOfStep(700), log);
    expect(log.log[f.row].step).toBe(700);
    expect(f.loss4).toBeCloseTo(1.003, 3);
    expect(log.thumbnails[f.thumb].step).toBe(700);
    expect(rowAt(log.log, 0)).toBe(0);
  });
  test('loss plateau then fall; attention on the query letter forms first', () => {
    const at = (s: number) => log.log[rowAt(log.log, s)];
    expect(at(450).loss4).toBeGreaterThan(2.06);
    expect(at(1000).loss4).toBeLessThan(0.31);
    expect(at(400).profile.query).toBeLessThan(0.05);
    expect(at(500).profile.query).toBeGreaterThan(0.45);
  });
  test('triangle is lower-triangular and holds the checkpoint numbers', () => {
    const m = triangle(log.thumbnails[1].weights);
    expect(m).toHaveLength(18);
    expect(m[0].slice(1).every((v) => v === 0)).toBe(true);
    expect(m[17][16]).toBeCloseTo(0.78, 2);
  });
  test('sweep: not monotone in size; exact match is near per-letter^4', () => {
    const ok = sweep.rows.map((r: { exact: number }) => r.exact > 0.9);
    expect(ok).toEqual([false, true, false, true, true, false]);
    const d48 = sweep.rows[5];
    expect(d48.exact).toBe(0.006);
    expect(d48.perLetter ** 4).toBeCloseTo(0.0042, 3);
    expect(CHANCE.meanLogp).toBeCloseTo(-8.318, 3);
    expect(norm('meanLogp', -9)).toBe(0);
    expect(norm('meanLogp', 0)).toBe(1);
  });
});
