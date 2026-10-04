import { describe, expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseModel } from '../../src/lib/transformer/load';
import * as S from '../../src/components/widgets/HallucinationLab/sim';

const manifest = JSON.parse(readFileSync('public/models/facts.json', 'utf8'));
const w = parseModel(manifest, new Uint8Array(readFileSync('public/models/facts.bin')));
const data = JSON.parse(readFileSync('public/models/facts-data.json', 'utf8'));
const people: S.Person[] = data.people.map((p: S.Person) => ({ name: p.name, month: p.month, count: p.count }));
const exp = S.runExperiment(w, people);
const unseen = exp.rows.filter((r) => r.count === 0);
const ctx = (extra: Partial<S.Context> = {}): S.Context => {
  const [a, b] = S.pickHook(exp.rows), row = S.pickWorked(unseen);
  const stated = Array.from({ length: 12 }, (_, m) => S.ask(w, row.name, { name: row.name, month: m }));
  return { exp, people, unseen, panels: [S.panelOf(a), S.panelOf(b)], worked: { row, stated, month: S.wrongMonth(row.month) }, ...extra };
};

describe('HallucinationLab sim', { timeout: 30000 }, () => {
  test('re-derived answers match the shipped report', () => {
    data.people.forEach((p: { pick: number; confidence: number }, i: number) => {
      expect(exp.rows[i].pick).toBe(p.pick); expect(Math.abs(exp.rows[i].conf - p.confidence)).toBeLessThan(1e-3);
    });
    const bc = S.byCount(exp.rows);
    manifest.metrics.byCount.forEach((m: { count: number; accuracy: number; people: number }) => {
      const b = bc.find((x) => x.count === m.count)!; expect(b.n).toBe(m.people); expect(b.accuracy).toBeCloseTo(m.accuracy, 3);
    });
    expect(S.reliability(exp.rows).ece).toBeCloseTo(manifest.metrics.ece, 3);
    expect(S.trainingFrequency(people).map((x) => +x.toFixed(4))).toEqual(manifest.metrics.trainingMonthFrequencies);
  });
  test('the numbers quoted in the lesson', () => {
    expect(S.byCount(exp.rows).map((b) => [b.count, b.n, b.right])).toEqual([[0, 50, 3], [1, 50, 17], [2, 40, 29], [5, 30, 30], [10, 20, 20], [20, 10, 10]]);
    const u = S.reliability(unseen);
    expect(u.meanConfidence).toBeCloseTo(0.354, 3); expect(u.accuracy).toBeCloseTo(0.06, 3);
    expect(S.mean(unseen.map((r) => r.unknown))).toBeLessThan(0.001);
    expect(S.selective(exp.rows, 0.5)).toMatchObject({ answered: 80, wrong: 10, abstained: 120 });
    expect(S.selective(exp.rows, 0.9)).toMatchObject({ answered: 12, wrong: 0 });
    expect(S.thresholdFor(exp.rows, 0.05)).toBe(0.72);
    const wk = S.pickWorked(unseen), i = unseen.indexOf(wk);
    expect(exp.right.every((a, j) => a.pick === unseen[j].month)).toBe(true);
    expect(exp.wrong.every((a, j) => a.pick === S.wrongMonth(unseen[j].month))).toBe(true);
    expect(i).toBeGreaterThanOrEqual(0);
    expect(S.flagStats(exp.rows, 10, 0.5).accuracy).toBeCloseTo(0.71, 2);
  });
  test('prompts: validated, and a stated line is prepended', () => {
    expect(S.validName('mira tolven')).toBe(true); expect(S.validName('Mira')).toBe(false); expect(S.validName('a'.repeat(15))).toBe(false); expect(S.validName('a  b')).toBe(false);
    expect(S.promptIds('ab', { name: 'ab', month: 2 }).length).toBe(2 * ('ab was born in '.length) + 3);
    expect(() => S.ask(w, 'Bad!')).toThrow();
    const a = S.ask(w, 'menel bunar');
    expect(a.dist.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 10);
  });
  test('frameAt: endpoints, determinism, and readouts agree with the stats', () => {
    for (const mode of Object.keys(S.DURATION) as S.Mode[]) {
      const c = ctx(), f0 = S.frameAt(0, mode, c), f1 = S.frameAt(S.DURATION[mode], mode, c);
      expect(S.frameAt(S.DURATION[mode] / 2, mode, c)).toEqual(S.frameAt(S.DURATION[mode] / 2, mode, c));
      expect(typeof S.describe(f0, c)).toBe('string'); expect(S.describe(f1, c).length).toBeGreaterThan(10);
    }
    const c = ctx();
    const cal = S.frameAt(10, 'calibration', c); if (cal.mode === 'calibration') { expect(cal.n).toBe(200); expect(cal.ece).toBeCloseTo(0.073, 3); }
    const un = S.frameAt(10, 'calibration', ctx({ unseenOnly: true })); if (un.mode === 'calibration') expect(un.n).toBe(50);
    const ab = S.frameAt(5, 'abstain', c); if (ab.mode === 'abstain') { expect(ab.answered).toBe(80); expect(ab.dots.filter((d) => d !== 2).length).toBe(80); }
    const ob = S.frameAt(12, 'objective', c); if (ob.mode === 'objective') { expect(ob.n).toBe(50); expect(ob.tv).toBeCloseTo(0.168, 2); }
    const re = S.frameAt(12, 'retrieval', c); if (re.mode === 'retrieval') { expect(re.acc).toEqual([0.06, 1, 0]); expect(re.copies).toBe(1); expect(re.bars).toEqual([0.06, 1, 0]); }
    const ve = S.frameAt(14, 'verify', c); if (ve.mode === 'verify') { expect(ve.n).toBe(200); expect(ve.levels.reduce((s, l) => s + l.right + l.wrong, 0)).toBe(200); }
  });
  test('sampling is seeded and agreement counts repeats of the main answer', () => {
    const d = exp.rows[0].dist;
    expect(S.drawSamples(d, 10, 5)).toEqual(S.drawSamples(d, 10, 5));
    expect(S.agreement([1, 1, 2, 1], 1)).toBe(0.75);
    expect(S.order(10, 3).slice().sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });
});
