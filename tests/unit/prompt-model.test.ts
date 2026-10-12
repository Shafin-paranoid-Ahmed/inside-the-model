import { describe, expect, test } from 'vitest';
import {
  EXAMPLES, ID, LETTERS, POOL, QUESTIONS, TASK_FN, TEST, VOCAB, answerBody, buildPrompt, decode, drawStrings, encode,
  formatMask, parseAnswer, parseCall, parses, permutations, promptText, resample, toolPrompt, toolSequence,
} from '../../src/lib/applied/promptmodel';

describe('vocabulary', () => {
  test('encode/decode round trip, word tokens are single ids', () => {
    const text = '[tools] 37*48>CALL calc(37*48) RESULT 1776 >1776;';
    const ids = encode(text);
    expect(ids).toContain(ID.call);
    expect(ids.filter((t) => t === ID.result)).toHaveLength(1);
    expect(encode(decode(ids))).toEqual(ids);
    expect(decode(encode('[rev] abc>cba;'))).toBe('[rev] abc>cba;');
    expect(encode('a = b | c')).toEqual([0, ID.equals, 1, ID.bar, 2]);
    expect(encode('a#<')).toEqual([0, ID.unk, ID.unk]);
    expect(new Set(VOCAB).size).toBe(VOCAB.length);
  });
});

describe('tasks and sets', () => {
  test('task functions', () => {
    expect(TASK_FN.rev('abc')).toBe('cba');
    expect(TASK_FN.shift('ahc')).toBe('bad');
    expect(TASK_FN.sort('hca')).toBe('ach');
    expect(TASK_FN.copy('dfa')).toBe('dfa');
    expect(TASK_FN.sr('abh')).toBe('acb');
  });
  test('pool of 336 distinct-letter strings; seeded test and example sets', () => {
    expect(POOL).toHaveLength(336);
    expect(POOL.every((s) => new Set(s).size === 3 && [...s].every((c) => LETTERS.includes(c)))).toBe(true);
    expect(new Set(TEST).size).toBe(50);
    expect(EXAMPLES.some((e) => TEST.includes(e))).toBe(false);
    expect(drawStrings(50, 1401)).toEqual(TEST);
    // every task is defined on every pool string, and sort answers stay distinct-letter strings
    expect(POOL.every((s) => POOL.includes(TASK_FN.sort(s)))).toBe(true);
  });
  test('permutations', () => {
    const p = permutations(['a', 'b', 'c', 'd']);
    expect(p).toHaveLength(24);
    expect(new Set(p.map((x) => x.join(''))).size).toBe(24);
  });
});

describe('prompt builders', () => {
  test('tag, examples, delimiters, scratchpad', () => {
    expect(promptText({ task: 'rev', query: 'abc' })).toBe('abc>');
    expect(promptText({ task: 'rev', query: 'abc', tag: true, examples: ['def'] })).toBe('[rev] def>fed;abc>');
    expect(promptText({ task: 'rev', query: 'abc', examples: ['def'], delims: ['=', '|'] })).toBe('def=fed|abc=');
    expect(promptText({ task: 'sr', query: 'abc', examples: ['def'], scratchpad: true })).toBe('def>efg>gfe;abc>');
    expect(answerBody({ task: 'sr', query: 'abc', scratchpad: true })).toBe('bcd>dcb');
    expect(buildPrompt({ task: 'sort', query: 'cab', tag: true })).toEqual(encode('[sort] cab>'));
    expect(toolPrompt(37, 48, true)).toEqual(encode('[tools]37*48>'));
    expect(decode(toolSequence(37, 48))).toBe('[tools] 37*48> CALL calc(37*48) RESULT 1776>1776;');
  });
  test('questions are seeded 2-digit pairs', () => {
    expect(QUESTIONS).toHaveLength(50);
    expect(QUESTIONS.flat().every((n) => n >= 10 && n <= 99)).toBe(true);
  });
});

describe('masks and parsers', () => {
  test('format mask and parse check', () => {
    expect(formatMask([])).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(formatMask([0, 1, 2])).toEqual([ID.semi]);
    expect(parses(encode('cba;'))).toBe(true);
    expect(parses(encode('cb;'))).toBe(false);
    expect(parses(encode('cba>dcb;'))).toBe(false);
  });
  test('calc call validator', () => {
    expect(parseCall(encode('CALL calc(37*48)'))).toMatchObject({ ok: true, a: 37, b: 48, length: 8 });
    expect(parseCall(encode('1776;'))).toMatchObject({ ok: false, error: expect.stringContaining('answered directly') });
    expect(parseCall(encode('CALL 37*48)'))).toMatchObject({ ok: false, error: expect.stringContaining('unknown tool') });
    expect(parseCall(encode('CALL calc(37*48'))).toMatchObject({ ok: false, error: expect.stringContaining('not closed') });
    expect(parseCall(encode('CALL calc(37**8)'))).toMatchObject({ ok: false, error: expect.stringContaining('invalid expression') });
    expect(parseCall(encode('CALL calc(37a8)'))).toMatchObject({ ok: false });
    expect(parseAnswer(encode('>1776;'))).toBe(1776);
    expect(parseAnswer(encode('1776;'))).toBe(1776);
    expect(parseAnswer(encode('17a6;'))).toBeNull();
    expect(parseAnswer(encode('1776'))).toBeNull();
  });
  test('resampling counts sets where B looks worse', () => {
    const A = Array(100).fill(true), B = Array.from({ length: 100 }, (_, i) => i % 10 !== 0);
    const r = resample(A, B, [5, 50], 200, 1);
    expect(r.bySize[0].diffs).toHaveLength(200);
    expect(r.bySize[0].worse).toBeGreaterThan(0.3);
    expect(r.bySize[1].worse).toBe(1);
    expect(resample(A, B, [5], 200, 1)).toEqual(resample(A, B, [5], 200, 1));
  });
});
