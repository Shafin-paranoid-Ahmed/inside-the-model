import { createRng, type Rng } from '../math/rng';

export const VOCAB = ['[start]', '.', 'this', 'that', 'a', 'these', 'those', 'the',
  'small', 'big', 'red', 'old', 'cat', 'cats', 'dog', 'dogs', 'fox', 'foxes',
  'bird', 'birds', 'sheep', 'apple', 'apples', 'seed', 'seeds', 'fish',
  'sees', 'see', 'chases', 'chase', 'eats', 'eat', 'likes', 'like'] as const;

type NumberClass = 'sg' | 'pl' | 'either';
export interface WordClass {
  kind: 'special' | 'det' | 'adj' | 'noun' | 'verb';
  number?: NumberClass;
  category?: 'animal' | 'food';
  object?: 'animal' | 'food' | 'any';
}
export const WORD_CLASS: readonly WordClass[] = VOCAB.map((word, id) => {
  if (id < 2) return { kind: 'special' };
  if (id < 8) return { kind: 'det', number: word === 'the' ? 'either' : id < 5 ? 'sg' : 'pl' };
  if (id < 12) return { kind: 'adj' };
  if (id < 26) return { kind: 'noun', category: id < 21 ? 'animal' : 'food',
    number: word === 'sheep' || word === 'fish' ? 'either' : word.endsWith('s') ? 'pl' : 'sg' };
  return { kind: 'verb', number: id % 2 === 0 ? 'sg' : 'pl',
    object: word.startsWith('chas') ? 'animal' : word.startsWith('eat') ? 'food' : 'any' };
});

const compatible = (a: NumberClass, b: NumberClass) => a === 'either' || b === 'either' || a === b;

/** Prefix grammar, including number agreement and verb/object selection.
 * Each NP allows 0–2 adjectives; at most three total keeps [start]…period within ten tokens.
 * Invalid prefixes throw instead of silently falling back to a permissive oracle.
 */
export function allowedNext(prefix: readonly number[]): Set<number> {
  let phase: 'start' | 'subject-det' | 'subject' | 'verb' | 'object-det' | 'object' | 'end' | 'done' = 'start';
  let number: NumberClass = 'either', subject: NumberClass = 'either';
  let object: 'animal' | 'food' | 'any' = 'any';
  let adjectives = 0, totalAdjectives = 0;
  const allowed = (): Set<number> => {
    const out = new Set<number>();
    for (let id = 0; id < VOCAB.length; id++) {
      const c = WORD_CLASS[id];
      if ((phase === 'start' && id === 0) ||
        ((phase === 'subject-det' || phase === 'object-det') && c.kind === 'det') ||
        ((phase === 'subject' || phase === 'object') && (
          (c.kind === 'adj' && adjectives < 2 && totalAdjectives < 3) ||
          (c.kind === 'noun' && compatible(number, c.number!) &&
            (phase === 'subject' ? c.category === 'animal' : object === 'any' || c.category === object)))) ||
        (phase === 'verb' && c.kind === 'verb' && compatible(subject, c.number!)) ||
        (phase === 'end' && id === 1)) out.add(id);
    }
    return out;
  };
  for (const id of prefix) {
    if (!allowed().has(id)) throw new Error(`Invalid grammar prefix at ${VOCAB[id] ?? id}`);
    const c = WORD_CLASS[id];
    switch (phase) {
      case 'start': phase = 'subject-det'; break;
      case 'subject-det': number = c.number!; adjectives = 0; phase = 'subject'; break;
      case 'subject':
        if (c.kind === 'adj') { adjectives++; totalAdjectives++; }
        else { subject = number === 'either' ? c.number! : number; phase = 'verb'; }
        break;
      case 'verb': object = c.object!; phase = 'object-det'; break;
      case 'object-det': number = c.number!; adjectives = 0; phase = 'object'; break;
      case 'object':
        if (c.kind === 'adj') { adjectives++; totalAdjectives++; }
        else phase = 'end';
        break;
      case 'end': phase = 'done'; break;
      default: throw new Error('Tokens after end of sentence');
    }
  }
  return allowed();
}

export function sampleSentence(rng: Rng): number[] {
  const ids: number[] = [];
  for (;;) {
    const legal = Array.from(allowedNext(ids));
    if (!legal.length) return ids;
    ids.push(legal[Math.floor(rng.next() * legal.length)]);
  }
}

export function encodeWords(text: string): number[] {
  const words = text.trim().replace(/\./g, ' . ').split(/\s+/).filter(Boolean);
  return words.map((word) => {
    const id = VOCAB.indexOf(word as typeof VOCAB[number]);
    if (id < 0) throw new Error(`Unknown word: ${word}`);
    return id;
  });
}
export const decodeIds = (ids: readonly number[]): string => ids.map((id) => {
  if (!VOCAB[id]) throw new Error(`Unknown token id: ${id}`);
  return VOCAB[id];
}).join(' ').replace(/ \./g, '.');

const evaluationRng = createRng(92026);
export const SAMPLE_200: readonly number[][] = Array.from({ length: 200 }, () => sampleSentence(evaluationRng));
