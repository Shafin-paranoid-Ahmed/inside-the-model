/** House-model character vocabulary: a–z, space, newline and ten punctuation marks, plus one unknown id. */
export const CHARSET = 'abcdefghijklmnopqrstuvwxyz \n.,;:\'!?-"';
export const UNKNOWN = CHARSET.length;
/** Model vocabulary (config.vocab): one entry per id; the last is the reserved unknown token. */
export const VOCAB: readonly string[] = [...CHARSET, '<unk>'];
const INDEX = new Map([...CHARSET].map((c, i) => [c, i]));

/** Lowercases, then maps every character outside CHARSET to the unknown id. */
export const encode = (text: string): number[] => [...text.toLowerCase()].map((c) => INDEX.get(c) ?? UNKNOWN);
/** The unknown id decodes to U+FFFD, which is not in CHARSET, so it never looks like real text. */
export const decode = (ids: readonly number[]): string => ids.map((id) => CHARSET[id] ?? '�').join('');
