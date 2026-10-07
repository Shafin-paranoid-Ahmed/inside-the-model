/**
 * Constitutional AI on the candidate bank: principles are checkable predicates on a response's features, a critique
 * samples one principle per round and a revision moves to the candidate that fixes it with the fewest other changes.
 * The AI labeler (reward.ts 'ai') prefers the response with fewer violations. In Bai et al. a language model reads
 * the principles; here this small rule checker plays that role so every judgement is visible. Pure TS.
 */
import { createRng, type Rng } from '../math/rng';
import { F_ANSWERS, F_CORRECT, F_FLATTERY, F_HEDGES, F_LENGTH, F_REFUSES, K, TRAIN, phi } from './bank';
import { sample, type Theta } from './policy';
import { trainSFT, type Demo } from './sft';

export interface Principle { id: string; text: string; violated: (f: ArrayLike<number>) => boolean }
export const PRINCIPLES: Principle[] = [
  { id: 'answer', text: 'Answer the question that was asked', violated: (f) => f[F_ANSWERS] === 0 },
  { id: 'accurate', text: 'Be accurate', violated: (f) => f[F_ANSWERS] === 1 && f[F_CORRECT] === 0 },
  { id: 'no-flattery', text: 'Do not pad the answer with flattery', violated: (f) => f[F_FLATTERY] === 1 },
  { id: 'no-refuse', text: 'Do not refuse a harmless request', violated: (f) => f[F_REFUSES] === 1 },
  { id: 'cautious', text: 'Avoid any answer that could be wrong', violated: (f) => f[F_ANSWERS] === 1 && f[F_HEDGES] === 0 },
];
/** The constitution used unless a lesson step edits it (the over-cautious principle is off). */
export const DEFAULT_PRINCIPLES = ['answer', 'accurate', 'no-flattery', 'no-refuse'];
const byId = (id: string): Principle => { const p = PRINCIPLES.find((q) => q.id === id); if (!p) throw new Error(`unknown principle ${id}`); return p; };

/** Ids of the active principles candidate y of prompt x violates. */
export const violations = (x: number, y: number, active: string[] = DEFAULT_PRINCIPLES): string[] => active.filter((id) => byId(id).violated(phi(x, y)));
export const violationCount = (x: number, y: number, active: string[] = DEFAULT_PRINCIPLES): number => violations(x, y, active).length;

/** Binary-feature differences between two candidates (answers, correct, flattery, refuses, hedges). */
const changes = (x: number, a: number, b: number): number => {
  const fa = phi(x, a), fb = phi(x, b);
  return [F_ANSWERS, F_CORRECT, F_FLATTERY, F_REFUSES, F_HEDGES].reduce((s, j) => s + (fa[j] !== fb[j] ? 1 : 0), 0);
};
/**
 * Revision for principle `id`: unchanged if y does not violate it, else the candidate that satisfies it with the fewest
 * other binary-feature changes (ties: closest length, then lowest index).
 */
export function revise(x: number, y: number, id: string): number {
  const pr = byId(id);
  if (!pr.violated(phi(x, y))) return y;
  let best = y, bc = Infinity, bl = Infinity;
  for (let z = 0; z < K; z++) {
    if (pr.violated(phi(x, z))) continue;
    const c = changes(x, y, z), l = Math.abs(phi(x, z)[F_LENGTH] - phi(x, y)[F_LENGTH]);
    if (c < bc || (c === bc && l < bl - 1e-12)) { best = z; bc = c; bl = l; }
  }
  return best;
}
export interface Round { principle: string; violated: boolean; from: number; to: number; before: number; after: number }
/** `rounds` critique-revise rounds from response y0, sampling one active principle uniformly per round. */
export function critiqueRevise(x: number, y0: number, rounds: number, rng: Rng, active: string[] = DEFAULT_PRINCIPLES): Round[] {
  const out: Round[] = [];
  let y = y0;
  for (let r = 0; r < rounds; r++) {
    const id = active[Math.floor(rng.next() * active.length)], to = revise(x, y, id);
    out.push({ principle: id, violated: to !== y, from: y, to, before: violationCount(x, y, active), after: violationCount(x, to, active) });
    y = to;
  }
  return out;
}
/** The AI judge: fewer violations wins; ties are broken by a seeded fair coin. True when a is preferred. */
export function aiPrefersA(x: number, a: number, b: number, rng: Rng, active: string[] = DEFAULT_PRINCIPLES): boolean {
  const va = violationCount(x, a, active), vb = violationCount(x, b, active);
  return va !== vb ? va < vb : rng.next() < 0.5;
}

export interface SLCAIResult {
  /** Each sampled start: prompt, the response the policy wrote, the critique-revise rounds and the final revision. */
  starts: { x: number; y0: number; rounds: Round[]; y: number }[];
  /** The revisions as an SFT dataset, and the SFT run on it from theta0. */
  demos: Demo[]; run: ReturnType<typeof trainSFT>;
}
/**
 * SL-CAI: sample n responses from `from` (default theta0; round robin over the train prompts), revise each for `rounds`
 * rounds, then fine-tune theta0 on the revisions (Adam, full batch).
 */
export function slCAI(theta0: Theta, o: { n?: number; rounds?: number; seed?: number; steps?: number; lr?: number; active?: string[]; from?: Theta } = {}): SLCAIResult {
  const n = o.n ?? 64, rounds = o.rounds ?? 4, rng = createRng(o.seed ?? 3), active = o.active ?? DEFAULT_PRINCIPLES;
  const starts = Array.from({ length: n }, (_, i) => {
    const x = TRAIN[i % TRAIN.length], y0 = sample(o.from ?? theta0, x, rng), rs = critiqueRevise(x, y0, rounds, rng, active);
    return { x, y0, rounds: rs, y: rs.length ? rs[rs.length - 1].to : y0 };
  });
  const demos = starts.map((s) => ({ x: s.x, y: s.y }));
  return { starts, demos, run: trainSFT(demos, { steps: o.steps ?? 150, lr: o.lr ?? 0.05, theta0 }) };
}
