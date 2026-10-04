/**
 * The Russell and Norvig 4x3 grid world (AIMA Fig. 17.1). Pure TS, no DOM.
 *
 * Cells are written (x, y), 1-based, with (1, 1) bottom-left, as in the book. State index s = (y - 1) * 4 + (x - 1):
 *   y=3:  8  9 10 11(+1)        start (1,1) = 0, wall (2,2) = 5, pit (4,2) = 7, goal (4,3) = 11.
 *   y=2:  4 [5] 6  7(-1)
 *   y=1:  0  1  2  3
 * Actions 0 up, 1 right, 2 down, 3 left. A move goes the intended way with probability 1 - slip and each
 * perpendicular way with slip / 2; bumping the wall or the edge leaves the agent where it is.
 *
 * Reward convention: every move costs `stepReward`; a move INTO a terminal also pays that terminal's reward
 * (+1 goal, -1 pit), so the last move of a goal episode earns stepReward + 1 (0.96 by default). Terminals have
 * value 0 and no actions. At gamma = 1 this reproduces the book's utilities exactly (U = R(s) + sum P U(s'),
 * terminal U = +-1). At gamma < 1 the terminal reward is discounted like any reward of that move.
 */
import type { Rng } from '../math/rng';

export const COLS = 4, ROWS = 3, N = 12, NA = 4;
export const ACTIONS = ['up', 'right', 'down', 'left'] as const;
export const ARROWS = ['↑', '→', '↓', '←'] as const;
export const DX = [0, 1, 0, -1] as const, DY = [1, 0, -1, 0] as const;
export const START = 0, WALL = 5, PIT = 7, GOAL = 11;

export const idx = (x: number, y: number): number => (y - 1) * COLS + (x - 1);
export const xy = (s: number): [number, number] => [(s % COLS) + 1, Math.floor(s / COLS) + 1];
/** "x,y" (1-based, as in the book) or [x, y] → state index; `fallback` when it is missing or not a usable cell. */
export function parseCell(v: unknown, fallback: number): number {
  const p = Array.isArray(v) ? v.map(Number) : typeof v === 'string' ? v.split(/[,\s]+/).map(Number) : [];
  if (p.length !== 2 || !p.every(Number.isInteger)) return fallback;
  const [x, y] = p;
  return x >= 1 && x <= COLS && y >= 1 && y <= ROWS ? idx(x, y) : fallback;
}
export const cellName = (s: number): string => `(${xy(s).join(',')})`;

export interface GridOpts {
  /** Reward of every move (default -0.04). */
  stepReward?: number;
  /** Probability the move goes sideways, split evenly between the two perpendicular directions (default 0.2). */
  slip?: number;
  /** Discount factor (default 1, the book's setting). */
  gamma?: number;
  goalReward?: number;
  pitReward?: number;
}
export interface Grid {
  stepReward: number; slip: number; gamma: number;
  /** Terminal reward per state (0 for non-terminals). */
  terminalReward: Float64Array;
  /** Non-terminal, non-wall states: the ones with actions. */
  states: number[];
}
/** One way a move can turn out. `dir` is the direction actually taken; `moved` is false after a bump. */
export interface Outcome { s2: number; p: number; r: number; dir: number; moved: boolean; done: boolean }

export function makeGrid(o: GridOpts = {}): Grid {
  const terminalReward = new Float64Array(N);
  terminalReward[GOAL] = o.goalReward ?? 1;
  terminalReward[PIT] = o.pitReward ?? -1;
  const states: number[] = [];
  for (let s = 0; s < N; s++) if (s !== WALL && s !== GOAL && s !== PIT) states.push(s);
  return { stepReward: o.stepReward ?? -0.04, slip: o.slip ?? 0.2, gamma: o.gamma ?? 1, terminalReward, states };
}

export const isTerminal = (s: number): boolean => s === GOAL || s === PIT;
export const isWall = (s: number): boolean => s === WALL;

/** Where a deterministic move in direction `dir` lands (stays put at the edge or the wall). */
export function move(s: number, dir: number): number {
  const [x, y] = xy(s), nx = x + DX[dir], ny = y + DY[dir];
  if (nx < 1 || nx > COLS || ny < 1 || ny > ROWS) return s;
  const s2 = idx(nx, ny);
  return s2 === WALL ? s : s2;
}

/** The three outcomes of action `a` in state `s`: intended, then its left and right turns (zero-probability ones dropped). */
export function transitions(g: Grid, s: number, a: number): Outcome[] {
  if (isTerminal(s) || isWall(s)) return [];
  const out: Outcome[] = [];
  for (const [dir, p] of [[a, 1 - g.slip], [(a + 3) % 4, g.slip / 2], [(a + 1) % 4, g.slip / 2]] as const) {
    if (p <= 0) continue;
    const s2 = move(s, dir), done = isTerminal(s2);
    out.push({ s2, p, r: g.stepReward + (done ? g.terminalReward[s2] : 0), dir, moved: s2 !== s, done });
  }
  return out;
}

export interface StepResult { s2: number; r: number; done: boolean; dir: number; slipped: boolean }
/** Sample one move. Uses exactly one rng draw. */
export function step(g: Grid, s: number, a: number, rng: Rng): StepResult {
  const u = rng.next(), side = g.slip / 2;
  const dir = u < 1 - g.slip ? a : u < 1 - side ? (a + 3) % 4 : (a + 1) % 4;
  const s2 = move(s, dir), done = isTerminal(s2);
  return { s2, r: g.stepReward + (done ? g.terminalReward[s2] : 0), done, dir, slipped: dir !== a };
}

export interface Episode {
  /** States visited: s_0 .. s_T (length T + 1). */
  states: number[];
  actions: number[];
  rewards: number[];
  /** Direction actually moved per step, and whether it differed from the action. */
  dirs: number[];
  slipped: boolean[];
  /** Ended in a terminal (true) or was cut off at maxSteps (false). */
  terminated: boolean;
}
export type Policy = (s: number, rng: Rng) => number;

/** Run one episode from `start` (default the start cell). */
export function rollout(g: Grid, policy: Policy, rng: Rng, maxSteps = 100, start = START): Episode {
  const ep: Episode = { states: [start], actions: [], rewards: [], dirs: [], slipped: [], terminated: false };
  let s = start;
  for (let t = 0; t < maxSteps; t++) {
    const a = policy(s, rng), r = step(g, s, a, rng);
    ep.actions.push(a); ep.rewards.push(r.r); ep.dirs.push(r.dir); ep.slipped.push(r.slipped); ep.states.push(r.s2);
    s = r.s2;
    if (r.done) { ep.terminated = true; break; }
  }
  return ep;
}

/** Uniformly random actions. */
export const randomPolicy: Policy = (_s, rng) => Math.floor(rng.next() * NA);
/** A deterministic policy from one action per state. */
export const fixedPolicy = (actions: ArrayLike<number>): Policy => (s) => actions[s];
/** Sample from per-state action probabilities (N x 4, row-major). */
export const stochasticPolicy = (probs: ArrayLike<number>): Policy => (s, rng) => sampleRow(probs, s, rng.next());
export function sampleRow(probs: ArrayLike<number>, s: number, u: number): number {
  let c = 0;
  for (let a = 0; a < NA; a++) { c += probs[s * NA + a]; if (u < c) return a; }
  return NA - 1;
}

/** Discounted returns G_t = r_{t+1} + gamma G_{t+1}, computed backward. */
export function returns(rewards: ArrayLike<number>, gamma: number): Float64Array {
  const G = new Float64Array(rewards.length);
  let acc = 0;
  for (let t = rewards.length - 1; t >= 0; t--) { acc = rewards[t] + gamma * acc; G[t] = acc; }
  return G;
}

/** Normalised coordinates in [-1, 1] (x: columns 1..4, y: rows 1..3), the DQN input. */
export const coords = (s: number): [number, number] => { const [x, y] = xy(s); return [(x - 2.5) / 1.5, y - 2]; };
