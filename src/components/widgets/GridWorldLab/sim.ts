/**
 * GridWorldLab: the Russell and Norvig 4x3 world (src/lib/rl) on one scrubbable timeline, one mode per lesson step.
 *
 * MODE-AUTHOR CONTRACT (shared by GridWorldLab, BanditLab and PPOLab; types in ../_rl/kit.ts)
 * You own two files per mode and never edit the shell (GridWorldLab.svelte, ../_rl/*), this sim.ts, draw.ts or src/lib/rl:
 *   m-<mode>.meta.ts  `export default { title, legend, aspect?, narrowAspect?, controls?, lanes?, pickable?, loading? } satisfies ModeMeta;`
 *                     A plain literal (only `import type`): it is bundled with the shell so the server-rendered card reserves
 *                     its final height, legend and control rows before your chunk loads (CLS 0). Default aspect 720 / 420,
 *                     narrow (stage < 540 px) 360 / 560. The lesson can still override `title`, `aspect`, `narrowAspect`.
 *   m-<mode>.ts       `const mode: ModeImpl<Scene, F> = { setup, timeline, frameAt, draw, label, restTime? }; export default mode;`
 *     setup(o, kit)        build the scene from the props `o` (typed KnownOpts + any other prop; MDX may pass strings, so
 *                          parse with num/int/bool/list/nums/str from ../_rl/kit and parseCell from src/lib/rl). Re-run when a
 *                          control changes. Heavy work: `kit.memo(key(...), () => …)` shares one result per page across
 *                          islands; `await kit.job('trainDQN', grid, opts)` runs ANY src/lib/rl export in the shared worker
 *                          (memoised by name + args; args/results must be cloneable). Keep every main-thread task < ~50 ms.
 *     timeline(s, o)       { duration (s), markers: [{ at, label }] } — labels name SEGMENTS (sentence case, short); the
 *                          shell highlights the last marker passed and announces "label: status" (aria-live) on change.
 *     frameAt(t, s, o, ui) pure; returns { status, detail?, readouts?: [label, value][] , ...your fields }. `status` is the
 *                          narration sentence of what is drawn NOW; numbers switch at segment boundaries (only geometry eases).
 *                          ui.pick = `pick` id of the clicked shape (meta.pickable), or null.
 *     draw(E, f, s, o)     push shapes into E.out using ../_rl/svg.ts (rect/line/path/circle/poly/text/arrow, axes, lineChart,
 *                          bars, binCounts, table, tok/mix/signedFill/seqFill/inkOn, fmt/signFmt, ease/lerp/seg) and ./draw.ts
 *                          (gridIn, twoBoxes, cellXY, at, drawGrid, drawValues, drawQ, triangle, drawPolicy, drawAgent,
 *                          drawOutcomes). E.W x E.H is the stage in CSS px; E.narrow when W < 540: lay out for a phone
 *                          (stack two grids with twoBoxes). Text >= 11 px (text() enforces it). Token colours only.
 *     label(s, o)          the figure's aria-label sentence.        restTime?(s, o)  paused frame for reduced motion (default: end).
 *   Controls: declare them in the meta (`{ key: 'stepReward', label: 'Step reward', kind: 'range', min, max, step, digits }`,
 *   kind 'select' with options, or 'toggle'); the shell renders them under the figure, puts the value in o[key] and calls
 *   setup again; `then: 'restart' | 'end' | 'keep'` sets the playhead afterwards (default restart).
 *   The shell owns FigureCard, Timeline (play/pause, scrub, speed, loop), autoplay on view, pause off-screen / hidden tab,
 *   reduced motion (starts paused at restTime), theme (colours are CSS variables, so nothing to re-read), the loading text.
 *   Performance: hold runs in the scene (plain objects; the shell keeps it in $state.raw), never recompute training in
 *   frameAt/draw, keep a frame under ~200 shapes. Measured setup costs (Node, plans/day-11.md §6): value iteration 7 ms,
 *   Q-learning 300 episodes 5 ms, 20-run Q-learning curves 30–45 ms per epsilon (memo; consider kit.job), REINFORCE
 *   or actor-critic 10 runs x 1000 episodes 85–110 ms (kit.job), threshold scans 100–190 ms (kit.job gammaSwitches /
 *   rewardSwitches), DQN single 1000-update runs 60–80 ms (kit.job), DQN four variants x 10 runs: seconds (kit.job,
 *   show the loading text), PPO 100 iterations 66 ms (kit.job). Add a check in tests/unit/gridworld-lab.test.ts when your
 *   frame logic is non-trivial (import your m-<mode>.ts directly; it has no DOM).
 * Props every mode may read: seed, gamma, stepReward, slip, policy, focus ("x,y" 1-based, (1,1) bottom-left), action
 * (up/right/down/left), gammas, sweeps, alpha, epsilon(s), episodes, runs, transitions, lr, baseline, episode, lrActor,
 * lrCritic, hidden, input, replay, targetNetwork, comparison, lrs, updates, capacity, batch, targetEvery/copyEvery,
 * variants, lambda, epochs, minibatch, iterations — and any other prop the lesson passes.
 */
import { ACTIONS, makeGrid, parseCell, type Grid } from '../../../lib/rl/gridworld';
import { num, type Opts } from '../_rl/kit';

export type { Opts };
/** The grid from the common props: stepReward (default -0.04), slip (0.2), gamma (default `gamma`, the book's 1). */
export const gridOf = (o: Opts, gamma = 1): Grid => makeGrid({ stepReward: num(o.stepReward, -0.04), slip: num(o.slip, 0.2), gamma: num(o.gamma, gamma) });
/** The focus cell ("x,y") as a state index. */
export const focusOf = (o: Opts, fallback = 0): number => parseCell(o.focus, fallback);
/** The action prop (name or 0..3) as an index. */
export const actionOf = (o: Opts, fallback = 0): number => {
  const i = ACTIONS.indexOf(o.action as (typeof ACTIONS)[number]);
  return i >= 0 ? i : Number.isInteger(Number(o.action)) && o.action !== undefined ? Math.max(0, Math.min(3, Number(o.action))) : fallback;
};
