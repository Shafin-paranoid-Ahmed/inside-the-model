/**
 * The mode contract shared by GridWorldLab, BanditLab and PPOLab (Day 11). Each widget's sim.ts documents it for
 * mode authors; this file holds the types, the prop parsers and the per-page memo/worker helpers. No DOM.
 */
import type { Marker } from '../_shared/clock';
import type { Env } from './svg';

/** Every prop the Day 11 lesson briefs name. All optional; MDX may pass strings, so parse with the helpers below. */
export interface KnownOpts {
  seed?: number | string;
  // environment
  gamma?: number | string; stepReward?: number | string; slip?: number | string;
  policy?: string; focus?: string; action?: string;
  gammas?: string | number[];
  // tabular learning
  sweeps?: number | string; alpha?: number | string; epsilon?: number | string; epsilons?: string | number[];
  episodes?: number | string; runs?: number | string; transitions?: string;
  // bandit
  arms?: number | string; ucb?: boolean | string; steps?: number | string; samples?: number | string; means?: string | number[]; c?: number | string;
  // policy gradients
  lr?: number | string; baseline?: string; episode?: number | string; lrActor?: number | string; lrCritic?: number | string;
  // DQN
  hidden?: number | string; input?: string; replay?: boolean | string; targetNetwork?: boolean | string; comparison?: string;
  lrs?: string | number[]; updates?: number | string; capacity?: number | string; batch?: number | string;
  targetEvery?: number | string; copyEvery?: number | string; variants?: string | string[];
  // PPO
  lambda?: number | string; epochs?: number | string; minibatch?: number | string; iterations?: number | string;
  advantage?: number | string; ratios?: string | number[];
}
/** What a mode receives: the lesson's props (known ones typed, any other prop passed through) plus control values. */
export type Opts = KnownOpts & { mode: string; [key: string]: unknown };
/** Props of GridWorldLab, BanditLab and PPOLab. */
export type WidgetProps = KnownOpts & { mode?: string; title?: string; aspect?: string; narrowAspect?: string; [key: string]: unknown };

// ---------- prop parsers (MDX may pass "0.5,0.99", [0.5, 0.99], "false" or false) ----------

export const num = (v: unknown, d: number): number => { const x = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN; return Number.isFinite(x) ? x : d; };
export const int = (v: unknown, d: number): number => Math.round(num(v, d));
export const bool = (v: unknown, d: boolean): boolean => (typeof v === 'boolean' ? v : v === 'true' ? true : v === 'false' ? false : d);
export const list = (v: unknown, d: string[]): string[] => (Array.isArray(v) ? v.map(String) : typeof v === 'string' && v.trim() ? v.split(',').map((x) => x.trim()) : d);
export const nums = (v: unknown, d: number[]): number[] => { const l = list(v, []).map(Number); return l.length && l.every(Number.isFinite) ? l : d; };
export const str = <T extends string>(v: unknown, allowed: readonly T[], d: T): T => (allowed.includes(v as T) ? (v as T) : d);
/** A stable cache key from any JSON-able parts. */
export const key = (...parts: unknown[]): string => JSON.stringify(parts);

// ---------- the mode contract ----------

export interface Frame {
  /** The narration sentence for what is visible now (body font, under the figure). Must describe the drawn state. */
  status: string;
  /** Optional second, muted line. */
  detail?: string;
  /** Small label/value pairs shown under the narration, e.g. [['Total reward', '−0.44']]. */
  readouts?: [string, string][];
}
export interface Timeline { duration: number; markers: Marker[] }
/** Interaction state passed to frameAt: the `pick` id of the last clicked shape (null when none). */
export interface UI { pick: string | null }
export interface Kit {
  /** Compute once per page: every island asking for the same key shares the value. */
  memo<T>(k: string, f: () => T): T;
  /** Run an exported function of src/lib/rl or src/lib/align in the shared worker (memoised per page by name + args). Args and result must be cloneable. */
  job<T>(fn: string, ...args: unknown[]): Promise<T>;
}
export interface ModeImpl<S, F extends Frame> {
  /** Build the scene from the props (and control values). May be async (worker jobs). Re-run when a control changes. */
  setup(o: Opts, kit: Kit): S | Promise<S>;
  timeline(s: S, o: Opts): Timeline;
  /** Pure: everything draw() needs at playhead t. */
  frameAt(t: number, s: S, o: Opts, ui: UI): F;
  /** Push shapes into E (helpers in ../_rl/svg.ts and the widget's draw.ts). Never touches the DOM. */
  draw(E: Env, f: F, s: S, o: Opts): void;
  /** aria-label of the figure (a full sentence describing what it shows). */
  label(s: S, o: Opts): string;
  /** Paused playhead for reduced motion (default: the end of the timeline). */
  restTime?(s: S, o: Opts): number;
}

export interface LegendItem { label: string; color: string; shape?: 'dot' | 'line' }
export type Control =
  | { key: string; label: string; kind: 'range'; min: number; max: number; step: number; digits?: number; unit?: string; then?: 'restart' | 'end' | 'keep' }
  | { key: string; label: string; kind: 'select'; options: { value: string | number; label: string }[]; then?: 'restart' | 'end' | 'keep' }
  | { key: string; label: string; kind: 'toggle'; then?: 'restart' | 'end' | 'keep' };
/**
 * Static facts about a mode, bundled with the shell so the server-rendered figure reserves its final size
 * (CLS 0) before the mode's chunk loads. A plain literal: no runtime imports (type imports are fine).
 */
export interface ModeMeta {
  title: string;
  legend: LegendItem[];
  /** Stage aspect ratio, CSS syntax (default from the widget). Narrow = stage under 540 px. */
  aspect?: string; narrowAspect?: string;
  controls?: Control[];
  /** Timeline marker label rows (1 to 3) when markers sit close together. */
  lanes?: 1 | 2 | 3;
  /** The stage has clickable shapes (shapes with `pick`): the figure becomes role="group". */
  pickable?: boolean;
  /** Quiet text shown until the scene is ready, e.g. 'Training 10 runs in your browser…'. */
  loading?: string;
}

// ---------- per-page memo and the worker ----------

const cache = new Map<string, unknown>();
let worker: Worker | null = null, nextId = 0;
const pending = new Map<number, { ok: (v: unknown) => void; fail: (e: unknown) => void }>();

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<{ id: number; result?: unknown; error?: string }>) => {
      const p = pending.get(e.data.id);
      pending.delete(e.data.id);
      if (p) { if (e.data.error) p.fail(new Error(e.data.error)); else p.ok(e.data.result); }
    };
    worker.onerror = () => { for (const p of pending.values()) p.fail(new Error('worker failed')); pending.clear(); };
  }
  return worker;
}

export const kit: Kit = {
  memo<T>(k: string, f: () => T): T {
    if (!cache.has(k)) cache.set(k, f());
    return cache.get(k) as T;
  },
  job<T>(fn: string, ...args: unknown[]): Promise<T> {
    return kit.memo(key('job', fn, args), () => {
      const p = new Promise<T>((ok, fail) => {
        const id = ++nextId;
        pending.set(id, { ok: ok as (v: unknown) => void, fail });
        getWorker().postMessage({ id, fn, args });
      });
      p.catch(() => cache.delete(key('job', fn, args))); // a failed job may be retried
      return p;
    });
  },
};
