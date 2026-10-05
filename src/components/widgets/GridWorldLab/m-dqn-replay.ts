/**
 * dqn-replay (deep-rl step 4): a random-walk stream (seed 19) fills a 256-slot FIFO buffer; at three moments a batch of 16
 * is sampled uniformly and compared with the 16 newest transitions in a row (what training without replay would see).
 * Measured (seed 19): batches after 256 / 340 / 420 steps touch 6 / 6 / 7 cells against 4 / 4 / 4; consecutive-column
 * correlation 0.71 in a row against 0.04 sampled (steps 256-999).
 * Props: seed (19), capacity (256), batch (16).
 */
import { createRng } from '../../../lib/math/rng';
import { WALL, isTerminal, xy } from '../../../lib/rl/gridworld';
import { ReplayBuffer, randomStream, type Transition } from '../../../lib/rl/dqn';
import { int, type Frame, type ModeImpl, type Opts } from '../_rl/kit';
import { ACCENT, MUTED, RULE, TEXT, BG, fmt, lerp, rect, seqFill, text, type Env } from '../_rl/svg';
import { cellXY, drawGrid, gridIn } from './draw';
import { dqnGrid } from './dqn-common';

const FILL = 5, HOLD = 3, SWEEP = 1, BEATS = [256, 340, 420];
interface Beat { tau: number; slots: number[]; ages: number[]; replay: Int32Array; recent: Int32Array; cells: number[]; seqCells: number }
interface Scene { stream: Transition[]; cap: number; batch: number; beats: Beat[]; corr: [number, number] }
interface F extends Frame { tau: number; beat: number | null; ages: Int32Array }

const corr = (xs: number[]) => {
  const a = xs.slice(0, -1), b = xs.slice(1), m = (v: number[]) => v.reduce((p, c) => p + c, 0) / v.length, ma = m(a), mb = m(b);
  let n = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) { n += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2; }
  return n / Math.sqrt(da * db);
};
/** Age (steps since collected) of every slot after tau transitions; -1 for an empty slot. */
const ringAges = (tau: number, cap: number) => {
  const out = new Int32Array(cap).fill(-1);
  for (let j = 0; j < cap && j < tau; j++) out[j] = tau - 1 - (j + cap * Math.floor((tau - 1 - j) / cap));
  return out;
};
const counts = (states: ArrayLike<number>) => { const c = new Int32Array(12); for (let i = 0; i < states.length; i++) c[states[i]]++; return c; };

const mode: ModeImpl<Scene, F> = {
  setup(o: Opts) {
    const g = dqnGrid(o), seed = int(o.seed, 19), cap = int(o.capacity, 256), B = int(o.batch, 16), stream = randomStream(g, 1000, seed);
    const beats = BEATS.map((tau) => {
      const buf = new ReplayBuffer(cap);
      for (let i = 0; i < tau; i++) buf.push(stream[i], i);
      const slots = Array.from(buf.sample(B, createRng(100 + tau))), ages = slots.map((j) => tau - 1 - buf.born[j]);
      const replay = Int32Array.from(slots, (j) => buf.s[j]), recent = Int32Array.from(stream.slice(tau - B, tau), (t) => t.s);
      return { tau, slots, ages, replay, recent, cells: Array.from(counts(replay)), seqCells: counts(recent).filter((x) => x > 0).length };
    });
    const buf = new ReplayBuffer(cap), rng = createRng(seed), col = (s: number) => xy(s)[0], samp: number[] = [];
    for (let i = 0; i < stream.length; i++) { buf.push(stream[i], i); if (i >= cap) samp.push(col(buf.s[buf.sample(1, rng)[0]])); }
    return { stream, cap, batch: B, beats, corr: [corr(stream.slice(cap).map((t) => col(t.s))), corr(samp)] };
  },
  timeline() {
    const m = [{ at: 0, label: 'Filling' }];
    BEATS.forEach((_, i) => { const t0 = FILL + i * (HOLD + SWEEP); m.push({ at: t0, label: `Batch ${i + 1}` }); if (i < BEATS.length - 1) m.push({ at: t0 + HOLD, label: 'New data' }); });
    return { duration: FILL + BEATS.length * HOLD + (BEATS.length - 1) * SWEEP + 0.8, markers: m };
  },
  frameAt(t, s) {
    const cap = s.cap;
    let tau: number, beat: number | null = null;
    if (t < FILL) tau = Math.round((t / FILL) * BEATS[0]);
    else {
      const k = Math.min(BEATS.length - 1, Math.floor((t - FILL) / (HOLD + SWEEP))), u = t - FILL - k * (HOLD + SWEEP);
      if (u < HOLD || k === BEATS.length - 1) { beat = k; tau = BEATS[k]; } else tau = Math.round(lerp(BEATS[k], BEATS[k + 1], (u - HOLD) / SWEEP));
    }
    const ages = ringAges(tau, cap), b = beat === null ? null : s.beats[beat], held = Math.min(tau, cap);
    let status: string;
    if (b) {
      const sc = b.cells.filter((x) => x > 0).length;
      status = `A batch of ${s.batch} is drawn at random from the buffer: its transitions are ${Math.min(...b.ages)} to ${Math.max(...b.ages)} steps old and come from ${sc} different cells. The ${s.batch} newest transitions in a row come from only ${b.seqCells}.`;
    } else if (beat === null && tau < BEATS[0]) status = `The agent wanders and stores every step it takes. ${tau} of ${cap} slots are filled.`;
    else status = 'The agent keeps moving. Each new transition overwrites the oldest slot, so the buffer always holds the latest 256 steps.';
    return {
      tau, beat, ages, status,
      readouts: b
        ? [['Steps taken', String(tau)], ['Ages in batch', `${Math.min(...b.ages)} to ${Math.max(...b.ages)}`], ['Cells in batch', `${b.cells.filter((x) => x > 0).length} (newest ${s.batch}: ${b.seqCells})`], ['Neighbour correlation', `${fmt(s.corr[0])} in a row, ${fmt(s.corr[1])} sampled`]]
        : [['Steps taken', String(tau)], ['In buffer', `${held} of ${cap}`]],
    };
  },
  draw(E: Env, f, s) {
    const nar = E.narrow, cap = s.cap, side = Math.round(Math.sqrt(cap));
    const rs = nar ? Math.min(E.W - 24, 190) : Math.min(E.H - 56, E.W * 0.36), cs = Math.floor(rs / side), rx = nar ? (E.W - cs * side) / 2 : 14, ry = nar ? 26 : 34;
    text(E, `Replay buffer: ${cap} slots, oldest overwritten first`, nar ? 8 : 14, 16, { fill: MUTED });
    for (let j = 0; j < cap; j++) {
      const age = f.ages[j], x = rx + (j % side) * cs, y = ry + Math.floor(j / side) * cs;
      rect(E, x, y, cs, cs, { fill: age < 0 ? BG : seqFill(1 - age / (cap - 1)), stroke: RULE, sw: 0.5 });
    }
    const b = f.beat === null ? null : s.beats[f.beat];
    if (f.tau >= cap) { const h = f.tau % cap; rect(E, rx + (h % side) * cs, ry + Math.floor(h / side) * cs, cs, cs, { fill: 'none', stroke: TEXT, sw: 2 }); }
    if (b) {
      for (let k = 1; k <= s.batch; k++) { const j = (b.tau - k) % cap; rect(E, rx + (j % side) * cs, ry + Math.floor(j / side) * cs, cs, cs, { fill: 'none', stroke: 'var(--viz-forward)', sw: 1.5 }); }
      for (const j of b.slots) rect(E, rx + (j % side) * cs + 0.5, ry + Math.floor(j / side) * cs + 0.5, cs - 1, cs - 1, { fill: 'none', stroke: ACCENT, sw: 2.2 });
    }
    // two small grids: how often each cell appears in the newest-in-a-row batch and in the sampled batch
    const area = nar ? { x: 8, y: ry + cs * side + 24, w: E.W - 16, h: E.H - (ry + cs * side + 24) - 8 } : { x: rx + cs * side + 30, y: 34, w: E.W - (rx + cs * side + 30) - 12, h: E.H - 48 };
    const half = (area.w - 14) / 2, boxes = [{ x: area.x, y: area.y + 26, w: half, h: nar ? half * 0.75 : area.h - 52 }, { x: area.x + half + 14, y: area.y + 26, w: half, h: nar ? half * 0.75 : area.h - 52 }];
    const names = [`Newest ${s.batch} in a row`, `${s.batch} sampled at random`], cnt = b ? [counts(b.recent), counts(b.replay)] : null;
    boxes.forEach((bx, i) => {
      const G = gridIn(bx);
      text(E, names[i], bx.x, area.y + 12, { fill: i ? ACCENT : 'var(--viz-forward)', weight: 600, size: 12 });
      drawGrid(E, G, { labels: false, fill: (c) => (cnt && cnt[i][c] > 0 ? seqFill(Math.min(1, cnt[i][c] / 5)) : undefined) });
      if (!cnt) return;
      for (let c = 0; c < 12; c++) if (cnt[i][c] > 0 && !isTerminal(c) && c !== WALL) { const p = cellXY(G, c); text(E, String(cnt[i][c]), p.cx, p.cy + 4, { anchor: 'middle', weight: 600, size: 12, fill: cnt[i][c] >= 4 ? BG : TEXT }); }
      const used = cnt[i].filter((x) => x > 0).length;
      text(E, `${used} different cell${used === 1 ? '' : 's'}`, bx.x, G.y0 + G.h + 18, { fill: MUTED, size: 12 });
    });
    if (!b) text(E, 'A batch appears once the buffer is full.', area.x, area.y + 36, { fill: MUTED });
  },
  label: (s) => `A ${s.cap}-slot replay buffer drawn as a square coloured by age, with the ${s.batch} newest transitions and ${s.batch} randomly sampled ones marked, and two small grids counting which cells each batch visits.`,
};
export default mode;
