<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, onThemeChange, prefersReducedMotion, readColors, readVar } from '../_shared';
  import { loadModel } from '../../../lib/transformer';
  import type { Weights } from '../../../lib/transformer';
  import type { FromWorker } from './worker';
  import {
    BEAT, GREEDY_N, LAB_BEAT, LAB_DEFAULTS, LAB_N, PROMPTS, SEED, SHIFT_MAX, TEXT_N, TEMPS,
    cleanPromptSettings, durationOf, emptyTemp, glyph, greedyFrame, kAt, labFrame, labText, logitsFrame, makePair,
    markersOf, nextOf, fmt, pct, restTime, tempFrame, topkFrame, toppFrame,
    type GreedyData, type LabData, type LabSettings, type Marker, type Mode, type Next, type PairData, type Panel, type PPanel, type TempData,
  } from './sim';

  interface Props {
    /** logits: scores to probabilities. greedy: greedy against sampled text. temperature. topk and topp: two contexts. lab: every control at once. */
    mode?: Mode;
    title?: string;
  }
  let { mode = 'logits', title = '' }: Props = $props();
  const init = untrack(() => ({
    mode,
    title: title || { logits: 'From scores to probabilities', greedy: 'Greedy and sampled text, one character at a time', temperature: 'Temperature reshapes the distribution', topk: 'Top-k keeps the k most likely characters', topp: 'Top-p keeps the smallest set that adds up to p', lab: 'Generate text and change the settings' }[mode],
  }));
  const M = init.mode;

  // Reserved stage size (logical units) for each mode: [wide, narrow].
  const HEIGHTS: Record<Mode, [number, number]> = { logits: [340, 340], greedy: [280, 400], temperature: [332, 336], topk: [300, 470], topp: [320, 560], lab: [214, 250] };
  const LEGENDS: Record<Mode, { label: string; color: string; shape?: 'dot' | 'line' }[]> = {
    logits: [{ label: 'positive score', color: '--viz-pos' }, { label: 'negative score', color: '--viz-neg' }, { label: 'positive number', color: '--viz-forward' }],
    greedy: [{ label: 'probability stretch', color: '--viz-seq-3' }, { label: 'chosen character', color: '--accent' }, { label: 'copied text', color: '--accent', shape: 'line' }],
    temperature: [{ label: 'probability', color: '--viz-forward' }, { label: 'uniform: 1 in 38', color: '--text-muted', shape: 'line' }],
    topk: [{ label: 'kept', color: '--viz-forward' }, { label: 'before cutting', color: '--text-muted', shape: 'line' }],
    topp: [{ label: 'kept', color: '--viz-forward' }, { label: 'cut', color: '--text-muted', shape: 'line' }, { label: 'running total', color: '--accent', shape: 'line' }],
    lab: [{ label: 'shade: faint = a surprise, strong = expected', color: '--viz-seq-4' }, { label: 'chosen', color: '--accent' }],
  };

  let w: Weights | null = null;
  let ready = $state(false), failed = $state(false);
  let playing = $state(false), speed = $state(1), uiT = $state(0), narrow = $state(false);
  let narration = $state(''), extra = $state(''), phaseName = $state('');
  let kNow = $state(5), pNow = $state(0.9);
  let markers = $state.raw<Marker[]>(markersOf(M));
  // controls
  let shift = $state(0), seed = $state(SEED);
  let manualK = $state<number | null>(null), manualP = $state<number | null>(null);
  let lab = $state<LabSettings>({ prompt: PROMPTS.main, ...LAB_DEFAULTS });
  const duration = durationOf(M);

  // model-derived data (plain variables: only the loop reads them)
  let nx: Next | null = null, pair: PairData | null = null, greedy: GreedyData | null = null, temp: TempData | null = null, labData: LabData | null = null;
  // Long generations run in a worker; the page itself only needs the model for single-step distributions.
  const needsModel = M !== 'greedy' && M !== 'lab';
  let worker: Worker | null = null, reqId = 0, setReady = () => {};
  const ask = (msg: { kind: 'greedy' | 'temp'; seed: number } | { kind: 'lab'; settings: LabSettings }) => worker?.postMessage({ id: ++reqId, ...msg });
  const build = () => {
    if (M === 'greedy') { greedy = null; ready = false; ask({ kind: 'greedy', seed }); return; }
    if (M === 'lab') { const settings = cleanPromptSettings(lab); labData = { settings, steps: [] }; ask({ kind: 'lab', settings }); return; }
    if (!w) return;
    if (M === 'logits') nx = nextOf(w, PROMPTS.main);
    else if (M === 'temperature') { temp = emptyTemp(nextOf(w, PROMPTS.main), seed); ask({ kind: 'temp', seed }); }
    else pair = makePair(w);
  };

  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let t = 0, poke = () => {};
  $effect(() => { playing; speed; if (playing) { manualK = null; manualP = null; } poke(); });
  const seek = (to: number) => { t = to; manualK = null; manualP = null; uiT = to; poke(); };
  const rebuild = () => { t = 0; build(); uiT = 0; if (!matchMedia('(prefers-reduced-motion: reduce)').matches) playing = true; poke(); };
  const newSeed = () => { seed = (seed % 99) + 1; if (M === 'lab') lab = { ...lab, seed }; rebuild(); };
  const setK = (v: number) => { manualK = v; playing = false; poke(); };
  const setP = (v: number) => { manualP = v; playing = false; poke(); };
  const onLab = () => { lab = cleanPromptSettings(lab); rebuild(); };
  const setLab = (patch: Partial<LabSettings>) => { lab = { ...lab, ...patch }; if (patch.seed !== undefined) seed = patch.seed; rebuild(); };

  onMount(() => {
    const reduced = prefersReducedMotion();
    const ctx = cv.getContext('2d')!;
    const C = { text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', sunken: '--bg-sunken', accent: '--accent', fwd: '--viz-forward', pos: '--viz-pos', neg: '--viz-neg', s1: '--viz-seq-1', s2: '--viz-seq-2', s3: '--viz-seq-3', s4: '--viz-seq-4', s5: '--viz-seq-5' };
    let c = readColors(C), code = readVar('--font-code'), ui = readVar('--font-ui');
    let kShown = 5, pShown = 0.9, Wd = 720, Hh = 300, lastUi = 0, lastNarr = '', lastExtra = '', lastPhase = '';
    const mono = (px: number, wt = 500) => `${wt} ${px}px ${code}`;
    const mixc = (a: number[], b: number[], u: number) => `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * u)).join(',')})`;
    const seqCol = (v: number) => mixc(c.s1.rgb, c.s5.rgb, Math.min(1, Math.max(0, v)));
    const lum = (rgb: number[]) => (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255;
    const txt = (s: string, x: number, y: number, col: string, px = 12, align: CanvasTextAlign = 'left', wt = 500, a = 1) => {
      if (a <= 0.02) return;
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align; ctx.font = mono(px, wt); ctx.fillText(s, x, y); ctx.globalAlpha = 1;
    };
    const head = (s: string, x: number, y: number, a = 1) => {
      ctx.globalAlpha = a; ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.font = `500 12px ${ui}`; ctx.fillText(s, x, y); ctx.globalAlpha = 1;
    };
    const rect = (x: number, y: number, ww: number, hh: number, col: string, a = 1) => { ctx.globalAlpha = a; ctx.fillStyle = col; ctx.fillRect(x, y, ww, hh); ctx.globalAlpha = 1; };
    const frame = (x: number, y: number, ww: number, hh: number, col: string, wd = 1, dash: number[] = []) => {
      ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.setLineDash(dash); ctx.strokeRect(x + 0.5, y + 0.5, ww - 1, hh - 1); ctx.setLineDash([]);
    };
    const hline = (x1: number, x2: number, y: number, col: string, dash: number[] = [], wd = 1) => {
      ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(x2, y); ctx.stroke(); ctx.setLineDash([]);
    };
    const tail = (s: string, n: number) => (s.length > n ? '…' + s.slice(-n) : s).replace(/\n/g, '↵');

    /** Draw characters on a fixed grid, wrapped at `cols`. `tint(i)` gives a background colour or null; `under(i)` an underline. Returns the y after the last line. */
    function lane(chars: string[], x: number, y: number, cols: number, cw: number, lines: number, tint?: (i: number) => string | null, under?: (i: number) => boolean, dimN = 0) {
      ctx.font = mono(13, 500); ctx.textAlign = 'left';
      chars.forEach((ch, i) => {
        const r = Math.floor(i / cols), cx = x + (i % cols) * cw, cy = y + r * 20;
        const bg = tint?.(i);
        let col = c.text.hex;
        if (bg) {
          ctx.fillStyle = bg; ctx.fillRect(cx - 0.5, cy - 14, cw + 1, 19);
          const m = bg.match(/\d+/g)!.map(Number), l = lum(m);
          col = Math.abs(l - lum(c.text.rgb)) > Math.abs(l - lum(c.raised.rgb)) ? c.text.hex : c.raised.hex;
        }
        ctx.fillStyle = i < dimN ? c.muted.hex : col; ctx.fillText(ch, cx, cy);
        if (under?.(i)) { ctx.fillStyle = c.accent.hex; ctx.fillRect(cx, cy + 3, cw, 2); }
      });
      return y + lines * 20;
    }
    const cwOf = () => { ctx.font = mono(13, 500); return ctx.measureText('M').width; };

    /** A strip with one stretch per character in id order, as wide as its probability; the picker `u` lands in one of them. */
    function strip(p: Float64Array, x: number, y: number, ww: number, hh: number, chosen: number, u: number | null, uShown = true) {
      let cum = 0;
      for (let i = 0; i < p.length; i++) {
        const x0 = x + cum * ww, wd = p[i] * ww; cum += p[i];
        if (wd <= 0.01) continue;
        ctx.fillStyle = i === chosen && uShown ? c.accent.hex : i % 2 ? c.s2.hex : c.s3.hex; ctx.fillRect(x0, y, Math.max(wd, 0.6), hh);
        if (wd >= 11) txt(glyph(i), x0 + wd / 2, y + hh / 2 + 4, i === chosen && uShown ? c.raised.hex : c.text.hex, 11, 'center', 600);
      }
      frame(x, y, ww, hh, c.rule.hex);
      if (u !== null) {
        const ux = x + u * ww;
        ctx.fillStyle = c.text.hex; ctx.beginPath(); ctx.moveTo(ux, y - 1); ctx.lineTo(ux - 5, y - 9); ctx.lineTo(ux + 5, y - 9); ctx.fill();
      }
    }

    // ---------------------------------------------------------------- 1. logits
    function drawLogits() {
      const f = logitsFrame(t, nx!, shift), order = nx!.order, V = order.length;
      const left = 14, right = Wd - 14, top = 58, bot = Hh - 34, ph = bot - top, slot = (right - left) / V, bw = slot * 0.72;
      const caption = f.phase === 0 ? (narrow ? 'Score for each character' : 'Score (logit) for each character: up is positive, down is negative')
        : f.phase === 1 ? (t < 5 ? 'Raising e to each score ...' : `e raised to the score. Full height = ${fmt(f.top)}`)
        : (t < 8 ? `The same bars, divided by ${fmt(f.divisor)} so far` : 'Probability = e^score divided by the total');
      head(caption, left, 24);
      const y0 = top + f.base * ph;
      hline(left, right, y0, c.rule.hex, f.phase === 0 ? [] : [], 1);
      if (f.phase === 0) txt('0', left - 4, y0 + 4, c.muted.hex, 10, 'right');
      let prevY = Infinity;
      for (let i = 0; i < V; i++) {
        const x = left + i * slot + (slot - bw) / 2, yE = top + f.ends[i] * ph, z = nx!.logits[order[i]] + shift;
        const col = mixc(z >= 0 ? c.pos.rgb : c.neg.rgb, c.fwd.rgb, f.mix);
        const ya = Math.min(y0, yE), hh = Math.max(Math.abs(yE - y0), 0.7);
        rect(x, ya, bw, hh, col);
        txt(glyph(order[i]), left + i * slot + slot / 2, Hh - 14, c.muted.hex, 11, 'center');
        if (i < 3 && f.labelAlpha > 0.02) {
          const v = f.vals[i], s = f.label === 2 ? pct(v) : fmt(v), ly = yE - 6 > prevY - 14 && yE - 6 < prevY + 12 ? prevY - 14 : yE - 6;
          prevY = ly;
          txt(s, x, ly, c.text.hex, 11, 'left', 600, f.labelAlpha);
        }
      }
      if (f.sum > 0) txt(`sum of the bars = ${f.sum >= 100 ? fmt(f.sum) : f.sum.toFixed(3)}`, right, 42, c.text.hex, 12, 'right', 600);
      narration = f.status; extra = f.sum > 0 ? `Sum of the bars: ${f.sum >= 100 ? fmt(f.sum) : f.sum.toFixed(3)}.` : `Shift ${shift >= 0 ? '+' : '−'}${Math.abs(shift)}: the order of the scores stays the same.`;
      phaseName = f.phase === 0 ? 'Scores' : f.phase === 1 ? 'Exponentiate' : 'Divide by the total';
    }

    // ---------------------------------------------------------------- 2. greedy versus sampled
    function drawGreedy() {
      const d = greedy!, f = greedyFrame(t, d), cw = cwOf(), x0 = 14, cols = Math.floor((Wd - 32) / cw), lines = Math.ceil(GREEDY_N / cols), R = Wd - 14;
      const so = f.shown >= GREEDY_N ? '' : ' so far', avg = (v: number) => (f.shown ? v.toFixed(2) : '-'), lbl = (v: number) => `average log-probability${so} ${avg(v)}`;
      head('Prompt (held-out Alice text)', x0, 20);
      txt(`“${tail(d.prompt, Math.floor((Wd - 40) / cw) - 2)}”`, x0, 42, c.muted.hex, 13);
      const o = narrow ? 14 : 0, stat = (v: number, y: number) => (narrow ? txt(lbl(v), x0, y + 14, c.muted.hex, 11) : txt(lbl(v), R, y, c.muted.hex, 11, 'right'));
      head('Greedy: the most likely character', x0, 84); stat(f.greedyMean, 84);
      const lp = d.loop, inLoop = (i: number) => !!lp && f.loopShown && ((i >= lp.start && i < lp.seenAt) || (i >= lp.from && i < lp.from + 12));
      let y = lane(Array.from(f.greedyText), x0, 106 + o, cols, cw, lines, undefined, inLoop) + 14;
      head(narrow ? 'Sampled: random, by probability' : 'Sampled: random, in proportion to probability', x0, y); stat(f.sampledMean, y);
      y = lane(Array.from(f.sampledText), x0, y + 22 + o, cols, cw, lines) + 18;
      head(narrow ? `Character ${f.i + 1}: stretch width = probability` : `Character ${f.i + 1}, sampled lane: stretch width = probability`, x0, y);
      strip(f.strip, x0, y + 16, Wd - 28, 28, f.chosen, f.uPos, f.landed);
      txt(`u = ${f.u.toFixed(3)}${f.landed ? `, which lands in “${glyph(f.chosen)}”` : ''}`, x0, y + 66, c.text.hex, 12, 'left', 600);
      narration = f.status;
      extra = `Average log-probability${so}: greedy lane ${avg(f.greedyMean)}, sampled lane ${avg(f.sampledMean)}.${f.loopShown ? ` The greedy lane is copying itself: characters ${lp!.start + 1} to ${lp!.seenAt} repeat characters ${lp!.from + 1} to ${lp!.from + 12}.` : ''}`;
      phaseName = f.loopShown ? 'Greedy repeats' : 'Writing';
    }

    // ---------------------------------------------------------------- 3. temperature
    function drawTemp() {
      const d = temp!, f = tempFrame(t, d), V = d.nx.V, left = 14, right = Wd - 14, top = 56, bot = top + (narrow ? 130 : 150), ph = bot - top, slot = (right - left) / V, bw = slot * 0.72;
      head(narrow ? 'Next-character probabilities' : 'Probability of each next character, most likely first', left, 22);
      txt(`T = ${f.T.toFixed(2)}`, right, 22, c.text.hex, 13, 'right', 700);
      txt(`entropy ${f.entropy.toFixed(2)} bits`, right, 40, c.muted.hex, 12, 'right');
      hline(left, right, bot, c.rule.hex);
      hline(left, right, bot - ph / 1 * (1 / V), c.muted.hex, [3, 3]);
      txt('uniform', right, bot - ph / V - 5, c.muted.hex, 10, 'right');
      for (let i = 0; i < V; i++) {
        const hh = Math.max(f.probs[i] * ph, 0.6), x = left + i * slot + (slot - bw) / 2;
        rect(x, bot - hh, bw, hh, c.fwd.hex);
        txt(glyph(d.nx.order[i]), left + i * slot + slot / 2, bot + 14, c.muted.hex, 11, 'center');
        if (i < 2) txt(pct(f.probs[i], 0), left + i * slot + slot / 2 + (i === 0 ? 2 : 0), bot - hh - 5, c.text.hex, 11, i === 0 ? 'left' : 'center', 600);
      }
      const cw = cwOf(), cols = Math.floor((Wd - 32) / cw), y = bot + 44;
      head(narrow ? `Sample at T = ${TEMPS[f.snap]}, the nearest preset` : `Sample of ${TEXT_N} characters at T = ${TEMPS[f.snap]}, the nearest preset (same random numbers at every temperature)`, left, y - 6);
      lane(Array.from(f.text || 'writing this sample ...'), left, y + 20, cols, cw, Math.ceil(TEXT_N / cols));
      head(`average log-probability of this text: ${Number.isNaN(f.mean) ? '-' : f.mean.toFixed(2)}`, left, y + (Math.ceil(TEXT_N / cols)) * 20 + 30);
      narration = f.status;
      extra = `Text at T = ${TEMPS[f.snap]}: average log-probability ${Number.isNaN(f.mean) ? '-' : f.mean.toFixed(2)} per character.`;
      phaseName = t < 8 ? 'Temperature rising' : 'Temperature falling';
    }

    // ---------------------------------------------------------------- 4 and 5. two contexts
    const panelBox = (i: number) => {
      const gap = 14;
      if (narrow) { const hh = (Hh - gap) / 2; return { x: 14, y: i * (hh + gap), ww: Wd - 28, hh }; }
      const ww = (Wd - 28 - gap) / 2; return { x: 14 + i * (ww + gap), y: 0, ww, hh: Hh };
    };
    const ctxTitle = (which: 0 | 1) => {
      const s = (which === 0 ? PROMPTS.sure : PROMPTS.unsure).slice(-18), cut = s.indexOf(' ') + 1;
      return `After “…${(cut > 0 ? s.slice(cut) : s).split('\n').join('↵')}_”`;
    };
    function drawKPanel(i: 0 | 1, p: Panel) {
      const { x, y, ww, hh } = panelBox(i), V = p.chars.length, slot = ww / V, bw = slot * 0.72, top = y + 84, bot = y + hh - 62, ph = bot - top;
      head(`${ctxTitle(i)}${i === 0 ? ': the model is sure' : ': not sure'}`, x, y + 20);
      hline(x, x + ww, bot, c.rule.hex);
      const max = Math.max(p.max, 1e-9);
      txt(`tallest: ${pct(p.probs[0])} before cut, ${pct(p.kept[0])} after`, x, y + 40, c.muted.hex, 11);
      let prevLy = 0;
      for (let r = 0; r < V; r++) {
        const bx = x + r * slot + (slot - bw) / 2, h0 = Math.max(p.probs[r] / max * ph, 0.6), h1 = p.kept[r] / max * ph;
        ctx.globalAlpha = 0.8; frame(bx, bot - h0, bw, h0, c.muted.hex); ctx.globalAlpha = 1;
        if (h1 > 0.3) rect(bx, bot - h1, bw, h1, c.fwd.hex);
        txt(p.chars[r], x + r * slot + slot / 2, bot + 13, c.muted.hex, 11, 'center');
        if (r < 3 && p.kept[r] > 0) { let ly = bot - h1 - 5; if (r && Math.abs(ly - prevLy) < 13) ly = prevLy - 13; prevLy = ly; txt(pct(p.kept[r], p.kept[r] < 0.095 ? 1 : 0), bx, ly, c.text.hex, 11, 'left', 600); }
      }
      txt(`kept ${p.keep} of ${V} characters`, x, bot + 36, c.text.hex, 13, 'left', 700);
      txt(`thrown away: ${pct(p.removed)} of the probability`, x, bot + 54, c.text.hex, 12, 'left', 600);
    }
    function drawPPanel(i: 0 | 1, p: PPanel, pv: number) {
      const { x, y, ww, hh } = panelBox(i), V = p.chars.length, slot = ww / V, bw = slot * 0.72;
      const curveTop = y + 68, curveBot = curveTop + (narrow ? 48 : 60), bot = y + hh - 62, barTop = curveBot + 16, ph = bot - barTop;
      head(`${ctxTitle(i)}${i === 0 ? ': the model is sure' : ': not sure'}`, x, y + 20);
      txt(`tallest bar: ${pct(p.probs[0])}`, x, y + 40, c.muted.hex, 11);
      // running total of probability, rank by rank
      hline(x, x + ww, curveBot, c.rule.hex); hline(x, x + ww, curveTop, c.rule.hex, [2, 3]);
      txt('100%', x, curveTop - 4, c.muted.hex, 10, 'left');
      const py = curveBot - pv * (curveBot - curveTop);
      hline(x, x + ww, py, c.accent.hex, [5, 4], 1.5);
      txt(`p = ${pv.toFixed(2)}`, x + ww, py - 5, c.accent.hex, 11, 'right', 700);
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, curveBot);
      for (let r = 0; r < Math.min(V, Math.ceil(p.head)); r++) {
        const frac = r + 1 <= p.head ? 1 : p.head - r, prev = r ? p.cum[r - 1] : 0, cumv = prev + (p.cum[r] - prev) * frac;
        ctx.lineTo(x + (r + frac) * slot, curveBot - cumv * (curveBot - curveTop));
      }
      ctx.stroke();
      const max = p.probs[0];
      for (let r = 0; r < V; r++) {
        const bx = x + r * slot + (slot - bw) / 2, h2 = Math.max(p.probs[r] / max * (bot - barTop), 0.6), st = p.state[r];
        if (st === 1) rect(bx, bot - h2, bw, h2, c.fwd.hex);
        else if (st === 0) rect(bx, bot - h2, bw, h2, c.s2.hex);
        else { ctx.globalAlpha = 0.8; frame(bx, bot - h2, bw, h2, c.muted.hex); ctx.globalAlpha = 1; }
        txt(p.chars[r], x + r * slot + slot / 2, bot + 13, c.muted.hex, 11, 'center');
      }
      hline(x, x + ww, bot, c.rule.hex);
      if (p.head >= V) { const cx = x + p.keep * slot; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(cx, curveTop); ctx.lineTo(cx, bot); ctx.stroke(); ctx.setLineDash([]); }
      txt(`kept ${p.keep} of ${V} characters`, x, bot + 36, c.text.hex, 13, 'left', 700);
      txt('cut at p, then rescaled to add up to 1', x, bot + 54, c.muted.hex, 11, 'left');
    }
    function drawK() {
      const f = topkFrame(t, pair!, manualK ?? undefined);
      drawKPanel(0, f.sure); drawKPanel(1, f.unsure);
      kShown = f.k; narration = f.status; extra = '';
      phaseName = manualK !== null ? `k = ${f.k}` : kAt(t).i < 7 ? 'Shrinking k' : 'Growing k';
    }
    function drawP() {
      const f = toppFrame(t, pair!, manualP ?? undefined);
      drawPPanel(0, f.sure, f.p); drawPPanel(1, f.unsure, f.p);
      pShown = f.p; narration = f.status; extra = f.sure.head >= 38 ? '' : 'Adding up the probabilities, most likely character first.';
      phaseName = manualP !== null ? `p = ${f.p.toFixed(2)}` : t < 5 ? 'Add up and cut' : t < 8 ? 'Lower p' : t < 12.5 ? 'Raise p' : 'Back to 0.9';
    }

    // ---------------------------------------------------------------- 6. lab
    function drawLab() {
      const d = labData!, f = labFrame(t, d), cw = cwOf(), x0 = 14, cols = Math.floor((Wd - 32) / cw);
      const pre = d.settings.prompt, shownChars = Array.from(labText(d, f.shown)), promptChars = Array.from(tail(pre, 30));
      const all = [...promptChars, ...shownChars], plen = promptChars.length;
      const lines = Math.ceil((32 + LAB_N) / cols);
      head(narrow ? 'Prompt in grey, then generated text' : 'Prompt in grey, then generated text shaded by the probability the model gave each character', x0, 20);
      lane(all, x0, 48, cols, cw, lines,
        (i) => (i < plen ? null : seqCol(Math.exp(d.steps[i - plen].logp))),
        (i) => i >= plen && !!f.repeat && f.shown >= f.repeat.seenAt && i - plen >= f.repeat.start && i - plen < f.repeat.seenAt, plen);
      const sy = Hh - 56;
      head(narrow ? `Chances for character ${f.shown || 1}` : f.shown ? `Chances for character ${f.shown}, alphabetical` : 'Chances for the first character, alphabetical', x0, sy - 16);
      // bars in id order
      const strip = f.strip ?? new Float64Array(38), V = strip.length, slot = (Wd - 28) / V, bw = slot * 0.72;
      let mx = 0; for (const v of strip) mx = Math.max(mx, v);
      hline(x0, Wd - 14, sy + 30, c.rule.hex);
      for (let i = 0; i < V; i++) {
        const hh = (strip[i] / (mx || 1)) * 30, bx = x0 + i * slot + (slot - bw) / 2;
        if (hh > 0.3) rect(bx, sy + 30 - hh, bw, hh, i === f.chosen ? c.accent.hex : c.s3.hex);
        txt(glyph(i), x0 + i * slot + slot / 2, sy + 44, i === f.chosen ? c.accent.hex : c.muted.hex, 11, 'center', i === f.chosen ? 700 : 500);
      }
      if (mx) txt(`tallest bar = ${pct(mx)}`, Wd - 14, sy - 16, c.muted.hex, 11, 'right');
      narration = f.status;
      extra = f.shown ? `Repeats itself: ${f.repeat ? `yes, from character ${f.repeat.start + 1}` : 'no'}.` : '';
      phaseName = 'Writing';
    }

    function loading() {
      txt(failed ? 'The model could not be loaded.' : needsModel ? 'Loading the model ...' : 'Writing the text ...', Wd / 2, Hh / 2, c.muted.hex, 13, 'center');
    }

    function render() {
      const d = Math.min(2, Math.max(1, window.devicePixelRatio || 1)), r = cv.getBoundingClientRect();
      const wpx = Math.round(r.width * d), hpx = Math.round(r.height * d);
      if (wpx && hpx && (cv.width !== wpx || cv.height !== hpx)) { cv.width = wpx; cv.height = hpx; }
      Wd = narrow ? 320 : 720; Hh = HEIGHTS[M][narrow ? 1 : 0];
      const k = cv.width / Wd;
      ctx.setTransform(k, 0, 0, k, 0, 0); ctx.clearRect(0, 0, Wd, Hh);
      if (!ready) { loading(); return; }
      if (M === 'logits') drawLogits(); else if (M === 'greedy') drawGreedy(); else if (M === 'temperature') drawTemp();
      else if (M === 'topk') drawK(); else if (M === 'topp') drawP(); else drawLab();
    }

    const loop = createLoop(cv, (dt, now) => {
      if (ready) t = advance({ t, playing, speed }, dt, { duration }).t;
      render();
      tl.paint(t);
      if (ready && (now - lastUi > 32 || !playing)) {
        lastUi = now; uiT = t;
        if (narration !== lastNarr) lastNarr = narration;
        if (extra !== lastExtra) lastExtra = extra;
        if (phaseName !== lastPhase) lastPhase = phaseName;
        roNarr = lastNarr; roExtra = lastExtra; roPhase = lastPhase; kNow = kShown; pNow = pShown;
      }
      return playing || !ready;
    }, { onView: (inV, first) => { if (inV && first && ready && !reduced.matches) playing = true; } });
    poke = loop.wake;
    setReady = () => {
      if (ready) { poke(); return; }
      ready = true;
      if (reduced.matches) { t = restTime(M); uiT = t; } else playing = true;
      poke();
    };
    if (M === 'greedy' || M === 'temperature' || M === 'lab') {
      worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      worker.onerror = () => { failed = true; poke(); };
      worker.onmessage = (e: MessageEvent<FromWorker>) => {
        const m = e.data;
        if (m.id !== reqId) return;
        if (m.kind === 'error') { failed = true; poke(); }
        else if (m.kind === 'greedy') { greedy = m.data; markers = markersOf(M, m.data); setReady(); }
        else if (m.kind === 'temp') { if (temp) { temp.texts[m.i] = m.text; temp.means[m.i] = m.mean; } poke(); }
        else if (labData && m.from === labData.steps.length) { labData.steps.push(...m.steps); setReady(); }
      };
    }
    if (needsModel) loadModel('house').then((m) => { w = m; build(); setReady(); }, () => { failed = true; poke(); });
    else build();

    const ro = new ResizeObserver(([e]) => { narrow = e.contentRect.width < 540; poke(); });
    ro.observe(cv.parentElement!);
    const off = onThemeChange(() => { c = readColors(C); code = readVar('--font-code'); ui = readVar('--font-ui'); poke(); });
    const onReduce = () => { if (reduced.matches) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    return () => { worker?.terminate(); loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); };
  });
  // readout text (updated about 30 times a second from the same frame the canvas drew)
  let roNarr = $state(''), roExtra = $state(''), roPhase = $state('');

  const legend = $derived(M !== 'logits' ? LEGENDS[M] : roPhase === 'Scores' || !roPhase ? LEGENDS.logits.slice(0, 2) : [{ label: roPhase === 'Exponentiate' ? 'e to the score' : 'probability', color: '--viz-forward' }]);
  const stageLabel = {
    logits: 'Bar chart of the next-character scores, then e raised to each score, then the probabilities after dividing by the total.',
    greedy: 'Two lines of generated text, greedy and sampled, and a bar whose stretches are the probabilities of each possible next character.',
    temperature: 'Bar chart of next-character probabilities that flattens as temperature rises, and a text sample at the nearest preset temperature.',
    topk: 'Two bar charts of next-character probabilities, after "q" and after "the ", with all but the k most likely bars cut.',
    topp: 'Two bar charts with running totals of probability, showing which characters make up the nucleus for the chosen p.',
    lab: 'Generated text with each character shaded by the probability the model gave it, and the distribution it was drawn from.',
  }[M];
</script>

<FigureCard title={init.title} legend={legend} aspect="720 / {HEIGHTS[M][0]}" narrowAspect="320 / {HEIGHTS[M][1]}" label={stageLabel}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} {markers} label="Timeline"
      onSeek={(to) => seek(to)} onReset={() => seek(0)} />
  {/snippet}
  {#snippet readout()}
    <p class="say">{ready ? roNarr : failed ? 'The model could not be loaded.' : needsModel ? 'Loading the model (35 KB) ...' : 'Writing the text ...'}</p>
    {#if roExtra}<p class="say">{roExtra}</p>{/if}
    <p class="sr-only" role="status" aria-live="polite">{roPhase}</p>
  {/snippet}
  {#snippet footer()}
    {#if M === 'logits'}
      <div class="ctl">
        <div class="ctrl"><span class="lbl">Add the same number to every score <output class="num">{shift > 0 ? '+' : shift < 0 ? '−' : ''}{Math.abs(shift)}</output></span>
          <input type="range" min={-SHIFT_MAX} max={SHIFT_MAX} step="0.5" bind:value={shift} oninput={() => poke()} aria-label="Number added to every score" /></div>
        <button type="button" class="btn" onclick={() => { shift = 0; poke(); }}>Reset</button>
      </div>
    {:else if M === 'greedy' || M === 'temperature'}
      <div class="ctl">
        <button type="button" class="btn" onclick={newSeed}>New random numbers</button>
        <span class="hint num">seed {seed}</span>
      </div>
    {:else if M === 'topk'}
      <div class="ctl">
        <div class="ctrl"><span class="lbl">k, how many characters to keep <output class="num">{manualK ?? kNow}</output></span>
          <input type="range" min="1" max="38" step="1" value={manualK ?? kNow} oninput={(e) => setK(+e.currentTarget.value)} aria-label="k, the number of characters to keep" /></div>
        <button type="button" class="btn" onclick={() => { manualK = null; playing = true; poke(); }} disabled={manualK === null}>Back to the sweep</button>
      </div>
    {:else if M === 'topp'}
      <div class="ctl">
        <div class="ctrl"><span class="lbl">p, how much probability to keep <output class="num">{(manualP ?? pNow).toFixed(2)}</output></span>
          <input type="range" min="0.05" max="1" step="0.01" value={manualP ?? pNow} oninput={(e) => setP(+e.currentTarget.value)} aria-label="p, the probability to keep" /></div>
        <button type="button" class="btn" onclick={() => { manualP = null; playing = true; poke(); }} disabled={manualP === null}>Back to the sweep</button>
      </div>
    {:else}
      <div class="ctl lab">
        <label class="wide"><span class="lbl">Prompt (letters and punctuation only)</span>
          <input class="txt" type="text" bind:value={lab.prompt} onchange={onLab} spellcheck="false" autocomplete="off" aria-label="Prompt" />
        </label>
        <div class="ctrl"><span class="lbl">Temperature <output class="num">{lab.temperature === 0 ? '0 (greedy)' : lab.temperature.toFixed(2)}</output></span>
          <input type="range" min="0" max="2" step="0.05" value={lab.temperature} oninput={(e) => setLab({ temperature: +e.currentTarget.value })} aria-label="Temperature" /></div>
        <div class="ctrl"><span class="lbl">Top-k <output class="num">{lab.topK >= 38 ? 'off' : lab.topK}</output></span>
          <input type="range" min="1" max="38" step="1" value={lab.topK} oninput={(e) => setLab({ topK: +e.currentTarget.value })} aria-label="Top-k, 38 means off" /></div>
        <div class="ctrl"><span class="lbl">Top-p <output class="num">{lab.topP >= 1 ? 'off' : lab.topP.toFixed(2)}</output></span>
          <input type="range" min="0.05" max="1" step="0.01" value={lab.topP} oninput={(e) => setLab({ topP: +e.currentTarget.value })} aria-label="Top-p, 1 means off" /></div>
        <div class="row">
          <button type="button" class="btn" onclick={newSeed}>New random numbers</button>
          <span class="hint num">seed {lab.seed}</span>
          <button type="button" class="btn" onclick={() => { lab = { prompt: PROMPTS.main, ...LAB_DEFAULTS }; seed = SEED; rebuild(); }}>Defaults</button>
        </div>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .say { margin: 0; font-size: 0.92rem; line-height: 1.45; font-variant-numeric: tabular-nums; min-height: 2.9em; }
  .say + .say { min-height: 1.45em; color: var(--text-muted); }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .ctrl { display: grid; gap: var(--s-1); min-width: 0; }
  .lbl { color: var(--text-muted); display: flex; gap: var(--s-2); justify-content: space-between; }
  .ctl input[type='range'] { width: min(16rem, 100%); accent-color: var(--accent); margin: 0; height: 1.25rem; background: transparent; appearance: none; -webkit-appearance: none; cursor: pointer; }
  .ctl input[type='range']::-webkit-slider-runnable-track { height: 4px; border-radius: 2px; background: var(--rule); }
  .ctl input[type='range']::-moz-range-track { height: 4px; border-radius: 2px; background: var(--rule); }
  .ctl input[type='range']::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; margin-top: -5px; border-radius: 50%; background: var(--accent); border: 2px solid var(--bg-raised); box-shadow: 0 0 0 1px var(--accent); }
  .ctl input[type='range']::-moz-range-thumb { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); border: 2px solid var(--bg-raised); box-shadow: 0 0 0 1px var(--accent); }
  .ctl output { color: var(--text); }
  .lab { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: var(--s-3) var(--s-4); }
  .lab .wide { grid-column: 1 / -1; display: grid; gap: var(--s-1); }
  .lab .ctrl input[type='range'] { width: 100%; }
  .row { grid-column: 1 / -1; display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); }
  .txt { width: 100%; box-sizing: border-box; font: 0.9rem var(--font-code); padding: 0.4em 0.55em; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); }
  .hint { font: var(--step--1) var(--font-ui); color: var(--text-muted); }
  button:disabled { opacity: 0.45; cursor: default; }
  button:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
</style>
