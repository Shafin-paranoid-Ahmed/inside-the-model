<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance as clock } from '../_shared';
  import { makePlan, frameAt, markersOf, phaseStart, gFrac, fmt, fmtG, showTok, clamp01, H, type Cell, type LossAt, type Phase, type Plan } from './sim';

  interface Props {
    /** 'rnn' | 'lstm' | 'gru'. */
    cell?: Cell;
    /** 'forward': the hidden state runs along the chain. 'bptt': then gradients flow backward through time. */
    mode?: 'forward' | 'bptt';
    /** Starting text (each character is one time step). */
    sequence?: string;
    /** Recurrent weight scale s: every singular value of W_hh (and of the gates' recurrent matrices) equals s. */
    scale?: number;
    /** Start folded (one cell with a loop) and animate the unrolling. false: start already unrolled. */
    folded?: boolean;
    /** Where the timeline starts. */
    initialPhase?: Phase;
    /** Loop only this phase. Default: loop the whole timeline. */
    loopPhase?: Phase;
    title?: string;
    /** Show the recurrent-weight-scale slider. */
    showScale?: boolean;
    /** Show the 'loss at every step' checkbox (bptt). */
    showLossToggle?: boolean;
    lossAt?: LossAt;
    /** Show the per-step inspector panel. */
    showPanel?: boolean;
    /** 1-based step to open in the inspector. */
    select?: number;
    /** Longest sequence (characters). */
    maxLen?: number;
    /** Allow editing the sequence. */
    editable?: boolean;
    /** Truncated BPTT: gradients stop every k steps (0 = full BPTT). */
    truncate?: number;
    /** Show the truncation control. */
    showTruncate?: boolean;
    /** RNN: mark, on every bar, the upper bound from the product of Jacobians. */
    showBound?: boolean;
    /** RNN: show gradient-norm clipping (threshold slider and readout). */
    showClip?: boolean;
    /** LSTM: forget-gate bias (1.5 gives f of about 0.82 on ordinary characters). */
    forgetBias?: number;
    /** LSTM: show the forget-gate bias slider. */
    showForget?: boolean;
  }
  let {
    cell = 'rnn', mode = 'forward', sequence = 'hello', scale = 1, folded = true, initialPhase, loopPhase,
    title = '', showScale = false, showLossToggle = false, lossAt = 'last', showPanel = true, select = 0,
    maxLen = 0, editable = true, truncate = 0, showTruncate = false, showBound = false, showClip = false,
    forgetBias = 1.5, showForget = false,
  }: Props = $props();

  const init = untrack(() => ({
    cell, bptt: mode === 'bptt', folded, loop: loopPhase, maxLen: maxLen || (cell === 'rnn' ? (mode === 'bptt' ? 12 : 8) : 6),
    start: initialPhase ?? loopPhase ?? (folded ? 'fold' : 'forward'),
    title: title || (cell === 'rnn' ? 'A recurrent cell unrolled through time' : cell === 'lstm' ? 'An LSTM cell: three gates on a conveyor' : 'A GRU cell'),
  }));
  const NAMES: Record<Cell, string> = { rnn: 'RNN', lstm: 'LSTM', gru: 'GRU' };
  const uid = $props.id();

  let seq = $state(untrack(() => sequence));
  let sc = $state(untrack(() => scale));
  let lossAll = $state(untrack(() => lossAt === 'all'));
  let tk = $state(untrack(() => truncate));
  let clipC = $state(5);
  let fb = $state(untrack(() => forgetBias));
  let narrow = $state(false);
  let playing = $state(false), speed = $state(0.5);
  let sel = $state(untrack(() => (select ? select - 1 : -1)));
  let uiT = $state(0), uiK = $state(0), phase = $state<Phase>('fold');
  let narration = $state('');

  const plan: Plan = $derived(makePlan(init.cell, sc, seq, lossAll ? 'all' : 'last', { folded: init.folded, bptt: init.bptt, maxLen: init.maxLen, truncate: tk, forgetBias: fb }));
  const markers = $derived(markersOf(plan.tm));
  const tr = $derived(plan.tr);
  const kShown = $derived(Math.min(plan.n - 1, sel >= 0 ? sel : uiK));

  // ---- layout (logical canvas units) ----
  const rows = {
    rnn: { h: 300, yOut: 24, yH: 76, yCell: 150, yTok: 232, yX: 262, yBelt: 0, yM: 0, mH: 0, yG: 0 },
    lstm: { h: 322, yOut: 0, yH: 214, yCell: 0, yTok: 264, yX: 294, yBelt: 40, yM: 70, mH: 48, yG: 180 },
    gru: { h: 290, yOut: 0, yH: 0, yCell: 0, yTok: 232, yX: 262, yBelt: 40, yM: 70, mH: 48, yG: 180 },
  } as const;
  const base = rows[init.cell];
  const CH = 130; // chart height
  const Hh = base.h + (init.bptt ? CH + 46 : 0);
  const chartTop = base.h + 40, chartBot = base.h + 40 + CH;
  const Wd = $derived(narrow ? 400 : 720);
  const cols = $derived.by(() => {
    const n = plan.n, left = narrow ? 26 : 40, avail = Wd - left - (narrow ? 10 : 16);
    const w = Math.min(init.cell === 'rnn' ? 100 : 118, avail / n);
    const x0 = left + (avail - w * n) / 2;
    return { w, x0, cx: (k: number) => x0 + w * (k + 0.5) };
  });

  // ---- canvas ----
  let cv: HTMLCanvasElement;
  let tl: ReturnType<typeof Timeline>;
  let t = 0, poke = () => {};
  let hoverK = -1;

  const legend = [
    { label: 'value', color: '--viz-forward' }, ...(init.bptt ? [{ label: 'gradient', color: '--viz-gradient' }] : []),
    { label: 'positive', color: '--viz-pos' }, { label: 'negative', color: '--viz-neg' },
    ...(init.cell !== 'rnn' ? [{ label: 'gate opening', color: '--viz-seq-4' }] : []),
  ];

  const cut = (k: number) => tk > 0 && k > 0 && k % tk === 0;
  const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
  const say = (ph: Phase, k: number): string => {
    const p = plan, r = p.tr, tok = showTok(r.toks[k] ?? '');
    if (ph === 'fold') return 'One cell with a loop. At each time step it reads an input and feeds its own state back into itself.';
    if (ph === 'unroll') return 'Unrolling: one copy of the cell per time step. Every copy uses the same weights.';
    if (ph === 'forward') return init.cell === 'rnn'
      ? `Step ${k + 1}: read "${tok}", mix it with h${k}, write the new hidden state h${k + 1}.`
      : init.cell === 'lstm'
        ? `Step ${k + 1}: read "${tok}". Forget ${fmt(mean(r.gates.f[k]))}, input ${fmt(mean(r.gates.i[k]))}, output ${fmt(mean(r.gates.o[k]))}.`
        : `Step ${k + 1}: read "${tok}". Update gate ${fmt(mean(r.gates.z[k]))}, reset gate ${fmt(mean(r.gates.r[k]))}.`;
    return `Backward: the gradient reaches step ${k + 1} with size ${fmtG(p.series[k])}.`;
  };

  function onSeek(to: number) { t = to; poke(); }
  const pick = (k: number) => { sel = sel === k ? -1 : k; if (sel >= 0) playing = false; poke(); };
  $effect(() => { playing; speed; sel; poke(); });
  $effect(() => { plan; poke(); });

  onMount(() => {
    const reduced = prefersReducedMotion();
    let reduce = reduced.matches;
    const ctx = cv.getContext('2d')!;
    const C = {
      pos: '--viz-pos', neg: '--viz-neg', fwd: '--viz-forward', grad: '--viz-gradient', text: '--text', muted: '--text-muted',
      rule: '--rule', raised: '--bg-raised', accent: '--accent', s1: '--viz-seq-1', s5: '--viz-seq-5',
    };
    let c = readColors(C), code = readVar('--font-code'), ui = readVar('--font-body');
    const dpr = () => Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const mono = (px: number, w = 500) => `${w} ${px}px ${code}`;
    const seqCol = (v: number) => {
      const a = c.s1.rgb, b = c.s5.rgb, u = clamp01(v);
      return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * u)).join(',')})`;
    };
    t = phaseStart(plan.tm, init.start);
    if (reduce) t = plan.tm.T - 0.3;
    let lastRead = 0, nowMs = 0;

    // ---- primitives ----
    const line = (pts: [number, number][], col: string, w: number, a = 1) => {
      ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineJoin = 'round';
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    };
    const dot = (x: number, y: number, r: number, col: string, a = 1) => {
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
    };
    const txt = (s: string, x: number, y: number, col: string, px = 12, align: CanvasTextAlign = 'center', w = 500, a = 1, ital = false) => {
      if (a <= 0.02) return;
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align;
      ctx.font = ital ? `italic ${px}px ${ui}` : mono(px, w); ctx.fillText(s, x, y);
    };
    const box = (x: number, y: number, w: number, h: number, r: number, a: number, stroke = c.rule.hex, sw = 1.2, fill: string | null = c.raised.hex) => {
      ctx.globalAlpha = a; ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke();
    };
    const tint = (x: number, y: number, w: number, h: number, r: number, col: string, a: number) => {
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.roundRect(x, y, w, h, r); ctx.fill();
    };
    /** Strip of colored cells centered at (x, y). kind 'div': signed, 'seq': 0..1. */
    const chips = (x: number, y: number, v: number[], s: number, a: number, kind: 'div' | 'seq' = 'div') => {
      if (a <= 0.02) return;
      const w = v.length * s;
      v.forEach((q, j) => {
        const rx = x - w / 2 + j * s;
        ctx.globalAlpha = a; ctx.fillStyle = c.raised.hex; ctx.fillRect(rx, y - s / 2, s, s);
        ctx.fillStyle = kind === 'seq' ? seqCol(q) : q >= 0 ? c.pos.hex : c.neg.hex;
        ctx.globalAlpha = a * (kind === 'seq' ? 1 : Math.min(1, Math.abs(q)));
        ctx.fillRect(rx, y - s / 2, s, s);
      });
      ctx.globalAlpha = a; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x - w / 2, y - s / 2, w, s);
    };
    const at = (pts: [number, number][], u: number): [number, number] => {
      const seg = pts.slice(1).map((p, i) => Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1]));
      let d = u * seg.reduce((a, b) => a + b, 0), i = 0;
      while (i < seg.length - 1 && d > seg[i]) { d -= seg[i]; i++; }
      const f = seg[i] ? Math.min(1, d / seg[i]) : 0;
      return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f];
    };
    /** An edge with an optional forward particle (u in 0..1) and a gradient overlay running backward (g in 0..1). */
    const edge = (pts: [number, number][], a: number, u: number, g: number, gw: number, w = 1.6) => {
      line(pts, c.muted.hex, w, 0.55 * a);
      if (u > 0 && u < 1 && !reduce) { const [x, y] = at(pts, u); dot(x, y, 4.5, c.fwd.hex, a); }
      if (g > 0) {
        const rev = [...pts].reverse() as [number, number][], n = 14, seg: [number, number][] = [];
        for (let i = 0; i <= n; i++) seg.push(at(rev, (g * i) / n));
        line(seg, c.grad.hex, gw, 0.9 * a);
        if (g < 1 && !reduce) dot(seg[n][0], seg[n][1], 4.5, c.grad.hex, a);
      }
    };

    // ---- drawing ----
    function render() {
      const P = plan, R = P.tr, n = P.n, fr = frameAt(Math.min(t, P.tm.T), n, P.tm);
      const k0 = cv.width / Wd;
      ctx.setTransform(k0, 0, 0, k0, 0, 0);
      ctx.clearRect(0, 0, Wd, Hh);
      const { w: cw, cx } = cols;
      const showNums = cw >= 56;
      const un = fr.unroll;
      const chip = Math.max(6, Math.min(11, cw / 6.4));
      const px = (k: number) => cx(0) + (cx(k) - cx(0)) * (k ? un : 0);

      if (init.cell === 'rnn') {
        const bw = Math.min(cw * 0.66, 62), bh = 44, y = base.yCell;
        if (un < 0.98) { // the folded loop
          const a = 1 - un, x = px(0);
          const p0 = [x + bw / 2, y - 10], p1 = [x + bw / 2 + 40, y - 78], p2 = [x - bw / 2 - 40, y - 78], p3 = [x - bw / 2, y - 10];
          const pts: [number, number][] = [];
          for (let i = 0; i <= 24; i++) {
            const u = i / 24, m = 1 - u;
            pts.push([0, 1].map((d) => m ** 3 * p0[d] + 3 * m * m * u * p1[d] + 3 * m * u * u * p2[d] + u ** 3 * p3[d]) as [number, number]);
          }
          line(pts, c.muted.hex, 1.8, 0.7 * a);
          ctx.globalAlpha = 0.7 * a; ctx.fillStyle = c.muted.hex; ctx.beginPath();
          ctx.moveTo(x - bw / 2, y - 10); ctx.lineTo(x - bw / 2 - 8, y - 20); ctx.lineTo(x - bw / 2 + 2, y - 21); ctx.fill();
          if (!reduce) { const [qx, qy] = at(pts, (t * 0.35) % 1); dot(qx, qy, 4.5, c.fwd.hex, a); }
          txt('h', x, y - 84, c.muted.hex, 13, 'center', 500, a);
        }
        for (let k = 0; k < n; k++) {
          const sf = fr.steps[k], a = sf.present, x = px(k), X = cx(k);
          const on = fr.active === k && (fr.phase === 'forward' || fr.phase === 'backward');
          edge([[X, base.yTok - 15], [x, y + bh / 2]], a, fr.phase === 'forward' ? sf.x : 0, 0, 1);
          if (k) {
            const prev = px(k - 1) + bw / 2, nx = x - bw / 2;
            if (nx - prev > 4) {
              edge([[prev, y], [nx, y]], a, sf.rec, cut(k) ? 0 : sf.flow, 1.5 + 6 * gFrac(P.series[k]), 1.8);
              if (init.bptt && cut(k)) { ctx.setLineDash([4, 4]); line([[(prev + nx) / 2, y - 34], [(prev + nx) / 2, y + 34]], c.grad.hex, 1.6, a); ctx.setLineDash([]); txt('stop', (prev + nx) / 2, y - 40, c.grad.hex, 10.5, 'center', 600, a); }
              if (nx - prev > 26) txt('W_hh', (prev + nx) / 2, y - 7, c.muted.hex, 10.5, 'center', 500, 0.9 * a);
            }
          } else {
            line([[x - bw / 2 - 20, y], [x - bw / 2, y]], c.muted.hex, 1.6, 0.55 * a);
            txt('h0', x - bw / 2 - 22, y + 4, c.muted.hex, 11, 'right', 500, a);
          }
          box(x - bw / 2, y - bh / 2, bw, bh, 7, a, on ? c.accent.hex : c.rule.hex, on ? 2.2 : 1.2);
          tint(x - bw / 2, y - bh / 2, bw, bh, 7, c.fwd.hex, a * 0.08 * sf.hP);
          txt('tanh', x, y + 4, c.text.hex, 12, 'center', 600, a);
          line([[x, y - bh / 2], [x, base.yH + chip / 2]], c.muted.hex, 1.4, 0.5 * a * sf.hP);
          chips(x, base.yH, R.h[k].map((q) => q * sf.hP), chip, a * (0.25 + 0.75 * sf.hP));
          if (showNums) txt(`h${k + 1}`, x + (H * chip) / 2 + 4, base.yH + 4, c.muted.hex, 10.5, 'left', 500, a * sf.hP);
          if (sf.outP > 0) {
            line([[x, base.yH - chip / 2], [x, base.yOut + 8]], c.muted.hex, 1.2, 0.5 * a * sf.outP);
            txt(`ŷ ${fmt(R.o[k])}`, x, base.yOut + 3, c.text.hex, showNums ? 12 : 10, 'center', 600, a * sf.outP);
            if (init.bptt && R.lossMask[k]) txt('L', x, base.yOut - 11, c.grad.hex, 12, 'center', 700, a * sf.outP);
          }
          if (init.bptt && sf.gradP > 0) {
            tint(x - bw / 2, y - bh / 2, bw, bh, 7, c.grad.hex, a * sf.gradP * 0.16);
            box(x - bw / 2, y - bh / 2, bw, bh, 7, a * sf.gradP, c.grad.hex, 1.4, null);
          }
        }
        if (un > 0.6) txt('the same W_xh, W_hh and b at every step', cx((n - 1) / 2), base.yX + 26, c.muted.hex, 12.5, 'center', 400, (un - 0.6) / 0.4, true);
      } else {
        drawGated(fr, n, chip);
      }

      for (let k = 0; k < n; k++) { // tokens + x chips
        const X = cx(k), sf = fr.steps[k], on = fr.active === k && fr.phase === 'forward';
        const a = k === 0 ? 1 : 0.5 + 0.5 * un;
        box(X - 14, base.yTok - 15, 28, 30, 5, a, on || hoverK === k || sel === k ? c.accent.hex : c.rule.hex, on || sel === k ? 2.2 : 1.2);
        txt(showTok(R.toks[k]), X, base.yTok + 6, c.text.hex, 16, 'center', 600, a);
        if (cw >= 40) chips(X, base.yX, R.xs[k], Math.min(chip, 9), a * (0.3 + 0.7 * Math.max(sf.x, fr.phase === 'backward' ? 1 : 0)));
        if (showNums) txt(`t=${k + 1}`, X + 20, base.yTok - 5, c.muted.hex, 10, 'left', 400, 0.8 * a);
      }
      if (init.bptt) drawChart(fr, n, showNums);
      ctx.globalAlpha = 1;
    }

    function drawGated(fr: ReturnType<typeof frameAt>, n: number, chip: number) {
      const P = plan, R = P.tr, { w: cw, cx, x0 } = cols, lstm = init.cell === 'lstm';
      const yB = base.yBelt, yM = base.yM, mH = base.mH, yG = base.yG, yHh = base.yH, un = fr.unroll, T = 11, G = R.gates;
      const px = (k: number) => cx(0) + (cx(k) - cx(0)) * (k ? un : 0);
      const xL = px(0) - cw / 2, xR = px(n - 1) + cw / 2, anim = playing && !reduce;
      const names = lstm ? ['f', 'i', 'o'] : ['z', 'r'];
      const gap = (lstm ? 0.27 : 0.36) * cw, mw = Math.max(5, Math.min(11, cw * 0.13)), showVal = gap >= 27, showLet = gap >= 9;
      const boxTop = yB + T / 2 + 8, boxBot = (lstm ? yHh : yG) + 16;
      // the belt: rails, a moving texture, and row names in the left margin
      line([[xL, yB - T / 2], [xR, yB - T / 2]], c.rule.hex, 1.2);
      line([[xL, yB + T / 2], [xR, yB + T / 2]], c.rule.hex, 1.2);
      ctx.globalAlpha = 0.45; ctx.fillStyle = c.muted.hex;
      const off = anim ? (nowMs / 40) % 22 : 0;
      for (let x = xL + off; x < xR - 6; x += 22) { ctx.beginPath(); ctx.moveTo(x, yB - 3); ctx.lineTo(x + 4, yB); ctx.lineTo(x, yB + 3); ctx.lineTo(x + 1.5, yB); ctx.fill(); }
      txt(lstm ? 'cell state c: the conveyor' : 'hidden state h: the highway', xL + 2, yB - T / 2 - 9, c.muted.hex, 12, 'left', 400, 0.9, true);
      const lx = x0 - 6;
      txt(lstm ? 'c' : 'h', lx, yB + 4, c.muted.hex, 11, 'right', 600);
      if (!narrow) txt('gates', lx, yM + mH / 2 + 4, c.muted.hex, 10, 'right', 500, 0.9);
      txt(lstm ? 'g' : 'n', lx, yG + 4, c.muted.hex, 11, 'right', 600);
      if (lstm) txt('h', lx, yHh + 4, c.muted.hex, 11, 'right', 600);
      for (let k = 0; k < n; k++) {
        const sf = fr.steps[k], a = sf.present, x = px(k), open = sf.hP;
        const on = fr.active === k && (fr.phase === 'forward' || fr.phase === 'backward');
        const gv = names.map((g) => mean(G[g][k]));
        const keep = lstm ? gv[0] : 1 - gv[0];
        // how much of the old state passes along the belt into this step
        ctx.globalAlpha = a * open * (0.12 + 0.6 * keep); ctx.fillStyle = c.fwd.hex;
        ctx.fillRect(x - cw / 2, yB - T / 2 + 1, cw, T - 2);
        box(x - cw * 0.47, boxTop, cw * 0.94, boxBot - boxTop, 8, a, on ? c.accent.hex : c.rule.hex, on ? 2.2 : 1.2, null);
        edge([[cx(k), base.yTok - 15], [x, boxBot]], a, fr.phase === 'forward' ? sf.x : 0, 0, 1);
        if (init.bptt && cut(k)) { ctx.setLineDash([4, 4]); line([[x - cw * 0.5, yB - 22], [x - cw * 0.5, boxBot]], c.grad.hex, 1.6, a); ctx.setLineDash([]); txt('stop', x - cw * 0.5, yB - 26, c.grad.hex, 10.5, 'center', 600, a); }
        // gate meters: the fill height is the gate value (0 closed, 1 open), averaged over the units
        gv.forEach((v, j) => {
          const mx = x + (lstm ? (j - 1) * 0.27 : (j - 0.5) * 0.36) * cw;
          ctx.globalAlpha = a; ctx.fillStyle = c.raised.hex; ctx.fillRect(mx - mw / 2, yM, mw, mH);
          ctx.globalAlpha = a * open; ctx.fillStyle = seqCol(0.35 + 0.65 * v);
          ctx.fillRect(mx - mw / 2, yM + mH * (1 - v), mw, mH * v);
          ctx.globalAlpha = a; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(mx - mw / 2, yM, mw, mH);
          if (showLet) txt(names[j], mx, yM + mH + 13, c.text.hex, 11, 'center', 700, a);
          if (showVal) txt(fmt(v), mx, yM + mH + 25, c.muted.hex, 10, 'center', 400, a * open);
        });
        // the belt value after this step, the candidate, and (LSTM) the read-out h
        chips(x, yB, (lstm ? R.c[k] : R.h[k]).map((q) => q * open), chip, a * (0.25 + 0.75 * open));
        chips(x, yG, (lstm ? G.g[k] : G.n[k]).map((q) => q * open), chip, a * (0.25 + 0.75 * open));
        if (lstm) {
          chips(x, yHh, R.h[k].map((q) => q * open), chip, a * (0.25 + 0.75 * open));
          if (k < n - 1) { // h feeds the next step's gates
            const nf = fr.steps[k + 1];
            edge([[x + (H * chip) / 2 + 2, yHh], [px(k + 1) - cw * 0.47, yHh]], nf.present, fr.phase === 'forward' ? nf.rec : 0, 0, 1);
          }
        }
        if (init.bptt && sf.gradP > 0) { // the gradient rides the belt backward
          line([[x + cw * 0.5, yB], [x - cw * 0.5, yB]], c.grad.hex, 1.5 + 7 * gFrac(P.series[k]), 0.85 * a * sf.gradP * sf.flow);
          if (sf.flow > 0 && sf.flow < 1 && !reduce) dot(x + cw * 0.5 - cw * sf.flow, yB, 4.5, c.grad.hex, a);
          tint(x - cw * 0.47, boxTop, cw * 0.94, boxBot - boxTop, 8, c.grad.hex, a * sf.gradP * 0.1);
        }
      }
    }

    function drawChart(fr: ReturnType<typeof frameAt>, n: number, showNums: boolean) {
      const P = plan, { w: cw, cx, x0 } = cols, ch = chartBot - chartTop;
      const left = x0 - 4, right = x0 + cw * n + 4;
      txt(init.cell === 'lstm' ? '|∂L/∂c_t| at each step (log scale)' : '|∂L/∂h_t| at each step (log scale)', left, chartTop - 14, c.muted.hex, 12, 'left', 500, 1);
      for (const e of [-3, -1, 1]) {
        const y = chartBot - gFrac(10 ** e) * ch;
        line([[left, y], [right, y]], c.rule.hex, 1, 0.8);
        txt(`1e${e < 0 ? '−' : ''}${Math.abs(e)}`, left, y - 3, c.muted.hex, 9.5, 'left', 400, 0.9);
      }
      line([[left, chartBot], [right, chartBot]], c.text.hex, 1.2, 0.6);
      for (let k = 0; k < n; k++) {
        const sf = fr.steps[k], a = sf.present * sf.gradP;
        if (a <= 0.01) continue;
        const bw = Math.min(26, cw * 0.5), hgt = gFrac(P.series[k]) * ch * sf.gradP;
        ctx.globalAlpha = 0.9 * a; ctx.fillStyle = c.grad.hex; ctx.fillRect(cx(k) - bw / 2, chartBot - hgt, bw, hgt);
        txt(fmtG(P.series[k]), cx(k), chartBot - hgt - 4, c.text.hex, showNums ? 11 : 9, 'center', 600, a);
        if (showBound && init.cell === 'rnn' && P.tr.jac.length && tk === 0 && !lossAll) {
          const by = chartBot - gFrac(P.series[P.n - 1] * P.tr.jac[k]) * ch;
          line([[cx(k) - bw / 2 - 4, by], [cx(k) + bw / 2 + 4, by]], c.accent.hex, 2.4, a);
        }
      }
      if (showBound && init.cell === 'rnn' && tk === 0 && !lossAll) txt('bound: |g at last step| × ‖∂h_last/∂h_t‖', right, chartTop - 14, c.accent.hex, 11.5, 'right', 500, 1);
    }

    function frame(dt: number, now: number) {
      nowMs = now;
      const P = plan, T = P.tm.T, from = init.loop ? phaseStart(P.tm, init.loop) : 0;
      if (t > T) t = from;
      t = clock({ t, playing, speed }, dt, { duration: T, loopFrom: from, loopTo: T }).t;
      render();
      tl.paint(t);
      // every frame, so the narration and panel never lag the canvas; only the time readout is throttled
      const fr = frameAt(Math.min(t, T), P.n, P.tm);
      let k = fr.phase === 'backward' ? P.n - 1 : fr.active;
      if (fr.phase === 'backward') fr.steps.forEach((s, i) => { if (s.gradP >= 1 && i < k) k = i; }); // the earliest bar fully drawn
      else if (k < 0) { // between steps: the last step that has finished
        k = 0;
        fr.steps.forEach((s, i) => { if (s.hP >= 1) k = i; });
      }
      if (fr.phase !== phase || k !== uiK) { phase = fr.phase; uiK = k; narration = say(fr.phase, Math.min(k, P.n - 1)); }
      if (now - lastRead > 100 || !playing) { lastRead = now; uiT = t; }
      return playing;
    }
    const loop = createLoop(cv, frame, { onView: (inV, first) => { if (inV && first && !reduce) playing = true; } });
    poke = loop.wake;

    const fit = () => {
      const d = dpr(), r = cv.getBoundingClientRect();
      const w = Math.round(r.width * d), h = Math.round(r.height * d);
      if (w && h && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
    };
    const ro = new ResizeObserver(([e]) => { narrow = e.contentRect.width < 540; fit(); poke(); });
    ro.observe(cv.parentElement!);
    const off = onThemeChange(() => { c = readColors(C); code = readVar('--font-code'); ui = readVar('--font-body'); poke(); });
    const onReduce = () => { reduce = reduced.matches; if (reduce) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    narration = say(phase, 0);
    fit();
    return () => { loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); };
  });

  // ---- inspector ----
  type Row = { name: string; v: number[]; kind?: 'seq' };
  const vecs = $derived.by((): Row[] => {
    const k = kShown, r = tr, z0 = Array(H).fill(0);
    const prevH = k ? r.h[k - 1] : z0, prevC = k ? r.c[k - 1] : z0;
    let rows: Row[];
    if (init.cell === 'rnn') rows = [
      { name: `x${k + 1}`, v: r.xs[k] }, { name: 'W_xh x', v: r.wx[k] }, { name: `W_hh h${k}`, v: r.wh[k] },
      { name: 'sum + b', v: r.z[k] }, { name: `h${k + 1} = tanh`, v: r.h[k] },
    ];
    else if (init.cell === 'lstm') rows = [
      { name: 'f forget', v: r.gates.f[k], kind: 'seq' }, { name: 'i input', v: r.gates.i[k], kind: 'seq' }, { name: 'o output', v: r.gates.o[k], kind: 'seq' },
      { name: 'g candidate', v: r.gates.g[k] }, { name: `c${k}`, v: prevC }, { name: `c${k + 1} = f·c + i·g`, v: r.c[k] }, { name: `h${k + 1} = o·tanh(c)`, v: r.h[k] },
    ];
    else rows = [
      { name: 'z update', v: r.gates.z[k], kind: 'seq' }, { name: 'r reset', v: r.gates.r[k], kind: 'seq' }, { name: 'n candidate', v: r.gates.n[k] },
      { name: `h${k}`, v: prevH }, { name: `h${k + 1}`, v: r.h[k] },
    ];
    if (init.bptt && init.cell === 'rnn') rows.splice(5, 0, { name: 'tanh slope 1−h²', v: r.h[k].map((q) => 1 - q * q), kind: 'seq' });
    if (init.bptt) rows.push({ name: `∂L/∂${init.cell === 'lstm' ? 'c' : 'h'}${k + 1}`, v: init.cell === 'lstm' ? r.dc[k] : r.dh[k] });
    return rows;
  });
  const pct = (s: number) => (s >= 1 ? `${s.toFixed(1)}× ` : `${(100 * s).toPrecision(2)}% `);
  const ratioText = $derived.by(() => {
    if (!init.bptt) return '';
    if (tk > 0) return lossAll ? `Truncated every ${tk} steps: the gradient stops at each dashed line, so each chunk learns only from its own losses.` : `Truncated every ${tk} steps: the gradient stops at each dashed line, so only the last chunk gets a signal from the final loss.`;
    const f = tr.factor;
    if (!Number.isFinite(f)) return 'Loss at every step: each step also receives its own local gradient, so the picture is a sum of many paths.';
    const kind = f < 0.95 ? 'shrinks' : f > 1.05 ? 'grows' : 'holds steady';
    const s0 = plan.series[0] / plan.series[plan.n - 1];
    const clipTxt = showClip && init.cell === 'rnn' ? clipText() : '';
    return `Going back one step the gradient ${kind} by ×${f.toFixed(2)} on average. Step 1 gets ${pct(s0)}of the size at step ${plan.n}.${clipTxt}`;
  });
  const gN = $derived(init.cell === 'rnn' ? Math.hypot(...tr.grads.Wh.h.flat()) : 0);
  const clipText = () => (gN > clipC ? ` Weight-gradient norm ‖∂L/∂W_hh‖ = ${gN.toFixed(1)} is above the clip threshold ${clipC}, so it is scaled by ${(clipC / gN).toFixed(2)} to length ${clipC}.` : ` Weight-gradient norm ‖∂L/∂W_hh‖ = ${gN.toFixed(1)} is below the clip threshold ${clipC}: no change.`);
  const scaleWord = $derived(sc < 0.9 ? 'vanishing regime' : sc > 1.3 ? 'exploding regime' : 'near the edge');
</script>

<FigureCard
  title={init.title} {legend} role="group"
  aspect="720 / {Hh}" narrowAspect="400 / {Hh}"
  label="{NAMES[init.cell]} cell reading the sequence '{seq}' one character at a time, shown folded and unrolled across time steps{init.bptt ? ', with gradients flowing backward through time' : ''}."
  asideLabel="Inspect one time step"
  aside={showPanel ? panel : undefined}
>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  <div class="hits">
    {#each { length: plan.n } as _, k (k)}
      <button type="button" class="hit" aria-pressed={sel === k}
        style:left="{((cols.x0 + cols.w * k) / Wd) * 100}%" style:width="{(cols.w / Wd) * 100}%"
        style:top="{((base.yTok - 20) / Hh) * 100}%" style:height="{(60 / Hh) * 100}%"
        aria-label="Time step {k + 1}, character {showTok(tr.toks[k])}: open in the inspector"
        onclick={() => pick(k)} onpointerenter={() => { hoverK = k; poke(); }} onpointerleave={() => { hoverK = -1; poke(); }}
        onfocus={() => { hoverK = k; poke(); }} onblur={() => { hoverK = -1; poke(); }}></button>
    {/each}
  </div>

  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={plan.tm.T} {markers} label="Timeline"
      {onSeek} onStep={(to) => { playing = false; t = to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}

  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    {#if init.bptt}<p class="next">{ratioText}</p>{/if}
  {/snippet}

  {#snippet footer()}
    <div class="ctl">
      {#if editable}
        <label>sequence
          <input class="seqin" type="text" bind:value={seq} maxlength={init.maxLen} spellcheck="false" autocomplete="off" aria-label="Input sequence, one character per time step, up to {init.maxLen} characters" />
        </label>
      {/if}
      {#if showScale}
        <label class="sc">recurrent weight scale s
          <input type="range" min="0.2" max="3" step="0.05" bind:value={sc} aria-describedby="{uid}-sc" />
          <output class="num" id="{uid}-sc">{sc.toFixed(2)}</output>
        </label>
        {#if init.cell === 'rnn'}<span class="regime">{scaleWord}</span>{/if}
      {/if}
      {#if showForget && init.cell === 'lstm'}
        <label class="sc">forget-gate bias
          <input type="range" min="0" max="6" step="0.5" bind:value={fb} aria-describedby="{uid}-fb" />
          <output class="num" id="{uid}-fb">{fb.toFixed(1)}</output>
        </label>
        <span class="regime">f ≈ {(1 / (1 + Math.exp(-fb))).toFixed(2)} on letters</span>
      {/if}
      {#if showTruncate}
        <label>truncate every
          <select bind:value={tk} aria-label="Truncated backpropagation: stop gradients every k steps">
            <option value={0}>never</option><option value={2}>2 steps</option><option value={3}>3 steps</option><option value={4}>4 steps</option><option value={6}>6 steps</option>
          </select>
        </label>
      {/if}
      {#if showClip}
        <label class="sc">clip threshold
          <input type="range" min="0.5" max="20" step="0.5" bind:value={clipC} />
          <output class="num">{clipC}</output>
        </label>
      {/if}
      {#if showLossToggle}
        <label class="chk"><input type="checkbox" bind:checked={lossAll} /> loss at every step</label>
      {/if}
    </div>
  {/snippet}
</FigureCard>

{#snippet panel()}
  <div class="panel">
    <header>
      <h4 class="pt">Step {kShown + 1}: reads “{showTok(tr.toks[kShown] ?? '')}”</h4>
      <span class="what">{sel >= 0 ? 'pinned' : 'follows the animation'}</span>
      {#if sel >= 0}<button type="button" class="x" onclick={() => (sel = -1)} aria-label="Unpin step">×</button>{/if}
    </header>
    <table class="vt">
      <tbody>
        {#each vecs as r (r.name)}
          <tr>
            <th scope="row">{r.name}</th>
            <td>
              <span class="vs">
                {#each r.v as q, j (j)}
                  <span class="vc num" style:--c={r.kind === 'seq' ? 'var(--viz-seq-4)' : q >= 0 ? 'var(--viz-pos)' : 'var(--viz-neg)'} style:--a="{r.kind === 'seq' ? 0.15 + 0.85 * q : Math.min(1, Math.abs(q))}">{fmt(q)}</span>
                {/each}
              </span>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
    {#if init.bptt && init.cell === 'rnn'}
      <p class="hint">|∂L/∂h{kShown + 1}| = <span class="num">{fmtG(tr.dhN[kShown])}</span>. This step adds <span class="num">{fmtG(tr.contrib[kShown])}</span> to the norm of ∂L/∂W_hh; the full gradient is the sum over all steps.</p>
    {:else if init.bptt}
      <p class="hint">Gradient norm at this step: <span class="num">{fmtG(plan.series[kShown])}</span>.</p>
    {:else}
      <p class="hint">Click a time step under the chain to pin it. Values are the real numbers from this tiny network (hidden size {H}).</p>
    {/if}
  </div>
{/snippet}

<style>
  .hits { position: absolute; inset: 0; }
  .hit { position: absolute; padding: 0; border: 0; background: transparent; cursor: pointer; border-radius: 6px; }
  .hit:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .say, .next { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-code); font-variant-numeric: tabular-nums; min-height: 2.6em; }
  .next { color: var(--text-muted); font-size: var(--step--1); margin-top: 0.2em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .seqin {
    width: 11rem; font: 0.9rem var(--font-code); padding: 0.3em 0.5em; color: var(--text);
    background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1);
  }
  .sc input { width: 8rem; accent-color: var(--accent); }
  .sc output { min-width: 2.6rem; color: var(--text); }
  .regime { font: var(--step--1) var(--font-ui); color: var(--text-muted); font-style: italic; }
  .chk input { accent-color: var(--accent); }
  .ctl select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.3em 0.4em; }

  .panel { font-family: var(--font-ui); min-width: 0; }
  .panel header { display: flex; align-items: baseline; gap: var(--s-2); margin-bottom: var(--s-2); }
  .pt { font: 650 1rem var(--font-ui); margin: 0; }
  .what { font: 0.78rem var(--font-ui); color: var(--text-muted); }
  .x { margin-left: auto; background: none; border: 0; cursor: pointer; font-size: 1.2rem; line-height: 1; color: var(--text-muted); padding: 0 0.2em; }
  .vt { border-collapse: collapse; width: 100%; }
  .vt th { font: 0.7rem / 1.25 var(--font-code); font-weight: 400; color: var(--text-muted); text-align: left; padding: 0.15em 0.5em 0.15em 0; max-width: 6.2rem; }
  .vs { display: inline-flex; gap: 2px; }
  .vc {
    display: inline-block; min-width: 2.55rem; box-sizing: border-box; text-align: center; font: 0.7rem var(--font-code); padding: 0.15em 0.1em;
    background: color-mix(in srgb, var(--c) calc(var(--a) * 100%), var(--bg-raised)); border: 1px solid var(--rule);
    color: var(--text); font-variant-numeric: tabular-nums;
  }
  .hint { color: var(--text-muted); font: 0.8rem / 1.45 var(--font-ui); margin: 0.6em 0 0; }
</style>
