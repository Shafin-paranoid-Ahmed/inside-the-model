<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels } from '../F1Shared/plot';
  import { T, MARKERS, N, P, LMAX, NAMES, TRUE_W, PATHS, W2_OLS, frameAt, ellipse, excess2, type Penalty } from './sim';

  interface Props {
    title?: string;
    /** Which penalty is shown first. */
    penalty?: Penalty;
    /** Show the L1 / L2 switch. */
    toggle?: boolean;
  }
  let { title = 'Regularization: the constraint region and the weight paths', penalty = 'l1', toggle = true }: Props = $props();
  const init = untrack(() => ({ pen: penalty, toggle }));

  let pen = $state<Penalty>(init.pen);
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let ui = $state(frameAt(0, init.pen));
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  $effect(() => { playing; speed; pen; wake(); });

  const name = $derived(pen === 'l1' ? 'L1' : 'L2');
  const narration = $derived(
    ui.phase === 'no penalty' ? 'No penalty yet: the optimum (filled dot) sits at the unpenalized fit (ringed diamond), and the region is just big enough to reach it. Every weight is free.'
      : ui.phase === 'strong penalty'
        ? pen === 'l1' ? 'Strong L1 penalty: every weight has been pushed to exactly zero.' : 'Strong L2 penalty: the weights are far smaller than at the start, but none is exactly zero.'
        : pen === 'l1'
          ? ui.w2.includes(0)
            ? `λ = ${f2(ui.lam)}. The optimum sits on a corner of the diamond, so one of the two weights is exactly zero. ${ui.zeros} of 6 weights in the full model are zero.`
            : `λ = ${f2(ui.lam)}. The diamond shrinks; the optimum is where a loss contour first touches it. ${ui.zeros} of 6 weights are zero.`
          : `λ = ${f2(ui.lam)}. The circle shrinks and every weight glides toward zero together. None reaches it. ${ui.zeros} of 6 weights are zero.`,
  );

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const LEVELS = [0.12, 0.35, 0.7, 1.2, 1.9, 2.8].map((d) => ellipse(d));
    const lay = () => {
      const { main, side, stacked } = splitPanels(stage.w, stage.h, stage.w < 540 ? 0.42 : 0.44);
      const a = { l: main.x + 30, r: main.x + main.w - 10, t: main.y + 30, b: main.y + main.h - 24 };
      // the largest region (L1 at lambda = 0) reaches 2.3 on each axis, so show -2.5..2.5 both ways
      const ppu = Math.min(a.r - a.l, a.b - a.t) / 5.1, cx = (a.l + a.r) / 2, cy = (a.t + a.b) / 2;
      const ph = Math.round(side.h * 0.62);
      const pa = { l: side.x + 42, r: side.x + side.w - 10, t: side.y + 30, b: side.y + ph - 26 };
      const va = { l: pa.l, r: pa.r, t: side.y + ph + 22, b: side.y + side.h - 22 };
      return { main, side, stacked, a, ppu, cx, cy, pa, va, sy: (v: number) => cy - v * ppu, sx: (v: number) => cx + v * ppu };
    };
    const feat = (j: number) => (TRUE_W[j] > 0 ? 'pos' : TRUE_W[j] < 0 ? 'neg' : 'muted') as 'pos' | 'neg' | 'muted';
    const val = (id: 'l1' | 'l2') => PATHS[id];
    const VY: [number, number] = [Math.min(...['l1', 'l2'].map((k) => Math.min(...val(k as Penalty).train))) - 0.4, 9];

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay(), f = frameAt(t, pen), p = PATHS[pen];
      ctx.clearRect(0, 0, W, H);
      const num = (v: number) => f2(v, v === Math.round(v) ? 0 : 1);
      ctx.lineWidth = 1; ctx.lineJoin = 'round';
      // ---- panel 1: contours + constraint region ----
      const { a } = L;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Two weights: loss contours and the penalty region', L.main.x + 2, L.main.y + 2);
      ctx.save(); ctx.beginPath(); ctx.rect(a.l, a.t, a.r - a.l, a.b - a.t); ctx.clip();
      ctx.strokeStyle = c.rule.hex; ctx.beginPath();
      ctx.moveTo(a.l, L.sy(0)); ctx.lineTo(a.r, L.sy(0)); ctx.moveTo(L.sx(0), a.t); ctx.lineTo(L.sx(0), a.b); ctx.stroke();
      const poly = (e: number[]) => { ctx.beginPath(); for (let i = 0; i < e.length; i += 2) (i ? ctx.lineTo : ctx.moveTo).call(ctx, L.sx(e[i]), L.sy(e[i + 1])); };
      LEVELS.forEach((e, i) => { poly(e); ctx.strokeStyle = c.s4.hex; ctx.globalAlpha = 0.3 + 0.06 * i; ctx.stroke(); });
      ctx.globalAlpha = 1;
      const ex = excess2(f.w2);
      if (ex > 1e-6) { poly(ellipse(ex)); ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1.6; ctx.stroke(); }
      const r = f.radius * L.ppu;
      if (r > 0.5) {
        ctx.beginPath();
        if (pen === 'l1') { ctx.moveTo(L.sx(0) + r, L.sy(0)); ctx.lineTo(L.sx(0), L.sy(0) - r); ctx.lineTo(L.sx(0) - r, L.sy(0)); ctx.lineTo(L.sx(0), L.sy(0) + r); ctx.closePath(); }
        else ctx.arc(L.sx(0), L.sy(0), r, 0, 6.2832);
        ctx.fillStyle = `rgba(${c.accent.rgb.join(',')},0.12)`; ctx.fill();
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.stroke();
      }
      // path of the solution so far
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.8; ctx.beginPath();
      const K = Math.floor(f.k);
      for (let i = 0; i <= K; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, L.sx(p.w2[i * 2]), L.sy(p.w2[i * 2 + 1]));
      ctx.lineTo(L.sx(f.w2[0]), L.sy(f.w2[1])); ctx.stroke(); ctx.globalAlpha = 1;
      const ox = L.sx(W2_OLS[0]), oy = L.sy(W2_OLS[1]);
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(ox, oy - 6); ctx.lineTo(ox + 6, oy); ctx.lineTo(ox, oy + 6); ctx.lineTo(ox - 6, oy); ctx.closePath(); ctx.stroke();
      ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(ox, oy, 11, 0, 6.2832); ctx.stroke();
      ctx.font = stage.body(11, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('unpenalized fit', ox - 12, oy - 10);
      ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(L.sx(f.w2[0]), L.sy(f.w2[1]), 6, 0, 6.2832); ctx.fill(); ctx.stroke();
      ctx.restore();
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [-2, -1, 0, 1, 2]) ctx.fillText(String(v).replace('-', '−'), L.sx(v), a.b + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [-2, -1, 1, 2]) ctx.fillText(String(v).replace('-', '−'), a.l - 5, L.sy(v));
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('w₁', a.r, L.main.y + L.main.h - 1);
      ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('w₂', L.main.x + 2, a.t - 1);

      // ---- panel 2: coefficient paths ----
      const { pa } = L, X = scale(0, LMAX, pa.l, pa.r), Y = scale(-2.4, 3.1, pa.b, pa.t), k = f.k;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(`${name} coefficient paths, six features`, pa.l - 30, L.side.y + 2);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(pa.l, pa.t, pa.r - pa.l, pa.b - pa.t);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [-2, 0, 2]) { ctx.strokeStyle = v ? c.rule.hex : c.muted.hex; ctx.globalAlpha = v ? 0.6 : 0.8; ctx.beginPath(); ctx.moveTo(pa.l, Y(v)); ctx.lineTo(pa.r, Y(v)); ctx.stroke(); ctx.globalAlpha = 1; ctx.fillText(String(v).replace('-', '−'), pa.l - 5, Y(v)); }
      ctx.save(); ctx.beginPath(); ctx.rect(pa.l, pa.t, pa.r - pa.l, pa.b - pa.t); ctx.clip();
      for (let j = 0; j < P; j++) {
        const col = c[feat(j)], line = (to: number) => { ctx.beginPath(); for (let i = 0; i <= to; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, X(p.lam[i]), Y(p.w6[i * P + j])); };
        ctx.strokeStyle = col.hex; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.28; line(N - 1); ctx.stroke();
        ctx.globalAlpha = 1; ctx.lineWidth = feat(j) === 'muted' ? 1.6 : 2.2; ctx.setLineDash(feat(j) === 'muted' ? [4, 3] : []);
        line(Math.floor(k)); ctx.lineTo(X(f.lam), Y(f.w6[j])); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(f.lam), pa.t); ctx.lineTo(X(f.lam), pa.b); ctx.stroke();
      for (let j = 0; j < P; j++) {
        ctx.beginPath(); ctx.arc(X(f.lam), Y(f.w6[j]), 3.6, 0, 6.2832);
        if (f.w6[j] === 0) { ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c[feat(j)].hex; ctx.lineWidth = 2; ctx.fill(); ctx.stroke(); } else { ctx.fillStyle = c[feat(j)].hex; ctx.fill(); }
      }
      ctx.restore();
      ctx.font = stage.mono(10.5); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      for (let j = 0; j < 3; j++) { ctx.fillStyle = c[feat(j)].hex; ctx.fillText(NAMES[j], pa.l + 4, Y(p.w6[j]) + (j === 2 ? 8 : -8)); }
      ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.fillText('x4, x5, x6: no true effect', pa.r - 6, Y(0) - 9);
      ctx.textBaseline = 'top'; ctx.textAlign = 'left'; ctx.fillText('λ = 0', pa.l, pa.b + 4);
      ctx.textAlign = 'right'; ctx.fillText(`λ = ${num(LMAX)}  (penalty strength →)`, pa.r, pa.b + 4);
      ctx.textAlign = 'left'; ctx.save(); ctx.translate(L.side.x + 8, (pa.t + pa.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('weight', 0, 0); ctx.restore();

      // ---- panel 3: error vs lambda ----
      const { va } = L, XV = scale(0, LMAX, va.l, va.r), YV = scale(VY[0], VY[1], va.b, va.t);
      ctx.font = stage.body(12); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Error as λ grows', pa.l - 30, va.t - 20);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(va.l, va.t, va.r - va.l, va.b - va.t);
      ctx.save(); ctx.beginPath(); ctx.rect(va.l, va.t, va.r - va.l, va.b - va.t); ctx.clip();
      for (const [arr, col, wd] of [[p.train, c.muted, 1.6], [p.val, c.text, 2.2]] as const) {
        ctx.strokeStyle = col.hex; ctx.lineWidth = wd; ctx.beginPath();
        for (let i = 0; i < N; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, XV(p.lam[i]), YV(arr[i]));
        ctx.stroke();
      }
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(XV(f.lam), va.t); ctx.lineTo(XV(f.lam), va.b); ctx.stroke();
      ctx.fillStyle = c.text.hex; ctx.beginPath(); ctx.arc(XV(f.lam), YV(f.val), 3.6, 0, 6.2832); ctx.fill();
      ctx.fillStyle = c.muted.hex; ctx.beginPath(); ctx.arc(XV(f.lam), YV(f.train), 3.2, 0, 6.2832); ctx.fill();
      const bx = XV(p.lam[p.best]), by = YV(p.val[p.best]);
      ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(bx, by - 6); ctx.lineTo(bx + 6, by); ctx.lineTo(bx, by + 6); ctx.lineTo(bx - 6, by); ctx.closePath(); ctx.stroke();
      ctx.restore();
      ctx.font = stage.mono(10.5); ctx.textBaseline = 'top';
      ctx.textAlign = 'left'; ctx.fillStyle = c.muted.hex; ctx.fillText('λ = 0', va.l, va.b + 4);
      ctx.textAlign = 'right'; ctx.fillText(`λ = ${num(LMAX)}`, va.r, va.b + 4);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [4, 8]) ctx.fillText(String(v), va.l - 5, YV(v));
      ctx.save(); ctx.translate(L.side.x + 8, (va.t + va.b) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('error', 0, 0); ctx.restore();
      ctx.font = stage.body(11.5, true); ctx.textAlign = 'left';
      const iv = Math.round(N * 0.16), it = Math.round(N * 0.16);
      ctx.fillStyle = c.text.hex; ctx.textBaseline = 'bottom'; ctx.fillText('validation', XV(p.lam[iv]), YV(p.val[iv]) - 6);
      ctx.fillStyle = c.muted.hex; ctx.textBaseline = 'top'; ctx.fillText('training', XV(p.lam[it]), YV(p.train[it]) + 6);
      ctx.textAlign = 'center'; ctx.fillStyle = c.text.hex; ctx.fillText('◇ best λ on validation', (va.l + va.r) / 2, va.b + 4);
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; ui = frameAt(t, pen); uiT = t; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });

  const choose = (p: Penalty) => { pen = p; ui = frameAt(t, p); };
</script>

<FigureCard {title} aspect="720 / 430" narrowAspect="360 / 720"
  label="Left: contours of a two-weight loss with the penalty region drawn as a diamond for L1 or a circle for L2, and the constrained optimum on its edge. Top right: how six coefficients change as the penalty strength lambda grows. Bottom right: training and validation error against lambda."
  legend={[
    { label: 'penalty region', color: '--accent', shape: 'line' },
    { label: 'constrained optimum', color: '--viz-forward' },
    { label: 'weight > 0', color: '--viz-pos', shape: 'line' },
    { label: 'weight < 0', color: '--viz-neg', shape: 'line' },
    { label: 'x4–x6: no true effect', color: '--text-muted', shape: 'line' },
  ]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Timeline: raising the penalty strength"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={() => { t = 0; playing = true; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">{name} · λ = <b>{f2(ui.lam)}</b> · zero weights = <b>{ui.zeros}</b> of {P} · error on new data = <b>{f2(ui.val)}</b> <span class="base">(training {f2(ui.train)})</span></p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if init.toggle}
      <div class="ctl" role="group" aria-label="Penalty type">
        <button type="button" class="btn" aria-pressed={pen === 'l1'} class:on={pen === 'l1'} onclick={() => choose('l1')}>L1 (diamond)</button>
        <button type="button" class="btn" aria-pressed={pen === 'l2'} class:on={pen === 'l2'} onclick={() => choose('l2')}>L2 (circle)</button>
        <span class="hint">Switch while it plays to compare the two.</span>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .base { font-size: 0.8rem; color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .ctl .btn { padding: 0.4em 0.9em; }
  .ctl .btn.on { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
