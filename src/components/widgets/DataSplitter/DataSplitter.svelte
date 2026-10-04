<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import {
    N, EPOCHS, NOISE_FLOOR, T, T_SPLIT, T_TUNE, NCAND, DEGREES, makeRun, makeTune, markers, tuneMarkers, frameAt, dotsAt, epochAt, uOfEpoch,
    curveEpochs, scores, pick, triedAt,
  } from './sim';

  interface Props {
    title?: string;
    /** 'curves': train one model, watch train vs validation loss. 'tune': try 30 settings and pick by validation or by (peeking at) test. */
    mode?: 'curves' | 'tune';
    /** Starting fraction of the data used for training / validation (test gets the rest). */
    trainFrac?: number;
    valFrac?: number;
    /** Hide the split sliders and drag handles (the lesson step fixes the split). */
    lockSplit?: boolean;
    /** Timeline position to start (and loop back) from; T_SPLIT skips the dot flow. */
    startAt?: number;
  }
  let { title = 'One dataset, three bins', mode = 'curves', trainFrac: tf0 = 0.4, valFrac: vf0 = 0.2, lockSplit = false, startAt = 0 }: Props = $props();
  const init = untrack(() => ({ tf: tf0, vf: vf0, lockSplit, t0: startAt, tune: mode === 'tune' }));
  const TEND = init.tune ? T_TUNE : T;

  let trainFrac = $state(init.tf), valFrac = $state(init.vf);
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let t = 0, wake = () => {}, drag = 0, first = true;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const R = $derived(init.tune ? null : makeRun(trainFrac, valFrac));
  const U = $derived(init.tune ? makeTune(trainFrac, valFrac) : null);
  const base = $derived((R ?? U)!);
  const MK = $derived(R ? markers(R) : tuneMarkers());
  const sc = $derived(R ? scores(R) : null);
  const ei = $derived(Math.min(EPOCHS, Math.round(epochAt(uiT))));
  const phase = $derived(R ? frameAt(uiT, R).phase : uiT < T_SPLIT ? 'split' : uiT < T_TUNE ? 'tuning' : 'picked');
  const chosen = $derived(R && sc ? epochAt(uiT) >= sc.epoch && uiT > T_SPLIT : uiT >= T_TUNE);
  const hon = $derived(U ? pick(U, false) : null);
  const cheat = $derived(U ? pick(U, true) : null);
  const narration = $derived(
    phase === 'split' ? `Splitting: ${base.split.nTrain} points for training, ${base.split.nVal} for validation, ${base.split.nTest} for test.`
      : phase === 'train' ? 'Training: both losses fall while the model learns the pattern.'
      : phase === 'stop' ? 'Past the early-stopping point: training loss keeps falling while validation loss rises. The model is memorizing noise.'
      : phase === 'overfit' ? 'End of training: the training loss is far below the validation loss.'
      : phase === 'tuning' ? 'Trying 30 candidate models, each scored on validation, on test and on fresh data.'
      : 'Picked. The test-picked model looks better on test than it is on fresh data; the validation-picked one is scored fairly.',
  );

  $effect(() => {
    base; // moving the split retrains from scratch
    untrack(() => {
      if (first) { first = false; t = reducedMotion() ? TEND : init.t0; playing = !reducedMotion(); }
      else { t = reducedMotion() ? TEND : T_SPLIT; playing = !reducedMotion(); }
      wake();
    });
  });
  $effect(() => { speed; wake(); });

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const strip = () => ({ x: 6, y: 20, w: stage.w - 12, h: Math.min(96, stage.h * (stage.w < 540 ? 0.3 : 0.28)) });
    const plot = () => {
      const s = strip(), top = s.y + s.h + 34, narrow = stage.w < 540;
      return { l: narrow ? 40 : 52, r: stage.w - 14, t: top, b: stage.h - 30 };
    };

    function render() {
      const { ctx, w: W, h: H, c } = stage, s = strip(), P = plot();
      const cols = [c.fwd, c.accent, c.neg], names = ['Train', 'Validation', 'Test'];
      const b0 = base, dots = dotsAt(t, b0);
      ctx.clearRect(0, 0, W, H);
      // ---- split strip ----
      const counts = [b0.split.nTrain, b0.split.nVal, b0.split.nTest];
      let x0 = s.x;
      ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
      counts.forEach((n, b) => {
        const w = (s.w * n) / N, sealed = b === 2 && !init.tune;
        ctx.fillStyle = cols[b].hex; ctx.globalAlpha = 0.1; ctx.fillRect(x0, s.y, w, s.h); ctx.globalAlpha = 1;
        ctx.strokeStyle = cols[b].hex; ctx.lineWidth = 1.2; ctx.setLineDash(sealed ? [4, 3] : []); ctx.strokeRect(x0 + 0.5, s.y + 0.5, w - 1, s.h - 1); ctx.setLineDash([]);
        ctx.font = stage.mono(11.5, 600); ctx.fillStyle = c.text.hex;
        const lab = W < 540 && w < 90 ? `${n}` : `${names[b]} ${n}${sealed ? ' · sealed' : ''}`;
        ctx.save(); ctx.beginPath(); ctx.rect(x0, 0, w, s.y); ctx.clip(); ctx.fillText(lab, x0 + 3, s.y - 6); ctx.restore();
        x0 += w;
      });
      for (let d = 0; d < N; d++) {
        const px = s.x + 4 + dots[2 * d] * (s.w - 8), py = s.y + 4 + dots[2 * d + 1] * (s.h - 8), b = b0.bin[d];
        ctx.fillStyle = cols[b].hex; ctx.globalAlpha = b === 2 && !init.tune && t >= T_SPLIT ? 0.45 : 1;
        ctx.beginPath(); ctx.arc(px, py, 3.2, 0, 6.2832); ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (!init.lockSplit) {
        for (const fx of [b0.split.nTrain / N, (b0.split.nTrain + b0.split.nVal) / N]) {
          const hx = s.x + fx * s.w;
          ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.roundRect(hx - 3.5, s.y + s.h / 2 - 11, 7, 22, 3); ctx.fill(); ctx.stroke();
        }
      }

      // ---- plot frame ----
      const pw = P.r - P.l, ph = P.b - P.t, ymax = b0.ymax;
      const Y = (v: number) => P.b - (v / ymax) * ph;
      ctx.font = stage.mono(11, 500); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1;
      const yt = ymax > 0.9 ? 0.25 : 0.2;
      for (let v = 0; v <= ymax + 1e-9; v += yt) { ctx.beginPath(); ctx.moveTo(P.l, Y(v)); ctx.lineTo(P.r, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(1), P.l - 6, Y(v)); }
      ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left'; ctx.font = stage.body(12, true);
      ctx.fillText('loss (mean squared error)', P.l, P.t - 6);
      ctx.setLineDash([2, 4]); ctx.strokeStyle = c.muted.hex; ctx.beginPath(); ctx.moveTo(P.l, Y(NOISE_FLOOR)); ctx.lineTo(P.r, Y(NOISE_FLOOR)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.font = stage.body(11.5, true);
      ctx.fillText('noise floor: no model can beat this', P.r, Y(NOISE_FLOOR) - 4);
      ctx.save(); ctx.beginPath(); ctx.rect(P.l, P.t - 2, pw, ph + 4); ctx.clip(); // off-scale values run off the top edge instead of flattening

      if (R && t > T_SPLIT) {
        const e = epochAt(t), uc = uOfEpoch(e), X = (u: number) => P.l + u * pw;
        const es = curveEpochs(e);
        ctx.lineJoin = 'round'; ctx.lineWidth = 2.2;
        for (const [arr, col] of [[R.train, c.fwd], [R.val, c.accent]] as const) {
          ctx.strokeStyle = col.hex; ctx.beginPath();
          es.forEach((ee, j) => {
            const i0 = Math.floor(ee), i1 = Math.min(EPOCHS, i0 + 1), v = arr[i0] + (arr[i1] - arr[i0]) * (ee - i0);
            if (j) ctx.lineTo(X(uOfEpoch(ee)), Y(v)); else ctx.moveTo(X(uOfEpoch(ee)), Y(v));
          });
          ctx.stroke();
          const i = Math.min(EPOCHS, Math.round(e));
          ctx.fillStyle = col.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(uc), Y(arr[i]), 4.5, 0, 6.2832); ctx.fill(); ctx.stroke(); ctx.lineWidth = 2.2;
        }
        const sx = X(uOfEpoch(R.bestVal));
        if (e >= R.bestVal) {
          ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(sx, P.t); ctx.lineTo(sx, P.b); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = c.accent.hex; ctx.font = stage.mono(11.5, 700); ctx.textBaseline = 'top';
          const left = sx > P.l + pw * 0.45; ctx.textAlign = left ? 'right' : 'left';
          ctx.fillText('early stop: validation lowest', sx + (left ? -6 : 6), P.t + 2);
        }
      }
      if (U) {
        // candidates in degree-major order: slot = degree index × resamples + resample
        const nd = DEGREES.length, rs = NCAND / nd, tried = triedAt(t);
        const XS = (c0: number) => P.l + ((((c0 % nd) * rs + Math.floor(c0 / nd)) + 0.5) / NCAND) * pw;
        const dot = (c0: number, v: number, col: string, hollow: boolean, square = false) => {
          const a = Math.min(1, Math.max(0, tried - c0));
          if (a <= 0) return;
          ctx.globalAlpha = a; ctx.lineWidth = 1.8; ctx.strokeStyle = col; ctx.fillStyle = hollow ? c.raised.hex : col;
          const r = hollow ? 4 : 3.2 + 1.2 * a; ctx.beginPath();
          if (square) ctx.rect(XS(c0) - r, Y(v) - r, 2 * r, 2 * r); else ctx.arc(XS(c0), Y(v), r, 0, 6.2832);
          ctx.fill(); if (hollow) ctx.stroke(); ctx.globalAlpha = 1;
        };
        for (let c0 = 0; c0 < NCAND; c0++) { dot(c0, U.val[c0], c.accent.hex, false); dot(c0, U.test[c0], c.neg.hex, false, true); dot(c0, U.fresh[c0], c.text.hex, true); }
        if (tried >= NCAND) {
          for (const [cand, col, lab, up] of [[U.bestVal, c.accent, 'picked by validation', false], [U.bestTest, c.neg, 'picked by test', true]] as const) {
            const x = XS(cand);
            ctx.strokeStyle = col.hex; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x, P.t); ctx.lineTo(x, P.b); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = col.hex; ctx.font = stage.mono(11, 700); ctx.textBaseline = 'top';
            const left = x > P.l + pw * 0.55; ctx.textAlign = left ? 'right' : 'left';
            ctx.fillText(lab, x + (left ? -5 : 5), P.t + (up ? 16 : 2));
          }
        }
      }
      ctx.restore();
      // ---- x axis ----
      ctx.fillStyle = c.muted.hex; ctx.font = stage.mono(11, 500); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      if (U) {
        const nd = DEGREES.length, rs = NCAND / nd;
        DEGREES.forEach((dg, i) => ctx.fillText(`deg ${dg}`, P.l + ((i * rs + rs / 2) / NCAND) * pw, P.b + 5));
        ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.fillText('degree × 5 resamples of the training bin', P.r, P.b + 16);
      } else {
        for (const e of [0, 10, 100, 1000]) ctx.fillText(String(e), P.l + uOfEpoch(e) * pw, P.b + 5);
        ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.fillText('training steps (epochs, stretched early)', P.r, P.b + 16);
      }
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: TEND, loopFrom: init.t0, loopTo: TEND + 1.5 }).t;
      const tt = Math.min(t, TEND);
      render(); tl.paint(tt);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = tt; }
      return playing || drag > 0;
    }, { onView: (_v, firstView) => { if (firstView && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const bounds = () => { const s = strip(), sp = base.split; return { s, b1: s.x + (sp.nTrain / N) * s.w, b2: s.x + ((sp.nTrain + sp.nVal) / N) * s.w }; };
    const down = (e: PointerEvent) => {
      if (init.lockSplit) return;
      const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, { s, b1, b2 } = bounds();
      if (y < s.y - 4 || y > s.y + s.h + 4) return;
      if (Math.abs(x - b1) < 14 || Math.abs(x - b2) < 14) { drag = Math.abs(x - b1) <= Math.abs(x - b2) ? 1 : 2; cv.setPointerCapture(e.pointerId); e.preventDefault(); }
    };
    const move = (e: PointerEvent) => {
      const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, { s, b1, b2 } = bounds();
      if (!drag) { cv.style.cursor = !init.lockSplit && y > s.y - 4 && y < s.y + s.h + 4 && (Math.abs(x - b1) < 14 || Math.abs(x - b2) < 14) ? 'ew-resize' : ''; return; }
      const f = Math.round(((x - s.x) / s.w) * N) / N;
      if (drag === 1) { trainFrac = Math.min(0.9, Math.max(0.05, f)); valFrac = Math.min(valFrac, 0.95 - trainFrac); }
      else valFrac = Math.min(0.95 - trainFrac, Math.max(0.05, f - trainFrac));
    };
    const up = () => { drag = 0; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard {title} role="group" aspect="720 / 460" narrowAspect="360 / 640"
  label={init.tune
    ? 'Top: 100 dots flow into training, validation and test bins whose widths you can drag. Bottom: 30 candidate models, each with a validation loss, a test loss and a loss on fresh data; the model picked by test looks better on test than on fresh data.'
    : 'Top: 100 dots flow from a pool into three bins, training, validation and test, whose widths you can drag. Bottom: training and validation loss over training time; the validation curve bottoms out and rises while training loss keeps falling, and the early-stopping point is marked.'}
  legend={init.tune
    ? [{ label: 'validation', color: '--accent' }, { label: 'test (squares)', color: '--viz-neg' }, { label: 'fresh data (ring)', color: '--text' }]
    : [{ label: 'train', color: '--viz-forward' }, { label: 'validation', color: '--accent' }, { label: 'test (sealed)', color: '--viz-neg' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={TEND} markers={MK} label={init.tune ? 'Timeline: split, then try settings' : 'Timeline: split, then train'}
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= TEND ? init.t0 : to; wake(); }} onReset={() => { t = init.t0; playing = true; wake(); }} />
  {/snippet}
  {#snippet readout()}
    {#if R}
      <p class="vals num">epoch {ei} · train loss {f2(R.train[ei], 3)} · validation loss {f2(R.val[ei], 3)}</p>
      {#if chosen && sc}
        <p class="vals num">stopped at epoch {sc.epoch} by validation loss · test score, read once: {f2(sc.reported, 3)} · fresh data: {f2(sc.real, 3)}</p>
      {/if}
    {:else if chosen && hon && cheat}
      <p class="vals num">pick by validation → reported test {f2(hon.reported, 3)} · fresh data {f2(hon.real, 3)}</p>
      <p class="vals num">pick by test (peek) → reported test {f2(cheat.reported, 3)} · fresh data {f2(cheat.real, 3)} · <span class="bad">looks {f2(cheat.real - cheat.reported, 3)} better than it is</span></p>
    {:else}
      <p class="vals num">candidates tried: {Math.floor(triedAt(uiT))} of {NCAND}</p>
    {/if}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if !init.lockSplit}
      <div class="ctl">
        <F1Slider label="train" value={Math.round(trainFrac * 100)} min={5} max={90} step={1} digits={0} oninput={(v) => { trainFrac = v / 100; valFrac = Math.min(valFrac, 0.95 - trainFrac); }} />
        <F1Slider label="valid." value={Math.round(valFrac * 100)} min={5} max={90} step={1} digits={0} oninput={(v) => (valFrac = Math.min(v / 100, 0.95 - trainFrac))} />
        <span class="base num">% · test gets {base.split.nTest}</span>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.85rem / 1.5 var(--font-code); font-variant-numeric: tabular-nums; }
  .bad { color: var(--viz-neg); font-weight: 700; }
  .base { color: var(--text-muted); font: 0.85rem var(--font-code); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
</style>
