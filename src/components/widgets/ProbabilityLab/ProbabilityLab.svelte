<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, splitPanels, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import {
    N, T_SAMPLE, T_BAYES, X_MIN, X_MAX, BINS, BIN_W, COLS, ROWS, POP, sampleMarkers, bayesMarkers,
    makeData, frameAt, countAt, normalPdf, binOf, trueMean, trueVar, bayesLayout, bayesFrameAt,
    type Mode,
  } from './sim';

  interface Props {
    title?: string;
    /** 'bernoulli': a biased coin. 'gaussian': a bell curve. 'bayes': a population grid and a test. */
    mode?: Mode;
    p?: number;
    mu?: number;
    sigma?: number;
    prevalence?: number;
    sensitivity?: number;
    fpr?: number;
  }
  let { title = '', mode = 'bernoulli', p = 0.3, mu = 0.5, sigma = 1.2, prevalence = 0.01, sensitivity = 0.9, fpr = 0.09 }: Props = $props();
  const init = untrack(() => ({
    mode, title: title || (mode === 'bayes' ? 'A test says positive: how likely is it that you are sick?'
      : mode === 'bernoulli' ? 'Samples fall into a histogram and settle on the true probabilities'
      : 'Samples fall into a histogram and trace out the bell curve'),
  }));
  const isBayes = init.mode === 'bayes', isCoin = init.mode === 'bernoulli';
  const T = isBayes ? T_BAYES : T_SAMPLE, MK = isBayes ? bayesMarkers : sampleMarkers;

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let pV = $state(untrack(() => p)), muV = $state(untrack(() => mu)), sgV = $state(untrack(() => sigma));
  let prV = $state(untrack(() => prevalence)), seV = $state(untrack(() => sensitivity)), fpV = $state(untrack(() => fpr));
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const cfg = $derived({ mode: isCoin ? ('bernoulli' as const) : ('gaussian' as const), p: pV, mu: muV, sigma: sgV });
  const data = $derived(makeData(cfg));
  const bcfg = $derived({ prevalence: prV, sensitivity: seV, fpr: fpV });
  const lay = $derived(bayesLayout(bcfg));
  $effect(() => { playing; speed; data; lay; wake(); });

  const phase = $derived(markerAt(uiT, MK)?.label ?? '');
  const k = $derived(Math.min(N, Math.floor(countAt(uiT) + 1e-9)));
  const mean = $derived(data.cumMean[k]), vari = $derived(data.cumVar[k]);
  const bc = $derived(lay.counts);
  const narration = $derived(
    isBayes ? ({
      'crowd': 'A thousand people appear.',
      'sick': 'A few of them are sick (indigo).',
      'test': 'Everyone takes the test. A ring marks a positive result.',
      'filter': 'Only the people who tested positive stay.',
      'fraction': 'Among the positives, the indigo dots are the ones truly sick.',
    } as Record<string, string>)[phase] ?? ''
      : phase === 'first samples' ? 'The first few samples land. The histogram is lumpy and the mean jumps around.'
        : phase === 'about 100' ? 'With a hundred samples the shape is clearer and the mean has stopped jumping.'
          : 'All 400 samples: the histogram matches the true distribution and the mean sits on the expectation.',
  );

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;

    function drawSamples() {
      const { ctx, w: W, h: H, c } = stage, d = data, cf = d.cfg, f = frameAt(d, t);
      const { main, side } = splitPanels(W, H, 0.62);
      const m = { l: main.x + 40, r: main.x + main.w - 10, t: main.y + 24, b: main.y + main.h - 30 };
      const [x0, x1] = isCoin ? [-0.6, 1.6] : [X_MIN, X_MAX];
      const sx = scale(x0, x1, m.l, m.r);
      // The axis grows to hold the tallest bar (and the sample about to land), so nothing clips.
      let ymax = 1;
      if (!isCoin) {
        let hi = 0;
        for (let i = 0; i < f.counts.length; i++) hi = Math.max(hi, f.counts[i]);
        const fb = f.fall >= 0 ? binOf(f.fallX) : -1;
        if (fb >= 0) hi = Math.max(hi, f.counts[fb] + 1);
        ymax = Math.max(normalPdf(cf.mu, cf.mu, cf.sigma) * 1.35, (hi / Math.max(1, f.k + 1) / BIN_W) * 1.08);
      }
      const sy = scale(0, ymax, m.b, m.t);
      const sd = Math.sqrt(f.variance), tm = trueMean(cf);
      ctx.lineWidth = 1; ctx.font = stage.mono(11);
      // axes
      ctx.strokeStyle = c.rule.hex; ctx.fillStyle = c.muted.hex; ctx.textBaseline = 'top'; ctx.textAlign = 'center';
      ctx.beginPath(); ctx.moveTo(m.l, m.b); ctx.lineTo(m.r, m.b); ctx.stroke();
      if (isCoin) {
        ctx.fillText('tails (0)', sx(0), m.b + 6); ctx.fillText('heads (1)', sx(1), m.b + 6);
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        for (const v of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(m.l, sy(v)); ctx.lineTo(m.r, sy(v)); ctx.stroke(); ctx.fillText(String(v), m.l - 6, sy(v)); }
      } else {
        for (let v = -8; v <= 8; v += 2) { ctx.beginPath(); ctx.moveTo(sx(v), m.b); ctx.lineTo(sx(v), m.b + 4); ctx.stroke(); ctx.fillText(String(v).replace('-', '−'), sx(v), m.b + 6); }
      }
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(isCoin ? 'fraction of samples' : 'relative frequency (density)', main.x + 4, main.y + 2);
      ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t - 4, m.r - m.l, m.b - m.t + 4); ctx.clip();
      // bars
      const val = (i: number, extra: number) => (f.counts[i] + extra) / Math.max(1, f.k + extra) / (isCoin ? 1 : BIN_W);
      ctx.fillStyle = `rgba(${c.fwd.rgb.join(',')},0.55)`; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1.2;
      const nb = f.counts.length;
      for (let i = 0; i < nb; i++) {
        if (!f.counts[i]) continue;
        const a = isCoin ? sx(i - 0.3) : sx(X_MIN + i * BIN_W) + 0.5, b = isCoin ? sx(i + 0.3) : sx(X_MIN + (i + 1) * BIN_W) - 0.5;
        ctx.beginPath(); ctx.rect(a, sy(val(i, 0)), b - a, m.b - sy(val(i, 0))); ctx.fill(); ctx.stroke();
      }
      // true distribution
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.8; ctx.setLineDash([5, 4]);
      if (isCoin) {
        for (const [i, v] of [[0, 1 - cf.p], [1, cf.p]] as const) { ctx.beginPath(); ctx.moveTo(sx(i - 0.38), sy(v)); ctx.lineTo(sx(i + 0.38), sy(v)); ctx.stroke(); }
      } else {
        ctx.setLineDash([]); ctx.beginPath();
        for (let i = 0; i <= 120; i++) { const x = X_MIN + ((X_MAX - X_MIN) * i) / 120, y = normalPdf(x, cf.mu, cf.sigma); i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)); }
        ctx.stroke();
      }
      ctx.setLineDash([]);
      // true expectation (dashed) and the running mean (solid)
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.2; ctx.setLineDash([2, 4]);
      ctx.beginPath(); ctx.moveTo(sx(tm), m.t); ctx.lineTo(sx(tm), m.b); ctx.stroke(); ctx.setLineDash([]);
      if (f.k > 0) { ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(sx(f.mean), m.t); ctx.lineTo(sx(f.mean), m.b); ctx.stroke(); }
      ctx.restore();
      // the sample in flight: starts above the chart and leaves a short trail
      if (f.fall >= 0) {
        const bi = isCoin ? f.fallX : binOf(f.fallX), inR = bi >= 0 && bi < nb;
        const y0 = m.t - 10, landY = inR ? sy(val(bi, 1)) : m.b - 4, fy = y0 + (landY - y0) * f.fall * f.fall, fx = sx(f.fallX);
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.lineCap = 'round';
        for (let j = 1; j <= 3; j++) { ctx.globalAlpha = 0.4 / j; ctx.beginPath(); ctx.moveTo(fx, fy - (j - 1) * 9 - 5); ctx.lineTo(fx, Math.max(y0, fy - j * 9 - 5)); ctx.stroke(); }
        ctx.globalAlpha = 1;
        ctx.fillStyle = c.accent.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(fx, fy, 5, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      if (f.k > 1) { // mean +- one standard deviation, as a bracket on the axis
        const a = clamp(sx(f.mean - sd), m.l, m.r), b = clamp(sx(f.mean + sd), m.l, m.r), by = m.b - 9;
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.5; ctx.lineCap = 'butt';
        ctx.beginPath(); ctx.moveTo(a, by); ctx.lineTo(b, by); ctx.moveTo(a, by - 5); ctx.lineTo(a, by + 5); ctx.moveTo(b, by - 5); ctx.lineTo(b, by + 5); ctx.stroke();
      }
      if (f.k > 0) { // mean marker under the axis
        const mx = clamp(sx(f.mean), m.l, m.r);
        ctx.fillStyle = c.accent.hex; ctx.beginPath(); ctx.moveTo(mx, m.b + 1); ctx.lineTo(mx - 5, m.b + 9); ctx.lineTo(mx + 5, m.b + 9); ctx.closePath(); ctx.fill();
      }

      // ---- side: the running mean settles on the expectation ----
      const s = { l: side.x + 40, r: side.x + side.w - 10, t: side.y + 24, b: side.y + side.h - 30 };
      const [ya, yb] = isCoin ? [0, 1] : [cf.mu - 2.5 * cf.sigma, cf.mu + 2.5 * cf.sigma];
      const cx = scale(0, N, s.l, s.r), cy = scale(ya, yb, s.b, s.t);
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('running mean after n samples', s.l - 40, side.y + 2);
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.font = stage.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [ya, tm, yb]) { ctx.beginPath(); ctx.moveTo(s.l, cy(v)); ctx.lineTo(s.r, cy(v)); ctx.stroke(); }
      for (const v of [ya, yb]) { ctx.fillStyle = c.muted.hex; ctx.fillText(f2(v, 1), s.l - 5, cy(v)); }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [0, 200, 400]) { ctx.fillStyle = c.muted.hex; ctx.fillText(String(v), cx(v), s.b + 5); }
      ctx.save(); ctx.beginPath(); ctx.rect(s.l, s.t - 2, s.r - s.l, s.b - s.t + 4); ctx.clip();
      // +- one standard error of the mean around the truth
      const se = Math.sqrt(trueVar(cf));
      ctx.fillStyle = `rgba(${c.fwd.rgb.join(',')},0.16)`; ctx.beginPath();
      for (let n = 4; n <= N; n += 4) { const y = cy(tm + se / Math.sqrt(n)); n === 4 ? ctx.moveTo(cx(n), y) : ctx.lineTo(cx(n), y); }
      for (let n = N; n >= 4; n -= 4) ctx.lineTo(cx(n), cy(tm - se / Math.sqrt(n)));
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.2; ctx.setLineDash([2, 4]);
      ctx.beginPath(); ctx.moveTo(s.l, cy(tm)); ctx.lineTo(s.r, cy(tm)); ctx.stroke(); ctx.setLineDash([]);
      if (f.k > 0) {
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath();
        for (let n = 1; n <= f.k; n++) n === 1 ? ctx.moveTo(cx(n), cy(d.cumMean[n])) : ctx.lineTo(cx(n), cy(d.cumMean[n]));
        ctx.stroke();
        ctx.fillStyle = c.accent.hex; ctx.beginPath(); ctx.arc(cx(f.k), cy(f.mean), 4, 0, 6.2832); ctx.fill();
      }
      ctx.restore();
    }

    function drawBayes() {
      const { ctx, w: W, h: H, c } = stage, L = lay, f = bayesFrameAt(t);
      const top = 8, cs = Math.min((W - 16) / COLS, (H - top - 8) / ROWS), ox = (W - cs * COLS) / 2, oy = top + (H - top - 8 - cs * ROWS) / 2;
      const rad = cs * 0.36;
      for (let pass = 0; pass < 2; pass++) {
        for (let i = 0; i < POP; i++) {
          const fl = L.flags[i], sick = fl & 1, pos = fl & 2;
          if ((pass === 1) !== !!sick) continue; // healthy first, sick on top
          const a = clamp(f.appear * 1.8 - L.r[i] * 0.8, 0, 1) * (pos ? 1 : 1 - f.leave);
          if (a <= 0.01) continue;
          const x = ox + cs * (L.x0[i] + (L.x1[i] - L.x0[i]) * f.pack), y = oy + cs * (L.y0[i] + (L.y1[i] - L.y0[i]) * f.pack);
          const r = rad * (1 + (L.scale - 1) * f.pack), col = sick ? c.accent : c.muted;
          ctx.globalAlpha = a * (sick ? 0.35 + 0.65 * f.sick : 0.6);
          ctx.fillStyle = sick && f.sick < 1 ? c.muted.hex : col.hex;
          ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
          if (sick && f.sick > 0) { ctx.globalAlpha = a * f.sick; ctx.fillStyle = c.accent.hex; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); }
          if (pos && f.test > 0) {
            ctx.globalAlpha = a * f.test; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = Math.max(1, r * 0.35);
            ctx.beginPath(); ctx.arc(x, y, r + Math.max(1, r * 0.35), 0, 6.2832); ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;
    }

    function render() {
      stage.ctx.clearRect(0, 0, stage.w, stage.h);
      if (isBayes) drawBayes(); else drawSamples();
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
</script>

<FigureCard title={init.title} aspect="720 / 400" narrowAspect="360 / 560"
  label={isBayes ? 'A grid of 1000 dots. A few are sick. After a test, only the positives remain, and the share of sick dots among them is the answer.'
    : isCoin ? 'A bar chart of coin flips, heads and tails, filling as samples fall. A small chart shows the running mean settling on the true probability.'
      : 'A histogram filling as samples fall, converging on a bell curve. A small chart shows the running mean settling on the true mean.'}
  legend={isBayes ? [{ label: 'healthy', color: '--text-muted' }, { label: 'sick', color: '--accent' }, { label: 'tested positive (ring)', color: '--viz-forward' }]
    : [{ label: 'samples', color: '--viz-forward' }, { label: 'true distribution', color: '--text', shape: 'line' }, { label: 'running mean, ±1σ bracket', color: '--accent', shape: 'line' }]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: samples arrive automatically"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">
      {#if isBayes}
        {bc.sick} sick of {POP} · positives: {bc.tp} sick + {bc.fp} healthy · P(sick | positive) = {bc.tp} / {bc.tp + bc.fp} = <b class="on">{pct(bc.posterior)}</b>
      {:else}
        n = {k} · mean <b class="on">{f2(mean, 3)}</b> <span class="base">(true {f2(trueMean(cfg), 3)})</span> · variance <b>{f2(vari, 3)}</b> <span class="base">(true {f2(trueVar(cfg), 3)})</span>
      {/if}
    </p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if isBayes}
        <F1Slider label="prevalence" value={prV} min={0.005} max={0.3} step={0.005} digits={3} oninput={(v) => (prV = v)} />
        <F1Slider label="sensitivity" value={seV} min={0.5} max={0.99} step={0.01} oninput={(v) => (seV = v)} />
        <F1Slider label="false positive rate" value={fpV} min={0.01} max={0.3} step={0.01} oninput={(v) => (fpV = v)} />
      {:else if isCoin}
        <F1Slider label="P(heads) = p" value={pV} min={0.05} max={0.95} step={0.05} oninput={(v) => (pV = v)} />
      {:else}
        <F1Slider label="mean μ" value={muV} min={-2} max={2} step={0.1} digits={1} oninput={(v) => (muV = v)} />
        <F1Slider label="spread σ" value={sgV} min={0.5} max={1.5} step={0.1} digits={1} oninput={(v) => (sgV = v)} />
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--text-muted); }
  .vals b { font-weight: 500; color: var(--text); } .vals b.on { color: var(--accent); font-weight: 700; }
  .base { font-size: 0.82rem; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
</style>
