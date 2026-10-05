<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
import { advance, createLoop, onThemeChange, prefersReducedMotion, readColors, readVar, reducedMotion } from '../_shared';
import { DURATION, makeScheduleRuns, frameAt, updateAt, type BatchComparison, type Mode } from './sim';
import { objective } from '../OptimizerLab/sim';

  interface Props { mode?: Mode; title?: string; unstable?: boolean; scaled?: boolean; only?: string[] }
  let { mode = 'rate', title = 'Learning-rate schedules', unstable = false, scaled = false, only }: Props = $props();
  const uid = $props.id();
  let comparison = $state<BatchComparison>('equal-updates');
  let batchScaling = $state(untrack(() => scaled));
  let includeUnstable = $state(untrack(() => unstable)), selectedRun = $state(0);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let announcedStep = $state(0);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, timeline: ReturnType<typeof Timeline>;
  let ctx: CanvasRenderingContext2D, width = 0, height = 0;
  let colors = {} as ReturnType<typeof readColors>;
  const markers = [
    { at: 0, label: 'Early' }, { at: DURATION / 3, label: 'Middle' },
    { at: 2 * DURATION / 3, label: 'Late' },
  ];
  const runs = $derived(makeScheduleRuns({ mode, batchComparison: comparison, batchScaling }));
  const shownRuns = $derived(runs.filter((run) => (mode !== 'rate' || includeUnstable || run.id !== 'unstable') && (!only || only.includes(run.id))));
  const focus = $derived(shownRuns[Math.min(selectedRun, shownRuns.length - 1)]);
  const update = $derived(updateAt(uiT, focus));
  const frames = $derived(shownRuns.map((run) => frameAt(uiT, run)));
  // The update in progress at the cursor; the worked box and every run row use this same number.
  const updateNumber = (step: number, run: { rates: number[] }) => Math.min(run.rates.length, step + 1);
  const gap = (loss: number) => Math.max(1e-4, loss - 1);
  const colorsForLegend = $derived(shownRuns.map((run) => ({ label: run.label, color: run.color, shape: 'line' as const })));

  function reset() { playing = false; t = uiT = 0; wake(); }
  function toggleUnstable() { selectedRun = Math.min(selectedRun, includeUnstable ? 2 : 1); reset(); }
  function seek(to: number) { playing = false; t = uiT = to; wake(); }
  function sizeCanvas() {
    const rect = cv.getBoundingClientRect();
    const dpr = Math.min(2, Math.max(1, devicePixelRatio || 1));
    width = rect.width; height = rect.height;
    const pixelWidth = Math.round(width * dpr), pixelHeight = Math.round(height * dpr);
    if (cv.width !== pixelWidth || cv.height !== pixelHeight) { cv.width = pixelWidth; cv.height = pixelHeight; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  const runColor = (token: string) => {
    const palette: Record<string, string> = { '--viz-forward': colors.forward.hex, '--viz-gradient': colors.gradient.hex, '--viz-pos': colors.positive.hex, '--viz-neg': colors.negative.hex };
    return palette[token] ?? colors.text.hex;
  };

  function drawChart(titleText: string, box: { x: number; y: number; w: number; h: number }, low: string, high: string, xEnd = 'run progress') {
    ctx.strokeStyle = colors.rule.hex; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(box.x, box.y); ctx.lineTo(box.x, box.y + box.h); ctx.lineTo(box.x + box.w, box.y + box.h); ctx.stroke();
    ctx.fillStyle = colors.text.hex; ctx.font = `600 11px ${readVar('--font-code')}`; ctx.fillText(titleText, box.x, box.y - 7);
    ctx.fillStyle = colors.muted.hex; ctx.font = `10px ${readVar('--font-code')}`;
    ctx.fillText(high, 2, box.y + 4); ctx.fillText(low, 2, box.y + box.h);
    ctx.fillText('0', box.x, box.y + box.h + 13); ctx.fillText(xEnd, box.x + box.w - 75, box.y + box.h + 13);
  }

  function draw() {
    if (!ctx || !width || !height) return;
    ctx.clearRect(0, 0, width, height);
    const pad = Math.max(42, width * 0.065), right = width - 12;
    const rateBox = { x: pad, y: height * 0.09, w: right - pad, h: height * 0.2 };
    const lossBox = { x: pad, y: height * 0.45, w: right - pad, h: height * 0.45 };
    // Loss above its minimum (1) on a log scale: early drops and late noise are both readable.
    const gaps = shownRuns.flatMap(run => run.states.map(state => gap(objective(state.point, 'ravine')))).filter(Number.isFinite);
    // Cap the axis at 1000 so a diverging run leaves through the top instead of flattening the stable runs.
    const top = Math.min(3, Math.ceil(Math.log10(Math.max(...gaps)))), bottom = Math.floor(Math.log10(Math.min(...gaps)));
    const maxRate = Math.max(0.01, ...shownRuns.flatMap(run => run.rates));
    const lossY = (value: number) => lossBox.y + lossBox.h * Math.max(0, top - Math.log10(gap(value))) / Math.max(1, top - bottom);
    const endLabel = mode === 'batch' ? (comparison === 'equal-updates' ? '80 updates' : '160 examples') : '80 updates';
    const power = (k: number) => k >= 4 || k <= -4 ? `1e${k}` : String(10 ** k);
    drawChart('1. Learning rate', rateBox, '0', maxRate.toFixed(2), endLabel);
    drawChart('2. Loss above its minimum (log scale)', lossBox, power(bottom), power(top), endLabel);
    // One faint gridline per factor of ten, labelled when there is room.
    const every = lossBox.h / Math.max(1, top - bottom) >= 14 ? 1 : 2;
    ctx.font = `10px ${readVar('--font-code')}`;
    for (let k = bottom + 1; k < top; k++) {
      const y = lossY(1 + 10 ** k);
      ctx.strokeStyle = colors.rule.hex; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.moveTo(lossBox.x, y); ctx.lineTo(lossBox.x + lossBox.w, y); ctx.stroke(); ctx.globalAlpha = 1;
      if ((k - bottom) % every === 0) { ctx.fillStyle = colors.muted.hex; ctx.fillText(power(k), 2, y + 3); }
    }
    for (const box of [rateBox, lossBox]) {
      const x = box.x + box.w * uiT / DURATION;
      ctx.strokeStyle = colors.muted.hex; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(x, box.y); ctx.lineTo(x, box.y + box.h); ctx.stroke(); ctx.setLineDash([]);
    }
    shownRuns.forEach((run, index) => {
      const color = runColor(run.color), f = frames[index], total = run.states.length - 1;
      ctx.strokeStyle = ctx.fillStyle = color; ctx.lineWidth = run.id === focus.id ? 2.6 : 1.5;
      // Future rates are a plan; future losses stay hidden until their update happens.
      ctx.globalAlpha = 0.25; ctx.beginPath();
      run.rates.forEach((rate, i) => { const x = rateBox.x + rateBox.w * i / total, y = rateBox.y + rateBox.h * (1 - rate / maxRate); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke(); ctx.globalAlpha = 1; ctx.beginPath();
      run.rates.slice(0, Math.min(total, f.step + 1)).forEach((rate, i) => { const x = rateBox.x + rateBox.w * i / total, y = rateBox.y + rateBox.h * (1 - rate / maxRate); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.lineTo(rateBox.x + rateBox.w * uiT / DURATION, rateBox.y + rateBox.h * (1 - f.learningRate / maxRate)); ctx.stroke();
      ctx.beginPath();
      run.states.slice(0, f.step + 1).forEach((state, i) => { const x = lossBox.x + lossBox.w * i / total, y = lossY(objective(state.point, 'ravine')); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.lineTo(lossBox.x + lossBox.w * uiT / DURATION, lossY(f.loss)); ctx.stroke();
      for (const [box, y] of [[rateBox, rateBox.y + rateBox.h * (1 - f.learningRate / maxRate)], [lossBox, lossY(f.loss)]] as const) { ctx.beginPath(); ctx.arc(box.x + box.w * uiT / DURATION, y, 4, 0, Math.PI * 2); ctx.fill(); }
    });
    ctx.globalAlpha = 1;
  }

  function readColorsNow() {
    colors = readColors({ forward: '--viz-forward', gradient: '--viz-gradient', positive: '--viz-pos', negative: '--viz-neg', rule: '--rule', muted: '--text-muted', text: '--text' });
  }

  onMount(() => {
    ctx = cv.getContext('2d')!;
    readColorsNow(); sizeCanvas(); draw();
    const figure = cv.closest('[data-figure]') ?? cv;
    const observer = new ResizeObserver(() => { sizeCanvas(); draw(); }); observer.observe(figure);
    const loop = createLoop(figure, (dt, _now) => {
      if (playing) t = advance({ t, playing, speed }, dt, { duration: DURATION }).t;
      uiT = t;
      announcedStep = frames[0]?.step ?? 0;
      draw();
      timeline?.paint(t);
      return playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) { playing = true; wake(); } } });
    wake = loop.wake;
    const off = onThemeChange(() => { readColorsNow(); wake(); });
    const motion = prefersReducedMotion();
    const onMotionChange = (event: MediaQueryListEvent) => { if (event.matches) { playing = false; t = uiT = 0; } wake(); };
    motion.addEventListener('change', onMotionChange);
    return () => { observer.disconnect(); loop.destroy(); off(); motion.removeEventListener('change', onMotionChange); };
  });

  $effect(() => { mode; comparison; batchScaling; includeUnstable; selectedRun; playing; speed; wake(); });
</script>

<FigureCard {title} label="Learning rate above and resulting loss below, connected by the same training progress cursor" asideLabel="How to read the plots" role="group" aspect="16 / 10" narrowAspect="1 / 1" legend={colorsForLegend}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={timeline} bind:t={uiT} bind:playing bind:speed duration={DURATION} {markers} label="Learning-rate schedule timeline" onSeek={seek} onStep={seek} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    <p class="explain">Read from top to bottom: the rate sets the scale of each update, and those updates change the loss. The vertical cursor connects the same training progress in both plots.</p>
    <div class="worked" aria-label="One recorded update behind the schedule">
      <strong>{focus.label} · update {update.index}</strong>
      <p>Learning rate <span class="num">{update.rate.toFixed(4)}</span> → weights move by <span class="num">({update.change.x.toFixed(4)}, {update.change.y.toFixed(4)})</span></p>
      <p>Loss before → after this update: <span class="num">{objective(update.before.point, 'ravine').toFixed(4)} → {objective(update.after.point, 'ravine').toFixed(4)}</span></p>
    </div>
    <div class="numbers">
      {#each shownRuns as run, index (run.id)}
        <p><span class="dot" style:--dot="var({run.color})"></span><strong>{run.label}</strong> · update {updateNumber(frames[index].step, run)}/{run.rates.length} · rate <span class="num">{frames[index].learningRate.toFixed(4)}</span> · loss <span class="num">{!Number.isFinite(frames[index].loss) ? 'diverged' : frames[index].loss >= 1e4 ? frames[index].loss.toExponential(2) : frames[index].loss.toFixed(4)}</span>
          {#if mode === 'batch'} · batch {frames[index].batchSize} · {frames[index].examplesSeen} examples seen{/if}
        </p>
      {/each}
      {#if mode === 'rate'}<p class="note">Every run starts at the same weights and uses all 16 examples on each update.</p>{/if}
      {#if mode === 'warmup'}<p class="note">Both runs use Adam on the same mini-batches. Only the learning rate differs.</p>{/if}
      {#if mode === 'decay'}<p class="note">Every run starts at 0.12. The decays end at 0.01; the constant run never changes.</p>{/if}
      {#if mode === 'batch'}<p class="note">Both runs draw examples from the same shuffled stream, so batch 8's first update sees the same 8 examples as batch 2's first four updates.</p>{/if}
    </div>
    <p class="sr-only" aria-live="polite">Update {Math.min(80, announcedStep + 1)}</p>
  {/snippet}
  {#snippet footer()}
    <p class="try"><strong>Try this.</strong> {mode === 'rate' ? (includeUnstable ? 'Follow the 0.7 run: every update overshoots further than the last, so its loss climbs to the top of the chart.' : 'Tick “Include the too-large rate” below and watch what a rate of 0.7 does.') : mode === 'warmup' ? 'Scrub to update 20, where the warmup reaches the full rate. Notice the fixed run bouncing back up around the same time.' : mode === 'decay' ? (only ? 'Scrub across a drop in the upper plot and watch how the loss below changes after it.' : 'Watch the last third: the constant run keeps bouncing while the decayed runs settle.') : 'Switch “Compare by” to Equal examples seen. Batch 8 now gets only 20 updates.'}</p>
    <div class="settings">
      <label for="{uid}-focus">Inspect a run<select id="{uid}-focus" bind:value={selectedRun} onchange={() => wake()}>{#each shownRuns as run, i (run.id)}<option value={i}>{run.label}</option>{/each}</select></label>
      {#if mode === 'rate'}<label class="toggle"><input type="checkbox" bind:checked={includeUnstable} onchange={toggleUnstable} /> Include the too-large rate (0.7)</label>{/if}
    </div>
    {#if mode === 'batch'}
      <div class="settings">
        <label for="{uid}-comparison">Compare by
          <select id="{uid}-comparison" bind:value={comparison} onchange={reset}>
            <option value="equal-updates">Equal updates</option><option value="equal-examples">Equal examples seen</option>
          </select>
        </label>
        <label for="{uid}-scaling">Batch 8 learning rate
          <select id="{uid}-scaling" bind:value={batchScaling} onchange={reset}>
            <option value={false}>Keep at 0.08</option><option value={true}>Linear scale to 0.32</option>
          </select>
        </label>
      </div>
    {/if}
  {/snippet}

</FigureCard>

<style>
  .explain, .try { margin: 0 0 var(--s-3); line-height: 1.6; }
  .try { padding: var(--s-3) var(--s-4) 0; }
  .worked { padding: var(--s-3); margin-bottom: var(--s-3); background: var(--bg-sunken); border-left: 2px solid var(--viz-forward); }
  .worked p { margin: var(--s-2) 0 0; line-height: 1.6; overflow-wrap: anywhere; }
  .settings .toggle { display: flex; align-items: center; gap: var(--s-2); margin-top: var(--s-3); }
  .toggle input { accent-color: var(--accent); }

  .numbers { display: grid; gap: var(--s-1); }
  .numbers p { margin: 0; line-height: 1.55; }
  .numbers strong { font-weight: 650; }
  .dot { display: inline-block; width: 0.8rem; height: 2px; margin: 0 0.45rem 0.2rem 0; background: var(--dot); }
  .num { font-family: var(--font-code); font-variant-numeric: tabular-nums; }
  .note { color: var(--text-muted); }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
  .note { font-size: 0.86em; }
  .settings { padding: var(--s-3) var(--s-4) 0; }
  .settings label { display: grid; gap: var(--s-1); max-width: 20rem; font-weight: 600; }
  .settings select { min-height: 2.5rem; padding: 0.35rem 0.5rem; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); font: inherit; }
</style>
