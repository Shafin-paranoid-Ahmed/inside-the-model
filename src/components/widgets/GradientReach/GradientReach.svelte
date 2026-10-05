<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt, readColors } from '../_shared';
  import { setupStage } from '../F1Shared/canvas';
  import { scale } from '../F1Shared/plot';
  import { SCALES, N_MIN, N_MAX, REACH, T, GROW, MARKERS, lengthAt, reachAt, share, pow10 } from './sim';

  let { title = 'How much gradient reaches the first step' }: { title?: string } = $props();
  const init = untrack(() => ({ title }));

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const n = $derived(Math.round(lengthAt(uiT)));
  const phase = $derived(markerAt(uiT, MARKERS)?.label ?? '');
  const narration = $derived(
    phase === 'length 30'
      ? 'At 30 steps: s = 0.5 leaves about 10⁻¹⁰ of the gradient for the first step, s = 1 about 0.03%. At s = 1.5 and 2 the saturated tanh keeps it between about 0.2× and 13× all the way: no steady fall, and no runaway growth either.'
      : 'The sequence grows one character at a time. Below s = 1.5 each extra step multiplies the first step’s share by about the same factor, so the lines fall straight on this log scale.',
  );
  const COLS = ['--viz-cat-1', '--viz-cat-2', '--viz-cat-3', '--viz-cat-4', '--viz-cat-5'];
  const readCols = () => { const r = readColors({ a: COLS[0], b: COLS[1], c: COLS[2], d: COLS[3], e: COLS[4] }); return [r.a, r.b, r.c, r.d, r.e]; };
  $effect(() => { playing; speed; wake(); });

  onMount(() => {
    let col = readCols();
    const stage = setupStage(cv, () => { col = readCols(); render(); });
    const LO = -11, HI = 2;
    let lastUi = 0;

    function render() {
      const { ctx, w: W, h: H, c } = stage, narrow = W < 500;
      const L = { l: narrow ? 46 : 58, r: W - (narrow ? 50 : 70), t: 14, b: H - 34 };
      const sx = scale(N_MIN, N_MAX, L.l, L.r), sy = scale(LO, HI, L.b, L.t), nNow = lengthAt(t);
      ctx.clearRect(0, 0, W, H);
      // grid: one line per power of ten, labelled as a share of the last step's gradient
      ctx.font = stage.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.lineWidth = 1;
      for (let e = LO + 1; e <= HI; e++) {
        ctx.strokeStyle = e === 0 ? c.muted.hex : c.rule.hex;
        ctx.beginPath(); ctx.moveTo(L.l, sy(e)); ctx.lineTo(L.r, sy(e)); ctx.stroke();
        if (e % (narrow ? 4 : 2) === 0 || e === 0) { ctx.fillStyle = c.muted.hex; ctx.fillText(pow10(e), L.l - 6, sy(e)); }
      }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (const v of [2, 10, 20, 30]) ctx.fillText(String(v), sx(v), L.b + 5);
      ctx.font = stage.body(12, true); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('sequence length (steps)', L.r, H - 1);
      // one line per weight scale, drawn up to the current length
      SCALES.forEach((s, i) => {
        ctx.strokeStyle = col[i].hex; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath();
        ctx.moveTo(sx(N_MIN), sy(REACH[i][0]));
        for (let m = N_MIN + 1; m <= nNow; m++) ctx.lineTo(sx(m), sy(REACH[i][m - N_MIN]));
        const y = sy(reachAt(i, nNow));
        ctx.lineTo(sx(nNow), y); ctx.stroke();
        ctx.fillStyle = col[i].hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(sx(nNow), y, 4.5, 0, 6.2832); ctx.fill(); ctx.stroke();
        if (nNow < 12) return; // the lines start bunched together; the legend names them until they separate
        ctx.font = stage.mono(11.5, 650); ctx.fillStyle = col[i].hex; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(`s = ${s}`, sx(nNow) + 8, i === 3 ? y + 7 : i === 4 ? y - 7 : y);
      });
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; else if (first) { t = GROW; uiT = t; } } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });
</script>

<FigureCard title={init.title} role="group" aspect="720 / 380" narrowAspect="360 / 420"
  label="Line chart, log scale: the share of the last step's gradient that reaches the first step of a tanh RNN, for sequence lengths 2 to 30 and weight scales 0.5, 0.8, 1, 1.5 and 2. The lines for 0.5, 0.8 and 1 fall steadily; the lines for 1.5 and 2 stay near 1."
  legend={SCALES.map((s, i) => ({ label: `s = ${s}`, color: COLS[i], shape: 'line' as const }))}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Timeline: the sequence grows"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">length {n} · first step gets {SCALES.map((s, i) => `${share(reachAt(i, n))} at s = ${s}`).join(', ')}</p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.88rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--text-muted); overflow-wrap: anywhere; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
</style>
