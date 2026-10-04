<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { HOUSES, EXAMPLES, X_MAX, Y_MAX, W_RANGE, B_RANGE, GOOD, predict, mse, duration, markers, frameAt, type Phase } from './sim';

  interface Props {
    title?: string;
    /** 1 = only the slope knob w; 2 = slope w and base price b. */
    knobs?: 1 | 2;
    /** Show the eight real houses behind the line. */
    showData?: boolean;
  }
  let { title = 'A model with knobs: price = w × area + b', knobs = 2, showData = false }: Props = $props();
  const init = untrack(() => ({ K: knobs, data: showData }));
  const T = duration(init.K), MK = markers(init.K);

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  /** Knob values the user set by hand; null while the timeline is driving them. */
  let over = $state<{ w: number; b: number } | null>(null);
  let phase = $state<Phase>('slope');
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const eff = $derived(over ?? frameAt(uiT, init.K));
  const narration = $derived(
    over ? 'You are turning the knobs by hand.'
      : phase === 'slope' ? 'The timeline turns the slope knob w: the whole line tilts.'
      : phase === 'intercept' ? 'Now the base-price knob b: the whole line slides up and down.' : 'Both knobs at once.',
  );
  $effect(() => { if (playing) over = null; speed; wake(); });

  const set = (k: 'w' | 'b', v: number) => { over = { ...(over ?? frameAt(t, init.K)), [k]: v }; playing = false; wake(); };

  onMount(() => {
    const stage = setupStage(cv, () => render());
    const P = { l: 50, r: 14, t: 14, b: 36 };
    let lastUi = 0;

    function render() {
      const { ctx, w: W, h: H, c } = stage;
      const f = over ?? frameAt(t, init.K);
      const sx = scale(0, X_MAX, P.l, W - P.r), sy = scale(0, Y_MAX, H - P.b, P.t);
      ctx.clearRect(0, 0, W, H);
      ctx.font = stage.mono(11); ctx.lineWidth = 1;
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let x = 0; x <= X_MAX; x += 50) {
        ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(sx(x), P.t); ctx.lineTo(sx(x), H - P.b); ctx.stroke();
        ctx.fillStyle = c.muted.hex; ctx.fillText(String(x), sx(x), H - P.b + 6);
      }
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (let y = 0; y <= 400; y += 100) {
        ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(P.l, sy(y)); ctx.lineTo(W - P.r, sy(y)); ctx.stroke();
        ctx.fillStyle = c.muted.hex; ctx.fillText(String(y), P.l - 6, sy(y));
      }
      ctx.fillStyle = c.muted.hex; ctx.font = stage.body(12, true);
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('area (m²)', W - P.r, H - 2);
      ctx.save(); ctx.translate(12, P.t + 4); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('price ($1000s)', 0, 0); ctx.restore();

      ctx.save();
      ctx.beginPath(); ctx.rect(P.l, P.t, W - P.l - P.r, H - P.t - P.b); ctx.clip();
      if (init.data) {
        for (const h of HOUSES) {
          const yp = predict(f.w, f.b, h.x);
          ctx.strokeStyle = c.muted.hex; ctx.globalAlpha = 0.5; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(sx(h.x), sy(h.y)); ctx.lineTo(sx(h.x), sy(yp)); ctx.stroke();
          ctx.globalAlpha = 1;
        }
        for (const h of HOUSES) {
          ctx.fillStyle = c.text.hex; ctx.beginPath(); ctx.arc(sx(h.x), sy(h.y), 4, 0, 6.2832); ctx.fill();
        }
      }
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 3; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(sx(0), sy(f.b)); ctx.lineTo(sx(X_MAX), sy(f.w * X_MAX + f.b)); ctx.stroke();
      ctx.restore();

      // predictions for three example houses: drop from the axis up to the line
      ctx.font = stage.mono(12, 600);
      for (const x of EXAMPLES) {
        const yp = predict(f.w, f.b, x), py = Math.min(H - P.b, Math.max(P.t + 2, sy(yp)));
        ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(sx(x), H - P.b); ctx.lineTo(sx(x), py); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = c.accent.hex; ctx.beginPath(); ctx.arc(sx(x), py, 5, 0, 6.2832); ctx.fill();
        ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.stroke();
        const s = `$${Math.round(yp)}k`, above = py > P.t + 22;
        ctx.textAlign = 'center'; ctx.textBaseline = above ? 'bottom' : 'top';
        ctx.lineWidth = 4; ctx.strokeStyle = c.raised.hex; ctx.lineJoin = 'round';
        ctx.strokeText(s, sx(x), py + (above ? -9 : 9)); ctx.fillStyle = c.accent.hex; ctx.fillText(s, sx(x), py + (above ? -9 : 9));
      }
    }

    const loop = createLoop(cv, (dt, now) => {
      if (!over) t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      const ph = frameAt(t, init.K).phase;
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; if (ph !== phase) phase = ph; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });
</script>

<FigureCard {title} label="A scatter of eight houses (area against price) with a straight line whose slope and height are set by knobs. Three example houses show the model's predicted prices."
  aspect="720 / 380" narrowAspect="360 / 300" legend={[{ label: 'model output', color: '--viz-forward', shape: 'line' }, { label: 'a prediction', color: '--accent' }]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: the knobs sweep automatically"
      onSeek={(to) => { over = null; t = to; wake(); }} onStep={(to) => { over = null; t = to >= T ? 0 : to; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="eq num" aria-live="off">price = <b>{f2(eff.w)}</b> × area + <b>{f2(eff.b, 0)}</b>
      {#if init.data}<span class="err">· typical miss ≈ ${Math.round(Math.sqrt(mse(eff.w, eff.b)))}k</span>{/if}</p>
    <p class="ex num">{#each EXAMPLES as x, i (x)}{i ? '  ·  ' : ''}{x} m² → ${Math.round(predict(eff.w, eff.b, x))}k{/each}</p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <F1Slider label="w" value={eff.w} min={W_RANGE[0]} max={W_RANGE[1]} step={0.05} oninput={(v) => set('w', v)} />
      {#if init.K === 2}<F1Slider label="b" value={eff.b} min={B_RANGE[0]} max={B_RANGE[1]} step={1} digits={0} oninput={(v) => set('b', v)} />{/if}
      <button type="button" class="btn" onclick={() => { over = { ...GOOD }; playing = false; wake(); }}>Good fit</button>
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .eq { font: 1rem var(--font-code); font-variant-numeric: tabular-nums; }
  .eq b { color: var(--viz-forward); font-weight: 650; }
  .err { color: var(--text-muted); font-size: 0.85rem; }
  .ex { font: 0.82rem var(--font-code); color: var(--text-muted); margin-top: 0.2em; white-space: pre-wrap; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl .btn { padding: 0.4em 0.9em; }
</style>
