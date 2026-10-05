<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { CLASSES, N, Z_RANGE, TAU_RANGE, duration, markers, frameAt, derive, type Mode, type Frame } from './sim';

  interface Props {
    title?: string;
    /** 'softmax': scores and probabilities ('swing' / 'temp' play only its first / second half). 'loss': adds the true class and cross-entropy. 'grad': adds the gradient bars. */
    mode?: Mode;
    /** Index of the true class (0 to 3). */
    truth?: number;
  }
  let { title = '', mode = 'softmax', truth: truth0 = 2 }: Props = $props();
  const init = untrack(() => ({ mode, title: title || 'Scores in, probabilities out', truth: truth0 }));
  const T = duration(init.mode), MK = markers(init.mode), hasLoss = init.mode === 'loss' || init.mode === 'grad', hasGrad = init.mode === 'grad';

  const uid = $props.id();
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let truth = $state(init.truth);
  /** Values the user set by hand; null while the timeline drives them. */
  let over = $state<{ z: number[]; tau: number } | null>(null);
  let ui = $state<Frame>(frameAt(init.mode, 0, init.truth));
  let t = 0, drag = -1, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const phase = $derived(over ? 'manual' : (markerAt(uiT, MK)?.label ?? ''));
  const narration = $derived(
    phase === 'manual' ? 'You are setting the scores by hand.'
      : phase === 'swing one score' ? 'Score A swings down and up. Its probability follows it, and the other three take up the slack so the bars still sum to 1.'
      : phase === 'temperature' ? 'Same scores, changing temperature. Low: the top class takes almost everything. High: the bars flatten.'
      : 'Each step nudges every score against the gradient. The true class rises, the others fall, and the loss drops.',
  );
  $effect(() => { if (playing) over = null; speed; truth; wake(); });

  const frame = (): Frame => (over ? derive(over.z, over.tau, truth) : frameAt(init.mode, t, truth));
  function take() {
    if (over) return;
    const f = frameAt(init.mode, t, truth);
    over = { z: f.z.slice(), tau: f.tau }; playing = false;
  }
  /** Round to the slider's shown precision; the + 0 turns negative zero into 0. */
  const r1 = (v: number, k = 10) => Math.round(v * k) / k + 0;
  const sync = () => { ui = frame(); uiT = t; };
  const setZ = (i: number, v: number) => { take(); const z = over!.z.slice(); z[i] = clamp(v, Z_RANGE[0], Z_RANGE[1]); over = { ...over!, z }; sync(); wake(); };
  const setTau = (v: number) => { take(); over = { ...over!, tau: v }; sync(); wake(); };

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const nP = hasGrad ? 3 : 2, GAP = 18;
    const lay = () => {
      const w = (stage.w - GAP * (nP - 1)) / nP;
      return Array.from({ length: nP }, (_, k) => {
        const x = k * (w + GAP);
        return { x, w, l: x + 32, r: x + w - 4, t: 34, b: stage.h - 26 };
      });
    };
    const RANGES = [Z_RANGE, [0, 1], [-1, 1]] as const;
    const colX = (P: { l: number; r: number }, i: number) => {
      const cw = (P.r - P.l) / N;
      return { cx: P.l + cw * (i + 0.5), bw: Math.min(44, cw * 0.62) };
    };

    function render() {
      const { ctx, w: W, h: H, c } = stage, f = frame(), P = lay();
      ctx.clearRect(0, 0, W, H);
      const titles = ['scores (logits) z', hasLoss ? 'probabilities p (outline = true)' : 'probabilities p', 'gradient p − y'];
      const vals = [f.z, f.p, f.grad];
      P.forEach((pn, k) => {
        const [a, b] = RANGES[k], y = scale(a, b, pn.b, pn.t), y0 = y(0);
        ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
        ctx.fillText(titles[k], pn.x + 2, 4, pn.w - 4);
        ctx.font = stage.mono(10.5); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.lineWidth = 1;
        const ticks = k === 0 ? [-4, -2, 0, 2, 4] : k === 1 ? [0, 0.5, 1] : [-1, -0.5, 0, 0.5, 1];
        for (const v of ticks) {
          ctx.strokeStyle = v === 0 ? c.muted.hex : c.rule.hex;
          ctx.beginPath(); ctx.moveTo(pn.l, y(v)); ctx.lineTo(pn.r, y(v)); ctx.stroke();
          ctx.fillStyle = c.muted.hex; ctx.fillText(String(v).replace('-', '−'), pn.l - 5, y(v));
        }
        for (let i = 0; i < N; i++) {
          const { cx, bw } = colX(pn, i), v = vals[k][i], yv = y(clamp(v, a, b));
          const col = k === 1 ? c.fwd : k === 2 ? c.grad : v >= 0 ? c.pos : c.neg;
          ctx.fillStyle = col.hex; ctx.globalAlpha = 0.85;
          ctx.fillRect(cx - bw / 2, Math.min(y0, yv), bw, Math.max(1, Math.abs(yv - y0)));
          ctx.globalAlpha = 1;
          if (k === 1 && hasLoss && i === truth) {
            ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.5; ctx.strokeRect(cx - bw / 2 - 2, y(1), bw + 4, y(0) - y(1));
          }
          if (k === 0) {
            ctx.fillStyle = c.raised.hex; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2.2;
            ctx.beginPath(); ctx.arc(cx, yv, 6.5, 0, 6.2832); ctx.fill(); ctx.stroke();
          } else {
            ctx.font = stage.mono(10.5, 600); ctx.textAlign = 'center'; ctx.textBaseline = v >= 0 ? 'bottom' : 'top';
            ctx.fillStyle = c.text.hex; ctx.fillText(f2(Math.round(v * 100) / 100 + 0, 2), cx, v >= 0 ? yv - 3 : yv + 3);
          }
          ctx.font = stage.body(12); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
          ctx.fillStyle = hasLoss && i === truth ? c.accent.hex : c.muted.hex;
          ctx.fillText(CLASSES[i], cx, pn.b + 6);
        }
      });
    }

    const loop = createLoop(cv, (dt, now) => {
      if (!over) t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; sync(); }
      return playing || drag >= 0;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const pos = (e: PointerEvent) => { const r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    const colAt = (x: number) => { const P = lay()[0]; return clamp(Math.floor(((x - P.l) / (P.r - P.l)) * N), 0, N - 1); };
    const inZone = (p: { x: number; y: number }) => { const P = lay()[0]; return p.x >= P.l - 4 && p.x <= P.r && p.y >= P.t - 10 && p.y <= P.b + 10; };
    const valAt = (py: number) => { const P = lay()[0]; return Z_RANGE[0] + ((P.b - py) / (P.b - P.t)) * (Z_RANGE[1] - Z_RANGE[0]); };
    const down = (e: PointerEvent) => {
      const p = pos(e); if (!inZone(p)) return;
      drag = colAt(p.x); cv.setPointerCapture(e.pointerId); setZ(drag, valAt(p.y)); e.preventDefault();
    };
    const move = (e: PointerEvent) => {
      const p = pos(e);
      if (drag < 0) { cv.style.cursor = inZone(p) ? 'ns-resize' : ''; return; }
      setZ(drag, valAt(p.y));
    };
    const up = () => { drag = -1; };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move);
    cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });
</script>

<FigureCard title={init.title} role="group" aspect="720 / 340" narrowAspect="360 / 340"
  label="Bar charts of four class scores, the probabilities softmax makes from them, and, in later figures, the gradient. Drag a score bar to change it."
  legend={[{ label: 'score up', color: '--viz-pos' }, { label: 'score down', color: '--viz-neg' }, { label: 'probability', color: '--viz-forward' },
    ...(hasGrad ? [{ label: 'gradient', color: '--viz-gradient' }] : []), ...(hasLoss ? [{ label: 'true class', color: '--accent' }] : [])]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: the scores move automatically"
      onSeek={(to) => { over = null; t = to; wake(); }} onStep={(to) => { over = null; t = to >= T ? 0 : to; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">
      {#if hasLoss}
        loss = −ln p<sub>{CLASSES[truth]}</sub> = <b class="on">{f2(ui.loss, 3)}</b>
        <span class="base">· guessing uniformly scores {f2(Math.log(N), 3)}</span>
      {:else}
        probabilities sum to <b>{f2(ui.p.reduce((a, b) => a + b, 0), 2)}</b> <span class="base">· temperature {f2(ui.tau, 2)}</span>
      {/if}
    </p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#each CLASSES as cl, i (cl)}
        <F1Slider label={'z ' + cl} value={r1(ui.z[i])} min={Z_RANGE[0]} max={Z_RANGE[1]} step={0.05} digits={1} oninput={(v) => setZ(i, v)} />
      {/each}
      <F1Slider label="temp" value={r1(ui.tau, 100)} min={TAU_RANGE[0]} max={TAU_RANGE[1]} step={0.05} oninput={setTau} />
      {#if hasLoss}
        <fieldset class="seg">
          <legend class="sr-only">True class</legend>
          {#each CLASSES as cl, i (cl)}
            <label><input type="radio" name="truth-{uid}" value={i} checked={truth === i} onchange={() => (truth = i)} /><span>{cl}</span></label>
          {/each}
        </fieldset>
        <span class="lbl">true class</span>
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
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
  .lbl { color: var(--text-muted); margin-left: calc(var(--s-3) * -1); }
</style>
