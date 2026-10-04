<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance } from '../_shared';
  import { SCENARIOS, duration, markersOf, frameAt, extentOf, summary, fmt, fmtSigned, type ScenarioId } from './sim';

  interface Props {
    /** 'two': y = (3x)². 'long': four links. 'paths': x feeds y through two paths. */
    scenario?: ScenarioId;
    title?: string;
    /** Starting value of x (default: the scenario's own). */
    x?: number;
    /** Size of the nudge to x. */
    dx?: number;
    /** Show the x slider. */
    controls?: boolean;
  }
  let { scenario = 'two', title = 'A nudge travels down a chain', x: xProp, dx = 0.02, controls: showControls = true }: Props = $props();

  const S = untrack(() => SCENARIOS[scenario]);
  const T = duration(S), MARKERS = markersOf(S);
  let xv = $state(untrack(() => xProp ?? S.x0));
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let narration = $state(S.say[0]);
  const eq = $derived(summary(S, xv, dx));
  const rows = Math.max(...S.nodes.map((n) => n.row)) + 1;

  let t = 0;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, poke = () => {};
  $effect(() => { xv; playing; speed; poke(); });

  const LEGEND = [
    { label: 'the nudge', color: '--viz-forward' },
    { label: 'rate of each link', color: '--viz-gradient', shape: 'line' as const },
  ];

  onMount(() => {
    const reduced = prefersReducedMotion();
    const ctx = cv.getContext('2d')!;
    const NAMES = { fwd: '--viz-forward', grad: '--viz-gradient', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised' };
    let c = readColors(NAMES), code = readVar('--font-code');
    let lastUi = 0, phase = -1, narrow = false;
    if (reduced.matches) t = T;

    const text = (s: string, x: number, y: number, col: string, size: number, align: CanvasTextAlign = 'left', a = 1) => {
      if (a <= 0.01) return;
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align;
      ctx.font = `600 ${size}px ${code}`;
      ctx.fillText(s, x, y);
    };

    function render(now: number) {
      const W = narrow ? 360 : 720, H = narrow ? 520 : 420;
      const k = cv.width / W;
      ctx.setTransform(k, 0, 0, k, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      const fr = frameAt(t, S, xv, dx);
      const fs = narrow ? 11 : 13;
      const gap = Math.min(120, (H - 90) / (rows - 1)), y0 = (H - gap * (rows - 1)) / 2;
      const pos = S.nodes.map((n) => ({ x: n.col * W, y: y0 + n.row * gap }));
      const wide = S.nodes.some((n) => n.col !== 0.5);
      const lw = W * (wide ? 0.4 : 0.6), half = lw / 2;
      const px = (0.88 * half) / extentOf(S, xv, dx);

      // links: connector, gear, rate label, particles
      fr.edges.forEach((ed) => {
        const a = pos[ed.from], b = pos[ed.to];
        const y1 = a.y + 8, y2 = b.y - 22;
        ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(a.x, y1); ctx.lineTo(b.x, y2); ctx.stroke();
        ctx.strokeStyle = c.grad.hex; ctx.globalAlpha = 0.85 * ed.r; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(a.x, y1); ctx.lineTo(a.x + (b.x - a.x) * ed.spin, y1 + (y2 - y1) * ed.spin); ctx.stroke();
        const mx = (a.x + b.x) / 2, my = (y1 + y2) / 2, R = 11;
        ctx.save(); ctx.translate(mx, my); ctx.rotate(ed.spin * 6.2832 * (ed.rate < 0 ? -1 : 1));
        ctx.globalAlpha = 0.35 + 0.65 * ed.r; ctx.strokeStyle = c.grad.hex; ctx.fillStyle = c.raised.hex; ctx.lineWidth = 1.8;
        ctx.beginPath(); ctx.arc(0, 0, R, 0, 6.2832); ctx.fill(); ctx.stroke();
        for (let i = 0; i < 8; i++) { ctx.rotate(0.7854); ctx.beginPath(); ctx.moveTo(R, 0); ctx.lineTo(R + 4, 0); ctx.stroke(); }
        ctx.restore();
        const side = b.x < a.x ? -1 : 1;
        text(`×${fmt(ed.rate)}`, mx + side * 22, my + 4, c.grad.hex, fs, side < 0 ? 'right' : 'left', ed.r);
        if (ed.flow >= 0 && !reduced.matches) {
          ctx.fillStyle = c.fwd.hex;
          for (let i = 0; i < 3; i++) {
            const u = ed.flow - i * 0.08;
            if (u < 0) continue;
            ctx.globalAlpha = 1 - i * 0.3;
            ctx.beginPath(); ctx.arc(a.x + (b.x - a.x) * u, y1 + (y2 - y1) * u, 3.4 - i * 0.7, 0, 6.2832); ctx.fill();
          }
        }
      });

      // number lines
      S.nodes.forEach((n, i) => {
        const p = pos[i], cm = fr.cum[i], rv = fr.reveal[i];
        ctx.globalAlpha = 1; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(p.x - half, p.y); ctx.lineTo(p.x + half, p.y); ctx.moveTo(p.x, p.y - 5); ctx.lineTo(p.x, p.y + 5); ctx.stroke();
        // stacked contributions (one segment per incoming link)
        for (let s = 0; s + 1 < cm.length; s++) {
          ctx.globalAlpha = s === 0 ? 0.35 : 0.2; ctx.fillStyle = c.fwd.hex;
          const l = p.x + Math.min(cm[s], cm[s + 1]) * px;
          ctx.fillRect(l, p.y - 6, Math.max(Math.abs(cm[s + 1] - cm[s]) * px, 0), 12);
        }
        // the value under the nudge: dot wiggles inside the bar
        const off = fr.amp[i] * fr.wig * px;
        ctx.globalAlpha = 1; ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(p.x + (rv > 0 ? off : 0), p.y, 5, 0, 6.2832); ctx.fill(); ctx.stroke();
        text(n.label, p.x - half, p.y - 14, c.text.hex, fs);
        text(`Δ${n.name} = ${fmtSigned(fr.amp[i])}`, p.x + half, p.y - 14, c.fwd.hex, fs, 'right', rv);
        text(`${n.name} = ${fmt(fr.e.vals[i])}`, p.x - half, p.y + 20, c.muted.hex, fs - 1);
      });
      ctx.globalAlpha = 1;

      tl.paint(t);
      if (now - lastUi > 100 || !playing) {
        lastUi = now; uiT = t;
        if (fr.phaseIdx !== phase) { phase = fr.phaseIdx; narration = fr.say; }
      }
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(now);
      return playing;
    }, { onView: (inView, first) => { if (inView && first && !reduced.matches) playing = true; } });
    poke = loop.wake;

    const fit = () => {
      const d = Math.min(2, Math.max(1, devicePixelRatio || 1)), r = cv.getBoundingClientRect();
      const w = Math.round(r.width * d), h = Math.round(r.height * d);
      if (w && h && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
    };
    const ro = new ResizeObserver(([e]) => { narrow = e.contentRect.width < 540; fit(); render(performance.now()); });
    ro.observe(cv.parentElement!);
    const off = onThemeChange(() => { c = readColors(NAMES); code = readVar('--font-code'); poke(); });
    const onReduce = () => { if (reduced.matches) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    fit();
    return () => { loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); };
  });
</script>

<FigureCard {title} legend={LEGEND} role="img" aspect="720 / 420" narrowAspect="360 / 520"
  label="Number lines for a chain of functions. A small nudge to x moves each variable in turn; each link multiplies the nudge by its rate, and where paths meet the contributions add.">
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Timeline of the nudge"
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { t = to >= T ? 0 : to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="num">{eq.line}</p>
    <p class="num muted">{eq.actual}</p>
  {/snippet}
  {#snippet footer()}
    {#if showControls}
      <div class="ctl">
        <label>x
          <input type="range" min={S.xRange[0]} max={S.xRange[1]} step="0.05" bind:value={xv} aria-label="Value of x" />
          <output class="num">{fmt(xv)}</output>
        </label>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-ui); min-height: 2.6em; }
  .num { font-size: var(--step--1); color: var(--text); }
  .muted { color: var(--text-muted); }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='range'] { width: 11rem; accent-color: var(--accent); }
  .ctl output { min-width: 3rem; color: var(--text); }
</style>
