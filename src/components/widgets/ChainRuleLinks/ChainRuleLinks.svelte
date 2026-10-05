<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance } from '../_shared';
  import { CHAINS, WIGGLE, duration, markers, frameAt, linkView, type ChainId, type View } from './sim';

  interface Props {
    /** 'wave': u = x², v = sin u, L = v². 'saturate': u = 2x, v = tanh u, L = 3v². */
    chain?: ChainId;
    title?: string;
    /** Starting value of x (default: the chain's own). */
    x?: number;
    /** Hide the x slider and chain switch. */
    controls?: boolean;
    /** One job per figure: 'wiggle' | 'tangent' | 'links' | 'full' (see sim.ts). */
    view?: View;
  }
  let { chain = 'wave', title = 'The chain rule as linked slopes', x: xProp, controls: showControls = true, view = 'full' }: Props = $props();
  const V = untrack(() => view), T = duration(V), MARKERS = markers(V);

  const uid = $props.id();
  const start = untrack(() => ({ chain, x: xProp ?? CHAINS[chain].x0 }));
  let cid = $state<ChainId>(start.chain);
  let xv = $state(start.x);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let vertical = $state(false);
  let narration = $state(untrack(() => view) === 'wiggle' ? 'x is wiggled by a small amount. Each point shakes, and the bars show how big the shake is at x, u, v and L.' : 'Wiggle x a little and watch each link scale the wiggle by its slope.');
  let numbers = $state('');
  const C = $derived(CHAINS[cid]);

  let t = 0;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, poke = () => {};
  $effect(() => { cid; xv; vertical; playing; speed; poke(); });

  const f2 = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)).replace('-', '−');
  const LEGEND = [
    { label: 'value (wiggling)', color: '--viz-forward' },
    { label: 'local slope', color: '--viz-gradient', shape: 'line' as const },
  ];
  const SAY: Record<string, string> = {
    wiggle: 'Wiggle x by a small amount and follow the wiggle through the chain.',
    'link 1': 'Link 1: the tangent slope says how much the wiggle in x is scaled to become a wiggle in u.',
    'link 2': 'Link 2: the wiggle in u is scaled again by the slope of the second function.',
    'link 3': 'Link 3: and once more into L.',
    product: 'Multiply the three slopes: that is dL/dx, matching the numerical derivative.',
    'shrinking x': 'x slides toward 0. Watch the first tangent flatten: the slope of x² is 2x.',
    'at x = 0': 'At x = 0 the first tangent is flat, so the wiggle in x makes almost no wiggle in u, and nothing downstream moves.',
    'growing x again': 'x goes negative and back. The first slope changes sign; the wiggle sizes grow again.',
  };

  onMount(() => {
    const reduced = prefersReducedMotion();
    const ctx = cv.getContext('2d')!;
    const NAMES = { fwd: '--viz-forward', grad: '--viz-gradient', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent', soft: '--accent-soft' };
    let c = readColors(NAMES), code = readVar('--font-code'), body = readVar('--font-body');
    let lastUi = 0, phase = '';
    if (reduced.matches) t = 9;
    let views = [0, 1, 2].map((k) => linkView(C, k));
    let viewsFor = cid;

    const lay = () => vertical
      ? { w: 360, h: 790, plots: [0, 1, 2].map((k) => ({ x: 34, y: 34 + k * 172, w: 296, h: 112 })), barY: 590, barX: 60, barW: 210, eqX: 30, eqY: 700 }
      : { w: 720, h: 400, plots: [0, 1, 2].map((k) => ({ x: 24 + k * 236, y: 40, w: 200, h: 170 })), barY: 274, barX: 70, barW: 330, eqX: 470, eqY: 280 };

    const text = (s: string, x: number, y: number, col: string, size = 13, align: CanvasTextAlign = 'left', mono = true, a = 1) => {
      if (a <= 0.01) return;
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align;
      ctx.font = mono ? `600 ${size}px ${code}` : `italic ${size}px ${body}`;
      ctx.fillText(s, x, y);
    };

    function render(now: number) {
      if (viewsFor !== cid) { views = [0, 1, 2].map((k) => linkView(C, k)); viewsFor = cid; }
      const L = lay(), k = cv.width / L.w;
      ctx.setTransform(k, 0, 0, k, 0, 0);
      ctx.clearRect(0, 0, L.w, L.h);
      const fr = frameAt(t, C, xv, V);
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';

      L.plots.forEach((p, i) => {
        const v = views[i], rv = fr.reveal[i];
        const X = (a: number) => p.x + ((a - v.d0) / (v.d1 - v.d0)) * p.w;
        const Y = (b: number) => p.y + p.h - ((b - v.r0) / (v.r1 - v.r0)) * p.h;
        ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1;
        ctx.strokeRect(p.x, p.y, p.w, p.h);
        text(`${C.names[i + 1]} = ${C.links[i].expr(C.names[i])}`, p.x, p.y - 10, c.text.hex, 14);
        ctx.save();
        ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
        // zero lines
        ctx.strokeStyle = c.rule.hex; ctx.globalAlpha = 0.6; ctx.setLineDash([2, 4]);
        if (v.r0 < 0 && v.r1 > 0) { ctx.beginPath(); ctx.moveTo(p.x, Y(0)); ctx.lineTo(p.x + p.w, Y(0)); ctx.stroke(); }
        if (v.d0 < 0 && v.d1 > 0) { ctx.beginPath(); ctx.moveTo(X(0), p.y); ctx.lineTo(X(0), p.y + p.h); ctx.stroke(); }
        ctx.setLineDash([]);
        // curve
        ctx.globalAlpha = 1; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.6; ctx.beginPath();
        v.grid.forEach((a, j) => (j ? ctx.lineTo(X(a), Y(v.ys[j])) : ctx.moveTo(X(a), Y(v.ys[j]))));
        ctx.stroke();
        // tangent at the base point
        const bx = X(fr.base[i]), by = Y(fr.base[i + 1]);
        const screenSlope = -fr.slopes[i] * (p.h / (v.r1 - v.r0)) / (p.w / (v.d1 - v.d0));
        const half = p.w * 0.3 * Math.min(1, Math.max(0.35, rv * 1.4));
        ctx.globalAlpha = 0.9 * rv; ctx.strokeStyle = c.grad.hex; ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.moveTo(bx - half, by - screenSlope * half); ctx.lineTo(bx + half, by + screenSlope * half); ctx.stroke();
        // moving point with guides
        const px = X(fr.pts[i]), py = Y(fr.pts[i + 1]);
        ctx.globalAlpha = 0.5; ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(px, p.y + p.h); ctx.lineTo(px, py); ctx.lineTo(p.x, py); ctx.stroke(); ctx.setLineDash([]);
        ctx.globalAlpha = 1; ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(px, py, 5, 0, 6.2832); ctx.fill(); ctx.stroke();
        ctx.restore();
        // labels
        text(`d${C.names[i + 1]}/d${C.names[i]} = ${f2(fr.slopes[i])}`, p.x + p.w, p.y + p.h + 16, c.grad.hex, 12.5, 'right', true, rv);
        text(`${C.names[i]} = ${f2(fr.base[i])}`, p.x, p.y + p.h + 16, c.muted.hex, 12.5, 'left', true, 1);
        // arrow to next plot
        if (i < 2) {
          const q = L.plots[i + 1], ax = vertical ? p.x + p.w / 2 : (p.x + p.w + q.x) / 2, ay = vertical ? (p.y + p.h + q.y) / 2 + 8 : p.y + p.h / 2;
          ctx.globalAlpha = 0.7; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.5; ctx.beginPath();
          if (vertical) { ctx.moveTo(ax, ay - 7); ctx.lineTo(ax, ay + 7); ctx.moveTo(ax - 4, ay + 3); ctx.lineTo(ax, ay + 7); ctx.lineTo(ax + 4, ay + 3); }
          else { ctx.moveTo(ax - 7, ay); ctx.lineTo(ax + 7, ay); ctx.moveTo(ax + 3, ay - 4); ctx.lineTo(ax + 7, ay); ctx.lineTo(ax + 3, ay + 4); }
          ctx.stroke();
        }
      });

      // wiggle bars: size of the wiggle at x, u, v, L
      const maxP = Math.max(...fr.prods.map(Math.abs), 1) * WIGGLE;
      const half = L.barW / 2, cx = L.barX + half;
      text('size of the wiggle', L.barX - (vertical ? 30 : 50), L.barY - 12, c.muted.hex, 13, 'left', false);
      for (let i = 0; i < 4; i++) {
        const y = L.barY + i * 24, rv = fr.bars[i];
        text(C.names[i], L.barX - 12, y + 12, c.text.hex, 13.5, 'right');
        ctx.globalAlpha = 1; ctx.fillStyle = c.rule.hex; ctx.fillRect(L.barX, y + 5, L.barW, 1);
        const a = (fr.amp[i] / maxP) * half * rv;
        ctx.globalAlpha = 0.25 * (rv > 0 ? 1 : 0); ctx.fillStyle = c.fwd.hex; ctx.fillRect(cx - a, y + 1, 2 * a, 14);
        if (rv > 0.02) {
          const d = ((fr.pts[i] - fr.base[i]) / maxP) * half * rv;
          ctx.globalAlpha = 1; ctx.fillStyle = c.fwd.hex; ctx.beginPath(); ctx.arc(cx + d, y + 8, 4.2, 0, 6.2832); ctx.fill();
        }
        if (V === 'wiggle') text(f2(fr.amp[i]), L.barX + L.barW + 10, y + 13, c.fwd.hex, 12.5, 'left', true, 1);
        else text(`×${f2(Math.abs(fr.prods[i]))}`, L.barX + L.barW + 10, y + 13, c.grad.hex, 12.5, 'left', true, rv);
      }
      // equation
      const ex = L.eqX, ey = L.eqY, r3 = fr.reveal[3];
      if (!vertical) text('multiply the slopes', ex, ey - 14, c.muted.hex, 13, 'left', false, fr.eq > 0 ? 1 : 0);
      const line = vertical ? [ey, ey + 26, ey + 52] : [ey + 14, ey + 44, ey + 74];
      text(`dL/dx = ${fr.slopes.map(f2).join(' × ')}`, ex, line[0], c.grad.hex, 13, 'left', true, fr.eq);
      text(`      = ${f2(fr.dLdx)}`, ex, line[1], c.grad.hex, 15, 'left', true, r3);
      text(`numerical: ${f2(fr.numeric)}`, ex, line[2], c.muted.hex, 13, 'left', true, r3);
      ctx.globalAlpha = 1;

      tl.paint(t);
      if (now - lastUi > 100 || !playing) {
        lastUi = now; uiT = t;
        if (fr.phase !== phase) { phase = fr.phase; narration = SAY[phase] ?? narration; }
        numbers = V === 'wiggle' ? `wiggle of ${f2(WIGGLE)} in x → ${f2(fr.amp[1])} in u → ${f2(fr.amp[2])} in v → ${f2(fr.amp[3])} in L`
          : V === 'tangent' ? `x = ${f2(fr.x)} · slopes ${fr.slopes.map(f2).join(', ')}`
          : `x = ${f2(xv)} · dL/dx = ${f2(fr.dLdx)} (product of slopes) · ${f2(fr.numeric)} (numerical)`;
      }
    }

    const loop = createLoop(cv.closest('[data-figure]') ?? cv, (dt, now) => {
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
    const ro = new ResizeObserver(([e]) => { vertical = e.contentRect.width < 540; fit(); render(performance.now()); });
    ro.observe(cv.parentElement!);
    const off = onThemeChange(() => { c = readColors(NAMES); code = readVar('--font-code'); body = readVar('--font-body'); poke(); });
    const onReduce = () => { if (reduced.matches) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    fit();
    return () => { loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); };
  });
</script>

<FigureCard {title} legend={LEGEND} role="img" aspect="720 / 400" narrowAspect="360 / 790"
  label="Three linked graphs for a chain of functions. Each shows a curve with a tangent line; a point wiggles along each curve, and the slopes multiply to the derivative of the whole chain.">
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Timeline of the chain rule"
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { t = to >= T ? 0 : to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="num muted">{numbers}</p>
  {/snippet}
  {#snippet footer()}
    {#if showControls}
      <div class="ctl">
        <label>x
          <input type="range" min={C.xRange[0]} max={C.xRange[1]} step="0.01" bind:value={xv} aria-label="Value of x" />
          <output class="num">{f2(xv)}</output>
        </label>
        <fieldset class="seg">
          <legend class="sr-only">Chain of functions</legend>
          <label><input type="radio" name="chain-{uid}" value="wave" bind:group={cid} onchange={() => (xv = CHAINS.wave.x0)} /><span>x² → sin → square</span></label>
          <label><input type="radio" name="chain-{uid}" value="saturate" bind:group={cid} onchange={() => (xv = CHAINS.saturate.x0)} /><span>2x → tanh → 3v²</span></label>
        </fieldset>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .say, .muted { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-ui); min-height: 2.6em; }
  .muted { color: var(--text-muted); font-size: var(--step--1); }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='range'] { width: 11rem; accent-color: var(--accent); }
  .ctl output { min-width: 3rem; color: var(--text); }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
