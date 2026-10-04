<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { splitPanels } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { T, MARKERS, DOM, RMAX, BIAS, frameAt, narrate, polar, angleOf, norm, type Mode, type Vec } from './sim';

  interface Props {
    title?: string;
    /** geometry: shadow + cosine curve. bars: shadow + elementwise products. neuron: w·x + bias with ReLU. */
    mode?: Mode;
  }
  let { title = '', mode = 'geometry' }: Props = $props();
  const init = untrack(() => ({
    mode,
    title: title || (mode === 'neuron' ? 'A neuron is a dot product' : mode === 'bars' ? 'Multiply, then add' : 'Two arrows and their shadow'),
  }));
  const nm = init.mode === 'neuron' ? { a: 'w', b: 'x' } : { a: 'a', b: 'b' };

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let over = $state<{ a: Vec; b: Vec } | null>(null);
  let ui = $state(frameAt(0, init.mode));
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

    const narration = $derived(over ? 'You placed the arrows. Press play to let the timeline take over again. ' + narrate(ui, init.mode) : narrate(ui, init.mode));
  $effect(() => { if (playing) over = null; speed; wake(); });

  const frame = () => (over ? frameAt(t, init.mode, over) : frameAt(t, init.mode));
  const place = (a: Vec, b: Vec) => {
    const cl = (v: Vec): Vec => { const n = norm(v); return n > RMAX ? [(v[0] * RMAX) / n, (v[1] * RMAX) / n] : v; };
    over = { a: cl(a), b: cl(b) }; playing = false; ui = frame(); wake();
  };

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let drag: 'a' | 'b' | null = null;
    const lay = () => {
      const { main, side, stacked } = splitPanels(stage.w, stage.h, 0.5);
      const s = Math.max(60, Math.min(main.w - 8, main.h - 8)), k = s / (2 * DOM);
      const cx = main.x + main.w / 2, cy = main.y + main.h / 2;
      return { main, side, stacked, s, k, cx, cy, px: (v: Vec): [number, number] => [cx + v[0] * k, cy - v[1] * k] };
    };
    const arrow = (x0: number, y0: number, x1: number, y1: number, col: string, wd: number) => {
      const { ctx } = stage, ang = Math.atan2(y1 - y0, x1 - x0), h = 9;
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = wd; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(ang) * h * 0.6, y1 - Math.sin(ang) * h * 0.6); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - h * Math.cos(ang - 0.4), y1 - h * Math.sin(ang - 0.4)); ctx.lineTo(x1 - h * Math.cos(ang + 0.4), y1 - h * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
    };

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay(), f = frame(), neuron = init.mode === 'neuron';
      ctx.clearRect(0, 0, W, H);
      const half = L.s / 2, x0 = L.cx - half, y0 = L.cy - half;
      const sign = (v: number) => (v >= 0 ? c.pos : c.neg);
      // ---- plane ----
      ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, L.s, L.s); ctx.clip();
      ctx.lineWidth = 1; ctx.strokeStyle = c.rule.hex;
      for (let g = -3; g <= 3; g++) { const [gx, gy] = L.px([g, g]); ctx.globalAlpha = g === 0 ? 1 : 0.45; ctx.beginPath(); ctx.moveTo(gx, y0); ctx.lineTo(gx, y0 + L.s); ctx.moveTo(x0, gy); ctx.lineTo(x0 + L.s, gy); ctx.stroke(); }
      ctx.globalAlpha = 1;
      const o = L.px([0, 0]), tipA = L.px(f.a), tipB = L.px(f.b);
      if (neuron) {
        // z = w·p + bias > 0 on one side of a line perpendicular to w
        const n2 = f.a[0] ** 2 + f.a[1] ** 2, base: Vec = [(-BIAS * f.a[0]) / n2, (-BIAS * f.a[1]) / n2];
        const pr: Vec = [-f.a[1] / Math.sqrt(n2), f.a[0] / Math.sqrt(n2)], nw: Vec = [f.a[0] / Math.sqrt(n2), f.a[1] / Math.sqrt(n2)], R = 12;
        const q = (s: number, d: number) => L.px([base[0] + pr[0] * s + nw[0] * d, base[1] + pr[1] * s + nw[1] * d]);
        const p1 = q(-R, 0), p2 = q(R, 0), p3 = q(R, R), p4 = q(-R, R);
        ctx.fillStyle = `rgba(${c.pos.rgb.join(',')},0.12)`;
        ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2); ctx.lineTo(...p3); ctx.lineTo(...p4); ctx.closePath(); ctx.fill();
        ctx.setLineDash([5, 5]); ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2); ctx.stroke(); ctx.setLineDash([]);
      } else {
        const pj = L.px(f.proj), col = sign(f.dot);
        // infinite line through a (faint), shadow of b on it (thick), drop line
        ctx.strokeStyle = c.muted.hex; ctx.globalAlpha = 0.35; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
        const ua: Vec = [f.a[0] / (norm(f.a) || 1), f.a[1] / (norm(f.a) || 1)], e1 = L.px([ua[0] * -8, ua[1] * -8]), e2 = L.px([ua[0] * 8, ua[1] * 8]);
        ctx.beginPath(); ctx.moveTo(...e1); ctx.lineTo(...e2); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
        ctx.strokeStyle = col.hex; ctx.globalAlpha = 0.55; ctx.lineWidth = 9; ctx.lineCap = 'butt';
        ctx.beginPath(); ctx.moveTo(...o); ctx.lineTo(...pj); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.setLineDash([4, 4]); ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(...tipB); ctx.lineTo(...pj); ctx.stroke(); ctx.setLineDash([]);
        // angle arc
        const sgn = f.theta360 <= 180 ? 1 : -1, aa = angleOf(f.a), r = Math.min(0.9 * L.k, 34);
        ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.4; ctx.globalAlpha = 0.8;
        ctx.beginPath(); ctx.arc(o[0], o[1], r, -aa, -(aa + sgn * f.theta), sgn > 0); ctx.stroke(); ctx.globalAlpha = 1;
      }
      arrow(o[0], o[1], tipA[0], tipA[1], c.fwd.hex, 3);
      arrow(o[0], o[1], tipB[0], tipB[1], c.accent.hex, 3);
      for (const [p, col, act] of [[tipA, c.fwd, drag === 'a'], [tipB, c.accent, drag === 'b']] as const) {
        ctx.fillStyle = col.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p[0], p[1], act ? 8 : 6.5, 0, 6.2832); ctx.fill(); ctx.stroke();
      }
      ctx.restore();
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, L.s, L.s);
      ctx.font = stage.body(13, true); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const lab = (tip: [number, number], text: string, col: string) => {
        const dx = tip[0] - o[0], dy = tip[1] - o[1], d = Math.hypot(dx, dy) || 1;
        ctx.fillStyle = col; ctx.fillText(text, tip[0] + (dx / d) * 15, tip[1] + (dy / d) * 15);
      };
      lab(tipA, nm.a, c.fwd.hex); lab(tipB, nm.b, c.accent.hex);
      ctx.font = stage.body(11.5); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(neuron ? 'shaded: z > 0, the neuron fires' : 'drag either arrow', x0 + 6, y0 + 5);

      // ---- side panel ----
      if (init.mode === 'geometry') curve(L, f); else bars(L, f);
    }

    function curve(L: ReturnType<typeof lay>, f: ReturnType<typeof frame>) {
      const { ctx, c } = stage, S = L.side;
      const m = norm(f.a) * norm(f.b) || 1;
      const l = S.x + 38, r = S.x + S.w - 12, tp = S.y + 34, bt = S.y + S.h - 34, mid = (tp + bt) / 2;
      ctx.fillStyle = c.text.hex; ctx.font = stage.body(12.5); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('a·b as b turns away from a', S.x + 4, S.y + 2);
      const X = (deg: number) => l + (deg / 180) * (r - l), Y = (v: number) => mid - (v / m) * ((bt - tp) / 2);
      ctx.lineWidth = 1; ctx.strokeStyle = c.rule.hex; ctx.strokeRect(l, tp, r - l, bt - tp);
      ctx.beginPath(); ctx.moveTo(l, mid); ctx.lineTo(r, mid); ctx.stroke();
      ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(X(90), tp); ctx.lineTo(X(90), bt); ctx.stroke(); ctx.setLineDash([]);
      ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textBaseline = 'top'; ctx.textAlign = 'center';
      for (const d of [0, 90, 180]) ctx.fillText(d + '°', X(d), bt + 5);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText('+' + f2(m, 1), l - 5, tp + 2); ctx.fillText('0', l - 5, mid); ctx.fillText('−' + f2(m, 1), l - 5, bt - 2);
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.font = stage.body(11.5); ctx.fillText('angle between a and b', r, S.y + S.h - 1);
      // curve, coloured by sign
      for (const [from, to, col] of [[0, 90, c.pos], [90, 180, c.neg]] as const) {
        ctx.strokeStyle = col.hex; ctx.lineWidth = 2.2; ctx.beginPath();
        for (let d = from; d <= to; d += 3) { const v = m * Math.cos((d * Math.PI) / 180); if (d === from) ctx.moveTo(X(d), Y(v)); else ctx.lineTo(X(d), Y(v)); }
        ctx.stroke();
      }
      const deg = (f.theta * 180) / Math.PI, px = X(deg), py = Y(f.dot);
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px, mid); ctx.lineTo(px, py); ctx.stroke();
      ctx.fillStyle = (f.dot >= 0 ? c.pos : c.neg).hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, 6, 0, 6.2832); ctx.fill(); ctx.stroke();
    }

    function bars(L: ReturnType<typeof lay>, f: ReturnType<typeof frame>) {
      const { ctx, c } = stage, S = L.side, neuron = init.mode === 'neuron';
      ctx.fillStyle = c.text.hex; ctx.font = stage.body(12.5); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(neuron ? 'the neuron adds three terms' : 'multiply matching entries, then add', S.x + 4, S.y + 2);
      const rows = f.items.length + 1 + (neuron ? 1 : 0), top = S.y + 30, rh = Math.min(54, (S.h - 40) / rows), bh = Math.min(28, rh - 12);
      const lw = 52, l = S.x + lw, r = S.x + S.w - 44, zero = (l + r) / 2, k = (r - l) / 2 / 8;
      const X = (v: number) => zero + v * k;
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(zero, top - 4); ctx.lineTo(zero, top + rows * rh + (neuron ? -2 : 2)); ctx.stroke();
      const row = (i: number, label: string, segs: { from: number; to: number; v: number }[], val: number, bold: boolean) => {
        const y = top + i * rh + (rh - bh) / 2;
        ctx.font = stage.mono(11, bold ? 650 : 500); ctx.fillStyle = bold ? c.text.hex : c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(label, S.x + 2, y + bh / 2);
        const n = segs.length, sh = n > 1 ? bh / n : bh;
        segs.forEach((s, j) => {
          const col = s.v >= 0 ? c.pos : c.neg, a = Math.min(X(s.from), X(s.to)), wd = Math.abs(X(s.to) - X(s.from)), yy = y + j * (n > 1 ? sh : 0);
          ctx.fillStyle = `rgba(${col.rgb.join(',')},${bold ? 0.85 : 0.7})`; ctx.fillRect(a, yy, wd, sh);
          ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 1; ctx.strokeRect(a, yy, wd, sh);
          if (n > 1 && wd > 14) { ctx.fillStyle = c.text.hex; ctx.font = stage.mono(9); ctx.textAlign = 'center'; ctx.fillText(s.v >= 0 ? '+' : '−', a + wd / 2, yy + sh / 2); ctx.font = stage.mono(11, bold ? 650 : 500); }
        });
        if (n > 1) { ctx.strokeStyle = c.text.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(val), y - 3); ctx.lineTo(X(val), y + bh + 3); ctx.stroke(); }
        const lo = Math.min(0, ...segs.flatMap((q) => [q.from, q.to])), hi = Math.max(0, ...segs.flatMap((q) => [q.from, q.to]));
        ctx.fillStyle = c.text.hex; ctx.textAlign = val >= 0 ? 'left' : 'right';
        ctx.fillText(f2(val), val >= 0 ? X(hi) + 5 : X(lo) - 5, y + bh / 2);
      };
      f.items.forEach((it, i) => row(i, it.label, [{ from: 0, to: it.v, v: it.v }], it.v, false));
      let cum = 0; const stack = f.items.map((it) => { const s = { from: cum, to: cum + it.v, v: it.v }; cum += it.v; return s; });
      const n = f.items.length;
      row(n, neuron ? 'sum z' : 'a·b', stack, f.total, true);
      if (neuron) row(n + 1, 'ReLU(z)', [{ from: 0, to: f.out!, v: 1 }], f.out!, true);
    }

    const loop = createLoop(cv, (dt, now) => {
      if (!over) t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      ui = frame(); uiT = t; // every frame: readout must equal what the canvas just drew
      return playing || drag !== null;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const at = (e: PointerEvent): { L: ReturnType<typeof lay>; v: Vec; hit: 'a' | 'b' | null } => {
      const r = cv.getBoundingClientRect(), L = lay(), x = e.clientX - r.left, y = e.clientY - r.top, f = frame();
      const near = (p: Vec) => { const [px, py] = L.px(p); return Math.hypot(px - x, py - y); };
      const da = near(f.a), db = near(f.b), R = 22;
      return { L, v: [(x - L.cx) / L.k, -(y - L.cy) / L.k], hit: Math.min(da, db) > R ? null : db <= da ? 'b' : 'a' };
    };
    const put = (v: Vec) => { const f = frame(); place(drag === 'a' ? v : f.a, drag === 'b' ? v : f.b); };
    const down = (e: PointerEvent) => { const { hit, v } = at(e); if (hit) { drag = hit; cv.setPointerCapture(e.pointerId); put(v); e.preventDefault(); } };
    const move = (e: PointerEvent) => { const { hit, v } = at(e); if (drag) put(v); else cv.style.cursor = hit ? 'grab' : ''; };
    const up = () => { drag = null; wake(); };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });

  const legend = init.mode === 'neuron'
    ? [{ label: 'weights w', color: '--viz-forward', shape: 'line' as const }, { label: 'input x', color: '--accent', shape: 'line' as const }, { label: 'positive', color: '--viz-pos' }, { label: 'negative', color: '--viz-neg' }]
    : [{ label: 'arrow a', color: '--viz-forward', shape: 'line' as const }, { label: 'arrow b', color: '--accent', shape: 'line' as const }, { label: 'positive', color: '--viz-pos' }, { label: 'negative', color: '--viz-neg' }];
</script>

<FigureCard title={init.title} role="group" aspect="720 / 400" narrowAspect="360 / 620" {legend}
  label={init.mode === 'neuron'
    ? 'Left: a weight arrow w and an input arrow x in the plane, with the region where the neuron fires shaded. Right: bars for each product w times x, the bias, and their sum, then the ReLU output.'
    : init.mode === 'bars'
      ? 'Left: two arrows a and b in the plane, with the shadow of b on a. Right: bars for each matching product a1 times b1 and a2 times b2, and a third bar showing they add up to the dot product.'
      : 'Left: two arrows a and b in the plane, with the angle between them and the shadow of b on a. Right: a curve of the dot product against the angle, with a dot marking the current value.'}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS} label="Timeline: b turns around a"
      onSeek={(to) => { over = null; t = to; wake(); }} onStep={(to) => { over = null; t = to >= T ? 0 : to; wake(); }} onReset={() => { over = null; t = 0; playing = true; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">
      {#if init.mode === 'neuron'}
        w·x = <b>{f2(ui.dot)}</b> · z = w·x + bias = <b>{f2(ui.total)}</b> (bias {f2(BIAS)}) · output = <b>{f2(ui.out ?? 0)}</b>
      {:else}
        a·b = <b>{f2(ui.dot)}</b> · cos θ = <b>{f2(ui.cos)}</b> · θ = <b>{Math.round((ui.theta * 180) / Math.PI)}°</b>
        <span class="base">· |a| = {f2(norm(ui.a))}, |b| = {f2(norm(ui.b))}</span>
      {/if}
    </p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      <F1Slider label="θ" value={ui.theta360} min={0} max={360} step={1} digits={0}
        oninput={(v) => place(ui.a, polar(norm(ui.b) || 1, angleOf(ui.a) + (v * Math.PI) / 180))} />
      <span class="hint">Drag the arrow tips, or use the angle slider.</span>
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.95rem var(--font-code); font-variant-numeric: tabular-nums; }
  .vals b { font-weight: 650; color: var(--viz-forward); }
  .base { font-size: 0.8rem; color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
</style>
