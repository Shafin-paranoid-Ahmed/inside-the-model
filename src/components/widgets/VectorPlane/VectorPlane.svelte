<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { splitPanels, clamp } from '../F1Shared/plot';
  import F1Slider from '../F1Shared/F1Slider.svelte';
  import { add, scaleBy, norm, dist, lerpV, ease, frameAt, duration, markers, DEFAULT_A, DEFAULT_B, LIM, FEATURES, HOUSE_POINT, type Scene, type V } from './sim';

  interface Props {
    title?: string;
    /** 'add' tip-to-tail, 'scale' c·a, 'length' norm and distance, 'features' a 2D point becoming a row of features. */
    scene?: Scene;
  }
  let { title = '', scene = 'add' }: Props = $props();
  const init = untrack(() => ({
    scene,
    title: title || ({ add: 'Adding two vectors', scale: 'Scaling a vector', length: 'Length and distance', features: 'From a point to a row of features' } as const)[scene],
  }));
  const S = init.scene, T = duration(S), MK = markers(S);
  const DRAG = S === 'add' || S === 'length' ? ['a', 'b'] : S === 'scale' ? ['a'] : [];

  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let a = $state.raw<V>({ ...DEFAULT_A }), b = $state.raw<V>({ ...DEFAULT_B });
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const ui = $derived(frameAt(uiT, S, a, b));
  const phase = $derived(markerAt(uiT, MK)?.label ?? '');
  const v2 = (v: V) => `[${f2(v.x, 1)}, ${f2(v.y, 1)}]`;
  const narration = $derived(
    S === 'add'
      ? phase === 'a and b' ? 'Two arrows from the origin.'
        : phase === 'slide b' ? 'Slide a copy of b, without turning it, until its tail sits on the tip of a.'
        : phase === 'a + b' ? 'The arrow from the origin to where you ended up is the sum. Its numbers are the entry-wise sums.'
        : 'Resetting: drag a tip and watch again.'
      : S === 'scale'
        ? phase === 'c = 1' ? 'With c = 1 the arrow is unchanged.'
          : phase === 'stretch' ? 'A factor above 1 stretches the arrow along the same line.'
          : phase === 'shrink' ? 'A factor between 0 and 1 shrinks it; at 0 it collapses to the origin.'
          : phase === 'flip' ? 'A negative factor flips the arrow to the opposite direction.'
          : 'Back to c = 1.'
      : S === 'length'
        ? phase === 'a' ? 'An arrow a with two coordinates.'
          : phase === 'length' ? 'Walk along x, then along y: the arrow is the diagonal of a right triangle, so its length is the square root of the two squares added.'
          : phase === 'distance' ? 'The distance between two vectors is the length of the arrow from one tip to the other.'
          : 'Resetting.'
      : phase === 'a point' ? 'A house as a point: its size is the x coordinate and its room count is y.'
        : phase === 'read off' ? 'Read off the coordinates: they are the first two entries of a list.'
        : phase === 'add features' ? 'Add more features. Each one is another coordinate, another dimension we can no longer draw.'
        : 'Six numbers: one point in a 6-dimensional space. Same operations, same formulas.',
  );
  const nums = $derived(
    S === 'add' ? `a = ${v2(a)} · b = ${v2(b)} · a + b = ${v2(add(a, b))}`
      : S === 'scale' ? `a = ${v2(a)} · c = ${f2(ui.c)} · c·a = ${v2(scaleBy(ui.c, a))} · |c·a| = ${f2(Math.abs(ui.c) * norm(a))}`
      : S === 'length' ? `|a| = ${f2(ui.lenA)} (final ${f2(norm(a))}) · |a − b| = ${f2(ui.distAB)} (final ${f2(dist(a, b))})`
      : `house = [${FEATURES.map((f) => f.value).join(', ')}]`,
  );
  $effect(() => { speed; wake(); });

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0, drag: 'a' | 'b' | null = null;
    const feat = S === 'features';
    /** Square view that fits what the scene draws (origin, a, b, a + b, or the extremes of c·a); snapped to whole steps so dragging doesn't jitter. */
    const dom = () => {
      if (feat) return { xl: -1, xh: 6.4, yl: -1, yh: 6.4 };
      const pts: V[] = [{ x: 0, y: 0 }, a];
      if (S !== 'scale') pts.push(b);
      if (S === 'add') pts.push(add(a, b));
      if (S === 'scale') pts.push(scaleBy(2, a), scaleBy(-1.5, a));
      const x0 = Math.min(...pts.map((p) => p.x)), x1 = Math.max(...pts.map((p) => p.x)), y0 = Math.min(...pts.map((p) => p.y)), y1 = Math.max(...pts.map((p) => p.y));
      const side = Math.ceil(Math.max(x1 - x0, y1 - y0, 3) + 1.6), cx = Math.round((x0 + x1) / 1) / 2, cy = Math.round((y0 + y1) / 1) / 2;
      return { xl: cx - side / 2, xh: cx + side / 2, yl: cy - side / 2, yh: cy + side / 2 };
    };

    const lay = () => {
      const { main, side, stacked } = splitPanels(stage.w, stage.h, stage.w < 540 ? 0.64 : 0.56);
      const pad = 22, size = Math.min(main.w - pad - 6, main.h - pad - 6), d = dom(), s = size / (d.xh - d.xl);
      const ox = main.x + pad + (main.w - pad - 6 - size) / 2, oy = main.y + 6 + (main.h - pad - 6 - size) / 2;
      return { main, side, stacked, size, s, ox, oy, ...d, px: (x: number) => ox + (x - d.xl) * s, py: (y: number) => oy + (d.yh - y) * s };
    };
    type L = ReturnType<typeof lay>;

    function arrow(L: L, from: V, to: V, color: string, w = 3, alpha = 1, dash = false) {
      const { ctx } = stage;
      const x0 = L.px(from.x), y0 = L.py(from.y), x1 = L.px(to.x), y1 = L.py(to.y), len = Math.hypot(x1 - x0, y1 - y0);
      if (len < 1.5) return;
      const ang = Math.atan2(y1 - y0, x1 - x0), hd = Math.min(10, len * 0.6);
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w; ctx.lineCap = 'round';
      if (dash) ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(ang) * hd * 0.6, y1 - Math.sin(ang) * hd * 0.6); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - hd * Math.cos(ang - 0.4), y1 - hd * Math.sin(ang - 0.4)); ctx.lineTo(x1 - hd * Math.cos(ang + 0.4), y1 - hd * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
      ctx.restore();
    }
    function tag(L: L, at: V, text: string, color: string, dx = 8, dy = -8) {
      const { ctx } = stage;
      ctx.font = stage.body(14, true); ctx.fillStyle = color; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(text, L.px(at.x) + dx, L.py(at.y) + dy);
    }
    function handle(L: L, at: V, color: string) {
      const { ctx, c } = stage;
      ctx.fillStyle = c.raised.hex; ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(L.px(at.x), L.py(at.y), 6.5, 0, 6.2832); ctx.fill(); ctx.stroke();
    }

    function plane(L: L, alpha: number) {
      const { ctx, c } = stage;
      ctx.save(); ctx.globalAlpha = alpha; ctx.lineWidth = 1;
      const step = feat || L.xh - L.xl <= 8 ? 1 : 2, fmt = (v: number) => String(v).replace('-', '−');
      ctx.font = stage.mono(10.5); ctx.fillStyle = c.muted.hex; ctx.strokeStyle = c.rule.hex;
      for (let v = Math.ceil(L.xl / step) * step; v <= L.xh; v += step) {
        ctx.globalAlpha = alpha * (v === 0 ? 1 : 0.55);
        ctx.beginPath(); ctx.moveTo(L.px(v), L.py(L.yl)); ctx.lineTo(L.px(v), L.py(L.yh)); ctx.stroke();
        ctx.globalAlpha = alpha;
        if ((v !== 0 || feat) && !(feat && v < 0)) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(feat ? String(v * 40) : fmt(v), L.px(v), L.py(L.yl) + 5); }
      }
      for (let v = Math.ceil(L.yl / step) * step; v <= L.yh; v += step) {
        ctx.globalAlpha = alpha * (v === 0 ? 1 : 0.55);
        ctx.beginPath(); ctx.moveTo(L.px(L.xl), L.py(v)); ctx.lineTo(L.px(L.xh), L.py(v)); ctx.stroke();
        ctx.globalAlpha = alpha;
        if ((v !== 0 || feat) && !(feat && v < 0)) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(fmt(v), L.px(L.xl) - 5, L.py(v)); }
      }
      ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2; ctx.globalAlpha = alpha * 0.8;
      ctx.beginPath(); ctx.moveTo(L.px(L.xl), L.py(0)); ctx.lineTo(L.px(L.xh), L.py(0)); ctx.moveTo(L.px(0), L.py(L.yl)); ctx.lineTo(L.px(0), L.py(L.yh)); ctx.stroke();
      ctx.restore();
    }

    /** Rows of "label  [numbers]" on the side panel. */
    function vecRows(L: L, f: ReturnType<typeof frameAt>) {
      const { ctx, c } = stage, sd = L.side, x0 = sd.x + 12;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('The same arrows, as lists of numbers', x0, sd.y + 6);
      const rows: [string, string, string][] = [];
      if (S === 'add') rows.push(['a', v2(a), c.fwd.hex], ['b', v2(b), c.accent.hex], ['a + b', v2(add(a, b)), c.text.hex]);
      if (S === 'scale') rows.push(['a', v2(a), c.fwd.hex], ['c', f2(f.c), f.c >= 0 ? c.pos.hex : c.neg.hex], ['c·a', v2(scaleBy(f.c, a)), f.c >= 0 ? c.pos.hex : c.neg.hex]);
      if (S === 'length') rows.push(['a', v2(a), c.fwd.hex], ['b', v2(b), c.accent.hex], ['|a|', f2(f.lenA), c.fwd.hex], ['|a − b|', f2(f.distAB), c.text.hex]);
      const gap = L.stacked ? 24 : 34;
      rows.forEach(([n, v, col], i) => {
        const y = sd.y + 32 + i * gap;
        ctx.font = stage.body(15, true); ctx.fillStyle = col; ctx.fillText(n, x0, y);
        ctx.font = stage.mono(L.stacked ? 13 : 15); ctx.fillStyle = c.text.hex; ctx.fillText(v, x0 + (L.stacked ? 56 : 74), y);
      });
      if (S === 'add') { ctx.font = stage.body(12); ctx.fillStyle = c.muted.hex; ctx.fillText('add matching entries', x0, sd.y + 32 + rows.length * gap + 4); }
      if (S === 'length') { ctx.font = stage.body(12); ctx.fillStyle = c.muted.hex; ctx.fillText(L.stacked ? '|a| = √(x² + y²)' : '|a| = √(x² + y²),  |a − b| = |a| of the difference', x0, sd.y + 32 + rows.length * gap + 4); }
      if (S === 'scale') { ctx.font = stage.body(12); ctx.fillStyle = c.muted.hex; ctx.fillText('multiply every entry by c', x0, sd.y + 32 + rows.length * gap + 4); }
    }

    /** cell i of the feature list: position of its label, for the flying chips. */
    const cellPos = (L: L, i: number) => {
      const sd = L.side, top = sd.y + (L.stacked ? 34 : 56), rh = Math.min(30, (sd.h - (top - sd.y) - 4) / FEATURES.length);
      return { x: sd.x + 12, y: top + i * rh, rh };
    };
    function featureRows(L: L, f: ReturnType<typeof frameAt>) {
      const { ctx, c } = stage, sd = L.side, x0 = sd.x + 12;
      ctx.font = stage.body(12.5); ctx.fillStyle = c.text.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('The same house, as a row of features', x0, sd.y + 6);
      const shown = FEATURES.map((ft, i) => (i < 2 ? ease((f.move - 0.85) / 0.15) : clamp(f.cells - i, 0, 1)));
      const n = shown.filter((v) => v > 0.5).length;
      ctx.font = stage.mono(L.stacked ? 12 : 13); ctx.fillStyle = c.fwd.hex;
      ctx.fillText(n ? `[${FEATURES.slice(0, n).map((ft) => ft.value).join(', ')}${n < FEATURES.length ? ', …' : ''}]` : '[ ]', x0, sd.y + 26);
      FEATURES.forEach((ft, i) => {
        const p = cellPos(L, i), al = shown[i];
        if (al <= 0) return;
        const bw = Math.max(20, sd.w - 24 - (L.stacked ? 150 : 190));
        ctx.save(); ctx.globalAlpha = al; ctx.textBaseline = 'middle';
        const yy = p.y + p.rh / 2;
        ctx.font = stage.body(12); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.fillText(ft.name, x0, yy);
        const bx = x0 + (L.stacked ? 92 : 106), bh = Math.min(10, p.rh - 8);
        ctx.fillStyle = c.rule.hex; ctx.fillRect(bx, yy - bh / 2, bw, bh);
        ctx.fillStyle = c.fwd.hex; ctx.fillRect(bx, yy - bh / 2, (bw * ft.value) / ft.max * al, bh);
        ctx.font = stage.mono(12); ctx.fillStyle = c.text.hex; ctx.fillText(String(ft.value), bx + bw + 8, yy);
        ctx.restore();
      });
    }

    function render() {
      const { ctx, w: W, h: H, c } = stage, L = lay();
      const f = frameAt(t, S, a, b);
      ctx.clearRect(0, 0, W, H);
      ctx.save(); ctx.beginPath(); ctx.rect(L.px(L.xl), L.py(L.yh), L.size, L.size); ctx.clip();
      plane(L, f.plane);
      const O: V = { x: 0, y: 0 };
      if (S === 'add') {
        const sum = add(a, b);
        if (f.sum > 0) { arrow(L, b, sum, c.fwd.hex, 2, 0.55 * f.sum, true); arrow(L, a, sum, c.accent.hex, 2, 0.55 * f.sum, true); }
        arrow(L, O, a, c.fwd.hex); arrow(L, O, b, c.accent.hex, f.slide > 0 ? 2 : 3, f.slide > 0 ? 0.45 : 1);
        if (f.slide > 0) { const o = scaleBy(f.slide, a); arrow(L, o, add(o, b), c.accent.hex, 3); }
        if (f.sum > 0) { arrow(L, O, lerpV(O, sum, f.sum), c.text.hex, 3.5); if (f.sum > 0.9) tag(L, sum, 'a + b', c.text.hex, 10, -6); }
      } else if (S === 'scale') {
        const line = 12 / Math.max(0.01, norm(a));
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.setLineDash([4, 5]); ctx.beginPath(); ctx.moveTo(L.px(-a.x * line), L.py(-a.y * line)); ctx.lineTo(L.px(a.x * line), L.py(a.y * line)); ctx.stroke(); ctx.setLineDash([]);
        arrow(L, O, a, c.fwd.hex, 2, 0.4);
        const ca = scaleBy(f.c, a), col = f.c >= 0 ? c.pos.hex : c.neg.hex;
        arrow(L, O, ca, col, 3.5);
        tag(L, a, 'a', c.fwd.hex, 8, -8);
        if (norm(ca) > 0.6) tag(L, ca, 'c·a', col, 10, -6);
      } else if (S === 'length') {
        const g = f.legs, leg1 = clamp(g * 2, 0, 1), leg2 = clamp(g * 2 - 1, 0, 1);
        ctx.lineWidth = 2; ctx.strokeStyle = c.muted.hex; ctx.setLineDash([5, 4]);
        ctx.beginPath(); ctx.moveTo(L.px(0), L.py(0)); ctx.lineTo(L.px(a.x * leg1), L.py(0)); if (leg2 > 0) ctx.lineTo(L.px(a.x), L.py(a.y * leg2)); ctx.stroke(); ctx.setLineDash([]);
        if (g > 0.5) { ctx.font = stage.mono(11); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(f2(a.x, 1), L.px(a.x / 2), L.py(0) + (a.y >= 0 ? 4 : -16)); }
        if (g > 0.98) { ctx.textAlign = a.x >= 0 ? 'left' : 'right'; ctx.textBaseline = 'middle'; ctx.fillText(f2(a.y, 1), L.px(a.x) + (a.x >= 0 ? 6 : -6), L.py(a.y / 2)); }
        arrow(L, O, a, c.fwd.hex, 3.5); arrow(L, O, b, c.accent.hex, 3.5);
        if (f.gap > 0) arrow(L, a, lerpV(a, b, f.gap), c.text.hex, 2.5, 1, true);
        tag(L, a, 'a', c.fwd.hex); tag(L, b, 'b', c.accent.hex);
      } else {
        const p = HOUSE_POINT, al = f.plane;
        ctx.globalAlpha = al; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(L.px(p.x), L.py(0)); ctx.lineTo(L.px(p.x), L.py(p.y)); ctx.lineTo(L.px(0), L.py(p.y)); ctx.stroke(); ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        arrow(L, O, p, c.fwd.hex, 3, al);
        ctx.globalAlpha = al; ctx.fillStyle = c.fwd.hex; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(L.px(p.x), L.py(p.y), 6, 0, 6.2832); ctx.fill(); ctx.stroke(); ctx.globalAlpha = 1;
      }
      ctx.restore();
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(L.px(L.xl), L.py(L.yh), L.size, L.size);
      if (feat) {
        ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.globalAlpha = f.plane;
        ctx.fillText('size (m²)', L.px(L.xh) - 4, L.py(L.yl) - 4); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('rooms', L.px(L.xl) + 4, L.py(L.yh) + 4); ctx.globalAlpha = 1;
      } else {
        ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
        ctx.fillText('x', L.px(L.xh) - 4, L.py(0) - 4); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('y', L.px(0) + 5, L.py(L.yh) + 3);
      }
      for (const k of DRAG) handle(L, k === 'a' ? a : b, k === 'a' ? c.fwd.hex : c.accent.hex);

      if (feat) {
        featureRows(L, f);
        // coordinates fly from the axes into the row
        if (f.move > 0 && f.move < 1) {
          const p = HOUSE_POINT;
          [{ x: L.px(p.x), y: L.py(L.yl) + 12, txt: String(FEATURES[0].value) }, { x: L.px(L.xl) - 12, y: L.py(p.y), txt: String(FEATURES[1].value) }].forEach((s, i) => {
            const dst = cellPos(L, i), u = ease(f.move);
            ctx.font = stage.mono(13); ctx.fillStyle = c.fwd.hex; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(s.txt, s.x + (dst.x + 8 - s.x) * u, s.y + (dst.y + dst.rh / 2 - s.y) * u);
          });
        }
      } else vecRows(L, f);
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return playing || drag !== null;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;

    const world = (e: PointerEvent) => { const r = cv.getBoundingClientRect(), L = lay(); return { L, x: L.xl + (e.clientX - r.left - L.ox) / L.s, y: L.yh - (e.clientY - r.top - L.oy) / L.s }; };
    const pick = (e: PointerEvent) => {
      const { L, x, y } = world(e);
      let best: 'a' | 'b' | null = null, bd = 22;
      for (const k of DRAG) { const v = k === 'a' ? a : b, d = Math.hypot((v.x - x) * L.s, (v.y - y) * L.s); if (d < bd) { bd = d; best = k as 'a' | 'b'; } }
      return best;
    };
    const set = (k: 'a' | 'b', x: number, y: number) => { const v = { x: clamp(Math.round(x * 20) / 20, -LIM, LIM), y: clamp(Math.round(y * 20) / 20, -LIM, LIM) }; if (k === 'a') a = v; else b = v; wake(); };
    const down = (e: PointerEvent) => { const k = pick(e); if (k) { drag = k; cv.setPointerCapture(e.pointerId); const w = world(e); set(k, w.x, w.y); e.preventDefault(); } };
    const move = (e: PointerEvent) => { if (drag) { const w = world(e); set(drag, w.x, w.y); } else cv.style.cursor = pick(e) ? 'grab' : ''; };
    const up = () => { drag = null; wake(); };
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
    return () => { loop.destroy(); stage.destroy(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); };
  });

  const setA = (x: number, y: number) => { a = { x, y }; wake(); };
  const setB = (x: number, y: number) => { b = { x, y }; wake(); };
</script>

<FigureCard title={init.title} role="group" aspect="720 / 420" narrowAspect="360 / 640"
  label={S === 'features'
    ? 'Left: a plane with a point for a house at 120 square metres and 4 rooms. Right: the same house as a list of six features shown as bars.'
    : 'A 2D plane with arrows a and b from the origin, and a list of their coordinates on the right. Drag the arrow tips.'}
  legend={S === 'features' ? [{ label: 'the house', color: '--viz-forward' }] : S === 'scale' ? [{ label: 'a', color: '--viz-forward', shape: 'line' }, { label: 'c·a, c ≥ 0', color: '--viz-pos', shape: 'line' }, { label: 'c·a, c < 0', color: '--viz-neg', shape: 'line' }] : [{ label: 'a', color: '--viz-forward', shape: 'line' }, { label: 'b', color: '--accent', shape: 'line' }]}>
  <canvas bind:this={cv} style="touch-action: none" aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MK} label="Timeline: vectors"
      onSeek={(to) => { t = to; wake(); }} onReset={() => { t = 0; playing = true; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">{nums}</p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if DRAG.length}
      <div class="ctl">
        <F1Slider label="a₁" value={a.x} min={-LIM} max={LIM} step={0.05} oninput={(v) => setA(v, a.y)} />
        <F1Slider label="a₂" value={a.y} min={-LIM} max={LIM} step={0.05} oninput={(v) => setA(a.x, v)} />
        {#if DRAG.length > 1}
          <F1Slider label="b₁" value={b.x} min={-LIM} max={LIM} step={0.05} oninput={(v) => setB(v, b.y)} />
          <F1Slider label="b₂" value={b.y} min={-LIM} max={LIM} step={0.05} oninput={(v) => setB(b.x, v)} />
        {/if}
        <span class="hint">You can also drag the arrow tips on the plane.</span>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.9rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--viz-forward); overflow-wrap: anywhere; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: grid; grid-template-columns: repeat(auto-fill, minmax(22rem, 1fr)); align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .hint { grid-column: 1 / -1; color: var(--text-muted); font-size: 0.8rem; }
</style>
