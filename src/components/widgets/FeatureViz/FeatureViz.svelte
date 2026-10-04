<script module lang="ts">
  import type { FromWorker } from './worker';
  import type { Trained } from './sim';
  let promise: Promise<Trained> | null = null;
  let progress = 0;
  const watchers = new Set<(f: number) => void>();
  /** Train once per page in a worker; every figure on the page shares the result. */
  function load(): Promise<Trained> {
    return (promise ??= new Promise((res, rej) => {
      const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (e: MessageEvent<FromWorker>) => {
        if (e.data.kind === 'progress') { progress = e.data.frac; watchers.forEach((f) => f(progress)); }
        else { res(e.data.trained); w.terminate(); }
      };
      w.onerror = () => rej(new Error('worker failed'));
      w.postMessage(0);
    }));
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale } from '../F1Shared/plot';
  import {
    SIZE, N, CLASSES, C1, C2, STEPS, STAGES, ALL_TARGETS, targetKey, targetName, roughness,
    duration, filterFrameAt, filterMarkers, ascentFrameAt, ascentMarkers, galleryFrameAt, galleryMarkers,
    saliencyMarkers, classMarkers, segmentAt, clamp01, type Mode, type Target,
  } from './sim';

  interface Props {
    title?: string;
    /** 'filters': first-layer filters while training. 'maximize': gradient ascent on the input. 'hierarchy': images that excite a channel. 'saliency': gradient of a class score with respect to pixels. */
    mode?: Mode;
    /** maximize: also show the cleaned-up (regularised) ascent next to the plain one. */
    show?: 'plain' | 'both';
    /** maximize: starting target, e.g. 'class3' (ring score), 'l22' (layer 2, channel 3). */
    target?: string;
    /** hierarchy: starting channels (0-based); -1 picks the most telling one automatically. */
    channel1?: number;
    channel2?: number;
    /** saliency: 'class' sweeps which class score is explained; 'sanity' randomises the weights layer by layer. */
    view?: 'class' | 'sanity';
    /** saliency: which test image (0-based, one per class). */
    image?: number;
  }
  let { title = '', mode = 'filters', show = 'both', target = 'class3', channel1 = -1, channel2 = -1, view = 'class', image = 0 }: Props = $props();
  const init = untrack(() => ({ mode, show, view, title: title || 'Inside a small CNN' }));
  const T = duration(init.mode);

  let tr = $state.raw<Trained | null>(null);
  let failed = $state(false);
  let sel = $state(untrack(() => ({ target, c1: channel1, c2: channel2, image })));
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let ui = $state.raw({ a: '', b: '', say: '' });
  let t = 0, wake = () => {}, prog = progress;
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const markers = $derived(
    !tr ? [] : init.mode === 'filters' ? filterMarkers(tr) : init.mode === 'maximize' ? ascentMarkers : init.mode === 'hierarchy' ? galleryMarkers
      : init.view === 'class' ? classMarkers : saliencyMarkers,
  );
  /** Targets whose activation actually rises from a blank start (a channel that never switches on has no gradient to follow). */
  const targets = $derived(ALL_TARGETS.filter((x) => !tr || tr.ascents[targetKey(x)].plain.act[STEPS] > 0.1));
  const tgt = $derived<Target>(targets.find((x) => targetKey(x) === sel.target) ?? targets[0]);
  const aliveL1 = $derived(tr ? tr.gallery.l1.map((e) => e[0].act > 1e-3) : []);
  const aliveL2 = $derived(tr ? tr.gallery.l2.map((e) => e[0].act > 1e-3) : []);
  /** auto pick: the layer 1 channel whose best examples share a class least; the layer 2 channel they share most. */
  const auto1 = $derived(tr ? tr.gallery.stat1.reduce((b, v, i) => (aliveL1[i] && (b < 0 || v.n < tr!.gallery.stat1[b].n) ? i : b), -1) : 0);
  const auto2 = $derived(tr ? tr.gallery.stat2.reduce((b, v, i) => (aliveL2[i] && (b < 0 || v.n > tr!.gallery.stat2[b].n) ? i : b), -1) : 0);
  const ch1 = $derived(sel.c1 >= 0 ? sel.c1 : auto1);
  const ch2 = $derived(sel.c2 >= 0 ? sel.c2 : auto2);
  const restart = () => { t = 0; playing = !reducedMotion(); wake(); };
  const choose = (k: 'target' | 'c1' | 'c2' | 'image', v: string | number) => { (sel as Record<string, string | number>)[k] = v; restart(); };

  const pct = (v: number) => Math.round(100 * v) + '%';
  /** All text for time t, from the same frame values the canvas draws. */
  function snapUi() {
    if (!tr) { ui = { a: failed ? 'Training failed in this browser.' : (prog >= 0.99 ? 'Preparing figures…' : `Training a small network in a background thread… ${Math.round(prog * 100)}%`), b: '', say: '' }; return; }
    const m = markerAt(t, markers)?.label ?? '';
    if (init.mode === 'filters') {
      const f = filterFrameAt(t, tr);
      ui = {
        a: `training step ${f.snap.iter} of ${tr.total} · accuracy on unseen images ${pct(f.snap.acc)} · loss ${f2(f.snap.loss)}`, b: '',
        say: m === 'guessing' ? 'Random starting weights. With four classes, guessing is right about one time in four.'
          : m === 'learning' ? 'The weights shift toward patterns that separate the classes, and accuracy climbs.'
          : 'Accuracy is above 90% and the filters keep refining.',
      };
    } else if (init.mode === 'maximize') {
      const A = tr.ascents[targetKey(tgt)], { step } = ascentFrameAt(t), fr = (k: 'plain' | 'reg') => A[k].frames.subarray(step * N, (step + 1) * N);
      ui = {
        a: `step ${step} of ${STEPS} · plain ${f2(A.plain.act[step], 1)} (jaggedness ${f2(roughness(fr('plain')), 2)})` + (init.show === 'both' ? ` · cleaned up ${f2(A.reg.act[step], 1)} (${f2(roughness(fr('reg')), 2)})` : ''),
        b: '',
        say: m === 'start' ? 'The image starts nearly blank, with faint noise.' : m === 'rising' ? 'Each step nudges every pixel in the direction that raises the number.' : 'The number is still creeping up, but more slowly.',
      };
    } else if (init.mode === 'hierarchy') {
      const s1 = tr.gallery.stat1[ch1], s2 = tr.gallery.stat2[ch2], g = galleryFrameAt(t);
      ui = {
        a: g.n1 >= tr.gallery.top ? `layer 1, channel ${ch1 + 1} sees 3×3 pixels · of its ${s1.of} best images, ${s1.n} are ${CLASSES[s1.cls]}s` : '',
        b: g.n2 >= tr.gallery.top ? `layer 2, channel ${ch2 + 1} sees 8×8 pixels · of its ${s2.of} best images, ${s2.n} are ${CLASSES[s2.cls]}s` : '',
        say: g.n2 > 0 ? 'Layer 2 sees a bigger patch, and its favourite images come mostly from one class.' : 'Layer 1 reacts to a tiny patch: a short stroke or edge, found in images of several classes.',
      };
    } else {
      const c = tr.saliency.cases[sel.image];
      if (init.view === 'class') {
        const k = segmentAt(t, T, 4);
        ui = { a: `score for "${CLASSES[k]}": ${f2(c.logits[k], 1)} · this image is a ${CLASSES[c.cls]}, the network says ${CLASSES[c.pred]}`, b: '',
          say: `Bright pixels are where a small change would move the "${CLASSES[k]}" score most.` };
      } else {
        const stage = segmentAt(t, T, 4);
        ui = {
          a: `${STAGES[stage]} · map vs trained map: ${f2(tr.saliency.meanCorrTrained[stage])} · map vs image edges: ${f2(tr.saliency.meanCorrEdge[stage])} (average of ${tr.saliency.nAvg} test images)`,
          b: stage ? `this image: the network now says "${CLASSES[c.predStage[stage]]}"` : '',
          say: stage === 0 ? 'The map of the trained network.' : 'Weights replaced by random numbers, from the last layer backward. The map is recomputed each time.',
        };
      }
    }
  }

  onMount(() => {
    const stage = setupStage(cv, () => render());
    let lastUi = 0;
    const off = document.createElement('canvas');
    off.width = off.height = SIZE;
    const octx = off.getContext('2d')!, px = octx.createImageData(SIZE, SIZE);
    const mix = (a: number[], b: number[], u: number) => a.map((v, i) => Math.round(v + (b[i] - v) * u));
    const rgba = (c: { rgb: number[] }, a: number) => `rgba(${c.rgb[0]},${c.rgb[1]},${c.rgb[2]},${a})`;

    /** Draw a 16x16 array as a crisp image, value 0 -> lo, value 1 -> hi. */
    function pic(data: ArrayLike<number>, x: number, y: number, s: number, lo: number[], hi: number[], vmax = 1) {
      for (let i = 0; i < N; i++) { const u = clamp01(data[i] / vmax), col = mix(lo, hi, u); px.data.set([col[0], col[1], col[2], 255], i * 4); }
      octx.putImageData(px, 0, 0);
      stage.ctx.imageSmoothingEnabled = false;
      stage.ctx.drawImage(off, x, y, s, s);
      stage.ctx.strokeStyle = stage.c.rule.hex; stage.ctx.lineWidth = 1; stage.ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
    }
    const gray = () => [stage.c.raised.rgb, stage.c.text.rgb] as const;
    const heat = () => [stage.c.raised.rgb, stage.c.s5.rgb] as const;
    const text = (s: string, x: number, y: number, o: { size?: number; color?: string; align?: CanvasTextAlign; base?: CanvasTextBaseline; mono?: boolean } = {}) => {
      const { ctx, c } = stage;
      ctx.font = o.mono ? stage.mono(o.size ?? 11) : stage.body(o.size ?? 12);
      ctx.fillStyle = o.color ?? c.muted.hex; ctx.textAlign = o.align ?? 'left'; ctx.textBaseline = o.base ?? 'alphabetic';
      ctx.fillText(s, x, y);
    };
    const axes = (L: number, R: number, Tp: number, B: number, yTicks: number[], Y: (v: number) => number, fmt: (v: number) => string) => {
      const { ctx, c } = stage;
      ctx.lineWidth = 1;
      for (const v of yTicks) { ctx.strokeStyle = c.rule.hex; ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(R, Y(v)); ctx.stroke(); text(fmt(v), L - 6, Y(v), { align: 'right', base: 'middle', mono: true, size: 10.5 }); }
      ctx.strokeStyle = c.muted.hex; ctx.beginPath(); ctx.moveTo(L, Tp); ctx.lineTo(L, B); ctx.lineTo(R, B); ctx.stroke();
    };
    const dot = (x: number, y: number, col: string, r = 5) => {
      const { ctx, c } = stage;
      ctx.fillStyle = col; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
    };

    function placeholder() {
      const { ctx, w, h, c } = stage;
      text(failed ? 'Training failed in this browser.' : prog >= 0.99 ? 'Preparing figures…' : 'Training a small network in your browser…', w / 2, h / 2 - 8, { align: 'center', base: 'middle', size: 13 });
      if (!failed) {
        const bw = Math.min(220, w - 60), x = (w - bw) / 2, y = h / 2 + 12;
        ctx.fillStyle = c.rule.hex; ctx.fillRect(x, y, bw, 5);
        ctx.fillStyle = c.fwd.hex; ctx.fillRect(x, y, bw * Math.min(prog, 0.99), 5);
      }
    }

    // ---------------- filters ----------------
    function filters() {
      const { ctx, w: W, h: H, c } = stage, Tr = tr!, narrow = W < 540, f = filterFrameAt(t, Tr);
      let scaleMax = 0;
      Tr.snaps.forEach((s) => s.filters.forEach((v) => { scaleMax = Math.max(scaleMax, Math.abs(v)); }));
      const gap = narrow ? 10 : 14, s = narrow ? Math.min(100, Math.floor((W - 16 - 2 * gap) / 3)) : Math.min(Math.floor((W * 0.46 - 16 - 2 * gap) / 3), Math.floor((H - 100) / 2));
      text(narrow ? 'First-layer filters (3×3 weights each)' : 'First-layer filters: the 3×3 weights each channel slides over the image', 8, 18, { size: 12.5, color: c.text.hex });
      for (let k = 0; k < C1; k++) {
        const col = k % 3, row = Math.floor(k / 3), x = 8 + col * (s + gap), y = 34 + row * (s + 30);
        ctx.fillStyle = c.raised.hex; ctx.fillRect(x, y, s, s);
        for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) {
          const v = f.snap.filters[k * 9 + r * 3 + q], a = Math.min(1, Math.abs(v) / scaleMax);
          ctx.fillStyle = rgba(v >= 0 ? c.pos : c.neg, 0.06 + 0.94 * a);
          ctx.fillRect(x + (q * s) / 3, y + (r * s) / 3, s / 3, s / 3);
        }
        ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1;
        for (let q = 0; q <= 3; q++) { ctx.beginPath(); ctx.moveTo(x + (q * s) / 3, y); ctx.lineTo(x + (q * s) / 3, y + s); ctx.moveTo(x, y + (q * s) / 3); ctx.lineTo(x + s, y + (q * s) / 3); ctx.stroke(); }
        ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
        text(`filter ${k + 1}`, x + s / 2, y + s + 14, { align: 'center', size: narrow ? 10 : 11.5 });
      }
      // accuracy chart; x is square-root of the training step so the playhead moves at an even pace
      const L = narrow ? 40 : Math.round(W * 0.5) + 30, R = W - 14, Tp = narrow ? 34 + 2 * (s + 30) + 30 : 40, B = H - 38;
      text('accuracy on unseen images', L - 32, Tp - 14, { size: 12, color: c.text.hex });
      const X = scale(0, 1, L, R), Y = scale(0, 1, B, Tp), xs = (iter: number) => X(Math.sqrt(iter / Tr.total));
      axes(L, R, Tp, B, [0, 0.25, 0.5, 0.75, 1], Y, (v) => String(Math.round(v * 100)));
      [0, 10, 50, Math.round(Tr.total / 2), Tr.total].forEach((it, i, a) => {
        if (i && it === a[i - 1]) return;
        text(String(it), xs(it), B + 14, { align: 'center', mono: true, size: 10.5 });
      });
      text('training step (square-root scale)', R, B + 30, { align: 'right', size: 11.5 });
      ctx.setLineDash([4, 4]); ctx.strokeStyle = c.muted.hex; ctx.beginPath(); ctx.moveTo(L, Y(0.25)); ctx.lineTo(R, Y(0.25)); ctx.stroke(); ctx.setLineDash([]);
      text('guessing', R, Y(0.25) - 4, { align: 'right', size: 11 });
      ctx.strokeStyle = c.fwd.hex; ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.beginPath();
      for (let i = 0; i <= f.i; i++) { const p = Tr.snaps[i]; if (i) ctx.lineTo(xs(p.iter), Y(p.acc)); else ctx.moveTo(xs(p.iter), Y(p.acc)); }
      ctx.stroke();
      dot(xs(f.snap.iter), Y(f.snap.acc), c.fwd.hex);
    }

    // ---------------- maximize ----------------
    function maximize() {
      const { ctx, w: W, h: H, c } = stage, Tr = tr!, narrow = W < 540, A = Tr.ascents[targetKey(tgt)], { step } = ascentFrameAt(t);
      const kinds = init.show === 'both' ? (['plain', 'reg'] as const) : (['plain'] as const), n = kinds.length;
      const colors = { plain: c.grad, reg: c.fwd }, names = { plain: 'plain ascent', reg: 'cleaned up' };
      const leftW = narrow ? W : Math.round(W * 0.5);
      const s = narrow ? Math.min(Math.floor((W - 8 * (n + 1)) / n), 176) : Math.min(Math.floor((leftW - 8 * (n + 1)) / n), H - 110);
      text(`Image that most excites: ${targetName(tgt)}`, 8, 18, { size: 12.5, color: c.text.hex });
      const [lo, hi] = gray();
      const gx = (leftW - n * s) / (n + 1);
      kinds.forEach((k, i) => {
        const x = gx + i * (s + gx), y = 34, fr = A[k].frames.subarray(step * N, (step + 1) * N);
        pic(fr, x, y, s, lo, hi);
        text(names[k], x, y + s + 16, { size: 12, color: colors[k].hex });
        text(`activation ${f2(A[k].act[step], 1)}`, x, y + s + 31, { mono: true, size: 11 });
      });
      // activation vs step
      const L = narrow ? 44 : leftW + 40, R = W - 14, Tp = narrow ? 34 + s + 84 : 44, B = H - 40;
      let lo2 = Infinity, hi2 = -Infinity;
      kinds.forEach((k) => { for (let i = 0; i <= STEPS; i++) { lo2 = Math.min(lo2, A[k].act[i]); hi2 = Math.max(hi2, A[k].act[i]); } });
      const span = hi2 - lo2 || 1, d0 = lo2 - 0.05 * span, d1 = hi2 + 0.08 * span;
      const X = scale(0, STEPS, L, R), Y = scale(d0, d1, B, Tp);
      const stepTick = Math.max(1, Math.round(span / 4 / (10 ** Math.floor(Math.log10(span / 4)))) ) * 10 ** Math.floor(Math.log10(span / 4));
      const ticks: number[] = []; for (let v = Math.ceil(d0 / stepTick) * stepTick; v <= d1; v += stepTick) ticks.push(Math.round(v * 1e6) / 1e6);
      text('activation being raised', L - 38, Tp - 14, { size: 12, color: c.text.hex });
      axes(L, R, Tp, B, ticks, Y, (v) => String(+v.toFixed(1)));
      [0, 12, 24, 36, 48].forEach((v) => text(String(v), X(v), B + 14, { align: 'center', mono: true, size: 10.5 }));
      text('ascent step', R, B + 30, { align: 'right', size: 11.5 });
      kinds.forEach((k) => {
        ctx.strokeStyle = colors[k].hex; ctx.lineWidth = 2.5; ctx.lineJoin = 'round'; ctx.beginPath();
        for (let i = 0; i <= step; i++) (i ? ctx.lineTo(X(i), Y(A[k].act[i])) : ctx.moveTo(X(i), Y(A[k].act[i])));
        ctx.stroke();
      });
      kinds.forEach((k) => dot(X(step), Y(A[k].act[step]), colors[k].hex));
    }

    // ---------------- hierarchy ----------------
    function hierarchy() {
      const { ctx, w: W, h: H, c } = stage, Tr = tr!, narrow = W < 540, f = galleryFrameAt(t), top = Tr.gallery.top;
      const rows = [
        { ex: Tr.gallery.l1[ch1], n: f.n1, name: `Layer 1, channel ${ch1 + 1}`, sees: 'sees 3×3 pixels', box: 3, off: 1 },
        { ex: Tr.gallery.l2[ch2], n: f.n2, name: `Layer 2, channel ${ch2 + 1}`, sees: 'sees 8×8 pixels', box: 8, off: 3 },
      ];
      const labelW = narrow ? 0 : 128, gap = narrow ? 6 : 10;
      const s = Math.floor((W - labelW - 12 - (top - 1) * gap) / top);
      const rowH = narrow ? s + 48 : s + 56, [lo, hi] = gray();
      rows.forEach((r, ri) => {
        const y0 = 8 + ri * rowH, x0 = labelW + 4, y = narrow ? y0 + 22 : y0 + 4;
        if (narrow) text(`${r.name}: ${r.sees}`, 6, y0 + 14, { size: 12.5, color: c.text.hex });
        else { text(r.name, 6, y0 + 22, { size: 13, color: c.text.hex }); text(r.sees, 6, y0 + 40, { size: 12 }); }
        r.ex.forEach((e, i) => {
          const x = x0 + i * (s + gap), a = clamp01(r.n - i);
          if (a <= 0) { ctx.setLineDash([3, 3]); ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1); ctx.setLineDash([]); return; }
          ctx.globalAlpha = a;
          pic(e.img, x, y, s, lo, hi);
          // the receptive field: the patch of the image this unit actually sees
          const bx = Math.max(0, e.x - r.off), by = Math.max(0, e.y - r.off), bx2 = Math.min(SIZE, e.x - r.off + r.box), by2 = Math.min(SIZE, e.y - r.off + r.box), k = s / SIZE;
          ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 2; ctx.strokeRect(x + bx * k, y + by * k, (bx2 - bx) * k, (by2 - by) * k);
          text(['vertical', 'horizontal', 'diagonal', 'ring'][e.cls], x + s / 2, y + s + 13, { align: 'center', size: narrow ? 10 : 11.5 });
          if (!narrow) text(f2(e.act, 1), x + s / 2, y + s + 27, { align: 'center', mono: true, size: 10.5 });
          ctx.globalAlpha = 1;
        });
      });
    }

    // ---------------- saliency ----------------
    function saliency() {
      const { ctx, w: W, h: H, c } = stage, Tr = tr!, narrow = W < 540, cs = Tr.saliency.cases[sel.image];
      const [glo, ghi] = gray(), [hlo, hhi] = heat();
      const map = (g: Float32Array) => { const a = Float32Array.from(g, Math.abs); let m = 1e-12; a.forEach((v) => { m = Math.max(m, v); }); return { a, m }; };
      if (init.view === 'class') {
        const k = segmentAt(t, T, 4);
        const s = narrow ? Math.floor((W - 30) / 2) : Math.min(Math.floor(W * 0.27), H - 80);
        const m = map(cs.byClass[k]);
        pic(cs.img, 8, 34, s, glo, ghi); text('the test image', 8, 34 + s + 15, { size: 12 });
        const x2 = 8 + s + 14;
        pic(m.a, x2, 34, s, hlo, hhi, m.m); text(`where "${CLASSES[k]}" is sensitive`, x2, 34 + s + 15, { size: 12, color: c.text.hex });
        text('Saliency: how much each pixel moves a class score', 8, 18, { size: 12.5, color: c.text.hex });
        // class scores
        const L = narrow ? 8 : x2 + s + 24, R = W - 44, Tp = narrow ? 34 + s + 50 : 58, B = narrow ? H - 10 : H - 30;
        text('the network’s four class scores for this image', L, Tp - 16, { size: 12, color: c.text.hex });
        const z = cs.logits, lo = Math.min(0, ...z), hi = Math.max(0, ...z), V = scale(lo, hi, L + 78, R), bh = (B - Tp) / 4;
        ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(V(0), Tp); ctx.lineTo(V(0), B); ctx.stroke();
        z.forEach((v, i) => {
          const yy = Tp + i * bh + bh * 0.2, hh = bh * 0.6, x0 = V(0), x1 = V(v);
          text(CLASSES[i].replace(' bar', ''), L + 70, yy + hh / 2, { align: 'right', base: 'middle', size: 11.5, color: i === k ? c.text.hex : c.muted.hex });
          ctx.fillStyle = i === k ? c.grad.hex : rgba(c.grad, 0.35); ctx.fillRect(Math.min(x0, x1), yy, Math.abs(x1 - x0), hh);
          text(f2(v, 1), v >= 0 ? x1 + 4 : x0 + 4, yy + hh / 2, { align: 'left', base: 'middle', mono: true, size: 10.5 });
        });
        return;
      }
      // sanity view
      const st = segmentAt(t, T, 4);
      const g = narrow ? 8 : 10, s = narrow ? Math.floor((W - 3 * g) / 2) : Math.min(Math.floor((W - 5 * g) / 4), Math.floor(H * 0.4));
      const pos = (i: number) => narrow ? { x: g + (i % 2) * (s + g), y: 30 + Math.floor(i / 2) * (s + 34) } : { x: g + i * (s + g), y: 34 };
      text(narrow ? 'Do scrambled weights change the map?' : 'Does the map change when the weights are scrambled?', 8, 18, { size: 12.5, color: c.text.hex });
      const panels: { name: string; data: ArrayLike<number>; img?: boolean }[] = [
        { name: 'test image', data: cs.img, img: true },
        { name: 'trained network', data: map(cs.maps[0]).a },
        { name: STAGES[st], data: map(cs.maps[st]).a },
        { name: 'edges of the image', data: cs.edges },
      ];
      panels.forEach((p, i) => {
        const { x, y } = pos(i);
        if (p.img) pic(p.data, x, y, s, glo, ghi);
        else { let m = 1e-12; for (let q = 0; q < N; q++) m = Math.max(m, p.data[q]); pic(p.data, x, y, s, hlo, hhi, m); }
        text(p.name, x, y + s + 15, { size: narrow ? 11.5 : 12, color: i === 2 ? c.text.hex : c.muted.hex });
      });
      // agreement across stages (mean over test images)
      const top = narrow ? pos(3).y + s + 40 : 34 + s + 40, L = 36, R = W - 14, B = H - 34, Tp = top + 24;
      text(narrow ? `correlation, average of ${Tr.saliency.nAvg} images` : `correlation of each map with the trained map and with the image edges (average of ${Tr.saliency.nAvg} test images)`, 8, top + 4, { size: 12, color: c.text.hex });
      const Y = scale(0, 1, B, Tp), gw = (R - L) / 4;
      axes(L, R, Tp, B, [0, 0.5, 1], Y, (v) => v.toFixed(1));
      const series = [{ v: Tr.saliency.meanCorrTrained, col: c.fwd }, { v: Tr.saliency.meanCorrEdge, col: c.s4 }];
      series.forEach((sr, si) => {
        sr.v.forEach((v, i) => {
          const bw = Math.min(40, gw * 0.3), x = L + i * gw + gw / 2 + (si ? 2 : -bw - 2);
          ctx.fillStyle = i === st ? sr.col.hex : rgba(sr.col, 0.35); ctx.fillRect(x, Y(Math.max(0, v)), bw, Y(0) - Y(Math.max(0, v)));
          text(f2(v), x + bw / 2, Y(Math.max(0, v)) - 3, { align: 'center', mono: true, size: 10.5, color: i === st ? c.text.hex : c.muted.hex });
        });
      });
      STAGES.forEach((nm, i) => text(narrow ? ['trained', 'last 1', 'last 2', 'all 3'][i] : nm, L + i * gw + gw / 2, B + 15, { align: 'center', size: narrow ? 10.5 : 11.5, color: i === st ? c.text.hex : c.muted.hex }));
    }

    function render() {
      const { ctx, w, h } = stage;
      ctx.clearRect(0, 0, w, h);
      if (!tr) return placeholder();
      if (init.mode === 'filters') filters();
      else if (init.mode === 'maximize') maximize();
      else if (init.mode === 'hierarchy') hierarchy();
      else saliency();
    }

    const loop = createLoop(cv, (dt, now) => {
      if (tr) t = advance({ t, playing, speed }, dt, { duration: T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; snapUi(); uiT = t; }
      return playing || !tr;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const watch = (f: number) => { prog = f; wake(); };
    watchers.add(watch);
    load().then((r) => { tr = r; t = 0; wake(); }, () => { failed = true; wake(); });
    return () => { loop.destroy(); stage.destroy(); watchers.delete(watch); };
  });
  $effect(() => { playing; speed; sel.target; sel.c1; sel.c2; sel.image; tr; wake(); });

  const LABELS: Record<Mode, string> = {
    filters: 'The six 3 by 3 first-layer filters of a small CNN as it trains, beside a curve of its accuracy on unseen images.',
    maximize: 'Images produced by gradient ascent on the input to excite one unit, with a curve of the unit activation against ascent step.',
    hierarchy: 'Test images that most excite one layer 1 channel and one layer 2 channel, with the patch each unit sees outlined.',
    saliency: 'A test image and heat maps of how sensitive a class score is to each pixel.',
  };
  const LEGENDS: Record<Mode, { label: string; color: string; shape?: 'dot' | 'line' }[]> = {
    filters: [{ label: 'positive weight', color: '--viz-pos' }, { label: 'negative weight', color: '--viz-neg' }, { label: 'accuracy', color: '--viz-forward', shape: 'line' }],
    maximize: init.show === 'both' ? [{ label: 'plain ascent', color: '--viz-gradient', shape: 'line' }, { label: 'cleaned up', color: '--viz-forward', shape: 'line' }] : [{ label: 'plain ascent', color: '--viz-gradient', shape: 'line' }],
    hierarchy: [{ label: 'patch the unit sees', color: '--accent', shape: 'line' }],
    saliency: init.view === 'class' ? [{ label: 'bright = more sensitive', color: '--viz-seq-5' }] : [{ label: 'correlation with trained map', color: '--viz-forward' }, { label: 'correlation with image edges', color: '--viz-seq-4' }],
  };
  const ASPECT: Record<string, [string, string]> = {
    filters: ['720 / 380', '360 / 560'], maximize: ['720 / 400', '360 / 560'], hierarchy: ['720 / 300', '360 / 215'],
    class: ['720 / 330', '360 / 420'], sanity: ['720 / 440', '360 / 640'],
  };
  const asp = ASPECT[init.mode === 'saliency' ? init.view : init.mode];
</script>

<FigureCard title={init.title} role="group" aspect={asp[0]} narrowAspect={asp[1]} label={LABELS[init.mode]} legend={LEGENDS[init.mode]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers} label="Timeline"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= T ? 0 : to; wake(); }} onReset={() => { t = 0; playing = !reducedMotion(); wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">{ui.a}</p>
    {#if ui.b}<p class="base num">{ui.b}</p>{/if}
    <p class="say" role="status" aria-live="polite">{ui.say}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if init.mode === 'maximize'}
        <label class="pick">Excite
          <select value={targetKey(tgt)} onchange={(e) => choose('target', e.currentTarget.value)}>
            <optgroup label="Class scores">
              {#each targets.filter((x) => x.kind === 'class') as x (targetKey(x))}<option value={targetKey(x)}>{CLASSES[x.index]}</option>{/each}
            </optgroup>
            <optgroup label="Layer 2 channels">
              {#each targets.filter((x) => x.kind === 'l2') as x (targetKey(x))}<option value={targetKey(x)}>channel {x.index + 1}</option>{/each}
            </optgroup>
          </select>
        </label>
      {:else if init.mode === 'hierarchy'}
        <fieldset class="seg">
          <legend class="sr-only">Layer 1 channel</legend>
          <span class="cap">Layer 1</span>
          {#each Array.from({ length: C1 }, (_, i) => i) as i (i)}
            <label><input type="radio" name="l1-{init.title}" checked={ch1 === i} disabled={!!tr && !aliveL1[i]} onchange={() => choose('c1', i)} /><span>{i + 1}</span></label>
          {/each}
        </fieldset>
        <fieldset class="seg">
          <legend class="sr-only">Layer 2 channel</legend>
          <span class="cap">Layer 2</span>
          {#each Array.from({ length: C2 }, (_, i) => i) as i (i)}
            <label><input type="radio" name="l2-{init.title}" checked={ch2 === i} disabled={!!tr && !aliveL2[i]} onchange={() => choose('c2', i)} /><span>{i + 1}</span></label>
          {/each}
        </fieldset>
      {:else if init.mode === 'saliency'}
        <fieldset class="seg">
          <legend class="sr-only">Test image</legend>
          <span class="cap">Image</span>
          {#each CLASSES as nm, i (nm)}
            <label><input type="radio" name="img-{init.title}" checked={sel.image === i} onchange={() => choose('image', i)} /><span>{nm.replace(' bar', '')}</span></label>
          {/each}
        </fieldset>
      {:else}
        <span class="hint">Drag the timeline to move through training.</span>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.88rem var(--font-code); font-variant-numeric: tabular-nums; }
  .base { font: 0.8rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--text-muted); }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .hint { color: var(--text-muted); font-size: 0.8rem; }
  .pick { display: inline-flex; align-items: center; gap: var(--s-2); font-size: 0.85rem; }
  .pick select { font: inherit; padding: 0.25em 0.5em; border: 1px solid var(--rule); border-radius: var(--r-2); background: var(--bg-raised); color: var(--text); }
  .seg { display: inline-flex; align-items: center; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; font-size: 0.85rem; }
  .seg .cap { padding: 0.3em 0.7em; color: var(--text-muted); background: var(--bg-sunken); }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span:not(.cap) { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:disabled + span { opacity: 0.35; cursor: not-allowed; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
