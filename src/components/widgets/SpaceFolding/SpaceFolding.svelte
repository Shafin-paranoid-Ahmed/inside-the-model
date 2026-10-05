<script module lang="ts">
  import type { Run, RunId } from './sim';
  import type { Reply } from './worker';
  const cache = new Map<RunId, Promise<Run>>();
  let worker: Worker | undefined;
  function load(id: RunId): Promise<Run> {
    const existing = cache.get(id);
    if (existing) return existing;
    const w = worker ??= new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    const pending = new Promise<Run>((resolve, reject) => {
      const cleanup = () => {
        w.removeEventListener('message', receive);
        w.removeEventListener('error', failed);
        w.removeEventListener('messageerror', failed);
      };
      const failed = (event: Event) => {
        cleanup();
        if (worker === w) { worker = undefined; w.terminate(); }
        reject(new Error(`Background training could not start or return its result. ${event instanceof ErrorEvent ? event.message : 'Worker communication failed.'}`));
      };
      const receive = (e: MessageEvent<Reply>) => {
        if (e.data.id !== id) return;
        cleanup();
        if ('error' in e.data) reject(new Error(e.data.error)); else resolve(e.data.run);
      };
      w.addEventListener('message', receive);
      w.addEventListener('error', failed);
      w.addEventListener('messageerror', failed);
      try { w.postMessage({ id }); }
      catch (error) { cleanup(); reject(error); }
    });
    cache.set(id, pending);
    void pending.catch(() => { if (cache.get(id) === pending) cache.delete(id); });
    return pending;
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt } from '../_shared';
  import { setupStage, type Stage } from '../F1Shared/canvas';
  import { RUNS_FOR, CFG, dataOf, fitFrame, field, makeFold, foldFrame, FIT_T, FOLD_T, DEPTH_T,
    FOLD_TIMING, DEPTH_TIMING, FOLD_MARKERS, DEPTH_MARKERS, LIM, paramCount, type Mode, type Fold, type View } from './sim';

  /** Fold mode only: 'affine' plays Input → Affine and holds; 'squash' starts at Affine and plays Tanh → Cut. */
  type Part = 'all' | 'affine' | 'squash';
  let { title = 'Learned coordinates', mode = 'fold', part = 'all' }: { title?: string; mode?: Mode; part?: Part } = $props();
  const init = untrack(() => mode), PART = untrack(() => (mode === 'fold' ? part : 'all'));
  /** Fold timeline time for widget time t (the parts are windows onto the full fold timeline). */
  const foldT = (t: number) => (PART === 'affine' ? Math.min(t, 4.5) : PART === 'squash' ? t + 4.5 : t);
  const duration = init === 'depth' ? DEPTH_T : PART === 'affine' ? 6 : PART === 'squash' ? FOLD_T - 4.5 : init === 'fold' ? FOLD_T : FIT_T;
  const markers = init === 'depth' ? DEPTH_MARKERS
    : PART === 'affine' ? FOLD_MARKERS.slice(0, 2)
    : PART === 'squash' ? FOLD_MARKERS.slice(2).map((m) => ({ ...m, at: m.at - 4.5 }))
    : init === 'fold' ? FOLD_MARKERS
    : [{ at: 0, label: 'Fit' }, { at: 6.8, label: 'Hold' }];
  let playing = $state(false), speed = $state(0.5), uiT = $state(0), unit = $state(0);
  let ready = $state(false), error = $state('');
  const phase = $derived(markerAt(uiT, markers)?.label ?? 'Input');
  let canvas: HTMLCanvasElement, values: HTMLParagraphElement, tl: ReturnType<typeof Timeline>;
  let t = 0, wake = () => {};
  const runs: Partial<Record<RunId, Run>> = {}, folds: Partial<Record<RunId, Fold>> = {};
  const seek = (to: number) => { t = to; uiT = to; wake(); };
  $effect(() => { playing; speed; unit; wake(); });

  onMount(() => {
    let disposed = false, stage: Stage;
    let lastUi = -Infinity;
    const draw = () => {
      if (!stage) return;
      const { ctx: c, w, h, c: colors } = stage;
      c.clearRect(0, 0, w, h);
      if (!ready) {
        c.font = stage.body(16); c.fillStyle = colors.muted.hex;
        if (values) values.textContent = error || 'Training the tiny model in a worker…';
        c.textAlign = 'center'; c.fillText(error || 'Training the tiny model in a worker…', w / 2, h / 2); return;
      }
      const descriptions: string[] = [];
      const mobile = w < 500, gap = 18;
      const pw = mobile ? w : (w - gap) / 2, ph = mobile ? (h - gap) / 2 : h;
      const panel = (index: number, name: string, view: View) => {
        const x = mobile ? 0 : index * (pw + gap), y = mobile ? index * (ph + gap) : 0;
        const side = Math.min(pw - 34, ph - 66), left = x + (pw - side) / 2, top = y + 30;
        c.font = stage.body(14); c.fillStyle = colors.text.hex; c.textAlign = 'center'; c.fillText(name, x + pw / 2, y + 17);
        c.strokeStyle = colors.rule.hex; c.lineWidth = 1; c.strokeRect(left, top, side, side);
        const sx = (a: number) => left + side * (.5 + (a - view.cx) / (2 * view.h));
        const sy = (a: number) => top + side * (.5 - (a - view.cy) / (2 * view.h));
        return { x, y, side, left, top, sx, sy, caption: top + side + 22 };
      };
      const points = (xs: Float64Array, labels: Float64Array | Uint8Array, p: ReturnType<typeof panel>, wrong?: Uint8Array) => {
        for (let i = 0; i < labels.length; i++) {
          const x = p.sx(xs[2 * i]), y = p.sy(xs[2 * i + 1]);
          c.fillStyle = labels[i] ? colors.pos.hex : colors.neg.hex;
          c.beginPath();
          if (labels[i]) c.arc(x, y, 3.1, 0, Math.PI * 2); else c.rect(x - 2.8, y - 2.8, 5.6, 5.6);
          c.fill();
          c.strokeStyle = colors.raised.hex; c.lineWidth = .8; c.stroke();
          if (wrong?.[i]) { c.strokeStyle = colors.text.hex; c.lineWidth = 1; c.beginPath(); c.arc(x, y, 5.2, 0, Math.PI * 2); c.stroke(); }
        }
      };
      const caption = (p: ReturnType<typeof panel>, text: string) => { c.font = stage.mono(12); c.fillStyle = colors.text.hex; c.textAlign = 'center'; c.fillText(text, p.x + pw / 2, p.caption); };
      if (init === 'fold' || init === 'depth') {
        const ids: RunId[] = init === 'fold' ? ['xor', 'xor'] : ['one', 'deep'];
        ids.forEach((id, index) => {
          const f = folds[id]!;
          const tm = init === 'depth' ? DEPTH_TIMING : FOLD_TIMING;
          const frame = foldFrame(f, init === 'fold' && index === 0 ? 0 : foldT(t), tm);
          const finished = frame.m === f.stages.length - 1;
          const name = init === 'fold' && index === 0 ? 'Original coordinates' : `${init === 'depth' ? (id === 'one' ? '1 hidden layer · ' : '3 hidden layers · ') : ''}${frame.m === 0 && frame.s === 0 ? 'Input' : finished ? 'Hidden coordinates' : `${frame.m % 2 ? 'Tanh' : 'Affine'} · layer ${Math.floor(frame.m / 2) + 1}`}`;
          const p = panel(index, name, frame.view);
          c.save(); c.beginPath(); c.rect(p.left, p.top, p.side, p.side); c.clip();
          for (const line of f.scene.lines) {
            c.strokeStyle = colors.rule.hex; c.lineWidth = line.axis ? 1.2 : .6; c.beginPath();
            for (let j = 0; j < line.count; j++) { const i = 2 * (line.start + j); if (j) c.lineTo(p.sx(frame.pos[i]), p.sy(frame.pos[i + 1])); else c.moveTo(p.sx(frame.pos[i]), p.sy(frame.pos[i + 1])); }
            c.stroke();
          }
          const showCut = finished && !(init === 'fold' && index === 0);
          if (showCut) {
            const [a, b, bias] = f.cut, v = frame.view;
            c.strokeStyle = colors.fwd.hex; c.lineWidth = 2; c.beginPath();
            if (Math.abs(b) > Math.abs(a)) { const x0 = v.cx - v.h, x1 = v.cx + v.h; c.moveTo(p.sx(x0), p.sy(-(a * x0 + bias) / b)); c.lineTo(p.sx(x1), p.sy(-(a * x1 + bias) / b)); }
            else if (Math.abs(a) > 1e-12) { const y0 = v.cy - v.h, y1 = v.cy + v.h; c.moveTo(p.sx(-(b * y0 + bias) / a), p.sy(y0)); c.lineTo(p.sx(-(b * y1 + bias) / a), p.sy(y1)); }
            c.stroke();
          }
          points(frame.pos, f.labels, p, showCut ? frame.wrong : undefined);
          c.restore();
          const text = showCut ? `${frame.right}/${f.scene.n} training points correct` : `${paramCount(f.sizes)} parameters · view ±${frame.view.h.toFixed(1)}`;
          caption(p, text);
          descriptions.push(`${name}: ${text}.`);
        });
      } else {
        const id: RunId = init === 'features' ? 'features' : 'line', r = runs[id]!, d = dataOf(id), frame = fitFrame(r, d, t), grid = field(r.sizes, frame.w, 32);
        [0, 1].forEach((index) => {
          const feature = init === 'features' && index === 0;
          const p = panel(index, feature ? `Hidden feature h${unit + 1}` : index === 0 ? 'Interleaved spirals' : 'Output probability', { cx: 0, cy: 0, h: LIM * 1.03 });
          if (index === 1 || feature) {
            const values = feature ? grid.units[unit] : grid.prob;
            const cell = p.side / grid.res;
            for (let j = 0; j < grid.res; j++) for (let i = 0; i < grid.res; i++) {
              const v = values[j * grid.res + i], sign = feature ? v : 2 * v - 1;
              c.fillStyle = sign >= 0 ? colors.pos.hex : colors.neg.hex;
              c.globalAlpha = .08 + .65 * Math.abs(sign);
              c.fillRect(p.left + cell * i, p.top + cell * (grid.res - 1 - j), cell + .5, cell + .5);
            }
            c.globalAlpha = 1;
            // Marching-square edge crossings show the actual p=0.5 contour.
            if (!feature) {
              c.strokeStyle = colors.fwd.hex; c.lineWidth = 1.5;
              for (let j = 0; j < grid.res - 1; j++) for (let i = 0; i < grid.res - 1; i++) {
                const coords = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], cross: number[][] = [];
                for (let q = 0; q < 4; q++) {
                  const a = coords[q], b = coords[(q + 1) % 4], va = grid.prob[a[1] * grid.res + a[0]] - .5, vb = grid.prob[b[1] * grid.res + b[0]] - .5;
                  if ((va >= 0) !== (vb >= 0)) { const f = va / (va - vb); cross.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]); }
                }
                if (cross.length >= 2) { c.beginPath(); c.moveTo(p.left + p.side * cross[0][0] / 31, p.top + p.side * (1 - cross[0][1] / 31)); c.lineTo(p.left + p.side * cross[1][0] / 31, p.top + p.side * (1 - cross[1][1] / 31)); c.stroke(); }
              }
            }
          }
          points(d.xs, d.ys, p);
          caption(p, feature ? 'tanh value: −1 to +1' : index === 0 ? `${d.n} labelled training points` : `${frame.correct}/${d.n} correct · BCE ${frame.loss.toFixed(3)}`);
        });
        descriptions.push(`${init === 'features' ? `Selected hidden feature h${unit + 1}, tanh values from −1 to +1. ` : ''}${frame.correct}/${d.n} training points correct. Binary cross-entropy ${frame.loss.toFixed(3)}. Interpolated training snapshots, approximately epoch ${frame.epoch}.`);
        c.font = stage.mono(11); c.fillStyle = colors.muted.hex; c.textAlign = 'center'; c.fillText(`Interpolated training snapshots · approx. epoch ${frame.epoch}`, w / 2, h - 2);
      }
      // Text and canvas use this exact frame; numerical changes are not live announcements.
      if (values) values.textContent = descriptions.join(' ');
    };
    stage = setupStage(canvas, () => wake());
    const loop = createLoop(canvas, (dt, now) => {
      if (ready) t = advance({ t, playing, speed }, dt, { duration }).t;
      draw(); tl?.paint(t);
      if (now - lastUi >= 100 || !playing || markerAt(t, markers)?.label !== phase) { uiT = t; lastUi = now; }
      return ready && playing;
    }, { onView: (_visible, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    void Promise.all(RUNS_FOR[init].map(async (id) => {
      const run = await load(id);
      if (disposed) return;
      runs[id] = run;
      if (init === 'fold' || init === 'depth') folds[id] = makeFold(CFG[id].sizes, run.weights.at(-1)!, dataOf(id));
    })).then(() => { if (!disposed) { ready = true; wake(); } }).catch((e) => { if (!disposed) { error = String(e); wake(); } });
    return () => { disposed = true; loop.destroy(); stage.destroy(); };
  });
</script>

<FigureCard {title} label="Real data points transformed by trained hidden layers; squares are class zero, circles class one" aspect="2 / 1" narrowAspect="1 / 1.95"
  legend={[{ label: '0 · squares', color: '--viz-neg' }, { label: '1 · circles', color: '--viz-pos' }, { label: 'output cut', color: '--viz-forward', shape: 'line' }]}>
  <canvas bind:this={canvas} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed {duration} {markers} onSeek={seek} onStep={seek} onReset={() => seek(0)} />
  {/snippet}
  {#snippet readout()}
    <p class="sr-only" bind:this={values}>Training the tiny model in a worker…</p>
    <p aria-live="polite">{error || (!ready ? 'Training in a background thread…' : `Phase: ${phase}.`)}
      {init === 'fold' || init === 'depth' ? 'The grid and points undergo the same coordinate map. A ring marks a mistake once the final cut appears.' : 'The colors and accuracy use the same interpolated weights. All scores are on the training points.'}</p>
    {#if init === 'depth'}<p>These are fixed runs with different parameter counts, not evidence that depth alone improves a model.</p>{/if}
  {/snippet}
  {#snippet footer()}
    {#if init === 'features'}
      <label>Inspect hidden unit <select bind:value={unit}>{#each [0, 1, 2, 3, 4, 5] as i (i)}<option value={i}>h{i + 1}</option>{/each}</select></label>
      <span>The output combines all six learned features.</span>
    {/if}
  {/snippet}
</FigureCard>

<style>
  p { margin: .2rem 0; font-size: .85rem; }
  label { display: inline-flex; gap: .7rem; align-items: center; }
  select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: .35rem; }
  span { margin-left: 1rem; font-size: .8rem; color: var(--text-muted); }
  select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
</style>
