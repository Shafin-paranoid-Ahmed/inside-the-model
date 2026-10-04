<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance } from '../_shared';
  import { buildTape, expr, costs, frameAt, durationOf, markersOf, PARAM_NAMES, REC, B0, type Frame } from './sim';

  interface Props {
    /** 'reverse': record, then walk the tape backward. 'forward': tangents ride along with the values. */
    mode?: 'reverse' | 'forward';
    title?: string;
    /** Show the cost comparison (forward vs reverse mode) with its slider. */
    showCost?: boolean;
    /** Initial number of inputs n in the cost comparison. */
    inputs?: number;
    /** Where the timeline starts: 'record' (t = 0) or 'backward' (after the tape is recorded). */
    phase?: 'record' | 'backward';
  }
  let { mode = 'reverse', title = '', showCost = true, inputs = 3, phase: startPhase = 'record' }: Props = $props();

  const uid = $props.id();
  const init = untrack(() => ({ mode, n: inputs, start: startPhase === 'backward' ? (mode === 'reverse' ? B0 : REC + 0.5) : 0, T: durationOf(mode), markers: markersOf(mode), title: title || (mode === 'reverse' ? 'A tape recorded forward, replayed backward' : 'Forward mode: one sweep per input') }));
  const tape = buildTape();
  const N = tape.N;
  let seed = $state(0); // index into PARAM_NAMES (forward mode)
  let logN = $state(Math.log10(Math.max(1, init.n)));
  const nIn = $derived(Math.round(10 ** logN));
  let playing = $state(false), speed = $state(1), uiT = $state(init.start);
  let vertical = $state(false);
  let narration = $state('');

  const tan = $derived(tape.tangents(tape.paramRows[seed]));
  const fin = tape.adj[N];
  const g4 = (v: number) => (Math.abs(v) < 5e-5 ? '0.0000' : v.toFixed(4)).replace('-', '−');
  const big = (v: number) => (v >= 1e6 ? `${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M` : v >= 1e4 ? `${Math.round(v / 1e3)}k` : String(Math.round(v)));
  const LEGEND = [
    { label: 'value', color: '--viz-forward' },
    { label: init.mode === 'reverse' ? 'adjoint ∂L/∂row' : 'tangent ∂row/∂input', color: '--viz-gradient' },
  ];
  const SAY = {
    record: 'Recording: each operation is appended to the tape, with its value, as the forward pass runs.',
    backward: 'Backward: walk the tape from the last row. Each row hands its adjoint to its inputs, times a local derivative, and shared inputs add up.',
    doneR: `Done: one backward sweep gave every gradient at once. ∂L/∂w1 = ${g4(fin[tape.paramRows[0]])}, ∂L/∂w2 = ${g4(fin[tape.paramRows[1]])}, ∂L/∂b = ${g4(fin[tape.paramRows[2]])}.`,
  };

  let t = init.start, phase = '';
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, poke = () => {};
  $effect(() => { seed; logN; vertical; playing; speed; poke(); });
  $effect(() => { seed; phase = ''; }); // forward-mode narration depends on the seed

  onMount(() => {
    const reduced = prefersReducedMotion();
    const ctx = cv.getContext('2d')!;
    const NAMES = { fwd: '--viz-forward', grad: '--viz-gradient', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent', soft: '--accent-soft' };
    let c = readColors(NAMES), code = readVar('--font-code'), body = readVar('--font-body');
    let lastUi = 0;
    if (reduced.matches) t = init.T;

    const lay = () => vertical
      ? { w: 360, h: showCost ? 590 : 350, rowY: 50, rh: 22, idx: 6, lab: 26, val: 218, col2: 304, arc: 316, costX: 20, costY: 372, costW: 320 }
      : showCost
        ? { w: 720, h: 380, rowY: 52, rh: 24, idx: 12, lab: 36, val: 330, col2: 424, arc: 440, costX: 480, costY: 40, costW: 220 }
        : { w: 720, h: 380, rowY: 52, rh: 24, idx: 12, lab: 36, val: 420, col2: 540, arc: 560, costX: 0, costY: 0, costW: 0 };

    const text = (s: string, x: number, y: number, col: string, size = 13, align: CanvasTextAlign = 'left', a = 1, mono = true, wt = 600) => {
      if (a <= 0.01) return;
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align;
      ctx.font = mono ? `${wt} ${size}px ${code}` : `italic ${size}px ${body}`;
      ctx.fillText(s, x, y);
    };

    function arcs(L: ReturnType<typeof lay>, fr: Frame, t0: number) {
      if (fr.cursor < 0) return;
      const row = tape.rows[fr.cursor];
      if (!row.args.length) return;
      const y = (i: number) => L.rowY + i * L.rh + 3;
      const rec = t0 < REC && (mode === 'forward' || t0 < REC);
      const col = rec ? c.fwd.hex : c.grad.hex;
      const p = rec ? Math.min(1, (Math.min(1, t0 / REC) * N) % 1) : fr.flow;
      for (const a of row.args) {
        const y0 = rec ? y(a) : y(fr.cursor), y1 = rec ? y(fr.cursor) : y(a);
        const bulge = L.arc + 6 + Math.min(24, Math.abs(y1 - y0) * 0.18);
        ctx.globalAlpha = 0.7; ctx.strokeStyle = col; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(L.arc - 8, y0); ctx.bezierCurveTo(bulge, y0, bulge, y1, L.arc - 8, y1); ctx.stroke();
        // dot flowing from y0 to y1
        const s = p, u = 1 - s;
        const px = u * u * u * (L.arc - 8) + 3 * u * u * s * bulge + 3 * u * s * s * bulge + s * s * s * (L.arc - 8);
        const py = u * u * u * y0 + 3 * u * u * s * y0 + 3 * u * s * s * y1 + s * s * s * y1;
        ctx.globalAlpha = 1; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px, py, 4, 0, 6.2832); ctx.fill();
        ctx.globalAlpha = 0.9; ctx.strokeStyle = c.accent.hex; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
        const ya = rec ? y(a) : y(a);
        ctx.strokeRect(L.idx - 4, ya - 11, L.col2 - L.idx + 16, L.rh - 2); ctx.setLineDash([]);
      }
    }

    function drawCost(L: ReturnType<typeof lay>) {
      const x = L.costX, y = L.costY, w = L.costW, cs = costs(N, nIn);
      text('Cost of all n derivatives', x, y - 8, c.text.hex, 13.5, 'left', 1, false);
      const axis = (v: number) => Math.max(0, Math.min(1, (Math.log10(v) - 1) / (7.4 - 1))) * w;
      const bars: [string, number, string, string][] = [
        ['forward mode', cs.forward, c.fwd.hex, `${N} × (${big(nIn)} + 1)`],
        ['reverse mode', cs.reverse, c.grad.hex, `2 × ${N}`],
      ];
      bars.forEach(([name, v, col, eq], i) => {
        const yy = y + 12 + i * 58;
        text(name, x, yy, c.muted.hex, 12.5, 'left', 1, false);
        text(eq, x + w, yy, c.muted.hex, 12, 'right');
        ctx.globalAlpha = 1; ctx.fillStyle = c.rule.hex; ctx.fillRect(x, yy + 8, w, 1);
        ctx.globalAlpha = 0.9; ctx.fillStyle = col; ctx.fillRect(x, yy + 10, Math.max(2, axis(v)), 14);
        text(`${big(v)} row visits`, x, yy + 40, c.text.hex, 13);
      });
      const r = cs.forward / cs.reverse;
      text(`n = ${big(nIn)} inputs`, x, y + 144, c.text.hex, 13.5);
      text(r > 1.05 ? `reverse mode is ${r >= 100 ? big(r) : r.toFixed(1)}× cheaper` : 'about equal for one input', x, y + 166, c.grad.hex, 12.5);
      text('log scale, up to a small constant', x, y + 188, c.muted.hex, 12, 'left', 1, false, 400);
    }

    function render(now: number) {
      const L = lay(), k = cv.width / L.w;
      ctx.setTransform(k, 0, 0, k, 0, 0);
      ctx.clearRect(0, 0, L.w, L.h);
      const fr = frameAt(t, tape, mode, tan);
      const col2 = mode === 'reverse' ? 'adjoint' : `tangent ∂/∂${PARAM_NAMES[seed]}`;
      text('tape (Wengert list)', L.idx, L.rowY - 30, c.text.hex, 13.5, 'left', 1, false);
      text('value', L.val, L.rowY - 12, c.fwd.hex, 12, 'right', 1, false, 400);
      text(col2, L.col2 + 10, L.rowY - 12, c.grad.hex, 12, 'right', 1, false, 400);
      ctx.globalAlpha = 1; ctx.fillStyle = c.rule.hex; ctx.fillRect(L.idx, L.rowY - 8, L.col2 + 10 - L.idx, 1);
      ctx.lineCap = 'round';

      for (const r of tape.rows) {
        const y = L.rowY + r.i * L.rh + 8, s = fr.show[r.i];
        if (s <= 0.01) continue;
        if (fr.cursor === r.i) { ctx.globalAlpha = 0.9; ctx.fillStyle = c.soft.hex; ctx.fillRect(L.idx - 4, y - 16, L.col2 - L.idx + 16, L.rh - 2); }
        text(String(r.i), L.idx + 14, y, c.muted.hex, 12, 'right', s, true, 400);
        const isIn = r.kind !== 'op';
        text(r.label, L.lab, y, isIn ? c.text.hex : c.text.hex, 13, 'left', s);
        const lw = ctx.measureText(r.label).width;
        text(isIn ? `  ${expr(tape, r)}` : ` = ${expr(tape, r)}`, L.lab + lw, y, c.muted.hex, 12, 'left', s, true, 400);
        text(g4(r.value), L.val, y, c.fwd.hex, 13, 'right', s);
        if (mode === 'forward') {
          text(g4(fr.tan[r.i]), L.col2 + 10, y, c.grad.hex, 13, 'right', s);
        } else if (fr.phase !== 'record' || t >= REC) {
          const a = fr.adj[r.i];
          if (t >= REC) text(g4(a), L.col2 + 10, y, c.grad.hex, 13, 'right', a === 0 ? 0.35 : 1);
        }
      }
      arcs(L, fr, t);
      if (showCost) drawCost(L);
      ctx.globalAlpha = 1;

      tl.paint(t);
      if (now - lastUi > 100 || !playing) {
        lastUi = now; uiT = t;
        const key = mode === 'forward' ? (fr.phase === 'record' ? 'record' : 'doneF') : fr.phase === 'record' ? 'record' : fr.phase === 'backward' ? 'backward' : t >= init.T - 0.01 || fr.phase === 'done' ? 'doneR' : 'backward';
        if (key !== phase) {
          phase = key;
          narration = key === 'doneF'
            ? `Done: with the tangent seeded on ${PARAM_NAMES[seed]}, this sweep gives ∂L/∂${PARAM_NAMES[seed]} = ${g4(tan[tape.root])}. Another input needs another sweep.`
            : SAY[key as keyof typeof SAY];
        }
      }
    }

    const loop = createLoop(cv.closest('.viz') ?? cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: init.T }).t;
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

<FigureCard title={init.title} legend={LEGEND} role="img" aspect="720 / 380" narrowAspect={showCost ? '360 / 590' : '360 / 350'}
  label="The tape of L equals (tanh(w1 x1 + w2 x2 + b) minus t) squared: thirteen rows, each an operation with its value. {mode === 'reverse' ? 'Adjoints accumulate as the tape is walked backward.' : 'Tangents are computed alongside the values.'}">
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={init.T} markers={init.markers} label="Timeline of the tape"
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { t = to >= init.T ? 0 : to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    <div class="ctl">
      {#if mode === 'forward'}
        <fieldset class="seg">
          <legend class="sr-only">Input to differentiate with respect to</legend>
          {#each PARAM_NAMES as p, i (p)}
            <label><input type="radio" name="seed-{uid}" value={i} bind:group={seed} /><span>∂/∂{p}</span></label>
          {/each}
        </fieldset>
      {/if}
      {#if showCost}
        <label>number of inputs n
          <input type="range" min="0" max="6" step="0.05" bind:value={logN} aria-label="Number of inputs, log scale" />
          <output class="num">{big(nIn)}</output>
        </label>
      {/if}
    </div>
  {/snippet}
</FigureCard>

<style>
  .say { margin: 0; font: 0.92rem / 1.45 var(--font-ui); min-height: 2.6em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; flex-wrap: wrap; max-width: 100%; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='range'] { width: 11rem; accent-color: var(--accent); }
  .ctl output { min-width: 3rem; color: var(--text); }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: var(--r-2); padding: 0; margin: 0; overflow: hidden; }
  .seg label { margin: 0; position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { padding: 0.3em 0.8em; cursor: pointer; display: block; }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: -2px; }
</style>
