<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, reducedMotion, advance, markerAt, readColors } from '../_shared';
  import { setupStage, f2 } from '../F1Shared/canvas';
  import { scale } from '../F1Shared/plot';
  import {
    STATE, POS_SEQ, POSITION, CONTRIB, CONTRIB_SUM_OF_NORMS, CONTRIB_NORM_OF_SUM, FACTOR_S, FACTOR, TIMING,
    stateSteps, positionShown, contribShown, factorS, factorAt, type Kind,
  } from './sim';

  interface Props {
    /** 'state': hidden units over "hello" and "help". 'position': how much each step can still move the final state.
     *  'contrib': each step's piece of ∂L/∂W_hh and their sum. 'factor': gradient factor per step against s, RNN vs LSTM. */
    kind: Kind;
    title?: string;
  }
  let { kind, title = '' }: Props = $props();
  const init = untrack(() => ({ kind, title }));
  const K = init.kind, TM = TIMING[K];

  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let t = 0, wake = () => {};
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;

  const HELLO = 'hello', HELP = 'help', POS = Array.from(POS_SEQ).map((ch) => (ch === ' ' ? '␣' : ch));
  const COLS = ['--viz-cat-1', '--viz-cat-2', '--viz-cat-3', '--viz-cat-4'];
  const readCols = () => { const r = readColors({ a: COLS[0], b: COLS[1], c: COLS[2], d: COLS[3] }); return [r.a, r.b, r.c, r.d]; };
  const vec = (v: readonly number[]) => `(${v.map((x) => f2(x)).join(', ')})`;

  const phase = $derived(markerAt(uiT, TM.markers)?.label ?? '');
  const nums = $derived.by(() => {
    if (K === 'state') {
      const k = Math.round(stateSteps(uiT));
      return `after “${HELLO.slice(0, k)}”: h = ${vec(STATE.hello[k - 1])}` + (k <= 4 ? ` · after “${HELP.slice(0, k)}”: h = ${vec(STATE.help[k - 1])}` : '');
    }
    if (K === 'position') {
      const i = Math.max(0, 12 - Math.ceil(positionShown(uiT)));
      return `a nudge to the state after “${POS[i]}” (step ${i + 1}) moves the final state by ${f2(POSITION[i])}× its size`;
    }
    if (K === 'contrib') {
      const c = contribShown(uiT), i = Math.max(0, 5 - Math.ceil(c.bars));
      return c.totals > 0.5 ? `sum of the five sizes ${f2(CONTRIB_SUM_OF_NORMS, 3)} · size of the sum ${f2(CONTRIB_NORM_OF_SUM, 3)}` : `step ${i + 1} (“${HELLO[i]}”) adds a piece of size ${f2(CONTRIB[i], 3)}`;
    }
    const s = factorS(uiT);
    return `s = ${f2(s)} · plain RNN ×${f2(factorAt(FACTOR.rnn, s))} per step · LSTM cell state ×${f2(factorAt(FACTOR.lstm, s))} per step`;
  });
  const narration = $derived(
    K === 'state'
      ? phase === 'reading' ? 'The first three characters are the same, so the dashed lines (help) lie exactly on the solid ones (hello).'
        : phase === 'l or p' ? 'Step 4 reads “l” in hello but “p” in help, and the states split. Earlier states never change: the state only looks backward.'
        : 'Still four numbers after five characters. The two l’s of hello (steps 3 and 4) are the same input but give different states.'
      : K === 'position'
        ? phase === 'last steps' ? 'Going back from the end: the last few states move the final state almost one for one.'
          : 'Twelve steps back, a nudge to the first state reaches the end at under a quarter of its size. Each copy of the cell in between shrinks it a little.'
        : K === 'contrib'
          ? phase === 'backward' ? 'The backward pass reaches step 5 first, then 4, 3, 2, 1. Each step adds its own piece to the shared weight’s gradient. Step 1 adds nothing, because h₀ = 0.'
            : 'The pieces point in different directions and partly cancel, so the size of the sum (0.400) is less than the sum of the sizes (0.629).'
          : phase === 'sweep s' ? 'As the recurrent weights grow, the plain RNN’s factor climbs with them. The LSTM’s cell-state factor hardly moves.'
            : 'From s = 0.5 to 2 the RNN goes from ×0.42 to ×0.98 per step, with a dip near 1.5 where some units saturate; the LSTM stays between ×0.81 and ×0.82, set by its forget gate.',
  );
  $effect(() => { playing; speed; wake(); });

  const LABEL: Record<Kind, string> = {
    state: 'Line chart of the four hidden-state numbers after each character, solid for “hello” and dashed for “help”. They coincide for the first three characters and split at the fourth.',
    position: 'Bar chart, one bar per character of “twelve steps”: how much a nudge to the state after that character moves the final state. Bars fall from 1 at the last step to 0.23 at the first.',
    contrib: 'Bar chart of each step’s contribution to the gradient of the recurrent weight for “hello”: 0, 0.085, 0.148, 0.197, 0.198, then two totals: the sum of their sizes, 0.629, and the size of their sum, 0.400.',
    factor: 'Line chart of the gradient factor per step against the recurrent weight scale s from 0.5 to 2: the plain RNN rises from 0.42 to 0.98, the LSTM cell state stays near 0.81.',
  };
  const LEGEND: Record<Kind, { label: string; color: string; shape?: 'line' | 'dot' }[]> = {
    state: [...COLS.map((c, i) => ({ label: `unit ${i + 1}`, color: c, shape: 'line' as const })), { label: 'dashed: “help”', color: '--text-muted', shape: 'line' }],
    position: [{ label: 'effect on the final state', color: '--viz-forward' }],
    contrib: [{ label: 'one step’s piece', color: '--viz-gradient' }, { label: 'totals', color: '--text-muted' }],
    factor: [{ label: 'plain RNN', color: '--viz-cat-1', shape: 'line' }, { label: 'LSTM cell state', color: '--viz-cat-2', shape: 'line' }],
  };

  onMount(() => {
    let col = readCols();
    const stage = setupStage(cv, () => { col = readCols(); render(); });
    let lastUi = 0;

    const box = () => {
      const W = stage.w, H = stage.h, narrow = W < 500;
      return { W, H, narrow, l: narrow ? 42 : 56, r: W - (narrow ? 14 : 24), t: 16, b: H - 40 };
    };
    const yAxis = (L: ReturnType<typeof box>, sy: (v: number) => number, ticks: number[], fmt = (v: number) => f2(v, 1)) => {
      const { ctx, c } = stage;
      ctx.font = stage.mono(11); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.lineWidth = 1;
      for (const v of ticks) {
        ctx.strokeStyle = v === 0 ? c.muted.hex : c.rule.hex;
        ctx.beginPath(); ctx.moveTo(L.l, sy(v)); ctx.lineTo(L.r, sy(v)); ctx.stroke();
        ctx.fillStyle = c.muted.hex; ctx.fillText(fmt(v).replace('-', '−'), L.l - 6, sy(v));
      }
    };
    const xLabel = (L: ReturnType<typeof box>, text: string) => {
      const { ctx, c } = stage;
      ctx.font = stage.body(12, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(text, L.r, L.H - 1);
    };
    const dot = (x: number, y: number, color: string) => {
      const { ctx, c } = stage;
      ctx.fillStyle = color; ctx.strokeStyle = c.raised.hex; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 6.2832); ctx.fill(); ctx.stroke();
    };

    function drawState() {
      const { ctx, c } = stage, L = box(), n = stateSteps(t);
      const sx = scale(1, 5, L.l + 20, L.r - 20), sy = scale(-1, 1, L.b, L.t);
      yAxis(L, sy, [-1, -0.5, 0, 0.5, 1]);
      ctx.font = stage.mono(13, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let k = 1; k <= 5; k++) {
        ctx.fillStyle = k <= n + 0.01 ? c.text.hex : c.muted.hex;
        ctx.fillText(k === 4 ? 'l / p' : HELLO[k - 1], sx(k), L.b + 6);
      }
      xLabel(L, 'character read (hello / help)');
      const line = (rows: number[][], u: number, upTo: number, dashed: boolean) => {
        ctx.strokeStyle = col[u].hex; ctx.lineWidth = dashed ? 2 : 2.6; ctx.setLineDash(dashed ? [6, 5] : []); ctx.lineJoin = 'round';
        ctx.beginPath(); ctx.moveTo(sx(1), sy(rows[0][u]));
        const m = Math.min(upTo, rows.length);
        for (let k = 2; k <= Math.floor(m); k++) ctx.lineTo(sx(k), sy(rows[k - 1][u]));
        if (m < rows.length && m % 1 > 0) { const k = Math.floor(m), f = m - k; ctx.lineTo(sx(k + f), sy(rows[k - 1][u] + (rows[k][u] - rows[k - 1][u]) * f)); }
        ctx.stroke(); ctx.setLineDash([]);
        const e = Math.min(Math.floor(m), rows.length);
        dot(sx(e), sy(rows[e - 1][u]), col[u].hex);
      };
      for (let u = 0; u < 4; u++) { line(STATE.hello, u, n, false); line(STATE.help, u, n, true); }
    }

    function drawPosition() {
      const { ctx, c } = stage, L = box(), shown = positionShown(t);
      const sy = scale(0, 1.1, L.b, L.t), bw = (L.r - L.l) / 12;
      yAxis(L, sy, [0, 0.25, 0.5, 0.75, 1], (v) => f2(v, 2));
      for (let i = 0; i < 12; i++) {
        const a = Math.min(1, Math.max(0, shown - (11 - i))), x = L.l + bw * (i + 0.5), w = Math.min(36, bw * 0.66);
        ctx.font = stage.mono(13, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = a > 0 ? c.text.hex : c.muted.hex;
        ctx.fillText(POS[i], x, L.b + 6);
        if (a <= 0) continue;
        const y = sy(POSITION[i] * a);
        ctx.fillStyle = c.fwd.hex; ctx.globalAlpha = 0.85; ctx.fillRect(x - w / 2, y, w, L.b - y); ctx.globalAlpha = 1;
        if (a > 0.9 && !L.narrow) { ctx.font = stage.mono(10.5, 600); ctx.fillStyle = c.text.hex; ctx.textBaseline = 'bottom'; ctx.fillText(f2(POSITION[i]), x, y - 3); }
      }
      xLabel(L, 'step whose state is nudged (“twelve steps”)');
    }

    function drawContrib() {
      const { ctx, c } = stage, L = box(), sh = contribShown(t);
      const sy = scale(0, 0.7, L.b, L.t), slots = 8, bw = (L.r - L.l) / slots, w = Math.min(44, bw * 0.62);
      yAxis(L, sy, [0, 0.2, 0.4, 0.6], (v) => f2(v, 1));
      const bar = (slot: number, v: number, a: number, color: string, label: string, sub: string) => {
        const x = L.l + bw * (slot + 0.5);
        ctx.font = stage.mono(12.5, 600); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = a > 0 ? c.text.hex : c.muted.hex;
        ctx.fillText(label, x, L.b + 6);
        if (sub) { ctx.font = stage.body(11, true); ctx.fillStyle = c.muted.hex; ctx.fillText(sub, x, L.b + 21); }
        if (a <= 0) return;
        const y = sy(v * a);
        ctx.fillStyle = color; ctx.globalAlpha = 0.85; ctx.fillRect(x - w / 2, y, w, Math.max(1, L.b - y)); ctx.globalAlpha = 1;
        if (a > 0.9) { ctx.font = stage.mono(L.narrow ? 9.5 : 11, 600); ctx.fillStyle = c.text.hex; ctx.textBaseline = 'bottom'; ctx.fillText(f2(v, L.narrow ? 2 : 3), x, y - 3); }
      };
      for (let i = 0; i < 5; i++) bar(i, CONTRIB[i], Math.min(1, Math.max(0, sh.bars - (4 - i))), c.grad.hex, HELLO[i], '');
      bar(6, CONTRIB_SUM_OF_NORMS, sh.totals, c.muted.hex, 'Σ', 'sizes');
      bar(7, CONTRIB_NORM_OF_SUM, sh.totals, c.text.hex, 'size', 'of Σ');
      ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(L.l + bw * 5.5, L.t); ctx.lineTo(L.l + bw * 5.5, L.b); ctx.stroke(); ctx.setLineDash([]);
    }

    function drawFactor() {
      const { ctx, c } = stage, L = box(), s = factorS(t);
      const sx = scale(0.5, 2, L.l + 8, L.r - 8), sy = scale(0.3, 1.05, L.b, L.t);
      yAxis(L, sy, [0.4, 0.6, 0.8, 1], (v) => `×${f2(v, 1)}`);
      ctx.font = stage.mono(11); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = c.muted.hex;
      for (const v of [0.5, 1, 1.5, 2]) ctx.fillText(String(v), sx(v), L.b + 6);
      xLabel(L, 'recurrent weight scale s');
      ctx.font = stage.body(11.5, true); ctx.fillStyle = c.muted.hex; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('×1: no shrinking', L.l + 6, sy(1) - 3);
      ([[FACTOR.rnn, col[0], 'plain RNN'], [FACTOR.lstm, col[1], 'LSTM']] as const).forEach(([row, cc, name], j) => {
        ctx.strokeStyle = cc.hex; ctx.lineWidth = 2.6; ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(sx(0.5), sy(row[0]));
        for (let i = 1; i < FACTOR_S.length && FACTOR_S[i] <= s + 1e-9; i++) ctx.lineTo(sx(FACTOR_S[i]), sy(row[i]));
        const y = sy(factorAt(row, s));
        ctx.lineTo(sx(s), y); ctx.stroke(); dot(sx(s), y, cc.hex);
        ctx.font = stage.mono(11.5, 650); ctx.fillStyle = cc.hex; ctx.textAlign = s > 1.6 ? 'right' : 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(name, sx(s) + (s > 1.6 ? -10 : 10), y + (j ? 12 : -12));
      });
    }

    function render() {
      stage.ctx.clearRect(0, 0, stage.w, stage.h);
      if (K === 'state') drawState(); else if (K === 'position') drawPosition(); else if (K === 'contrib') drawContrib(); else drawFactor();
    }

    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: TM.T }).t;
      render(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return playing;
    }, { onView: (_v, first) => { if (first && !reducedMotion()) playing = true; else if (first) { t = TM.grow; uiT = t; } } });
    wake = loop.wake;
    return () => { loop.destroy(); stage.destroy(); };
  });
</script>

<FigureCard title={init.title} role="group" aspect="720 / 360" narrowAspect="360 / 380" label={LABEL[K]} legend={LEGEND[K]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={TM.T} markers={TM.markers} label="Timeline"
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to >= TM.T ? 0 : to; wake(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="vals num">{nums}</p>
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
</FigureCard>

<style>
  p { margin: 0; }
  .vals { font: 0.88rem var(--font-code); font-variant-numeric: tabular-nums; color: var(--text-muted); overflow-wrap: anywhere; }
  .say { font: var(--step--1) var(--font-ui); color: var(--text-muted); margin-top: 0.3em; }
</style>
