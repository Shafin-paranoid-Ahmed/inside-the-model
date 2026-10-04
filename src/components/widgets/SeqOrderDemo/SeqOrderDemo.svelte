<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance as clock } from '../_shared';
  import { makeDemo, frameAt, timing, markers as mk, H, type Demo } from './sim';

  interface Props {
    /** First sentence. */
    a?: string;
    /** Second sentence: same words, other order. */
    b?: string;
    /** 'both' | 'bow' (bag of words only) | 'rnn' (recurrent only). */
    show?: 'both' | 'bow' | 'rnn';
    title?: string;
    editable?: boolean;
  }
  let { a = 'the dog chased the cat', b = 'the cat chased the dog', show = 'both', title = '', editable = true }: Props = $props();

  const init = untrack(() => ({ show, title: title || 'Same words, different order' }));
  const groups = init.show === 'both' ? (['bow', 'rnn'] as const) : ([init.show] as const);
  const Hh = 20 + groups.length * 140;
  let sa = $state(untrack(() => a)), sb = $state(untrack(() => b));
  let narrow = $state(false), playing = $state(false), speed = $state(1);
  let uiT = $state(0), narration = $state('');
  const demo: Demo = $derived(makeDemo(sa, sb));
  const tm = $derived(timing(demo.n));
  const mks = $derived(mk(demo.n));
  const Wd = $derived(narrow ? 400 : 720);

  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let t = 0, poke = () => {};
  $effect(() => { playing; speed; poke(); });
  $effect(() => { demo; poke(); });
  const same = $derived(demo.dBow < 1e-9);
  const summary = $derived(
    `Bag of words: ‖A − B‖ = ${demo.dBow.toFixed(3)}${demo.dBow < 1e-9 ? ' (identical)' : ''}. Recurrent: ‖A − B‖ = ${demo.dRnn.toFixed(3)}.`,
  );

  onMount(() => {
    const reduced = prefersReducedMotion();
    let reduce = reduced.matches;
    const ctx = cv.getContext('2d')!;
    const C = { pos: '--viz-pos', neg: '--viz-neg', fwd: '--viz-forward', grad: '--viz-gradient', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', accent: '--accent' };
    let c = readColors(C), code = readVar('--font-code'), ui = readVar('--font-body');
    if (reduce) t = tm.T - 0.3;
    let lastRead = 0, lastPhase = '';
    const txt = (s: string, x: number, y: number, col: string, px = 12, align: CanvasTextAlign = 'center', w = 500, a2 = 1, ital = false) => {
      if (a2 <= 0.02) return;
      ctx.globalAlpha = a2; ctx.fillStyle = col; ctx.textAlign = align;
      ctx.font = ital ? `italic ${px}px ${ui}` : `${w} ${px}px ${code}`; ctx.fillText(s, x, y);
    };
    const chips = (x: number, y: number, v: number[], s: number, a2: number, hi: boolean) => {
      const w = v.length * s;
      v.forEach((q, j) => {
        ctx.globalAlpha = a2; ctx.fillStyle = c.raised.hex; ctx.fillRect(x - w / 2 + j * s, y - s / 2, s, s);
        ctx.fillStyle = q >= 0 ? c.pos.hex : c.neg.hex; ctx.globalAlpha = a2 * Math.min(1, Math.abs(q));
        ctx.fillRect(x - w / 2 + j * s, y - s / 2, s, s);
      });
      ctx.globalAlpha = a2; ctx.strokeStyle = hi ? c.accent.hex : c.rule.hex; ctx.lineWidth = hi ? 2 : 1;
      ctx.strokeRect(x - w / 2, y - s / 2, w, s);
    };

    function render() {
      const D = demo, n = D.n, W = Wd, fr = frameAt(Math.min(t, tm.T), n);
      const k0 = cv.width / W;
      ctx.setTransform(k0, 0, 0, k0, 0, 0);
      ctx.clearRect(0, 0, W, Hh);
      const left = 34, avail = W - left - 14, cw = Math.min(104, avail / n), x0 = left + (avail - cw * n) / 2;
      const cx = (k: number) => x0 + cw * (k + 0.5), chip = Math.max(7, Math.min(13, cw / 5));
      groups.forEach((g, gi) => {
        const y0 = 8 + gi * 140;
        const head = g === 'bow' ? (narrow ? 'Bag of words: tanh(W Σ x + b)' : 'Bag of words: feature = tanh(W Σ xᵢ + b), a sum ignores order') : (narrow ? 'Recurrent: hₜ = tanh(W_xh xₜ + W_hh hₜ₋₁ + b)' : 'Recurrent: hₜ = tanh(W_xh xₜ + W_hh hₜ₋₁ + b), state carries order');
        txt(head, left - 22, y0 + 12, c.text.hex, narrow ? 10.5 : 12, 'left', 600);
        const dv = g === 'bow' ? D.dBow : D.dRnn;
        [D.a, D.b].forEach((s, ri) => {
          const y = y0 + 34 + ri * 58;
          txt(ri ? 'B' : 'A', 10, y + 22, c.muted.hex, 13, 'center', 700);
          const feats = g === 'bow' ? s.bow : s.rnn;
          for (let k = 0; k < s.toks.length; k++) {
            const f = fr.fill[k], last = k === s.toks.length - 1;
            txt(s.toks[k], cx(k), y + 8, c.text.hex, narrow ? 10.5 : 13, 'center', 600, 0.4 + 0.6 * Math.min(1, f * 3));
            if (k > 0) {
              ctx.globalAlpha = 0.5 * f; ctx.strokeStyle = c.muted.hex; ctx.lineWidth = 1.4; ctx.beginPath();
              ctx.moveTo(cx(k - 1) + (H * chip) / 2 + 2, y + 24); ctx.lineTo(cx(k) - (H * chip) / 2 - 2, y + 24); ctx.stroke();
              if (g === 'bow') txt('+', (cx(k - 1) + cx(k)) / 2, y + 28, c.muted.hex, 12, 'center', 500, 0.8 * f);
              else if (!reduce && f > 0 && f < 1) { ctx.globalAlpha = 1; ctx.fillStyle = c.fwd.hex; ctx.beginPath(); ctx.arc(cx(k - 1) + (H * chip) / 2 + 2 + (cx(k) - cx(k - 1) - H * chip - 4) * f, y + 24, 3.5, 0, 6.2832); ctx.fill(); }
            }
            chips(cx(k), y + 24, feats[k], chip, 0.2 + 0.8 * f, last && fr.cmp > 0.5);
          }
        });
        // equal / not equal between the two final vectors
        const xe = cx(Math.max(D.a.toks.length, D.b.toks.length) - 1) + (H * chip) / 2 + 14;
        const same2 = dv < 1e-9;
        txt(same2 ? '=' : '≠', Math.min(xe, W - 10), y0 + 34 + 52, same2 ? c.fwd.hex : c.grad.hex, 20, 'center', 700, fr.cmp);
        txt(`‖A − B‖ = ${dv.toFixed(3)}`, W - 10, y0 + 12, same2 ? c.fwd.hex : c.grad.hex, narrow ? 10.5 : 12.5, 'right', 700, fr.cmp);
      });
      ctx.globalAlpha = 1;
    }

    function frame(dt: number, now: number) {
      t = clock({ t, playing, speed }, dt, { duration: tm.T }).t;
      if (t > tm.T) t = 0;
      render(); tl.paint(t);
      if (now - lastRead > 100 || !playing) {
        lastRead = now; uiT = t;
        const ph = t < tm.cmpAt ? 'read' : 'compare';
        if (ph !== lastPhase) { lastPhase = ph; narration = ph === 'read' ? 'Both sentences are read one word at a time. Each strip is the feature vector after that word.' : summary; }
      }
      return playing;
    }
    const loop = createLoop(cv, frame, { onView: (v, first) => { if (v && first && !reduce) playing = true; } });
    poke = loop.wake;
    const fit = () => {
      const d = Math.min(2, Math.max(1, window.devicePixelRatio || 1)), r = cv.getBoundingClientRect();
      const w = Math.round(r.width * d), h = Math.round(r.height * d);
      if (w && h && (cv.width !== w || cv.height !== h)) { cv.width = w; cv.height = h; }
    };
    const ro = new ResizeObserver(([e]) => { narrow = e.contentRect.width < 540; fit(); poke(); });
    ro.observe(cv.parentElement!);
    const off = onThemeChange(() => { c = readColors(C); code = readVar('--font-code'); ui = readVar('--font-body'); poke(); });
    const onReduce = () => { reduce = reduced.matches; if (reduce) playing = false; poke(); };
    reduced.addEventListener('change', onReduce);
    document.fonts.ready.then(() => poke());
    narration = 'Both sentences are read one word at a time. Each strip is the feature vector after that word.';
    fit();
    return () => { loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); };
  });
</script>

<FigureCard
  title={init.title} role="img" aspect="720 / {Hh}" narrowAspect="400 / {Hh}"
  legend={[{ label: 'positive', color: '--viz-pos' }, { label: 'negative', color: '--viz-neg' }, { label: 'state flows', color: '--viz-forward' }]}
  label="Two sentences with the same words in different order, encoded by a bag-of-words feature and by a recurrent network. {summary}"
>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={tm.T} markers={mks} label="Timeline"
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { playing = false; t = to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
    <p class="next">{same ? 'The two bags are the same set of words, so a bag-of-words model sees one input.' : 'The two sentences are different word multisets, so the bag of words differs too.'}</p>
  {/snippet}
  {#snippet footer()}
    {#if editable}
      <div class="ctl">
        <label>A <input type="text" bind:value={sa} maxlength="40" spellcheck="false" autocomplete="off" aria-label="Sentence A, up to 6 words" /></label>
        <label>B <input type="text" bind:value={sb} maxlength="40" spellcheck="false" autocomplete="off" aria-label="Sentence B, up to 6 words" /></label>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .say, .next { margin: 0; }
  .say { font: 0.92rem / 1.45 var(--font-code); font-variant-numeric: tabular-nums; min-height: 2.6em; }
  .next { color: var(--text-muted); font-size: var(--step--1); margin-top: 0.2em; }
  .ctl { display: flex; flex-wrap: wrap; gap: var(--s-2) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input {
    width: 15rem; max-width: 100%; font: 0.9rem var(--font-code); padding: 0.3em 0.5em; color: var(--text);
    background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1);
  }
</style>
