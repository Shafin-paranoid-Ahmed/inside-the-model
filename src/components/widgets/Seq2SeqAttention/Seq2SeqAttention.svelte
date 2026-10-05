<script module lang="ts">
  import type { FromWorker } from './worker';
  import type { Run } from './sim';
  const runs: Partial<Record<'plain' | 'attn', Promise<Run>>> = {};
  const progress = { plain: 0, attn: 0 };
  /** Train each model once per page in its own worker; every figure on the page shares the result. */
  function load(kind: 'plain' | 'attn'): Promise<Run> {
    return (runs[kind] ??= new Promise((res, rej) => {
      const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      w.onmessage = (e: MessageEvent<FromWorker>) => {
        if (e.data.kind === 'progress') progress[kind] = e.data.frac;
        else { progress[kind] = 1; res(e.data.run); w.terminate(); }
      };
      w.onerror = () => rej(new Error('worker failed'));
      w.postMessage({ attn: kind === 'attn' });
    }));
  }
</script>

<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, readVar, onThemeChange, prefersReducedMotion, advance } from '../_shared';
  import {
    ALPHABET, MIN_LEN, MAX_LEN, LENS, encode, decode, reverse, translate, alignment, argmax, testSet,
    encdecTiming, encdecMarkers, encdecFrame, bottleneckT, bottleneckMarkers, bottleneckFrame,
    attnTiming, attnMarkers, attnFrame, subP, AT_SUB, alignT, alignMarkers, alignFrame, clamp01,
    type Mode, type Run, type Pass,
  } from './sim';

  interface Props {
    /** 'encdec': read, hand over one vector, write. 'bottleneck': accuracy against length. 'attention': one decoding step at a time. 'alignment': the attention matrix as training goes on. */
    mode?: Mode;
    title?: string;
    /** Starting string (letters a to h). */
    sequence?: string;
    /** bottleneck: also draw the model with attention. */
    compare?: boolean;
    editable?: boolean;
  }
  let { mode = 'encdec', title = '', sequence = '', compare = false, editable = true }: Props = $props();

  const init = untrack(() => ({
    mode, compare,
    title: title || { encdec: 'An encoder-decoder reversing a string', bottleneck: 'Accuracy against length', attention: 'Attention, one output letter at a time', alignment: 'The attention matrix as the network trains' }[mode],
    seq: sequence || { encdec: 'hdfab', bottleneck: '', attention: 'cegbhadf', alignment: 'cegbhadf' }[mode],
  }));
  const needPlain = init.mode === 'encdec' || init.mode === 'bottleneck';
  const needAttn = init.mode === 'attention' || init.mode === 'alignment' || init.compare;

  let text = $state(untrack(() => init.seq));
  let src = $state(untrack(() => encode(init.seq)));
  const valid = (s: string) => encode(s).length === s.length && s.length >= MIN_LEN && s.length <= MAX_LEN;
  const onText = () => { const s = text.toLowerCase(); if (valid(s)) { src = encode(s); t = 0; poke(); } };

  let plain = $state.raw<Run | null>(null), attn = $state.raw<Run | null>(null);
  let failed = $state(false);
  const ready = $derived((!needPlain || !!plain) && (!needAttn || !!attn));

  const L = $derived(src.length);
  const tgt = $derived(reverse(src));
  const timing = $derived.by(() => {
    if (init.mode === 'encdec') { const tm = encdecTiming(L); return { T: tm.T, markers: encdecMarkers(tm), ed: tm, at: null }; }
    if (init.mode === 'attention') { const tm = attnTiming(L); return { T: tm.T, markers: attnMarkers(tm), ed: null, at: tm }; }
    if (init.mode === 'alignment') { const n = attn?.snaps.length ?? 31; return { T: alignT(n), markers: attn ? alignMarkers(attn.snapStep) : [], ed: null, at: null }; }
    return { T: bottleneckT, markers: bottleneckMarkers, ed: null, at: null };
  });
  const passP: Pass | null = $derived(plain && init.mode === 'encdec' ? translate(plain.params, false, src) : null);
  const passA: Pass | null = $derived(attn && init.mode === 'attention' ? translate(attn.params, true, src) : null);
  const aligns = $derived(attn && init.mode === 'alignment' ? attn.snaps.map((p) => alignment(p, src)) : []);
  const examples = $derived.by(() => {
    if (init.mode !== 'bottleneck') return [];
    const test = testSet();
    return LENS.map((_, k) => {
      const x = test[k][0];
      return { x, plain: plain ? translate(plain.params, false, x).probs.map(argmax) : null, attn: attn ? translate(attn.params, true, x).probs.map(argmax) : null };
    });
  });

  let narrow = $state(false);
  let playing = $state(false), speed = $state(0.5), uiT = $state(0);
  let narration = $state('');
  const Wd = $derived(narrow ? 400 : 720);
  const Hh = { encdec: 286, bottleneck: 320, attention: 312, alignment: 372 }[untrack(() => init.mode)];

  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let t = 0, poke = () => {};
  $effect(() => { playing; speed; poke(); });
  $effect(() => { src; plain; attn; poke(); });

  const legend = {
    encdec: [{ label: 'positive', color: '--viz-pos' }, { label: 'negative', color: '--viz-neg' }, { label: 'context vector', color: '--accent' }],
    bottleneck: [{ label: 'fixed vector', color: '--text-muted', shape: 'line' as const }, ...(untrack(() => compare) ? [{ label: 'with attention', color: '--viz-seq-4', shape: 'line' as const }] : [])],
    attention: [{ label: 'positive', color: '--viz-pos' }, { label: 'negative', color: '--viz-neg' }, { label: 'attention weight', color: '--viz-seq-4' }],
    alignment: [{ label: 'attention weight', color: '--viz-seq-4' }],
  }[untrack(() => init.mode)];

  const pct = (v: number) => `${Math.round(100 * v)}%`;
  const letter = (i: number) => ALPHABET[i] ?? '?';
  const rightCount = (out: number[], y: number[]) => out.reduce((a, o, i) => a + (o === y[i] ? 1 : 0), 0);

  function say(): string {
    if (!ready) return failed ? 'Training failed in this browser.' : 'Training the networks in your browser…';
    const tt = t;
    if (init.mode === 'encdec' && passP && timing.ed) {
      const f = encdecFrame(tt, timing.ed), out = passP.probs.map(argmax);
      if (f.phase === 'encode') return `Encoder step ${f.active + 1}: reads "${letter(src[f.active])}" into h${f.active + 1}.`;
      if (f.phase === 'hand over') return `The whole input is now one vector, h${L}: 16 numbers. It is all the decoder gets.`;
      if (tt > timing.ed.decStart + L * 0.75) return `Output "${decode(out)}", target "${decode(tgt)}": ${rightCount(out, tgt)} of ${L} letters right.`;
      return `Decoder step ${f.active + 1}: writes "${letter(out[f.active])}" (target "${letter(tgt[f.active])}").`;
    }
    if (init.mode === 'bottleneck' && plain) {
      const { k, L: n } = bottleneckFrame(tt);
      return `Strings of ${n} letters: the fixed-vector model reverses ${pct(plain.seqAcc[k])} of them exactly${init.compare && attn ? `, the model with attention ${pct(attn.seqAcc[k])}` : ''}.`;
    }
    if (init.mode === 'attention' && passA && timing.at) {
      const f = attnFrame(tt, timing.at);
      if (f.i < 0) return 'The encoder reads the input and keeps every state h1 … h' + L + '.';
      const al = passA.alpha[f.i], j = argmax(al), out = argmax(passA.probs[f.i]);
      const n = Math.min(3, Math.floor(f.sub / 0.25));
      const head = `Output ${f.i + 1}: `;
      if (n === 0) return head + `the decoder's state s${f.i} scores each encoder state.`;
      if (n === 1) return head + `softmax turns the scores into weights that sum to 1. The largest, ${pct(al[j])}, is on h${j + 1} ("${letter(src[j])}").`;
      if (n === 2) return head + `the context c${f.i + 1} is the weighted sum of the encoder states, so it is mostly h${j + 1}.`;
      return head + `writes "${letter(out)}" (target "${letter(tgt[f.i])}").`;
    }
    if (init.mode === 'alignment' && attn && aligns.length) {
      const { k } = alignFrame(tt, aligns.length), a = aligns[k];
      const mirror = a.alpha.reduce((s, r, i) => s + r[L - 1 - i], 0) / L;
      return `After ${attn.snapStep[k]} updates: ${pct(mirror)} of each output's weight is on the mirrored input letter. Output "${decode(a.out)}", ${rightCount(a.out, tgt)} of ${L} right.`;
    }
    return '';
  }

  onMount(() => {
    const reduced = prefersReducedMotion();
    let reduce = reduced.matches;
    const ctx = cv.getContext('2d')!;
    const C = {
      pos: '--viz-pos', neg: '--viz-neg', fwd: '--viz-forward', text: '--text', muted: '--text-muted', rule: '--rule',
      raised: '--bg-raised', sunken: '--bg-sunken', accent: '--accent', s1: '--viz-seq-1', s4: '--viz-seq-4', s5: '--viz-seq-5',
    };
    let c = readColors(C), code = readVar('--font-code'), ui = readVar('--font-body');
    const mono = (px: number, w = 500) => `${w} ${px}px ${code}`;
    const seqCol = (v: number) => {
      const a = c.s1.rgb, b = c.s5.rgb, u = clamp01(v);
      return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * u)).join(',')})`;
    };
    let lastUi = 0;

    const line = (pts: [number, number][], col: string, w: number, a = 1) => {
      ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    };
    const dot = (x: number, y: number, r: number, col: string, a = 1) => { ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); };
    const txt = (s: string, x: number, y: number, col: string, px = 12, align: CanvasTextAlign = 'center', w = 500, a = 1, ital = false) => {
      if (a <= 0.02) return;
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.textAlign = align;
      ctx.font = ital ? `italic ${px}px ${ui}` : mono(px, w); ctx.fillText(s, x, y);
    };
    const rbox = (x: number, y: number, w: number, h: number, r: number, a: number, stroke: string, sw = 1.2, fill: string | null = c.raised.hex) => {
      ctx.globalAlpha = a; ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke();
    };
    /** A state vector as a strip of cells centered at (x, y): blue positive, orange negative, opacity = size. */
    const strip = (x: number, y: number, v: ArrayLike<number>, w: number, h: number, a: number, outline = c.rule.hex, ow = 1) => {
      if (a <= 0.02) return;
      const n = v.length, cw = w / n, x0 = x - w / 2;
      ctx.globalAlpha = a; ctx.fillStyle = c.raised.hex; ctx.fillRect(x0, y - h / 2, w, h);
      for (let k = 0; k < n; k++) {
        ctx.globalAlpha = a * Math.min(1, Math.abs(v[k])); ctx.fillStyle = v[k] >= 0 ? c.pos.hex : c.neg.hex;
        ctx.fillRect(x0 + k * cw, y - h / 2, cw + 0.3, h);
      }
      ctx.globalAlpha = a; ctx.strokeStyle = outline; ctx.lineWidth = ow; ctx.strokeRect(x0, y - h / 2, w, h);
    };
    const tokenBox = (x: number, y: number, s: string, a: number, stroke = c.rule.hex, sw = 1.2, col = c.text.hex, size = 26) => {
      rbox(x - size / 2, y - size / 2, size, size, 5, a, stroke, sw);
      txt(s, x, y + 5.5, col, 15, 'center', 600, a);
    };
    const arrowHead = (x: number, y: number, col: string, a: number) => {
      ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - 6, y - 3.5); ctx.lineTo(x - 6, y + 3.5); ctx.fill();
    };
    const cols = (n: number, left: number) => {
      const avail = Wd - left - 14, w = Math.min(84, avail / n), x0 = left + (avail - w * n) / 2;
      return { w, x: (k: number) => x0 + w * (k + 0.5), x0 };
    };

    function drawLoading() {
      const p = (needPlain ? progress.plain : 0) + (needAttn ? progress.attn : 0), n = (needPlain ? 1 : 0) + (needAttn ? 1 : 0);
      const loadP = p / n;
      const w = Math.min(280, Wd - 80), x = (Wd - w) / 2, y = Hh / 2;
      txt(failed ? 'Training failed in this browser.' : `Training the network in your browser: ${Math.round(100 * loadP)}%`, Wd / 2, y - 14, c.muted.hex, 13, 'center', 500);
      rbox(x, y, w, 8, 4, 1, c.rule.hex, 1, c.sunken.hex);
      ctx.globalAlpha = 1; ctx.fillStyle = c.fwd.hex; ctx.fillRect(x + 1, y + 1, (w - 2) * loadP, 6);
    }

    // ------------------------------------------------ encoder-decoder (fixed vector)
    function drawEncDec() {
      if (!passP || !timing.ed) return;
      const f = encdecFrame(t, timing.ed), left = narrow ? 60 : 76, g = cols(L, left), out = passP.probs.map(argmax);
      const sw = Math.min(g.w - 10, 70), sh = 12;
      const yIn = 26, yEnc = 72, yBus = 122, yDec = 170, yOut = 214, yTgt = 250;
      const lab = (s: string, y: number) => txt(s, left - 10, y + 4, c.muted.hex, narrow ? 10 : 11, 'right', 500);
      lab('input', yIn); lab('encoder', yEnc); lab('decoder', yDec); lab('output', yOut); lab('target', yTgt);
      for (let j = 0; j < L; j++) {
        const x = g.x(j), a = f.enc[j], on = f.phase === 'encode' && f.active === j;
        tokenBox(x, yIn, letter(src[j]), 1, on ? c.accent.hex : c.rule.hex, on ? 2.2 : 1.2);
        line([[x, yIn + 14], [x, yEnc - sh / 2 - 2]], c.muted.hex, 1.2, 0.5 * a);
        if (j > 0) { line([[g.x(j - 1) + sw / 2 + 2, yEnc], [x - sw / 2 - 3, yEnc]], c.muted.hex, 1.4, 0.6 * a); arrowHead(x - sw / 2 - 2, yEnc, c.muted.hex, 0.6 * a); }
        const last = j === L - 1;
        strip(x, yEnc, passP.hs[j], sw, sh, 0.15 + 0.85 * a, last ? c.accent.hex : c.rule.hex, last ? 2 : 1);
        if (g.w >= 40) txt(`h${j + 1}`, x + sw / 2, yEnc - sh / 2 - 5, c.muted.hex, 10, 'right', 500, a);
      }
      // the context bus: h_L to every decoder step
      const xL = g.x(L - 1), hp = f.hand;
      if (hp > 0) {
        const xa = g.x(0);
        const bus: [number, number][] = [[xL, yEnc + sh / 2 + 2], [xL, yBus], [xa, yBus]];
        line(bus.slice(0, 2), c.accent.hex, 2, hp);
        line([[xL, yBus], [xL + (xa - xL) * hp, yBus]], c.accent.hex, 2, hp);
        txt(`context = h${L}, the same 16 numbers at every step`, (xa + xL) / 2, yBus - 6, c.accent.hex, narrow ? 10 : 11, 'center', 500, hp);
        for (let i = 0; i < L; i++) { const x = g.x(i); if ((xL - x) / Math.max(1, xL - xa) <= hp + 1e-6 || L === 1) line([[x, yBus], [x, yDec - sh / 2 - 3]], c.accent.hex, 1.4, 0.7 * hp); }
        if (!reduce && hp < 1) dot(xL + (xa - xL) * hp, yBus, 4.5, c.accent.hex);
      }
      for (let i = 0; i < L; i++) {
        const x = g.x(i), a = f.dec[i], on = f.phase === 'decode' && f.active === i;
        if (i > 0) { line([[g.x(i - 1) + sw / 2 + 2, yDec], [x - sw / 2 - 3, yDec]], c.muted.hex, 1.4, 0.6 * a); arrowHead(x - sw / 2 - 2, yDec, c.muted.hex, 0.6 * a); }
        strip(x, yDec, passP.ss[i + 1], sw, sh, 0.15 + 0.85 * a);
        if (g.w >= 40) txt(`s${i + 1}`, x + sw / 2, yDec - sh / 2 - 5, c.muted.hex, 10, 'right', 500, a);
        line([[x, yDec + sh / 2 + 2], [x, yOut - 14]], c.muted.hex, 1.2, 0.5 * a);
        const ok = out[i] === tgt[i];
        tokenBox(x, yOut, letter(out[i]), a, on ? c.accent.hex : ok ? c.rule.hex : c.neg.hex, on || !ok ? 2.2 : 1.2, ok ? c.text.hex : c.neg.hex);
        txt(letter(tgt[i]), x, yTgt + 4, c.muted.hex, 13, 'center', 500, 0.85);
      }
    }

    // ------------------------------------------------ accuracy against length
    function drawBottleneck() {
      if (!plain) return;
      const f = bottleneckFrame(t), ex = examples[f.k];
      const left = narrow ? 104 : 128, top = 156, bot = Hh - 34, right = Wd - 20;
      const X = (k: number) => left + ((right - left) * k) / (LENS.length - 1), Y = (v: number) => bot - (bot - top) * v;
      // the example at this length
      if (ex) {
        const n = ex.x.length, s = Math.min(26, (Wd - left - 10) / n), x0 = left;
        const rows: [string, number[] | null, boolean][] = [['input', ex.x, false], ['target', reverse(ex.x), false], ['fixed vector', ex.plain, true]];
        if (init.compare) rows.push(['with attention', ex.attn, true]);
        const rowH = init.compare ? 28 : 34;
        rows.forEach(([name, v, check], r) => {
          const y = 22 + r * rowH;
          txt(name, left - 10, y + 4, c.muted.hex, narrow ? 10 : 11, 'right', 500);
          const al = check ? f.reveal : 1;
          v?.forEach((q, i) => {
            const bad = check && q !== reverse(ex.x)[i];
            rbox(x0 + i * s + 1, y - s / 2 + 2, s - 2, s - 4, 4, al, bad ? c.neg.hex : c.rule.hex, bad ? 2 : 1);
            txt(letter(q), x0 + i * s + s / 2, y + 5, bad ? c.neg.hex : c.text.hex, 13, 'center', 600, al);
          });
        });
      }
      // axes
      for (const v of [0, 0.5, 1]) { line([[left, Y(v)], [right, Y(v)]], c.rule.hex, 1, 0.9); txt(pct(v), left - 8, Y(v) + 4, c.muted.hex, 10, 'right', 400); }
      LENS.forEach((n, k) => txt(String(n), X(k), bot + 16, k === f.k ? c.text.hex : c.muted.hex, 11, 'center', k === f.k ? 700 : 400));
      txt('string length', (left + right) / 2, bot + 30, c.muted.hex, 11, 'center', 400, 1, true);
      txt('reversed exactly', left - 8, top - 18, c.muted.hex, 11, 'right', 400, 1, true);
      const curve = (acc: number[], col: string, w: number, below: boolean) => {
        // points up to the current length; the newest segment draws in as the length is revealed
        const pts: [number, number][] = [];
        for (let k = 0; k < f.k; k++) pts.push([X(k), Y(acc[k])]);
        const r = f.k === 0 ? 1 : f.reveal, kp = Math.max(0, f.k - 1);
        pts.push([X(kp) + (X(f.k) - X(kp)) * r, Y(acc[kp] + (acc[f.k] - acc[kp]) * r)]);
        line(pts, col, w);
        for (let k = 0; k < f.k; k++) dot(X(k), Y(acc[k]), 3.5, col);
        dot(pts[pts.length - 1][0], pts[pts.length - 1][1], 4.5, col);
        if (r >= 1) {
          const atEnd = f.k === LENS.length - 1, y = Y(acc[f.k]), dy = below && y + 15 < bot - 4 ? 15 : -8;
          txt(pct(acc[f.k]), X(f.k) + (atEnd ? -8 : 8), y + dy, col, 11, atEnd ? 'right' : 'left', 700);
        }
      };
      if (init.compare && attn) curve(attn.seqAcc, c.s4.hex, 2.4, false);
      curve(plain.seqAcc, c.muted.hex, 2.4, !!init.compare);
    }

    // ------------------------------------------------ one decoding step with attention
    function drawAttention() {
      if (!passA || !timing.at) return;
      const f = attnFrame(t, timing.at), left = narrow ? 52 : 82, g = cols(L, left), out = passA.probs.map(argmax);
      const sw = Math.min(g.w - 10, 64), sh = 12;
      const yIn = 22, yEnc = 58, yBarT = 76, yBarB = 120, yNum = 134, yCtx = 190, yDec = 236, yOut = 276;
      const lab = (s: string, y: number) => txt(s, left - 10, y + 4, c.muted.hex, narrow ? 10 : 11, 'right', 500);
      lab('input', yIn); lab('encoder', yEnc); lab('weights', (yBarT + yBarB) / 2); lab('context', yCtx); lab('decoder', yDec); lab('output', yOut);
      const i = f.i, u = f.sub;
      const pS = i >= 0 ? subP(u, 0) : 0, pW = i >= 0 ? subP(u, 1) : 0, pC = i >= 0 ? subP(u, 2) : 0, pO = i >= 0 ? subP(u, 3) : 0;
      const al = i >= 0 ? passA.alpha[i] : null, sc = i >= 0 ? passA.scores[i] : null;
      for (let j = 0; j < L; j++) {
        const x = g.x(j), a = f.enc[j];
        tokenBox(x, yIn, letter(src[j]), 1);
        strip(x, yEnc, passA.hs[j], sw, sh, 0.15 + 0.85 * a);
        if (g.w >= 40) txt(`h${j + 1}`, x + sw / 2, yEnc - sh / 2 - 4, c.muted.hex, 9.5, 'right', 500, a);
        // weight bar
        const bw = Math.min(16, g.w * 0.3);
        ctx.globalAlpha = 1; ctx.fillStyle = c.sunken.hex; ctx.fillRect(x - bw / 2, yBarT, bw, yBarB - yBarT);
        if (al && sc) {
          const v = al[j] * pW;
          ctx.globalAlpha = 1; ctx.fillStyle = c.s4.hex; ctx.fillRect(x - bw / 2, yBarB - (yBarB - yBarT) * v, bw, (yBarB - yBarT) * v);
          if (g.w >= 34) {
            if (pW < 0.5) txt(sc[j].toFixed(1), x, yNum, c.muted.hex, 10.5, 'center', 500, pS * (1 - 2 * pW));
            else txt(pct(al[j]), x, yNum, al[j] > 0.3 ? c.text.hex : c.muted.hex, 10.5, 'center', al[j] > 0.3 ? 700 : 400, 2 * pW - 1);
          }
        }
      }
      if (i >= 0 && al) {
        const xi = g.x(i);
        // query: the previous decoder state reaches up to every encoder state
        const qx = i === 0 ? g.x(L - 1) : g.x(i - 1), qy = i === 0 ? yEnc : yDec;
        if (pS > 0 && pW < 1) for (let j = 0; j < L; j++) line([[qx, qy + (i === 0 ? sh / 2 + 2 : -sh / 2 - 2)], [g.x(j), yBarB + 2]], c.muted.hex, 1, 0.35 * pS * (1 - pW));
        if (pS > 0 && pW < 1) txt(i === 0 ? `query s0 = h${L}` : `query s${i}`, qx, qy + (i === 0 ? sh / 2 + 14 : sh / 2 + 14), c.accent.hex, 10.5, 'center', 600, pS * (1 - pW));
        // weighted lines from every encoder state into the context
        if (pW > 0) for (let j = 0; j < L; j++) {
          const pts: [number, number][] = [[g.x(j), yNum + 6], [xi, yCtx - sh / 2 - 3]];
          line(pts, c.s4.hex, 0.6 + 7 * al[j], (0.15 + 0.85 * al[j]) * pW);
          if (!reduce && pC > 0 && pC < 1 && al[j] > 0.05) dot(pts[0][0] + (pts[1][0] - pts[0][0]) * pC, pts[0][1] + (pts[1][1] - pts[0][1]) * pC, 2 + 3 * al[j], c.s4.hex);
        }
        strip(xi, yCtx, passA.ctx[i], sw, sh, pC, c.s4.hex, 1.6);
        if (pC > 0.5) txt(`c${i + 1}`, xi + sw / 2 + 4, yCtx + 4, c.s4.hex, 10, 'left', 600, pC);
        line([[xi, yCtx + sh / 2 + 2], [xi, yDec - sh / 2 - 2]], c.s4.hex, 1.4, pO);
      }
      for (let k = 0; k < L; k++) {
        const x = g.x(k), a = Math.max(f.done[k], k === i ? subP(u, 3) : 0), on = k === i;
        if (k > 0) { line([[g.x(k - 1) + sw / 2 + 2, yDec], [x - sw / 2 - 3, yDec]], c.muted.hex, 1.4, 0.6 * a); arrowHead(x - sw / 2 - 2, yDec, c.muted.hex, 0.6 * a); }
        strip(x, yDec, passA.ss[k + 1], sw, sh, 0.12 + 0.88 * a, on ? c.accent.hex : c.rule.hex, on ? 2 : 1);
        const ok = out[k] === tgt[k];
        if (a > 0.02) tokenBox(x, yOut, letter(out[k]), a, ok ? c.rule.hex : c.neg.hex, ok ? 1.2 : 2.2, ok ? c.text.hex : c.neg.hex);
      }
    }

    // ------------------------------------------------ alignment matrix over training
    function drawAlignment() {
      if (!attn || !aligns.length) return;
      const { k } = alignFrame(t, aligns.length), a = aligns[k];
      const cs = Math.min(30, (Wd - 130) / L, (Hh - 120) / L), w = cs * L;
      const x0 = (Wd - w) / 2 + (narrow ? 10 : 0), y0 = 58;
      txt('input letter (encoder state)', x0 + w / 2, 18, c.muted.hex, 11, 'center', 400, 1, true);
      for (let j = 0; j < L; j++) txt(letter(src[j]), x0 + (j + 0.5) * cs, y0 - 10, c.text.hex, 13, 'center', 600);
      ctx.save(); ctx.translate(x0 - 40, y0 + w / 2); ctx.rotate(-Math.PI / 2);
      txt('output step', 0, 0, c.muted.hex, 11, 'center', 400, 1, true); ctx.restore();
      for (let i = 0; i < L; i++) {
        const ok = a.out[i] === tgt[i];
        txt(letter(a.out[i]), x0 - 12, y0 + (i + 0.5) * cs + 5, ok ? c.text.hex : c.neg.hex, 13, 'center', 700);
        for (let j = 0; j < L; j++) {
          ctx.globalAlpha = 1; ctx.fillStyle = seqCol(a.alpha[i][j]);
          ctx.fillRect(x0 + j * cs + 0.5, y0 + i * cs + 0.5, cs - 1, cs - 1);
        }
      }
      ctx.globalAlpha = 1; ctx.strokeStyle = c.rule.hex; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, w, w);
      // colour scale
      const sx = x0 + w + 22, sy = y0, sh = Math.min(w, 140);
      if (sx + 40 < Wd) {
        for (let q = 0; q < 40; q++) { ctx.fillStyle = seqCol(1 - q / 39); ctx.fillRect(sx, sy + (sh * q) / 40, 10, sh / 40 + 0.5); }
        ctx.strokeStyle = c.rule.hex; ctx.strokeRect(sx, sy, 10, sh);
        txt('1', sx + 15, sy + 9, c.muted.hex, 10, 'left', 400); txt('0', sx + 15, sy + sh, c.muted.hex, 10, 'left', 400);
      }
      txt(`target: ${decode(tgt)}`, x0 + w / 2, y0 + w + 24, c.muted.hex, 12, 'center', 500);
      txt(`after ${attn.snapStep[k]} updates`, x0 + w / 2, y0 + w + 42, c.text.hex, 12, 'center', 700);
    }

    function render() {
      const k0 = cv.width / Wd;
      ctx.setTransform(k0, 0, 0, k0, 0, 0);
      ctx.clearRect(0, 0, Wd, Hh);
      if (!ready) drawLoading();
      else if (init.mode === 'encdec') drawEncDec();
      else if (init.mode === 'bottleneck') drawBottleneck();
      else if (init.mode === 'attention') drawAttention();
      else drawAlignment();
      ctx.globalAlpha = 1;
    }

    const loop = createLoop(cv, (dt, now) => {
      const T = timing.T;
      if (ready) {
        if (t > T) t = 0;
        t = advance({ t, playing, speed }, dt, { duration: T }).t;
      }
      render();
      tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; const s = say(); if (s !== narration) narration = s; }
      return playing || !ready;
    }, { onView: (inV, first) => { if (inV && first && !reduce) playing = true; } });
    poke = loop.wake;

    const got = (kind: 'plain' | 'attn') => (r: Run) => {
      if (kind === 'plain') plain = r; else attn = r;
      if (reduce && (!needPlain || plain) && (!needAttn || attn)) t = timing.T;
      poke();
    };
    if (needPlain) load('plain').then(got('plain'), () => { failed = true; poke(); });
    if (needAttn) load('attn').then(got('attn'), () => { failed = true; poke(); });

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
    fit();
    return () => { loop.destroy(); ro.disconnect(); off(); reduced.removeEventListener('change', onReduce); };
  });

  const stageLabel = $derived({
    encdec: `An encoder reads "${decode(src)}" into a chain of state vectors; its last state is handed to a decoder that writes the reversed string.`,
    bottleneck: 'Line chart of the share of test strings reversed exactly, against string length from 3 to 10, with an example at each length.',
    attention: `A decoder writing the reverse of "${decode(src)}". At each output step, bars show the attention weight on every encoder state and lines carry them into the context vector.`,
    alignment: `Heatmap of attention weights: one row per output letter, one column per input letter of "${decode(src)}", as training goes on.`,
  }[init.mode]);
</script>

<FigureCard title={init.title} {legend} aspect="720 / {Hh}" narrowAspect="400 / {Hh}" label={stageLabel}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={timing.T} markers={timing.markers} label="Timeline"
      onSeek={(to) => { t = to; poke(); }} onStep={(to) => { playing = false; t = to; poke(); }} onReset={() => { t = 0; poke(); }} />
  {/snippet}
  {#snippet readout()}
    <p class="say" role="status" aria-live="polite">{narration}</p>
  {/snippet}
  {#snippet footer()}
    {#if editable && init.mode !== 'bottleneck'}
      <div class="ctl">
        <label>input
          <input class="seqin" type="text" bind:value={text} oninput={onText} maxlength={MAX_LEN} spellcheck="false" autocomplete="off"
            aria-label="Input string: {MIN_LEN} to {MAX_LEN} letters from a to h" aria-invalid={!valid(text.toLowerCase())} />
        </label>
        <span class="hint" class:bad={!valid(text.toLowerCase())}>{MIN_LEN} to {MAX_LEN} letters from a to h</span>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .say { margin: 0; font: 0.92rem / 1.45 var(--font-code); font-variant-numeric: tabular-nums; min-height: 2.9em; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .seqin {
    width: 9rem; font: 0.9rem var(--font-code); padding: 0.3em 0.5em; color: var(--text);
    background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1);
  }
  .hint { font: var(--step--1) var(--font-ui); color: var(--text-muted); }
  .hint.bad { color: var(--viz-neg); }
</style>
