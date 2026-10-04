<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, onThemeChange, prefersReducedMotion, readColors, readVar } from '../_shared';
  import { T, MARKERS, INPUT, frameAt, forward, type AugmentTask, type Mode } from './sim';

  interface Props { mode?: Mode; title?: string }
  let { mode = 'mask', title = '' }: Props = $props();
  const init = untrack(() => ({ mode, title: title || defaultTitle(mode) }));
  const uid = $props.id();
  let keep = $state(0.5);
  let task = $state<AugmentTask>('presence');
  let playing = $state(false), speed = $state(1), uiT = $state(0);
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>;
  let t = 0, wake = () => {};
  let colors: ReturnType<typeof readColors>;
  let codeFont = '', bodyFont = '';

  const markers = MARKERS[init.mode];
  const current = $derived(frameAt(uiT, init.mode, keep, task));
  const narration = $derived(phaseText(current));
  const h1 = forward().hidden[0];
  const f3 = (v: number) => v.toFixed(3);

  const LEGENDS: Record<Mode, { label: string; color: string; shape?: 'dot' | 'line' }[]> = {
    mask: [{ label: 'kept unit', color: '--viz-forward' }, { label: 'dropped unit', color: '--text-muted' }, { label: 'output', color: '--accent' }],
    scaling: [{ label: 'not scaled', color: '--text-muted' }, { label: 'scaled by 1 / keep', color: '--viz-forward' }, { label: 'full network', color: '--accent', shape: 'line' }],
    ensemble: [{ label: 'one mask (bigger = more likely)', color: '--viz-forward' }, { label: 'weighted mean', color: '--viz-forward', shape: 'line' }, { label: 'full network', color: '--accent', shape: 'line' }],
    decay: [{ label: 'weight after one update', color: '--viz-forward' }, { label: 'starting weight', color: '--text-muted', shape: 'line' }],
    'early-stop': [{ label: 'training loss', color: '--viz-forward', shape: 'line' }, { label: 'validation loss', color: '--viz-gradient', shape: 'line' }, { label: 'best checkpoint', color: '--accent', shape: 'line' }, { label: 'stop', color: '--viz-neg', shape: 'line' }],
    augment: [{ label: 'bright pixel', color: '--viz-forward' }],
  };
  const LABELS: Record<Mode, string> = {
    mask: 'A network with two inputs, four hidden units and one output. Dropped hidden units are greyed out and pass on zero.',
    scaling: 'The output of each of the 16 possible dropout masks, with and without scaling, and their averages.',
    ensemble: 'The output of each of the 16 possible dropout masks, sized by probability, with their weighted mean and the full network output.',
    decay: 'Four number lines showing a weight of 1.00 after one SGD, SGD with L2, Adam with L2, and AdamW update.',
    'early-stop': 'Training and validation loss by epoch on a log scale, with the best checkpoint and the stopping epoch marked.',
    augment: 'A 2 by 2 image and its horizontal flip, with the label before and after.',
  };

  function defaultTitle(value: Mode) {
    return ({ mask: 'Dropout on one training example', scaling: 'Scaling kept units keeps the average', ensemble: 'Sixteen subnetworks and their average', decay: 'One update with SGD, L2 and AdamW', 'early-stop': 'Keep the checkpoint with the best validation loss', augment: 'Flip an image, then check the label' })[value];
  }

  function phaseText(fr: ReturnType<typeof frameAt>) {
    const d = fr.decay;
    switch (init.mode) {
      case 'mask': return fr.inference
        ? `Inference: no dropout. All four units pass on their ordinary activation and the output is ${f3(fr.full)}.`
        : `Training: every example gets a new random mask. Kept units are multiplied by 1 / ${keep.toFixed(2)}; dropped units pass on zero.`;
      case 'scaling': return fr.phase === 'Unscaled'
        ? 'Grey dots: each mask’s output if dropped units are simply zeroed. Their average sits well below the full network.'
        : fr.phase === 'Inverted' ? 'Teal dots: the same masks with kept units divided by the keep probability. Their average moves up.'
        : 'The full network, with no dropout, lands exactly on the average of the scaled outputs.';
      case 'ensemble': return fr.phase === 'Masks'
        ? 'Each dot is one of the 16 subnetworks dropout can pick. A bigger dot is a mask that is picked more often.'
        : fr.phase === 'Mean' ? 'Weight each output by how often its mask is picked and add them up: the result equals the full network.'
        : 'Put a sigmoid after the output and the two numbers no longer match exactly.';
      case 'decay': return fr.decayRows === 1 ? `SGD: 1.00 − ${d.learningRate} × ${d.gradient} = ${f3(d.sgd)}.`
        : fr.decayRows === 2 ? `L2 adds ${d.lambda} × 1.00 to the gradient, so SGD steps by ${d.learningRate} × ${(d.gradient + d.lambda).toFixed(1)}: ${f3(d.sgdL2)}.`
        : fr.decayRows === 3 ? `Adam’s first step is ${d.learningRate} long whatever the gradient, so the L2 term has no effect: ${f3(d.adamCoupled)}.`
        : `AdamW takes the same Adam step, then shrinks the weight by ${d.learningRate} × ${d.lambda} × 1.00 = ${(d.learningRate * d.lambda).toFixed(2)}: ${f3(d.adamW)}.`;
      case 'early-stop': return fr.phase === 'Improving'
        ? 'Both losses fall. Each new low in validation loss is saved as the best checkpoint.'
        : fr.phase === 'Waiting' ? 'Training loss keeps falling, but validation loss has started to rise. Count the epochs without a new best.'
        : `Three epochs without a new best: stop, and go back to the weights saved at epoch ${fr.checkpoint.epoch}.`;
      default: return fr.phase === 'Original' ? 'One tiny training image and its label.'
        : fr.phase === 'Flip' ? 'Flip it left to right to make a new training example.'
        : fr.augmentation.labelPreserved ? 'The flipped image still has a bright pixel, so the old label still fits.' : 'The bright pixel is now on the right. Keeping the old label would teach the model something false.';
    }
  }

  const text = (ctx: CanvasRenderingContext2D, s: string, x: number, y: number, color = colors.text.hex, size = 11, align: CanvasTextAlign = 'center', font = codeFont) => {
    ctx.fillStyle = color; ctx.font = `500 ${size}px ${font}`; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y);
  };
  const dot = (ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); };
  const hline = (ctx: CanvasRenderingContext2D, x0: number, x1: number, y: number, color: string, dash: number[] = [], width = 1) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  };

  function renderNetwork(ctx: CanvasRenderingContext2D, fr: ReturnType<typeof frameAt>, w: number, h: number) {
    const inputX = w < 500 ? 43 : 90, hiddenX = w / 2, outputX = w - inputX;
    const inputY = [h * 0.39, h * 0.65], hiddenY = [h * 0.2, h * 0.39, h * 0.58, h * 0.77];
    for (let j = 0; j < 4; j++) {
      const kept = fr.units[j].kept;
      ctx.strokeStyle = kept ? colors.fwd.hex : colors.rule.hex; ctx.globalAlpha = kept ? 0.75 : 0.5; ctx.lineWidth = 1.2;
      ctx.setLineDash(kept ? [] : [3, 4]);
      for (let i = 0; i < 2; i++) { ctx.beginPath(); ctx.moveTo(inputX + 15, inputY[i]); ctx.lineTo(hiddenX - 20, hiddenY[j]); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(hiddenX + 20, hiddenY[j]); ctx.lineTo(outputX - 26, h * 0.5); ctx.stroke();
    }
    ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.lineWidth = 1.5;
    inputY.forEach((y, i) => { ctx.fillStyle = colors.raised.hex; ctx.strokeStyle = colors.fwd.hex; ctx.beginPath(); ctx.arc(inputX, y, 15, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); text(ctx, INPUT[i].toFixed(1), inputX, y); });
    hiddenY.forEach((y, i) => {
      const u = fr.units[i];
      ctx.fillStyle = u.kept ? colors.raised.hex : colors.bg.hex; ctx.strokeStyle = u.kept ? colors.fwd.hex : colors.muted.hex;
      ctx.setLineDash(u.kept ? [] : [3, 3]); ctx.beginPath(); ctx.arc(hiddenX, y, 19, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
      text(ctx, u.kept ? f3(u.activation) : '0', hiddenX, y, u.kept ? colors.text.hex : colors.muted.hex, 10);
      text(ctx, `h${i + 1} ${u.kept ? 'kept' : 'dropped'}`, hiddenX, y + 29, colors.muted.hex, 9);
    });
    ctx.fillStyle = colors.raised.hex; ctx.strokeStyle = colors.accent.hex; ctx.beginPath(); ctx.roundRect(outputX - 26, h * 0.5 - 17, 52, 34, 4); ctx.fill(); ctx.stroke();
    text(ctx, f3(fr.network.output), outputX, h * 0.5, colors.accent.hex, 11);
    text(ctx, 'input', inputX, 30, colors.muted.hex, 10);
    text(ctx, 'hidden units', hiddenX, 30, colors.muted.hex, 10);
    text(ctx, 'output', outputX, 30, colors.muted.hex, 10);
    const mask = fr.units.map((u) => (u.kept ? 1 : 0)).join('');
    text(ctx, fr.inference ? 'inference · all units kept, no scaling' : `training example ${fr.maskIndex + 1} · mask ${mask} (1 = kept)`, w / 2, h - 22, colors.text.hex, 11);
  }

  function renderMaskChart(ctx: CanvasRenderingContext2D, fr: ReturnType<typeof frameAt>, w: number, h: number, scaling: boolean) {
    const narrow = w < 500, left = narrow ? 44 : 60, right = w - 16, top = 56, bottom = h - 54;
    const raw = fr.masks.map((m) => forward(INPUT, m.mask, keep, false).output);
    const inv = fr.masks.map((m) => m.output);
    const values = scaling ? [...raw, ...inv, fr.full] : [...inv, fr.full];
    const low = Math.min(...values) - 0.12, high = Math.max(...values) + 0.12;
    const y = (v: number) => bottom - ((v - low) / (high - low)) * (bottom - top);
    const x = (i: number) => left + 10 + (i / 15) * (right - left - 20);
    const maxP = Math.max(...fr.masks.map((m) => m.probability));
    const radius = (p: number) => (p > 0 ? 1.8 + 3.6 * Math.sqrt(p / maxP) : 0);
    const step = high - low > 1.2 ? 0.5 : 0.25;
    for (let v = Math.ceil(low / step) * step; v <= high; v += step) {
      hline(ctx, left, right, y(v), colors.rule.hex, Math.abs(v) < 1e-9 ? [] : [3, 4]);
      text(ctx, v.toFixed(2), left - 6, y(v), colors.muted.hex, 9, 'right');
    }
    text(ctx, 'output of each mask', left, 30, colors.text.hex, 11, 'left', bodyFont);
    text(ctx, `the 16 masks, from 0000 (all dropped) to 1111 (all kept)`, (left + right) / 2, h - 30, colors.muted.hex, narrow ? 9 : 10);
    const meanLabel = (s: string, yy: number, color: string, above: boolean, align: CanvasTextAlign = 'right') => {
      const lx = align === 'right' ? right - 2 : left + 4, ly = yy + (above ? -9 : 10);
      ctx.font = `500 10px ${codeFont}`; const tw = ctx.measureText(s).width;
      ctx.globalAlpha = 0.9; ctx.fillStyle = colors.raised.hex; ctx.fillRect(align === 'right' ? lx - tw - 3 : lx - 3, ly - 7, tw + 6, 14); ctx.globalAlpha = 1;
      text(ctx, s, lx, ly, color, 10, align);
    };
    if (scaling) {
      for (let i = 0; i < fr.unscaledShown; i++) dot(ctx, x(i) - 2.5, y(raw[i]), radius(fr.masks[i].probability), colors.muted.hex);
      for (let i = 0; i < fr.invertedShown; i++) dot(ctx, x(i) + 2.5, y(inv[i]), radius(fr.masks[i].probability), colors.fwd.hex);
      if (fr.showUnscaledMean) { hline(ctx, left, right, y(fr.unscaledMean), colors.muted.hex, [5, 4], 1.5); meanLabel(`average, not scaled ${f3(fr.unscaledMean)}`, y(fr.unscaledMean), colors.muted.hex, false); }
      if (fr.showInvertedMean) { hline(ctx, left, right, y(fr.weightedMean), colors.fwd.hex, [5, 4], 1.5); meanLabel(`average, scaled ${f3(fr.weightedMean)}`, y(fr.weightedMean), colors.fwd.hex, true); }
      if (fr.showFull) { hline(ctx, left, right, y(fr.full), colors.accent.hex, [], 1); meanLabel(`full network ${f3(fr.full)}`, y(fr.full), colors.accent.hex, true, 'left'); }
    } else {
      for (let i = 0; i < fr.ensembleShown; i++) { ctx.globalAlpha = 0.85; dot(ctx, x(i), y(inv[i]), radius(fr.masks[i].probability) * 1.3, colors.fwd.hex); }
      ctx.globalAlpha = 1;
      if (fr.showMean) {
        hline(ctx, left, right, y(fr.weightedMean), colors.fwd.hex, [], 3);
        hline(ctx, left, right, y(fr.full), colors.accent.hex, [5, 4], 1.5);
        meanLabel(`weighted mean ${f3(fr.weightedMean)}`, y(fr.weightedMean), colors.fwd.hex, true, 'left');
        meanLabel(`full network ${f3(fr.full)}`, y(fr.full), colors.accent.hex, false);
      }
      if (fr.showSigmoid) {
        const s = `after a sigmoid: average ${f3(fr.nonlinearMean)} vs full network ${f3(fr.sigmoidFull)}`;
        if (narrow) text(ctx, s, left, 46, colors.text.hex, 9, 'left'); else text(ctx, s, right, 30, colors.text.hex, 10, 'right');
      }
    }
  }

  function renderDecay(ctx: CanvasRenderingContext2D, fr: ReturnType<typeof frameAt>, w: number, h: number) {
    const d = fr.decay, narrow = w < 500;
    const rows = [
      ['SGD', `step ${d.learningRate} × ${d.gradient}`, d.sgd],
      ['SGD + L2', `step ${d.learningRate} × (${d.gradient} + ${d.lambda} × 1)`, d.sgdL2],
      ['Adam + L2', `Adam’s first step is ${d.learningRate}`, d.adamCoupled],
      ['AdamW', `Adam step, then − ${d.learningRate} × ${d.lambda} × 1`, d.adamW],
    ] as const;
    const lo = 0.86, hi = 1.01;
    const x0 = narrow ? 20 : 250, x1 = w - (narrow ? 56 : 80), top = narrow ? 70 : 64, bottom = h - 46;
    const x = (v: number) => x0 + ((v - lo) / (hi - lo)) * (x1 - x0);
    const band = (bottom - top) / 4;
    const start = `start: weight 1.00 · data gradient ${d.gradient.toFixed(2)}`, settings = `learning rate ${d.learningRate.toFixed(2)} · λ ${d.lambda.toFixed(2)}`;
    if (narrow) { text(ctx, start, 16, 22, colors.text.hex, 10, 'left'); text(ctx, settings, 16, 38, colors.text.hex, 10, 'left'); }
    else text(ctx, `${start} · ${settings}`, 24, 30, colors.text.hex, 11, 'left', bodyFont);
    for (let v = 0.88; v <= 1.0001; v += 0.02) {
      ctx.strokeStyle = colors.rule.hex; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(x(v), top); ctx.lineTo(x(v), bottom); ctx.stroke(); ctx.setLineDash([]);
      text(ctx, v.toFixed(2), x(v), bottom + 14, colors.muted.hex, 9);
    }
    ctx.strokeStyle = colors.muted.hex; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x(1), top - 6); ctx.lineTo(x(1), bottom); ctx.stroke(); ctx.lineWidth = 1;
    rows.forEach(([label, how, value], i) => {
      const shown = i < fr.decayRows, yy = top + band * (i + 0.5) + (narrow ? 8 : 0);
      const lx = narrow ? 16 : 24, ly = narrow ? yy - 20 : yy - 8;
      text(ctx, label, lx, ly, shown ? colors.text.hex : colors.muted.hex, 12, 'left', bodyFont);
      text(ctx, how, narrow ? lx + 70 : lx, narrow ? ly : yy + 9, colors.muted.hex, 9.5, 'left');
      hline(ctx, x0, x1, yy, colors.rule.hex);
      if (!shown) return;
      const grow = i === fr.decayRows - 1 ? fr.decayGrow : 1, end = 1 + (value - 1) * grow, px = x(end);
      ctx.strokeStyle = colors.fwd.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x(1), yy); ctx.lineTo(px, yy); ctx.stroke(); ctx.lineWidth = 1;
      dot(ctx, px, yy, 6, colors.fwd.hex);
      if (grow === 1) {
        text(ctx, `−${(1 - value).toFixed(3)}`, (x(1) + px) / 2, yy + (narrow ? 12 : -11), colors.muted.hex, 9.5);
        text(ctx, f3(value), w - 16, yy, colors.text.hex, 12, 'right');
      }
    });
  }

  function renderEarlyStop(ctx: CanvasRenderingContext2D, fr: ReturnType<typeof frameAt>, w: number, h: number) {
    const narrow = w < 500, run = fr.run, history = run.history.slice(0, fr.visibleEpoch + 1);
    const left = narrow ? 40 : 56, right = w - 20, top = narrow ? 74 : 60, bottom = h - 48;
    const lo = Math.log10(0.03), hi = Math.log10(15);
    const x = (epoch: number) => left + (epoch / run.stopEpoch) * (right - left);
    const y = (loss: number) => bottom - ((Math.log10(loss) - lo) / (hi - lo)) * (bottom - top);
    for (const v of [0.1, 1, 10]) { hline(ctx, left, right, y(v), colors.rule.hex, [3, 4]); text(ctx, String(v), left - 6, y(v), colors.muted.hex, 9, 'right'); }
    text(ctx, 'loss (log scale)', left, top - 14, colors.muted.hex, 9, 'left');
    for (let e = 0; e <= run.stopEpoch; e++) text(ctx, String(e), x(e), bottom + 14, colors.muted.hex, 9);
    text(ctx, 'epoch', right, bottom + 30, colors.muted.hex, 9, 'right');
    const bx = x(fr.checkpoint.epoch);
    ctx.strokeStyle = colors.accent.hex; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(bx, top); ctx.lineTo(bx, bottom); ctx.stroke();
    const stopped = fr.visibleEpoch >= run.stopEpoch;
    if (stopped) { ctx.strokeStyle = colors.neg.hex; ctx.beginPath(); ctx.moveTo(x(run.stopEpoch), top); ctx.lineTo(x(run.stopEpoch), bottom); ctx.stroke(); }
    ctx.setLineDash([]);
    const series = (key: 'train' | 'validation', color: string) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
      history.forEach((p, i) => (i ? ctx.lineTo(x(p.epoch), y(p[key])) : ctx.moveTo(x(p.epoch), y(p[key]))));
      ctx.stroke(); ctx.lineWidth = 1; history.forEach((p) => dot(ctx, x(p.epoch), y(p[key]), 3, color));
    };
    series('train', colors.fwd.hex); series('validation', colors.grad.hex);
    ctx.strokeStyle = colors.accent.hex; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(bx, y(fr.checkpoint.validation), 7, 0, Math.PI * 2); ctx.stroke(); ctx.lineWidth = 1;
    const best = `best: epoch ${fr.checkpoint.epoch}, validation ${f3(fr.checkpoint.validation)}`;
    const wait = stopped ? `3 of 3 epochs without a new best: stop at epoch ${run.stopEpoch}` : `${fr.staleChecks} of 3 epochs without a new best`;
    if (narrow) { text(ctx, best, left, 22, colors.accent.hex, 10, 'left'); text(ctx, wait, left, 40, stopped ? colors.neg.hex : colors.text.hex, 10, 'left'); }
    else { text(ctx, best, left, 26, colors.accent.hex, 11, 'left'); text(ctx, wait, right, 26, stopped ? colors.neg.hex : colors.text.hex, 11, 'right'); }
  }

  function renderAugment(ctx: CanvasRenderingContext2D, fr: ReturnType<typeof frameAt>, w: number, h: number) {
    const a = fr.augmentation, narrow = w < 500, cell = narrow ? 50 : 72, gap = 5, top = narrow ? 110 : 84, xA = w * 0.27, xB = w * 0.73;
    const gx = (cx: number, c: number) => cx + (c - 1) * (cell + gap), gy = (r: number) => top + r * (cell + gap);
    const grid = (cx: number, faint: boolean) => {
      for (let r = 0; r < 2; r++) for (let c = 0; c < 2; c++) {
        ctx.fillStyle = colors.bg.hex; ctx.strokeStyle = faint ? colors.rule.hex : colors.muted.hex; ctx.setLineDash(faint ? [3, 3] : []);
        ctx.beginPath(); ctx.roundRect(gx(cx, c), gy(r), cell, cell, 3); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
      }
    };
    const bright = (cx: number, col: number) => { ctx.fillStyle = colors.fwd.hex; ctx.beginPath(); ctx.roundRect(gx(cx, 0) + col * (cell + gap), gy(0), cell, cell, 3); ctx.fill(); };
    text(ctx, `Label rule: ${a.task === 'presence' ? 'is any pixel bright?' : 'which side is the bright pixel on?'}`, w / 2, 40, colors.text.hex, narrow ? 11 : 13, 'center', bodyFont);
    grid(xA, false); bright(xA, 0);
    text(ctx, 'original', xA, gy(2) + 12, colors.muted.hex, 10);
    grid(xB, fr.flip === 0);
    if (fr.flip > 0) { bright(xB, fr.flip); text(ctx, 'flipped left to right', xB, gy(2) + 12, colors.muted.hex, 10); }
    const ay = gy(1) + gap / 2;
    ctx.strokeStyle = colors.muted.hex; ctx.beginPath(); ctx.moveTo(xA + cell + 18, ay); ctx.lineTo(xB - cell - 18, ay); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(xB - cell - 18, ay); ctx.lineTo(xB - cell - 25, ay - 4); ctx.moveTo(xB - cell - 18, ay); ctx.lineTo(xB - cell - 25, ay + 4); ctx.stroke();
    const ly = gy(2) + 44;
    const label = (s: string, cx: number) => {
      if (!narrow) return text(ctx, `label: ${s}`, cx, ly, colors.fwd.hex, 12);
      text(ctx, 'label', cx, ly - 8, colors.muted.hex, 9); text(ctx, s, cx, ly + 7, colors.fwd.hex, 10);
    };
    label(a.labelBefore, xA);
    if (fr.showAfterLabel) {
      label(a.labelAfter, xB);
      if (a.labelPreserved) text(ctx, 'same label as before: this flip is a safe new example', w / 2, ly + 34, colors.text.hex, narrow ? 10 : 11);
      else text(ctx, `copying the old label would be wrong: ${a.wrongLabel}`, w / 2, ly + 34, colors.neg.hex, narrow ? 10 : 11);
    }
  }

  function render() {
    const rect = cv.getBoundingClientRect();
    if (!rect.width || !rect.height) return false;
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    const W = Math.round(rect.width * dpr), H = Math.round(rect.height * dpr);
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const ctx = cv.getContext('2d'); if (!ctx) return false;
    const w = rect.width < 500 ? 360 : 720, h = 410;
    ctx.setTransform(cv.width / w, 0, 0, cv.height / h, 0, 0); ctx.clearRect(0, 0, w, h);
    const fr = frameAt(t, init.mode, keep, task);
    if (init.mode === 'mask') renderNetwork(ctx, fr, w, h);
    else if (init.mode === 'scaling' || init.mode === 'ensemble') renderMaskChart(ctx, fr, w, h, init.mode === 'scaling');
    else if (init.mode === 'decay') renderDecay(ctx, fr, w, h);
    else if (init.mode === 'early-stop') renderEarlyStop(ctx, fr, w, h);
    else renderAugment(ctx, fr, w, h);
    uiT = t; tl?.paint(t);
    return playing;
  }

  $effect(() => { keep; task; playing; speed; wake(); });

  onMount(() => {
    const reduced = prefersReducedMotion();
    const read = () => readColors({ fwd: '--viz-forward', grad: '--viz-gradient', neg: '--viz-neg', text: '--text', muted: '--text-muted', rule: '--rule', raised: '--bg-raised', bg: '--bg', accent: '--accent' });
    colors = read(); codeFont = readVar('--font-code'); bodyFont = readVar('--font-body');
    const loop = createLoop(cv.closest('[data-figure]') ?? cv, (dt) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      return render();
    }, { onView: (inView, first) => { if (inView && first && !reduced.matches) playing = true; } });
    wake = loop.wake;
    const offTheme = onThemeChange(() => { colors = read(); codeFont = readVar('--font-code'); bodyFont = readVar('--font-body'); wake(); });
    const onReduce = () => { if (reduced.matches) playing = false; wake(); };
    reduced.addEventListener('change', onReduce);
    const ro = new ResizeObserver(() => wake()); ro.observe(cv.parentElement!);
    document.fonts.ready.then(() => wake());
    render();
    return () => { loop.destroy(); offTheme(); reduced.removeEventListener('change', onReduce); ro.disconnect(); };
  });

  function seek(to: number) { playing = false; t = to; uiT = to; wake(); }
  function reset() { seek(0); }
  const dash = '—';
</script>

<FigureCard title={init.title} legend={LEGENDS[init.mode]} role="group" aspect="720 / 410" narrowAspect="360 / 410" label={LABELS[init.mode]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} {markers}
      label="Dropout and regularization timeline" onSeek={(to) => seek(to)} onStep={seek} onReset={reset} />
  {/snippet}
  {#snippet readout()}
    <p class="say" aria-live="polite" role="status">{narration}</p>
    {#if init.mode === 'mask'}
      <div class="contributions" aria-label="How each hidden unit adds to the output">
        {#each current.units as unit, i (i)}
          <div class:off={!unit.kept}><strong>h{i + 1} {unit.kept ? 'kept' : 'dropped'}</strong><span class="num">{f3(unit.activation)} × weight {unit.weight.toFixed(2)} = {f3(unit.contribution)}</span></div>
        {/each}
      </div>
      <p class="sum">Add the four contributions: output <strong>{f3(current.network.output)}</strong>{current.inference ? '' : ` (the full network gives ${f3(current.full)})`}.</p>
    {:else if init.mode === 'scaling'}
      <p class="worked">Take unit h1. In the full network its activation is {f3(h1)}. If it is kept with probability {keep.toFixed(2)} and simply zeroed otherwise, its average is {keep.toFixed(2)} × {f3(h1)} = {f3(keep * h1)}. Divide it by {keep.toFixed(2)} whenever it is kept, and the average is back to {f3(h1)}.</p>
      <p class="readout">Average over the 16 masks · not scaled {current.showUnscaledMean ? f3(current.unscaledMean) : dash} · scaled {current.showInvertedMean ? f3(current.weightedMean) : dash} · full network {current.showFull ? f3(current.full) : dash}</p>
    {:else if init.mode === 'ensemble'}
      <p class="readout">Weighted mean of the 16 outputs {current.showMean ? f3(current.weightedMean) : dash} · full network {current.showMean ? f3(current.full) : dash}</p>
      <p class="readout">After a sigmoid · average {current.showSigmoid ? f3(current.nonlinearMean) : dash} · full network {current.showSigmoid ? f3(current.sigmoidFull) : dash}</p>
      <details class="table-wrap">
        <summary>Table: all 16 masks</summary>
        <table>
          <caption>Probability and output of each mask at keep probability {keep.toFixed(2)}</caption>
          <thead><tr><th scope="col">Mask</th><th scope="col">Probability</th><th scope="col">Output</th><th scope="col">Probability × output</th></tr></thead>
          <tbody>
            {#each current.masks as mask (mask.index)}
              <tr><th scope="row">{mask.mask.join('')}</th><td>{mask.probability.toFixed(4)}</td><td>{mask.output.toFixed(4)}</td><td>{(mask.probability * mask.output).toFixed(4)}</td></tr>
            {/each}
          </tbody>
        </table>
      </details>
    {:else if init.mode === 'decay'}
      {@const d = current.decay}
      {@const vals = [['SGD', d.sgd], ['SGD + L2', d.sgdL2], ['Adam + L2', d.adamCoupled], ['AdamW', d.adamW]] as const}
      <p class="readout">Weight after one update · {#each vals as [name, v], i (name)}{i ? ' · ' : ''}{name} {i < current.decayRows ? f3(v) : dash}{/each}</p>
    {:else if init.mode === 'early-stop'}
      {@const point = current.run.history[current.visibleEpoch]}
      <p class="readout">Epoch {point.epoch} · training loss {f3(point.train)} · validation loss {f3(point.validation)}</p>
      <p class="readout">Best so far: epoch {current.checkpoint.epoch} · {current.staleChecks} of 3 epochs without a new best</p>
    {:else}
      <p class="readout">Before the flip: {current.augmentation.labelBefore} · after: {current.showAfterLabel ? current.augmentation.labelAfter : dash}</p>
    {/if}
  {/snippet}
  {#snippet footer()}
    <p class="try"><strong>Try this.</strong> {init.mode === 'mask' ? 'Pause during training and drag the timeline slowly. Each new example gets a new mask, and the output jumps around. Then drag into inference, where all four units come back.' : init.mode === 'scaling' ? 'Lower the keep probability to 0.25. The grey average drops to a quarter of the full network, while the teal average stays on it.' : init.mode === 'ensemble' ? 'Raise the keep probability to 0.9. Masks with many kept units grow, the rest shrink, and the weighted mean still lands on the full network.' : init.mode === 'early-stop' ? 'Drag to epoch 4, the lowest validation loss. Then drag forward and watch training loss keep falling while validation loss climbs.' : init.mode === 'augment' ? 'Switch the label rule to "Left or right" and play the flip again.' : 'Click each marker on the timeline and compare how far the weight moves from 1.00.'}</p>
    {#if init.mode === 'mask' || init.mode === 'scaling' || init.mode === 'ensemble'}
      <div class="ctl">
        <label class="keep">Keep probability
          <input type="range" min="0.1" max="1" step="0.05" bind:value={keep} aria-label="Dropout keep probability" />
          <output>{keep.toFixed(2)}</output>
        </label>
      </div>
    {:else if init.mode === 'augment'}
      <div class="ctl">
        <fieldset class="seg">
          <legend>Label rule</legend>
          <label><input type="radio" name="augment-{uid}" value="presence" bind:group={task} /><span>Bright pixel present</span></label>
          <label><input type="radio" name="augment-{uid}" value="side" bind:group={task} /><span>Left or right</span></label>
        </fieldset>
      </div>
    {/if}
  {/snippet}
</FigureCard>

<style>
  .sum, .worked, .try { line-height: 1.6; margin: var(--s-2) 0; }
  .worked { padding: var(--s-3); background: var(--bg-sunken); border-left: 2px solid var(--viz-forward); }
  .try { padding: var(--s-3) var(--s-4) 0; }
  .contributions { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: var(--s-1); margin-block: var(--s-3); }
  .contributions > div { display: flex; flex-wrap: wrap; gap: var(--s-2); padding: var(--s-2); border-left: 2px solid var(--viz-forward); background: var(--bg-sunken); font-size: var(--step--1); }
  .contributions strong { min-width: 6.5em; }
  .contributions .off { border-left-style: dashed; border-left-color: var(--text-muted); color: var(--text-muted); }
  .num { font-family: var(--font-code); font-variant-numeric: tabular-nums; }
  .sum strong { color: var(--accent); font-family: var(--font-code); }
  .say, .readout { margin: 0 0 var(--s-2); }
  .say { font: 0.92rem / 1.45 var(--font-ui); }
  .readout { font: var(--step--1) / 1.5 var(--font-code); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
  .table-wrap { margin-top: var(--s-2); }
  table { width: 100%; border-collapse: collapse; font: 0.75rem / 1.4 var(--font-code); font-variant-numeric: tabular-nums; }
  caption { text-align: left; color: var(--text-muted); padding: var(--s-2) 0; }
  th, td { text-align: right; padding: 0.25rem 0.35rem; border-bottom: 1px solid var(--rule); }
  th:first-child { text-align: left; }
  thead th { color: var(--text-muted); }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-3) var(--s-5); padding: var(--s-3) var(--s-4); }
  .ctl label { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .ctl input[type='range'] { width: min(13rem, 38vw); accent-color: var(--accent); }
  .ctl output { min-width: 3.4rem; color: var(--text); font: var(--step--1) var(--font-code); font-variant-numeric: tabular-nums; }
  .seg { display: inline-flex; flex-wrap: wrap; gap: var(--s-2); border: 1px solid var(--rule); border-radius: var(--r-2); padding: var(--s-2); margin: 0; }
  .seg legend { padding-inline: var(--s-1); color: var(--text-muted); font-size: 0.8rem; }
  .seg label { position: relative; margin: 0; }
  .seg input { position: absolute; opacity: 0; inset: 0; }
  .seg span { display: block; padding: 0.35rem 0.5rem; cursor: pointer; border-radius: var(--r-1); }
  .seg input:checked + span { background: var(--accent-soft); color: var(--text); font-weight: 600; }
  .seg input:focus-visible + span { outline: 2px solid var(--accent); outline-offset: 2px; }
  @media (max-width: 40em) { .ctl { gap: var(--s-3); } .keep { width: 100%; } .ctl input[type='range'] { flex: 1; min-width: 6rem; } table { font-size: 0.65rem; } th, td { padding: 0.2rem 0.15rem; } }
</style>
