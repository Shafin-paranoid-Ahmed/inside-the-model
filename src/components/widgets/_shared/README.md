# Widget building blocks

Import TS helpers from `'../_shared'`, components from `'../_shared/FigureCard.svelte'` / `Timeline.svelte`.
Rules: `LESSONS.md` §2. Reference implementation: `../BackpropViz/`.

| Piece | API |
|---|---|
| `FigureCard` | props `title`, `label` (stage aria-label), `legend?: {label, color: '--viz-forward', shape?: 'dot'\|'line'}[]`, `aspect?: '16 / 9'`, `narrowAspect?` (stage < 540px), `role?: 'img'\|'group'` (use `group` if the stage holds buttons), `asideLabel?`. Snippets: `children` (stage: `<canvas>`/`<svg>` fill the box), `controls?`, `readout?`, `aside?`, `footer?` |
| `Timeline` | `bind:t` `bind:playing` `bind:speed` (0.5/1/2), `duration`, `markers?: {at, label}[]`, `label?`, `onSeek?(t, smooth)`, `onStep?(to)` (shows Step), `onReset?` (shows Reset). `tl.paint(t)` = cheap per-frame thumb update via `bind:this` |
| `createLoop(el, frame, {maxDt?, onView?})` | `frame(dt_s, now_ms)`; return `false` to idle; returns `{ wake(), destroy() }`. Runs only in view + tab visible |
| `readColors({k: '--css-var'})` | `{k: {hex, rgb}}`; redo inside `onThemeChange(cb)` (returns unsubscribe). `readVar('--font-code')` for fonts |
| `advance(state, dt, {duration, loopFrom?, loopTo?})` | pure; loops in range; `next.t < s.t` means a cycle ended. `nextMarker`, `markerAt` |
| `reducedMotion()`, `prefersReducedMotion()` | boolean / MediaQueryList |

## Minimal widget

`Pulse/sim.ts` (pure, no DOM):
```ts
export const T = 6;
export const MARKERS = [{ at: 0, label: 'rise' }, { at: 3, label: 'fall' }];
/** Everything needed to draw at time t. Scrubbing is free. */
export const frameAt = (t: number) => ({ v: t < 3 ? t / 3 : 1 - (t - 3) / 3, phase: t < 3 ? 'rise' : 'fall' });
```

`Pulse/Pulse.svelte`:
```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { createLoop, readColors, onThemeChange, reducedMotion, advance } from '../_shared';
  import { T, MARKERS, frameAt } from './sim';

  let { title = 'A pulse' }: { title?: string } = $props();
  let playing = $state(false), speed = $state(0.5), uiT = $state(0); // uiT: readout only, ~10x/s
  let t = 0;                                   // truth for the canvas: plain variable
  let cv: HTMLCanvasElement, tl: ReturnType<typeof Timeline>, wake = () => {};
  $effect(() => { playing; speed; wake(); });  // any control change restarts an idle loop

  onMount(() => {
    const ctx = cv.getContext('2d')!;
    let c = readColors({ fwd: '--viz-forward' }), lastUi = 0;
    const draw = () => {
      const dpr = Math.min(2, devicePixelRatio || 1), r = cv.getBoundingClientRect();
      cv.width = r.width * dpr; cv.height = r.height * dpr;
      const { v } = frameAt(t);
      ctx.fillStyle = c.fwd.hex; ctx.fillRect(0, 0, cv.width * v, cv.height);
    };
    const loop = createLoop(cv, (dt, now) => {
      t = advance({ t, playing, speed }, dt, { duration: T }).t;
      draw(); tl.paint(t);
      if (now - lastUi > 100 || !playing) { lastUi = now; uiT = t; }
      return playing;                          // false → idle until wake()
    }, { onView: (_in, first) => { if (first && !reducedMotion()) playing = true; } });
    wake = loop.wake;
    const off = onThemeChange(() => { c = readColors({ fwd: '--viz-forward' }); wake(); });
    return () => { loop.destroy(); off(); };
  });
</script>

<FigureCard {title} label="A bar that fills then empties" aspect="3 / 1" legend={[{ label: 'value', color: '--viz-forward' }]}>
  <canvas bind:this={cv} aria-hidden="true"></canvas>
  {#snippet controls()}
    <Timeline bind:this={tl} bind:t={uiT} bind:playing bind:speed duration={T} markers={MARKERS}
      onSeek={(to) => { t = to; wake(); }} onStep={(to) => { t = to; wake(); }} onReset={() => { t = 0; wake(); }} />
  {/snippet}
  {#snippet readout()}<p aria-live="polite">Phase: {frameAt(uiT).phase}</p>{/snippet}
</FigureCard>
```

MDX: `import Pulse from '../../../components/widgets/Pulse/Pulse.svelte';` then `<Pulse client:visible title="…" />`.

Notes: Timeline pauses playback on any user seek. Do not write `$state` from the frame loop more than ~10×/s. Never bind ←/→. Ease jumps yourself if you want (BackpropViz `glide`).
