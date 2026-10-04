<script lang="ts">
  import { markerAt, nextMarker, type Marker } from './clock';

  interface Props {
    /** Playhead in seconds. Parent writes it at most ~10x/s (the canvas keeps its own copy). */
    t: number;
    playing: boolean;
    /** Speed multiplier: 0.5 / 1 / 2. */
    speed?: number;
    duration: number;
    /** Labeled phase ticks on the scrubber; click to seek. */
    markers?: Marker[];
    /** aria-label of the scrubber. */
    label?: string;
    /** Called after a user seek. `smooth` is true for tick clicks (ease there if you like), false while dragging. Default: just set `t`. */
    onSeek?: (t: number, smooth: boolean) => void;
    /** Shows a secondary Step button; called with the next marker time (or `duration`). */
    onStep?: (to: number) => void;
    /** Shows a Reset button. */
    onReset?: () => void;
  }
  let { t = $bindable(0), playing = $bindable(false), speed = $bindable(1), duration, markers = [], label = 'Timeline', onSeek, onStep, onReset }: Props = $props();

  const clamp = (v: number) => Math.min(Math.max(v, 0), duration);
  const shown = $derived(clamp(t));
  const cur = $derived(markerAt(shown, markers));

  let scrub: HTMLInputElement;
  let track: HTMLElement;
  let dragging = false, lastV = '';

  /**
   * Cheap direct DOM update of the thumb and fill. Call it every frame with the live `t`,
   * and set the bindable `t` only ~10x/s for the text readout.
   */
  export function paint(v: number) {
    const c = clamp(v), s = c.toFixed(2);
    if (s === lastV) return;
    lastV = s;
    if (!dragging) scrub.value = s;
    track.style.setProperty('--p', `${(c / duration) * 100}%`);
  }

  function onInput() {
    playing = false;
    t = +scrub.value;
    onSeek?.(t, false);
  }
  function seekTo(to: number, smooth: boolean) {
    playing = false;
    if (onSeek) onSeek(to, smooth); else t = to;
  }
</script>

<svelte:window onpointerup={() => (dragging = false)} onpointercancel={() => (dragging = false)} />

<div class="tl">
  <button type="button" class="btn btn-primary play" onclick={() => (playing = !playing)}>{playing ? 'Pause' : 'Play'}</button>
  <div class="track" bind:this={track} style:--p="{(shown / duration) * 100}%">
    <input bind:this={scrub} class="scrub" type="range" min="0" max={duration} step="0.01" value={shown}
      aria-label={label} aria-valuetext="{cur ? cur.label + ', ' : ''}t = {shown.toFixed(2)} seconds"
      onpointerdown={() => (dragging = true)} oninput={onInput} />
    <div class="ticks">
      {#each markers as m (m.at)}
        <button type="button" class="tick" class:on={cur === m} class:edge={m.at / duration > 0.8} style:left="{(m.at / duration) * 100}%" onclick={() => seekTo(m.at, true)}>{m.label}</button>
      {/each}
    </div>
  </div>
  <p class="tro num">t = <span>{shown.toFixed(2)}</span> s <span class="phn">{cur?.label ?? ''}</span></p>
  <label class="spd">Speed
    <select bind:value={speed}>
      <option value={0.5}>Slow 0.5×</option>
      <option value={1}>Normal</option>
      <option value={2}>Fast 2×</option>
    </select>
  </label>
  {#if onStep}
    <button type="button" class="btn" title="Glide to the next phase marker" onclick={() => { playing = false; onStep(nextMarker(shown, markers) ?? duration); }}>Step</button>
  {/if}
  {#if onReset}<button type="button" class="btn" onclick={onReset}>Reset</button>{/if}
</div>

<style>
  .tl { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s-2) var(--s-4); padding: var(--s-3) var(--s-4); border-top: 1px solid var(--rule); font-family: var(--font-ui); font-size: var(--step--1); color: var(--text); }
  .tl .btn { padding: 0.45em 0.9em; }
  .play { min-width: 5.4rem; }
  .track { --p: 0%; position: relative; flex: 1 1 15rem; min-width: 14rem; padding-bottom: 1.3rem; }
  .scrub { display: block; width: 100%; height: 1.25rem; margin: 0; background: transparent; appearance: none; -webkit-appearance: none; cursor: pointer; }
  .scrub::-webkit-slider-runnable-track { height: 4px; border-radius: 2px; background: linear-gradient(to right, var(--accent) var(--p), var(--rule) var(--p)); }
  .scrub::-moz-range-track { height: 4px; border-radius: 2px; background: linear-gradient(to right, var(--accent) var(--p), var(--rule) var(--p)); }
  .scrub::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; margin-top: -5px; border-radius: 50%; background: var(--accent); border: 2px solid var(--bg-raised); box-shadow: 0 0 0 1px var(--accent); }
  .scrub::-moz-range-thumb { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); border: 2px solid var(--bg-raised); box-shadow: 0 0 0 1px var(--accent); }
  .scrub:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 4px; }
  .ticks { position: absolute; left: 7px; right: 7px; top: 1.15rem; height: 1.2rem; }
  .tick { position: absolute; top: 0; padding: 0.25rem 0.1rem 0 0.3rem; background: none; border: 0; cursor: pointer; white-space: nowrap; font: 0.72rem var(--font-ui); color: var(--text-muted); }
  .tick::before { content: ''; position: absolute; left: 0; top: -0.2rem; height: 0.5rem; border-left: 1px solid var(--text-muted); }
  .tick.edge { transform: translateX(-100%); padding-left: 0.1rem; padding-right: 0.3rem; }
  .tick.edge::before { left: auto; right: 0; }
  .tick.on { color: var(--text); font-weight: 600; }
  .tick:hover { color: var(--text); }
  .tro { margin: 0; display: inline-flex; gap: 0.5ch; font-size: 0.85rem; white-space: nowrap; }
  .tro span:first-child { display: inline-block; min-width: 4.2ch; text-align: right; }
  .phn { display: inline-block; min-width: 7.5ch; color: var(--text-muted); }
  .tick::first-letter, .phn::first-letter { text-transform: uppercase; }
  .spd { display: inline-flex; align-items: center; gap: var(--s-2); color: var(--text-muted); }
  .spd select { font: inherit; color: var(--text); background: var(--bg-raised); border: 1px solid var(--rule); border-radius: var(--r-1); padding: 0.3em 0.4em; }
</style>
