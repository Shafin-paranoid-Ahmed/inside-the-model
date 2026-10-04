<script lang="ts">
  import type { Snippet } from 'svelte';

  interface LegendItem {
    label: string;
    /** CSS custom property name, e.g. '--viz-forward'. */
    color: string;
    shape?: 'dot' | 'line';
  }
  interface Props {
    title: string;
    legend?: LegendItem[];
    /** Reserved stage aspect ratio, e.g. '720 / 400' (CSS aspect-ratio syntax). Keeps CLS at 0. */
    aspect?: string;
    /** Aspect ratio when the stage is narrower than 540px (defaults to `aspect`). */
    narrowAspect?: string;
    /** aria-label of the stage. */
    label: string;
    /** 'img' for a passive figure; 'group' when the stage contains buttons/inputs. */
    role?: 'img' | 'group';
    /** aria-label of the aside. */
    asideLabel?: string;
    /** The stage: put a <canvas>/<svg> here (it is stretched to fill the reserved box). */
    children: Snippet;
    /** Control bar(s) under the stage, e.g. <Timeline>. */
    controls?: Snippet;
    /** Live caption line(s) under the controls. */
    readout?: Snippet;
    /** Side panel: beside the stage when wide, stacked under it when narrow. */
    aside?: Snippet;
    /** Extra parameter inputs, shown in a sunken bar at the bottom. */
    footer?: Snippet;
  }
  let { title, legend = [], aspect = '16 / 9', narrowAspect, label, role = 'img', asideLabel, children, controls, readout, aside, footer }: Props = $props();
  const uid = $props.id();
</script>

<div class="viz" data-figure role="group" aria-labelledby="{uid}-title">
  <header class="head">
    <p class="ttl" id="{uid}-title">{title}</p>
    {#if legend.length}
      <p class="legend" aria-hidden="true">
        {#each legend as k (k.label)}
          <span class="k {k.shape ?? 'dot'}" style:--c="var({k.color})">{k.label}</span>
        {/each}
      </p>
    {/if}
  </header>

  <div class="grid" class:has-aside={!!aside}>
    <div class="stage">
      <div class="fig" {role} aria-label={label} style:--aspect={aspect} style:--aspect-narrow={narrowAspect ?? aspect}>
        {@render children()}
      </div>
    </div>
    {#if aside}
      <aside class="aside" aria-label={asideLabel}>{@render aside()}</aside>
    {/if}
  </div>

  {@render controls?.()}
  {#if readout}<div class="cap">{@render readout()}</div>{/if}
  {#if footer}<div class="foot">{@render footer()}</div>{/if}
</div>

<style>
  .viz {
    container-type: inline-size;
    width: max(100%, min(60rem, calc(100vw - 3rem - 17px)));
    margin-block: var(--s-5);
    border: 1px solid var(--rule); border-radius: var(--r-2);
    background: var(--bg-raised);
    font-family: var(--font-ui); font-size: var(--step--1); color: var(--text);
  }
  @media (max-width: 40em) { .viz { width: 100%; } }

  .head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: var(--s-1) var(--s-4); padding: var(--s-3) var(--s-4); border-bottom: 1px solid var(--rule); }
  .ttl { margin: 0; font: 650 1.05rem / 1.3 var(--font-ui); }
  .legend { display: flex; flex-wrap: wrap; gap: 0.2rem var(--s-4); margin: 0; font: 0.78rem var(--font-ui); color: var(--text-muted); }
  .k::before { content: ''; display: inline-block; width: 0.7em; height: 0.7em; border-radius: 50%; margin-right: 0.4em; vertical-align: -0.05em; background: var(--c, var(--text-muted)); }
  .k.line::before { width: 1.2em; height: 2px; border-radius: 1px; vertical-align: 0.25em; }

  .grid { display: grid; grid-template-columns: minmax(0, 1fr); }
  .stage { container-type: inline-size; padding: var(--s-3) var(--s-3) var(--s-2); min-width: 0; }
  .fig { position: relative; aspect-ratio: var(--aspect); }
  @container (width < 540px) { .fig { aspect-ratio: var(--aspect-narrow); } }
  .fig :global(canvas), .fig :global(svg) { position: absolute; inset: 0; display: block; width: 100%; height: 100%; }

  .aside { border-top: 1px solid var(--rule); padding: var(--s-3) var(--s-4); height: 13rem; overflow: auto; min-width: 0; }
  @container (min-width: 58rem) {
    .grid.has-aside { grid-template-columns: minmax(0, 1fr) 19rem; }
    .grid.has-aside .aside { border-left: 1px solid var(--rule); border-top: 0; height: auto; contain: size; }
  }

  .cap { padding: var(--s-2) var(--s-4) var(--s-3); border-top: 1px solid var(--rule); min-height: 4.6rem; }
  .foot { border-top: 1px solid var(--rule); background: var(--bg-sunken); border-radius: 0 0 var(--r-2) var(--r-2); }
</style>
