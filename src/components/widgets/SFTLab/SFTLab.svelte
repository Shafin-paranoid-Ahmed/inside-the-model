<script lang="ts">
  // Day 12 figure shell on the Day 11 shell (../_rl/Shell.svelte): one lazily loaded chunk per mode (m-<mode>.ts) plus
  // its bundled meta (m-<mode>.meta.ts). The mode contract and this widget's helpers are at the top of ./sim.ts.
  import Shell from '../_rl/Shell.svelte';
  import type { Frame, ModeImpl, ModeMeta, WidgetProps } from '../_rl/kit';

  const modes = import.meta.glob<{ default: ModeImpl<unknown, Frame> }>(['./m-*.ts', '!./m-*.meta.ts']);
  const metas = import.meta.glob<ModeMeta>('./m-*.meta.ts', { eager: true, import: 'default' });

  const { mode = 'base', title, aspect, narrowAspect, ...rest }: WidgetProps = $props();
</script>

<Shell name="SFTLab" load={modes[`./m-${mode}.ts`]} meta={metas[`./m-${mode}.meta.ts`]} opts={{ ...rest, mode }} {title} {aspect} {narrowAspect} />
