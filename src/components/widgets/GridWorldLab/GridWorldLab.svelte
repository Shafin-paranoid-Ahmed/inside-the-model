<script lang="ts">
  // Day 11 figure shell: one lazily loaded chunk per mode (m-<mode>.ts) plus its bundled meta (m-<mode>.meta.ts).
  // The mode contract is at the top of ./sim.ts.
  import Shell from '../_rl/Shell.svelte';
  import type { Frame, ModeImpl, ModeMeta, WidgetProps } from '../_rl/kit';

  const modes = import.meta.glob<{ default: ModeImpl<unknown, Frame> }>(['./m-*.ts', '!./m-*.meta.ts']);
  const metas = import.meta.glob<ModeMeta>('./m-*.meta.ts', { eager: true, import: 'default' });

  const { mode = 'loop', title, aspect, narrowAspect, ...rest }: WidgetProps = $props();
</script>

<Shell name="GridWorldLab" load={modes[`./m-${mode}.ts`]} meta={metas[`./m-${mode}.meta.ts`]} opts={{ ...rest, mode }} {title} {aspect} {narrowAspect} />
