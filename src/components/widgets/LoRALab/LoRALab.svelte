<script module lang="ts">
  let pending: Promise<Run> | undefined;
  function loadRun(): Promise<Run> {
    return pending ??= new Promise((resolve, reject) => {
      const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = ({ data }) => { worker.terminate(); data.error ? reject(new Error(data.error)) : resolve(data.run); };
      worker.onerror = () => { worker.terminate(); reject(new Error('Training could not load. Reload to retry.')); };
      worker.postMessage(null);
    });
  }
</script>
<script lang="ts">
  import { onMount } from 'svelte';
  import FigureCard from '../_shared/FigureCard.svelte';
  import Timeline from '../_shared/Timeline.svelte';
  import { advance, createLoop, reducedMotion } from '../_shared';
  import * as S from './sim';
  import type { Run, Mode, Rank, Curve, Scheme } from './sim';
  let { title, mode = 'train', r = 2 }: { title: string; mode?: Mode; r?: Rank } = $props();
  let run = $state.raw<Run>();
  let error = $state(''), playing = $state(false), speed = $state(0.5), t = $state(0);
  let rank = $state<Rank>(2), model = $state('7B'), dense = $state(false), adapterC = $state(false), scheme = $state<Scheme>('nf4');
  let host: HTMLDivElement, wake = () => {};
  const W = S.base();
  let time = $derived(S.modeTime(mode, rank));
  let ck = $derived(S.checkpointAt(t));
  let task = $derived(run && (dense ? run.fullrank : run.low));
  let curve = $derived(task?.lora[rank]);
  let q = $derived(S.quantize(W, scheme));
  let preset = $derived(S.PRESETS.find(p => p.id === model)!);
  let merge = $derived.by(() => { const c = adapterC ? run?.c : run?.low.lora[4]; return c ? S.mergeCheck(W, c.B!, c.A!, 4) : undefined; });
  let series = $derived.by((): { label: string; curve: Curve; color: string }[] => {
    if (!run) return [];
    if (mode === 'qlora') return [{label:'16-bit base + adapter',curve:run.qlora.fp,color:'--viz-cat-1'}, {label:'NF4 base + adapter',curve:run.qlora.nf4,color:'--viz-cat-2'}];
    if (mode === 'data') { const c = run.data[S.dataFrame(t).i].curve; return [{label:'Training examples',curve:c,color:'--viz-cat-1'},{label:'Fresh examples',curve:c,color:'--viz-cat-2'}]; }
    return [{label:'Full: 1024 parameters',curve:run.low.full,color:'--viz-cat-1'},...S.RANKS.map((r,i)=>({label:`Rank ${r}: ${64*r} parameters`,curve:run.low.lora[r],color:`--viz-cat-${i+2}`}))];
  });
  let index = $derived(mode === 'data' ? S.dataFrame(t).update / S.EVERY : ck);
  const lossValues = (c: Curve, i: number) => mode === 'data' && i === 0 ? c.train : c.test;
  const y = (v: number) => 180 - 140 * (Math.log10(Math.max(0.001,v)) + 3) / 3.3;
  const path = (c: Curve, i: number) => Array.from(lossValues(c,i).slice(0,index+1), (v,j)=>`${j?'L':'M'}${35+j*250/60},${y(v)}`).join(' ');
  $effect(() => { playing; speed; rank; model; dense; adapterC; scheme; wake(); });
  onMount(() => {
    rank = r;
    let alive = true;
    if (mode !== 'memory' && mode !== 'quantize') loadRun().then(value => { if(alive) {run=value; wake();} }).catch(e => {if(alive) error=String(e);});
    const loop = createLoop(host, dt => { t = advance({t,playing,speed},dt,{duration:time.dur}).t; return playing; }, {onView:(_visible,first)=>{if(first && !reducedMotion()) playing=true;}});
    wake=loop.wake;
    return () => {alive=false;loop.destroy();};
  });
</script>
<div bind:this={host}>
<FigureCard {title} label={title} aspect="320 / 230" narrowAspect="320 / 230">
  {#if error}<p>{error}</p>
  {:else if mode === 'memory'}
    {@const m = S.memory('full',preset)}
    {@const f = S.memoryFrame(t)}
    <svg viewBox="0 0 320 230" aria-hidden="true">
      {#each [['Weights',m.weights,f.weights],['Gradients',m.grads,f.grads],['Optimizer',m.optimizer,f.optimizer]] as row,i (i)}
        <text x="8" y={30+i*60}>{row[0]}: {Number(row[1]).toFixed(0)} GB</text>
        <rect x="8" y={38+i*60} width={290*Number(row[1])/m.total} height="20" fill={`var(--viz-cat-${i+1})`} opacity={0.2+0.8*Number(row[2])}/>
      {/each}
      <text x="8" y="224">Total model states: {m.total.toFixed(0)} GB</text>
    </svg>
  {:else if mode === 'quantize'}
    {@const j=320}
    {@const f=S.quantFrame(t)}
    <svg viewBox="0 0 320 230" aria-hidden="true">
      <text x="8" y="20">Block 6 · normalized weights</text>
      {#each S.levelsOf(scheme) as l (l)}<line x1={160+l*140} x2={160+l*140} y1="42" y2="170" stroke="var(--rule)"/>{/each}
      {#each Array.from(q.norm.slice(j,j+64)) as v,i (i)}
        <circle cx={160+140*(v+(q.snapped[j+i]-v)*f.slide)} cy={55+(i%8)*14} r="2.5" fill="var(--viz-forward)"/>
      {/each}
      <text x="8" y="195">-1</text><text x="155" y="195">0</text><text x="295" y="195">1</text>
      <text x="8" y="222">Whole-matrix weight MSE: {q.mse.toFixed(6)}</text>
    </svg>
  {:else if !run}<p class="loading">Training the comparison…</p>
  {:else if mode === 'lowrank' && curve}
    {@const f = S.lowrankFrame(t,rank)}
    {@const product = S.correctionFrame(t,curve)}
    <svg viewBox="0 0 320 230" aria-hidden="true">
      <text x="8" y="20">B (32 × {rank}) · A ({rank} × 32)</text>
      {#each Array.from(product) as v,i (i)}
        <rect x={72+(i%32)*5} y={32+Math.floor(i/32)*5} width="4.8" height="4.8" fill={v>=0?'var(--viz-pos)':'var(--viz-neg)'} opacity={Math.min(1,Math.abs(v)/0.65)}/>
      {/each}
      <text x="8" y="214">Layer {f.k+1}/{rank} of 2BA · {64*rank} values</text>
    </svg>
  {:else if mode === 'spectrum' && task}
    <svg viewBox="0 0 320 230" aria-hidden="true">
      <text x="8" y="20">Change directions · log size</text>
      {#each Array.from(task.sv) as v,i (i)}
        <rect x={12+i*9.2} y={175-45*Math.log10(1+v/0.01)} width="7" height={45*Math.log10(1+v/0.01)} fill="var(--viz-forward)" opacity={i<S.spectrumFrame(t).bars?1:0.15}/>
      {/each}
      <text x="8" y="200">Top 2: {(100*S.energyTop(task.sv,2)).toFixed(1)}% squared size</text>
      <text x="8" y="223">Rank 2 fresh MSE: {task.lora[2].test[60].toFixed(3)}</text>
    </svg>
  {:else if mode === 'merge' && merge}
    <svg viewBox="0 0 320 230" aria-hidden="true">
      <text x="8" y="25">Before: Wx + 2B(Ax)</text>
      <text x="8" y="62">After: (W + 2BA)x</text>
      <text x="8" y="106">First output, before: {merge.unmerged[0].toFixed(6)}</text>
      <text x="8" y="135">First output, after: {merge.merged[0].toFixed(6)}</text>
      <rect x="8" y="152" width={300*S.mergeFrame(t).fold} height="10" fill="var(--viz-forward)"/>
      <text x="8" y="191">Largest difference: {merge.maxDiff.toExponential(1)}</text>
      <text x="8" y="222">Multiply-adds: {merge.madUnmerged} ? {merge.madMerged}</text>
    </svg>
  {:else}
    <svg viewBox="0 0 320 230" aria-hidden="true">
      <text x="8" y="18">MSE per output · log scale</text>
      {#each [0.001,0.01,0.1,1] as v (v)}<line x1="35" x2="285" y1={y(v)} y2={y(v)} stroke="var(--rule)"/><text x="1" y={y(v)+4}>{v}</text>{/each}
      {#each series as s,i (s.label)}<path d={path(s.curve,i)} fill="none" stroke={`var(${s.color})`} stroke-width="2"/>{/each}
      <line x1={35+index*250/60} x2={35+index*250/60} y1="25" y2="180" stroke="var(--text-muted)" stroke-dasharray="3 3"/>
      <text x="35" y="200">0</text><text x="248" y="200">300</text>
      <text x="35" y="224">Update {index*5}{mode==='data'?` · ${run.data[S.dataFrame(t).i].n} examples`:''}</text>
    </svg>
  {/if}
  {#snippet controls()}
    <Timeline bind:t bind:playing bind:speed duration={time.dur} markers={time.markers} onSeek={()=>wake()} onReset={()=>{t=0;wake();}} />
  {/snippet}
  {#snippet readout()}
    <div class="readout">
    <p aria-live="polite">{time.markers.filter(m=>m.at<=t).at(-1)?.label}</p>
    {#if mode === 'memory'}<p>2 bytes weights + 2 bytes gradients + 12 bytes optimizer = 16 bytes per parameter. Activations and temporary buffers not counted. 1 GB = 10? bytes.</p>
    {:else if mode==='lowrank' && curve}
      <p>Heat map adds the rank-one layers of the trained correction. Positive and negative changes use their signed colors. Completed first update entry = 2 × ({Array.from({length:rank},(_,k)=>`${curve.B![k].toFixed(4)} × ${curve.A![k*32].toFixed(4)}`).join(' + ')}) = {S.lowRank(curve.B!,curve.A!,rank)[0].toFixed(4)}.</p>
      <p>Base {W[0].toFixed(4)} + update {S.lowRank(curve.B!,curve.A!,rank)[0].toFixed(4)} = adapted {curve.W[0].toFixed(4)}. Entries above are rounded.</p>
    {:else if mode==='quantize'}
      <p>One weight: {W[320].toFixed(5)} ÷ scale {q.absmax[5].toFixed(5)} = {q.norm[320].toFixed(5)} ? nearest level {q.snapped[320].toFixed(5)} ? × {q.absmax[5].toFixed(5)} = {q.deq[320].toFixed(5)}.</p>
      <p>{scheme==='nf4'?'NF4: 16 uneven levels':'Symmetric int4: 15 levels (-7…7), one code unused'}. Dots move to their stored levels; the caption gives the completed conversion.</p>
    {:else if mode==='train' || mode==='qlora' || mode==='data'}
      <p>Loss below 0.001 is drawn on the bottom line.</p>{#each series as s,i (s.label)}<p><span class="key" style:background={`var(${s.color})`}></span>{s.label}: <strong>{lossValues(s.curve,i)[index].toFixed(3)}</strong></p>{/each}
      {#if mode==='qlora' && run}<p>NF4 base without adaptation: {run.qlora.baseNf4.toFixed(3)} MSE.</p><p>{model} model states (rank 16 adapters): {(['full','lora','qlora'] as const).map(s=>`${s} ${S.memory(s,preset).total.toFixed(2)} GB`).join(' · ')}. Activations excluded; quantization scales included.</p>{/if}
    {:else if mode==='spectrum'}<p>A larger bar means the learned change uses that direction more strongly. The top-two share uses squared singular values.</p>
    {:else if mode==='merge'}<p>The same trained rank-4 adapter and input go through both expressions. Folding the adapter removes its extra matrix products.</p>{/if}
    </div>
  {/snippet}
  {#snippet footer()}
    <div class="options">
    {#if mode==='memory' || mode==='qlora'}<label>Model size <select bind:value={model}>{#each S.PRESETS as p (p.id)}<option value={p.id}>{p.label}</option>{/each}</select></label>{/if}
    {#if mode==='lowrank'}<label>Rank <select bind:value={rank}>{#each S.RANKS as rr (rr)}<option value={rr}>{rr}</option>{/each}</select></label>{/if}
    {#if mode==='spectrum'}<label><input type="checkbox" bind:checked={dense}/> Use full-rank task change</label>{/if}
    {#if mode==='merge'}<label><input type="checkbox" bind:checked={adapterC}/> Switch to task C adapter</label>{/if}
    {#if mode==='quantize'}<label>Stored levels <select bind:value={scheme}><option value="nf4">NF4</option><option value="int4">Symmetric int4</option></select></label>{/if}
    </div>
  {/snippet}
</FigureCard>
</div>
<style>
  svg text { fill: var(--text); font: 12px var(--font-ui); font-variant-numeric: tabular-nums; }
  .loading { padding: 2rem; }
  .readout { font-variant-numeric: tabular-nums; min-height: 5rem; overflow-wrap:anywhere; }
  .readout p { margin: .4rem 0; }
  .key { display:inline-block; width:.65rem; height:.65rem; margin-right:.5rem; }
  .options {padding:.7rem 1rem;}
  select {background:var(--bg);color:var(--text);border:1px solid var(--rule);padding:.3rem;}
  input {accent-color:var(--accent);}
</style>

