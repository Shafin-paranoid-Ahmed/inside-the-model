<script lang="ts">
 import {onMount} from 'svelte';
 import FigureCard from '../_shared/FigureCard.svelte';
 import Timeline from '../_shared/Timeline.svelte';
 import {createLoop,reducedMotion} from '../_shared/loop';
 import {searchModel,frameAt,QUERY} from './sim';
 import type {Index} from '../../../lib/applied/embed';
 let {mode='keyword',title='Search the invented Elm Street Tool Library handbook'}:{mode?:string;title?:string}=$props();
 let query=$state(QUERY),index=$state.raw<Index>(),t=$state(0),playing=$state(false),speed=$state(.5),host:HTMLDivElement,wake=()=>{};
 const model=$derived(index?searchModel(query,index):null),frame=$derived(frameAt(t));
 $effect(()=>{playing;speed;wake();});
 onMount(()=>{const worker=new Worker(new URL('./worker.ts',import.meta.url),{type:'module'});worker.onmessage=e=>{index=e.data;worker.terminate();};worker.postMessage('index');let live=0,last=0;const loop=createLoop(host,(dt,now)=>{if(playing)live=(live+dt*speed)%8;else live=t;if(now-last>100||!playing){t=live;last=now;}return playing;},{onView:(_,first)=>{if(first&&!reducedMotion())playing=true;}});wake=loop.wake;return()=>{worker.terminate();loop.destroy();};});
 const label=(id:string)=>index?.passages.find(p=>p.id===id)?.title??id;
</script>
<div bind:this={host}>
<FigureCard {title} label="Actual handbook search scores" role="group" aspect={mode==='embed'?'2 / 1':'1.6 / 1'} narrowAspect={mode==='embed'?'0.85 / 1':'0.7 / 1'}>
 <div class="scene">
 {#if !model}<p>Building the text vectors…</p>
 {:else if mode==='embed'}
 <p><b>16 coordinates for this question</b> · {index?.vocab.length} vocabulary words</p>
 <div class="coordinates">{#each Array.from(model.v) as value,i}<div class="coordinate"><span>{i+1}</span><div class="track"><i style:transform="scaleX({i<frame.shown?Math.abs(value)/Math.max(...Array.from(model.v,Math.abs)):0})" style:background="var({value<0?'--viz-neg':'--viz-pos'})"></i></div><span>{i<frame.shown?value.toFixed(2):'…'}</span></div>{/each}</div>
 <p>Similar usage in the background sentences gives related words shared directions. These are LSA vectors, not neural embeddings.</p>
 {:else if mode==='similarity'}
 <p><b>Same passage, twice the vector length</b></p>
 <p>Question and “{label(model.meaning[0].id)}”</p>
 {@const d=index!.docVecs[index!.passages.findIndex(p=>p.id===model.meaning[0].id)]}
 {@const raw=model.raw.find(p=>p.id===model.meaning[0].id)!}
 <div class="math">Dot product: {raw.score.toFixed(3)} × {t<4?'1':'2'} = {(raw.score*(t<4?1:2)).toFixed(3)}<br/>Passage length: {raw.norm.toFixed(3)} × {t<4?'1':'2'} = {(raw.norm*(t<4?1:2)).toFixed(3)}<br/>Cosine stays {raw.cos.toFixed(3)}</div>
 <div class="track big"><i style:transform="scaleX({t<4?.5:1})"></i></div>
 <p>Only the vector's length changes. Its direction and cosine do not.</p>
 {:else}
 <div class="lists">{#each [{name:'Keyword · BM25',rows:model.words},{name:'Embedding · cosine',rows:model.meaning}] as list}<section><b>{list.name}</b>{#each list.rows.slice(0,7) as row,i}<div class:focus={row.id==='overdue'} class="row"><span>{i+1}. {label(row.id)}</span><span>{row.score.toFixed(3)}</span><div class="track"><i style:transform="scaleX({Math.min(1,t/3)*Math.max(0,row.score)/(list.rows[0].score||1)})"></i></div></div>{/each}</section>{/each}</div>
 {/if}
 </div>
 {#snippet controls()}<Timeline bind:t bind:playing bind:speed duration={8} markers={mode==='similarity'?[{at:0,label:'Original'},{at:4,label:'Double length'}]:[{at:0,label:'Words'},{at:3,label:'Compare'},{at:6,label:'Rank'}]} onSeek={v=>{t=v;wake();}} onReset={()=>{t=0;wake();}} />{/snippet}
 {#snippet readout()}<p>{mode==='keyword'?'The outline marks the general overdue rule in both rankings.':mode==='embed'?'Coordinates are revealed one by one; each number is the actual computed coordinate.':'The bar shows passage length; multiplying by 2 doubles dot product too.'}</p>{/snippet}
 {#snippet footer()}<label class="query">Question <input value={query} oninput={e=>{query=e.currentTarget.value;t=0;wake();}} /></label>{/snippet}
</FigureCard>
</div>
<style>
.scene{position:absolute;inset:0;padding:.5rem;overflow:auto;font-family:var(--font-body);font-size:.94rem}.scene p{margin:.4rem 0 1rem}.lists{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.row{display:grid;grid-template-columns:1fr auto;gap:.2rem;padding:.42rem .25rem;border-bottom:1px solid var(--rule);font-size:.9rem}.focus{outline:2px solid var(--accent);outline-offset:-2px}.track{height:6px;background:var(--bg-sunken);grid-column:1/-1;overflow:hidden}.track i{display:block;height:100%;background:var(--viz-forward);transform-origin:left;transition:transform .1s linear}.big{height:24px;margin:1rem 0}.coordinates{display:grid;grid-template-columns:1fr 1fr;gap:.4rem 1rem}.coordinate{display:grid;grid-template-columns:1.4rem 1fr 3rem;align-items:center;gap:.4rem}.coordinate .track{grid-column:auto;height:10px}.math{font-size:1.1rem;line-height:2;padding:1rem 0}.query{display:grid;gap:.4rem;padding:1rem}.query input{width:100%;min-width:0;padding:.5rem;background:var(--bg);border:1px solid var(--rule);color:var(--text)}@container (width < 540px){.lists{gap:.6rem}.row{font-size:.8rem}.coordinates{grid-template-columns:1fr}.scene{font-size:.9rem}}
</style>

