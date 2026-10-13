import { createRng } from '../math/rng';
export type Point=number[];
export const distance=(a:Point,b:Point)=>a.reduce((s,x,j)=>s+(x-b[j])**2,0);
export function clustered(seed=19,n=2000,dim=2):Point[] {const rng=createRng(seed),centers=Array.from({length:12},()=>Array.from({length:dim},()=>rng.next()*8-4));return Array.from({length:n},()=>{const c=centers[Math.floor(rng.next()*centers.length)];return c.map(v=>v+rng.normal(0,1));});}
export interface Visit {id:number; layer:number; distance:number}
export interface Search {ids:number[]; comparisons:number; trace:Visit[]}
export function exactTopK(pts:Point[],q:Point,k=5):Search {const trace=pts.map((p,id)=>({id,layer:0,distance:distance(p,q)}));return {ids:[...trace].sort((a,b)=>a.distance-b.distance).slice(0,k).map(x=>x.id),comparisons:pts.length,trace};}
export interface IVF {pts:Point[]; centers:Point[]; lists:number[][]}
export function buildIVF(pts:Point[],nlist=16,seed=7):IVF {
 const rng=createRng(seed),centers=[pts[Math.floor(rng.next()*pts.length)].slice()];
 while(centers.length<nlist){const ds=pts.map(p=>Math.min(...centers.map(c=>distance(p,c)))),sum=ds.reduce((a,b)=>a+b,0);let r=rng.next()*sum,i=0;while(i<ds.length-1 && (r-=ds[i])>0)i++;centers.push(pts[i].slice());}
 let lists:number[][]=[];
 for(let step=0;step<12;step++){lists=centers.map(()=>[]);pts.forEach((p,i)=>{let best=0;for(let j=1;j<centers.length;j++)if(distance(p,centers[j])<distance(p,centers[best]))best=j;lists[best].push(i);});if(step<11)lists.forEach((ids,j)=>{if(ids.length)centers[j]=centers[j].map((_,d)=>ids.reduce((s,i)=>s+pts[i][d],0)/ids.length);});}
 return {pts,centers,lists};
}
export function searchIVF(g:IVF,q:Point,k=5,nprobe=1):Search {const cells=g.centers.map((c,id)=>({id,d:distance(c,q)})).sort((a,b)=>a.d-b.d).slice(0,nprobe);const trace=cells.flatMap(c=>g.lists[c.id].map(id=>({id,layer:0,distance:distance(g.pts[id],q)})));return {ids:[...trace].sort((a,b)=>a.distance-b.distance).slice(0,k).map(x=>x.id),comparisons:g.centers.length+trace.length,trace};}
export interface HNSW {pts:Point[]; levels:number[]; edges:number[][][]; entry:number; M:number}
function layerSearch(g:HNSW,q:Point,entry:number,layer:number,ef:number,trace:Visit[]):number[] {
 const seen=new Set([entry]),first={id:entry,d:distance(g.pts[entry],q)},todo=[first],best=[first];trace.push({id:entry,layer,distance:first.d});
 while(todo.length){todo.sort((a,b)=>a.d-b.d);const cur=todo.shift()!;best.sort((a,b)=>a.d-b.d);if(best.length>=ef && cur.d>best[best.length-1].d)break;
 for(const id of g.edges[layer]?.[cur.id]??[]){if(seen.has(id))continue;seen.add(id);const d=distance(g.pts[id],q);trace.push({id,layer,distance:d});if(best.length<ef || d<best[best.length-1].d){todo.push({id,d});best.push({id,d});best.sort((a,b)=>a.d-b.d);if(best.length>ef)best.pop();}}}return best.sort((a,b)=>a.d-b.d).map(x=>x.id);
}
export function buildHNSW(pts:Point[],M=6,seed=17):HNSW {
 const rng=createRng(seed),levels=pts.map(()=>Math.min(5,Math.floor(-Math.log(Math.max(1e-9,rng.next()))/Math.log(M)))),g:HNSW={pts,levels,edges:Array.from({length:6},()=>pts.map(()=>[])),entry:0,M};
 for(let i=1;i<pts.length;i++){let ep=g.entry;for(let l=levels[ep];l>levels[i];l--)ep=layerSearch(g,pts[i],ep,l,1,[])[0];for(let l=Math.min(levels[i],levels[g.entry]);l>=0;l--){const near=layerSearch(g,pts[i],ep,l,Math.max(32,M*4),[]).slice(0,M);g.edges[l][i]=near;for(const j of near){g.edges[l][j].push(i);g.edges[l][j].sort((a,b)=>distance(pts[j],pts[a])-distance(pts[j],pts[b]));g.edges[l][j]=g.edges[l][j].slice(0,l===0?M*2:M);}ep=near[0]??ep;}if(levels[i]>levels[g.entry])g.entry=i;}
 return g;
}
export function searchHNSW(g:HNSW,q:Point,k=5,ef=8):Search {const trace:Visit[]=[];let ep=g.entry;for(let l=g.levels[g.entry];l>0;l--)ep=layerSearch(g,q,ep,l,1,trace)[0];const ids=layerSearch(g,q,ep,0,Math.max(k,ef),trace).slice(0,k);return {ids,comparisons:trace.length,trace};}
export const recallAtK=(found:number[],truth:number[])=>truth.length?found.filter(id=>truth.includes(id)).length/truth.length:1;

