import { clustered, buildIVF, buildHNSW, exactTopK, searchIVF, searchHNSW, recallAtK } from '../../../lib/applied/ann';
export { recallAtK };
export const QUERY=[.3,.4];
export function makeData(n=2000,dim=2){const pts=clustered(19,n,dim);return {pts,ivf:buildIVF(pts),graph:buildHNSW(pts)}}
export function run(data:ReturnType<typeof makeData>,mode:string,q=QUERY,probe=1,ef=8){return mode==='exact'?exactTopK(data.pts,q):mode==='ivf'?searchIVF(data.ivf,q,5,probe):searchHNSW(data.graph,q,5,ef);}
export function frameAt(t:number,result:ReturnType<typeof run>){const count=Math.min(result.trace.length,Math.floor(Math.max(0,t)/8*result.trace.length));const seen=result.trace.slice(0,count);return {count,seen,ids:[...new Map(seen.map(v=>[v.id,v])).values()].sort((a,b)=>a.distance-b.distance).slice(0,5).map(v=>v.id),layer:seen.at(-1)?.layer??0};}
export function sweep(){const data=makeData(5000,32),queries=clustered(23,200,32),truth=queries.map(q=>exactTopK(data.pts,q,10).ids);return [1,2,4,8,16].map(setting=>{let ic=0,ir=0,hc=0,hr=0;queries.forEach((q,j)=>{const a=searchIVF(data.ivf,q,10,setting),b=searchHNSW(data.graph,q,10,setting*8);ic+=a.comparisons;ir+=recallAtK(a.ids,truth[j]);hc+=b.comparisons;hr+=recallAtK(b.ids,truth[j]);});return {setting,ivfCost:ic/200,ivfRecall:ir/200,hnswCost:hc/200,hnswRecall:hr/200};});}
