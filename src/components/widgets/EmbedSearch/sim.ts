import { handbookIndex, semantic, bm25, embedText, dot, cosine } from '../../../lib/applied/embed';
export const QUERY='what happens if I bring the drill back late';
export function searchModel(q=QUERY,index=handbookIndex()){const v=embedText(index,q);return {index,v,words:bm25(index,q),meaning:semantic(index,q),raw:index.docVecs.map((d,i)=>({id:index.passages[i].id,score:dot(v,d),norm:Math.sqrt(dot(d,d)),cos:cosine(v,d)})).sort((a,b)=>b.score-a.score)};}
export function frameAt(t:number,n=16){return {shown:Math.min(n,Math.floor(Math.max(0,t)/8*n)+1),phase:t<3?'Read the words':t<6?'Compare numbers':'Rank the passages'};}

