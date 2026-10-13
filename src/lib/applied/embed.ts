import { eigSym } from '../math/linalg';
import { PASSAGES, BACKGROUND, type Passage } from './corpus';
const STOP = new Set('a an the is are to of for if what how much happens i my can on in it and with does say about handbook'.split(' '));
export const tokenize = (text:string) => (text.toLowerCase().match(/[a-z0-9]+(?:-[a-z0-9]+)*/g) ?? []).filter(x=>!STOP.has(x));
export const dot = (a:ArrayLike<number>,b:ArrayLike<number>) => Array.from(a).reduce((s,x,i)=>s+x*b[i],0);
export const cosine = (a:ArrayLike<number>,b:ArrayLike<number>) => dot(a,b)/(Math.sqrt(dot(a,a)*dot(b,b))||1);
export interface Index { passages:Passage[]; vocab:string[]; idf:number[]; U:number[][]; S:number[]; docVecs:Float64Array[]; terms:string[][]; avgLength:number }
export function buildIndex(passages=PASSAGES, background=BACKGROUND, k=16):Index {
 const all=[...passages.map(p=>p.text),...background], terms=all.map(tokenize), vocab=[...new Set(terms.flat())].sort();
 const idf=vocab.map(w=>Math.log((1+terms.length)/(1+terms.filter(t=>t.includes(w)).length))+1);
 const raw=(t:string[])=>vocab.map((w,j)=>t.filter(x=>x===w).length*idf[j]);
 // Collapse identical background rows, preserving their multiplicity in X'X.
 const unique=[...new Set(all)], rows=unique.map(s=>raw(tokenize(s)).map(x=>x*Math.sqrt(all.filter(a=>a===s).length)));
 const gram=rows.map(a=>rows.map(b=>dot(a,b))), e=eigSym(gram);
 const S=e.values.slice(0,k).map(x=>Math.sqrt(Math.max(0,x)));
 const U=S.map((s,i)=>vocab.map((_,j)=>s>1e-10 ? rows.reduce((sum,r,n)=>sum+r[j]*e.vectors[i][n],0)/s : 0));
 const index:Index={passages,vocab,idf,U,S,docVecs:[],terms:terms.slice(0,passages.length),avgLength:terms.slice(0,passages.length).reduce((s,t)=>s+t.length,0)/passages.length};
 index.docVecs=passages.map(p=>embedText(index,p.text));return index;
}
export function embedText(index:Index,text:string):Float64Array {
 const terms=tokenize(text), x=index.vocab.map((w,j)=>terms.filter(t=>t===w).length*index.idf[j]);
 return Float64Array.from(index.U.map(u=>dot(u,x)));
}
export interface Ranked {id:string; score:number}
export const semantic=(index:Index,q:string):Ranked[]=>{const v=embedText(index,q);return index.docVecs.map((d,i)=>({id:index.passages[i].id,score:cosine(v,d)})).sort((a,b)=>b.score-a.score);};
export function bm25(index:Index,query:string):Ranked[] {
 const q=tokenize(query),N=index.passages.length;
 return index.terms.map((t,i)=>({id:index.passages[i].id,score:q.reduce((s,w)=>{const f=t.filter(x=>x===w).length,df=index.terms.filter(x=>x.includes(w)).length;return s+Math.log(1+(N-df+.5)/(df+.5))*f*2.2/(f+1.2*(.25+.75*t.length/index.avgLength));},0)})).sort((a,b)=>b.score-a.score);
}
export function rrf(rankings:Ranked[][]):Ranked[] { const scores=new Map<string,number>();for(const ranking of rankings)ranking.forEach((p,i)=>scores.set(p.id,(scores.get(p.id)||0)+1/(60+i+1)));return [...scores].map(([id,score])=>({id,score})).sort((a,b)=>b.score-a.score); }
let cached:Index|undefined;
export const handbookIndex=()=>cached??=buildIndex();
