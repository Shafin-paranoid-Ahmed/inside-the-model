import {makeReader,evaluate} from './sim';
self.onmessage=()=>{const reader=makeReader();self.postMessage({reader,evaluation:Array.from({length:8},(_,i)=>({k:i+1,semantic:evaluate(reader,i+1),hybrid:evaluate(reader,i+1,true)}))});};
