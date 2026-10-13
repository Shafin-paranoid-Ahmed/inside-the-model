import {makeData,sweep} from './sim';
self.onmessage=(e:MessageEvent)=>{self.postMessage(e.data==='sweep'?sweep():makeData());};
