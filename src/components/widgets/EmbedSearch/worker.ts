import {buildIndex} from '../../../lib/applied/embed';
self.onmessage=()=>self.postMessage(buildIndex());
