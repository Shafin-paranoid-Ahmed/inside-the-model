import { train } from './sim';
self.onmessage = () => { try { self.postMessage({ run: train() }); } catch (error) { self.postMessage({ error: String(error) }); } };
