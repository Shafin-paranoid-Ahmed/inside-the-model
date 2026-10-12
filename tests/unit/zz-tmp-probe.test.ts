import { it } from 'vitest';
import * as S from '../../src/components/widgets/ClipLab/sim';
it('probe', () => {
  const run = S.trainClip(); const p = run.snaps[run.snaps.length-1];
  const cls = S.classEmbeddings(p,'plain');
  let ok=0; for (const it of S.TRAIN_ITEMS) { const r=S.classify(p,S.itemImage(it),cls); if(r.pred===S.shapeIndex(it)) ok++; }
  const hs=(['plain','article','sized'] as const).map(t=>S.heldOutAccuracy(p,t).acc);
  process.stdout.write(`PROBE train ${ok}/35 held ${hs.join(' ')} loss ${run.evalLoss[run.evalLoss.length-1].toFixed(3)} diag ${run.evalDiag[run.evalDiag.length-1].toFixed(3)}\n`);
});
