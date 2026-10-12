import { it } from 'vitest';
import * as S from '../../src/components/widgets/LoRALab/sim';
it('probe', () => {
  const t0 = performance.now();
  const run = S.computeRun();
  const P = (s: string) => process.stdout.write('LPROBE ' + s + '\n');
  P('ms ' + (performance.now() - t0).toFixed(0));
  P('nf4 ' + S.NF4_LEVELS.join(','));
  const f = (c: S.Curve) => `test ${c.test[0].toFixed(3)}->${c.test[c.test.length - 1].toFixed(3)} @60:${c.test[12].toFixed(3)} train end ${c.train[c.train.length - 1].toFixed(3)} params ${c.params}`;
  for (const [n, T] of [['low', run.low], ['fullrank', run.fullrank]] as const) {
    P(n + ' full ' + f(T.full));
    for (const r of S.RANKS) P(n + ' r' + r + ' ' + f(T.lora[r]));
    P(n + ' sv ' + Array.from(T.sv.slice(0, 6)).map((v) => v.toFixed(2)).join(' ') + ' top2 ' + S.energyTop(T.sv, 2).toFixed(3) + ' top4 ' + S.energyTop(T.sv, 4).toFixed(3));
  }
  P('qfp ' + f(run.qlora.fp)); P('qnf4 ' + f(run.qlora.nf4)); P('qint4 ' + f(run.qlora.int4)); P(`base ${run.qlora.baseNf4} ${run.qlora.baseInt4}`);
  for (const d of run.data) P('n' + d.n + ' ' + f(d.curve));
  const W0 = S.base();
  for (const s of ['int4', 'nf4'] as const) { const q = S.quantize(W0, s); P(s + ' mse ' + q.mse + ' rel ' + q.mse / S.variance(W0)); }
  for (const m of ['1B', '7B', '65B', '70B']) { const p = S.PRESETS.find((x) => x.id === m)!; P(m + ' ' + (['full', 'lora', 'qlora'] as const).map((s) => s + ' ' + S.memory(s, p).total.toFixed(2)).join(' ') + ' adapters ' + S.adapterParams(p)); }
  const m = S.mergeCheck(W0, run.low.lora[4].B!, run.low.lora[4].A!, 4); P('merge ' + m.maxDiff);
});
