import { describe, expect, it } from 'vitest';
import * as S from '../../src/components/widgets/LoRALab/sim';
const run = S.computeRun();
describe('LoRA mechanics', () => {
  it('starts at the frozen base and leaves the caller base unchanged', () => {
    const W=S.base(), before=W.slice(), st=S.dataStats('low',64);
    const c=S.train(W,S.target('low'),st,2,7,10);
    expect(W).toEqual(before);
    expect(c.test[0]).toBeCloseTo(S.testLoss(W,S.target('low')),12);
    expect(c.train[0]).toBeCloseTo(S.trainLoss(W,st),12);
  });
  it('sums rank-one layers to the scaled product', () => {
    const c=run.low.lora[4], sum=new Float64Array(1024);
    for(let k=0;k<4;k++) S.layer(c.B!,c.A!,4,k).forEach((v,i)=>sum[i]+=v);
    const p=S.lowRank(c.B!,c.A!,4);
    sum.forEach((v,i)=>expect(v).toBeCloseTo(p[i],12));
  });
  it('matrix multiplication agrees with a hand calculation', () => {
    expect([...S.matmul(new Float64Array([1,2,3,4]),new Float64Array([5,6,7,8]),2,2,2)]).toEqual([19,22,43,50]);
  });
  it('merging preserves every output and swapping replaces the correction', () => {
    const a=run.low.lora[4], b=run.c;
    const m=S.mergeCheck(S.base(),a.B!,a.A!,4);
    expect(m.maxDiff).toBeLessThan(1e-12);
    expect(m.madUnmerged).toBe(1280); expect(m.madMerged).toBe(1024);
    S.swapAdapter(a.W,a,b).forEach((v,i)=>expect(v).toBeCloseTo(b.W[i],12));
  });
  it('shows the measured low-rank benefit and full-rank counterexample', () => {
    expect(run.low.lora[2].test[60]).toBeCloseTo(.103,3);
    expect(run.low.lora[1].test[60]).toBeCloseTo(.403,3);
    expect(run.fullrank.lora[2].test[60]).toBeCloseTo(.862,3);
    expect(S.energyTop(run.low.sv,2)).toBeCloseTo(.975,3);
    expect(S.energyTop(run.fullrank.sv,2)).toBeCloseTo(.227,3);
  });
  it('exposes overfitting instead of calling training loss quality', () => {
    expect(run.data[0].curve.train[60]).toBeLessThan(.001);
    expect(run.data[0].curve.test[60]).toBeGreaterThan(run.data[0].curve.test[0]);
    expect(run.data[3].curve.test[60]).toBeCloseTo(.103,3);
  });
  it('training is deterministic', () => {
    const a=S.train(S.base(),S.target('low'),S.dataStats('low',64),2,5,10);
    const b=S.train(S.base(),S.target('low'),S.dataStats('low',64),2,5,10);
    expect(a).toEqual(b);
  });
});
describe('quantization and storage', () => {
  it('reproduces published NF4 reconstruction levels', () => {
    const expected=[-1,-.6961928,-.5250731,-.3949175,-.2844414,-.1847734,-.09105,0,.0795803,.1609302,.2461123,.3379152,.4407098,.562617,.7229568,1];
    S.NF4_LEVELS.forEach((v,i)=>expect(v).toBeCloseTo(expected[i],6));
  });
  it('handles zero blocks and partial blocks without NaN or discarded values', () => {
    for(const scheme of ['nf4','int4'] as const){
      expect([...S.quantize(new Float64Array(64),scheme).deq]).toEqual(Array(64).fill(0));
      const q=S.quantize(new Float64Array([0,.25,-.8]),scheme);
      expect(q.absmax[0]).toBe(.8); expect(q.deq[2]).toBe(-.8);
      expect(Number.isFinite(q.mse)).toBe(true);
      expect(S.quantize(new Float64Array(),scheme).mse).toBe(0);
    }
  });
  it('chooses the nearest level and reports actual squared error', () => {
    const w=S.base(),q=S.quantize(w,'nf4'); let e=0;
    for(let i=0;i<w.length;i++){
      expect(q.deq[i]).toBe(q.snapped[i]*q.absmax[Math.floor(i/64)]);
      expect(Math.abs(q.norm[i]-q.snapped[i])).toBeCloseTo(Math.min(...S.NF4_LEVELS.map(v=>Math.abs(v-q.norm[i]))),12);
      e+=(w[i]-q.deq[i])**2;
    }
    expect(q.mse).toBeCloseTo(e/w.length,14);
    expect(q.mse).toBeCloseTo(.00082071072758,12);
    expect(S.runningMse(w,q,16)).toBeCloseTo(q.mse,12);
    expect(run.qlora.nf4.test[60]).toBeCloseTo(.119,3);
  });
  it('keeps decimal GB, scale overhead and adapters in the memory sum', () => {
    const p=S.PRESETS[1]; expect(S.memory('full',p).total).toBe(112);
    expect(S.adapterParams(p)).toBe(16777216);
    expect(S.memory('qlora',p).total).toBeCloseTo(4.205935456,9);
    expect(S.memory('lora',p).total).toBeCloseTo(14.268435456,9);
  });
  it('clamps the training timeline and reaches the final checkpoint', () => {
    expect(S.checkpointAt(-10)).toBe(0); expect(S.checkpointAt(6)).toBe(60); expect(S.checkpointAt(100)).toBe(60);
    expect(S.memoryFrame(7).optimizer).toBe(1);
    expect(S.quantFrame(10).blocks).toBe(16);
    expect(S.dataFrame(8).update).toBe(300);
  });
});
