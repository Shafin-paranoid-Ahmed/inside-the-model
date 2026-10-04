import { parameterLayout, type Config } from '../transformer/model';

/** Matrix multiply-adds (MACs; FLOP ≈ 2 × MACs) to process `nNew` tokens whose last one sits at
 * position `ctxLen` (so ctxLen − nNew tokens are already cached). Counts Q/K/V/O projections,
 * Q·K scores and value mixing over the visible (causal) keys only, the MLP and the unembedding of
 * every new token. Embedding lookups, LayerNorm, softmax and residual adds are not MACs and are
 * not counted. A full uncached pass of T tokens is macsForward(cfg, T, T).
 */
export function macsForward(cfg: Pick<Config, 'd' | 'ff' | 'layers' | 'vocab' | 'causal'>, nNew: number, ctxLen: number): number {
  if (!Number.isInteger(nNew) || !Number.isInteger(ctxLen) || nNew < 1 || ctxLen < nNew) throw new Error('Need 1 <= nNew <= ctxLen');
  const past = ctxLen - nNew, { d, ff, layers } = cfg;
  const visible = cfg.causal ? nNew * past + nNew * (nNew + 1) / 2 : nNew * ctxLen;
  return layers * (4 * nNew * d * d + 2 * d * visible + 2 * nNew * d * ff) + nNew * d * cfg.vocab.length;
}
export const weightBytes = (model: number | Config, bytesPerParam: number): number =>
  (typeof model === 'number' ? model : parameterLayout(model).size) * bytesPerParam;
/** K and V: 2 × layers × kvDim × tokens × batch × bytes per number. */
export const kvBytes = ({ layers, kvDim, tokens, batch = 1, bytes }: { layers: number; kvDim: number; tokens: number; batch?: number; bytes: number }): number =>
  2 * layers * kvDim * tokens * batch * bytes;
/** Roofline (Williams, Waterman and Patterson 2009): attainable FLOP/s at an arithmetic intensity in FLOP per byte. */
export function roofline(hw: { peakFlops: number; bandwidth: number }, intensity: number): { flops: number; ridge: number; bound: 'memory' | 'compute' } {
  const ridge = hw.peakFlops / hw.bandwidth;
  return { flops: Math.min(hw.peakFlops, hw.bandwidth * intensity), ridge, bound: intensity < ridge ? 'memory' : 'compute' };
}

export interface PublishedModel {
  kind: 'model'; name: string; layers: number; d: number; heads: number; kvHeads: number; kvDim: number;
  context: number; trainingTokens: number; params: number; source: string; verified: boolean; unverified: readonly string[]; note: string;
}
export interface PublishedAccelerator {
  kind: 'accelerator'; name: string; peakFlops: number; bandwidth: number; memoryBytes: number;
  source: string; verified: boolean; note: string;
}
/** Published configs. `verified` = every field except those in `unverified` checked against the primary source on 2026-10-04 (see `note`). */
export const CONFIGS = {
  llama2_7b: {
    kind: 'model', name: 'Llama 2 7B', layers: 32, d: 4096, heads: 32, kvHeads: 32, kvDim: 4096, context: 4096,
    trainingTokens: 2.0e12, params: 6.74e9, source: 'https://arxiv.org/abs/2307.09288', verified: true, unverified: ['params'],
    note: 'Llama 2 paper Table 1: 7B, 4k context, no GQA, 2.0T tokens; §2.2 adopts the Llama 1 architecture, whose Table 2 (arXiv:2302.13971) gives 4096 dim, 32 heads, 32 layers for 7B. params 6.74e9 is not in either paper (it is the count of the released checkpoint); treat it as unverified.',
  },
  llama3_8b: {
    kind: 'model', name: 'Llama 3 8B', layers: 32, d: 4096, heads: 32, kvHeads: 8, kvDim: 1024, context: 8192,
    trainingTokens: 15e12, params: 8.03e9, source: 'https://arxiv.org/abs/2407.21783', verified: true, unverified: ['params'],
    note: 'Llama 3 paper Table 3: 32 layers, dim 4096, FFN 14336, 32 heads, 8 KV heads (kvDim 8 × 128 = 1024), vocab 128K; "about 15T multilingual tokens". Model card: 8k context, 15T+ tokens. params 8.03e9 is the released checkpoint count, not stated in the paper.',
  },
  h100_sxm: {
    kind: 'accelerator', name: 'NVIDIA H100 SXM', peakFlops: 989.5e12, bandwidth: 3.35e12, memoryBytes: 80e9,
    source: 'https://www.nvidia.com/en-us/data-center/h100/', verified: true,
    note: 'Spec sheet: BF16 Tensor Core 1,979 TFLOPS "with sparsity" (dense is one half: 989.5 TFLOPS), 80 GB, 3.35 TB/s.',
  },
} as const satisfies Record<string, PublishedModel | PublishedAccelerator>;
