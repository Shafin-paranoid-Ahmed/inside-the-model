/**
 * AttentionExplorer: every view is a pure function of the playhead `t` over one real forward pass of TinyLM
 * (src/lib/transformer). No DOM, no fetch. All layer/head numbers are ZERO-based; the widget shows them ONE-based.
 * core.ts is shared; each figure lives in its own m-<mode>.ts and is loaded on demand by the component.
 * This file re-exports everything for tests and tooling; the component does not import it.
 */
import type { Frame, Opts, Scene } from './core';
import * as m_arcs from './m-arcs';
import * as m_qkv from './m-qkv';
import * as m_scores from './m-scores';
import * as m_softmax from './m-softmax';
import * as m_mix from './m-mix';
import * as m_explorer from './m-explorer';
import * as m_reach from './m-reach';
import * as m_compare from './m-compare';
import * as m_split from './m-split';
import * as m_heads from './m-heads';
import * as m_concat from './m-concat';
import * as m_gallery from './m-gallery';
import * as m_ablate from './m-ablate';
export * from './core';
export * from './frames-a';
export * from './derived';
export * from './stats';
const frames: Record<string, (t: number, s: Scene, o: Opts) => Frame> = { arcs: m_arcs.frameAt, qkv: m_qkv.frameAt, scores: m_scores.frameAt, softmax: m_softmax.frameAt, mix: m_mix.frameAt, explorer: m_explorer.frameAt, reach: m_reach.frameAt, compare: m_compare.frameAt, split: m_split.frameAt, heads: m_heads.frameAt, concat: m_concat.frameAt, gallery: m_gallery.frameAt, ablate: m_ablate.frameAt };
export const frameAt = (t: number, s: Scene, o: Opts): Frame => frames[o.mode](t, s, o);
