/** Logical canvas size per mode (width fixed: 720 wide, 330 narrow), reserved before any mode code loads. */
import type { Mode } from './core';

export function sizeOf(mode: Mode, T: number, narrow: boolean): { W: number; H: number } {
  const W = narrow ? 330 : 720;
  switch (mode) {
    case 'arcs': return { W, H: narrow ? 270 : 270 };
    case 'qkv': return { W, H: narrow ? 36 + T * 14 + 52 + T * 11 + 16 : 66 + Math.max(T * 15, 80) + 30 };
    case 'scores': return { W, H: narrow ? 58 + T * 24 + 40 + 232 : 66 + T * 30 + 36 };
    case 'softmax': return { W, H: narrow ? 24 + 3 * (92 + 26) + 12 : 24 + 3 * (104 + 26) + 10 };
    case 'mix': return { W, H: narrow ? 34 + T * 26 + 40 + 96 + 26 : 34 + T * 32 + 24 };
    case 'explorer': return { W, H: narrow ? 224 + 52 + T * 22 + 30 + 4 * (52 + 24) + 40 : Math.max(520, 200 + 52 + T * 26 + 24) };
    case 'reach': return { W, H: narrow ? 204 + 78 + 12 * T + 40 : 270 };
    case 'compare': return { W, H: 40 + 2 * ((narrow ? 96 : 104) + 52) + 4 };
    case 'split': return { W, H: narrow ? 62 + 16 * 5.2 + 8 + 12 + 34 + 34 + 60 : 62 + 16 * 9.5 + 8 + 12 + 52 + 40 + 40 };
    case 'heads': return { W, H: narrow ? 96 + T * 14 + 38 + T * 8 + 34 : 78 + T * 26 + 50 + T * 12 + 34 };
    case 'concat': return { W, H: narrow ? 54 + 16 * 7 + 8 + 10 + 54 + 3 * 50 + 6 : 66 + 16 * 10 + 8 + 14 + 44 };
    case 'gallery': { const cs = narrow ? Math.max(10, Math.min(14, Math.floor((W / 2 - 52) / T))) : 22, hdr = narrow ? 46 : 40, ph = 54 + hdr + T * cs + (narrow ? 12 : 18); return { W, H: 20 + 2 * ph }; }
    case 'ablate': return { W, H: 56 + (narrow ? 150 + 16 : 190) + 46 };
  }
}