import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Clipping a positive advantage',
  legend: [
    { label: 'Clipped objective', color: '--viz-pos', shape: 'line' },
    { label: 'Unclipped ρ·A', color: '--text-muted', shape: 'line' },
    { label: 'Clip band', color: '--accent' },
  ],
  aspect: '720 / 380',
  lanes: 2,
  narrowAspect: '360 / 420',
} satisfies ModeMeta;
