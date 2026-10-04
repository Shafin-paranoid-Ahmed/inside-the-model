import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Ratios and drift for three clip widths',
  legend: [
    { label: 'Positive advantage', color: '--viz-pos', shape: 'dot' },
    { label: 'Negative advantage', color: '--viz-neg', shape: 'dot' },
    { label: 'Clip band', color: '--accent' },
  ],
  aspect: '720 / 440',
  narrowAspect: '360 / 640',
  lanes: 2,
} satisfies ModeMeta;
