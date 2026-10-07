import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Eight generations on an easy and a hard test',
  legend: [
    { label: 'Score with 95% interval', color: '--viz-forward', shape: 'line' },
    { label: 'Interval overlaps the previous generation', color: '--accent', shape: 'line' },
    { label: 'Best possible score', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 380',
  narrowAspect: '360 / 560',
  lanes: 2,
} satisfies ModeMeta;
