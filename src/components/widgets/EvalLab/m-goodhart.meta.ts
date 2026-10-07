import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Fifty tries at the same public test',
  legend: [
    { label: 'Public score of each try', color: '--viz-cat-1' },
    { label: 'Best public score so far', color: '--accent', shape: 'line' },
    { label: 'Private score of that best try', color: '--viz-cat-2', shape: 'line' },
    { label: 'True skill', color: '--text', shape: 'line' },
  ],
  aspect: '720 / 380',
  narrowAspect: '360 / 420',
  lanes: 2,
} satisfies ModeMeta;
