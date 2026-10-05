import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'A replay buffer mixes old and new experience',
  legend: [
    { label: 'Recent (dark) to old (light)', color: '--viz-seq-5' },
    { label: 'Sampled at random', color: '--accent' },
    { label: 'Newest 16 in a row', color: '--viz-forward' },
  ],
  aspect: '720 / 400',
  narrowAspect: '360 / 420',
  lanes: 2,
} satisfies ModeMeta;
