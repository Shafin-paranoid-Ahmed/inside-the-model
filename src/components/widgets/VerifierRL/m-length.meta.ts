import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Response length during training',
  legend: [
    { label: '1 digit', color: '--viz-cat-1' },
    { label: '2 digits', color: '--viz-cat-2' },
    { label: '3 digits', color: '--viz-cat-3' },
    { label: '4 digits', color: '--viz-cat-4' },
  ],
  aspect: '720 / 380',
  narrowAspect: '360 / 600',
  loading: 'Training 150 iterations in your browser…',
} satisfies ModeMeta;
