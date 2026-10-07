import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'Training against the verifier',
  legend: [
    { label: '1 digit', color: '--viz-cat-1' },
    { label: '2 digits', color: '--viz-cat-2' },
    { label: '3 digits', color: '--viz-cat-3' },
    { label: '4 digits', color: '--viz-cat-4' },
    { label: 'Policy probability', color: '--viz-forward' },
  ],
  aspect: '720 / 460',
  narrowAspect: '360 / 720',
  lanes: 1,
  loading: 'Training 150 iterations in your browser…',
} satisfies ModeMeta;
