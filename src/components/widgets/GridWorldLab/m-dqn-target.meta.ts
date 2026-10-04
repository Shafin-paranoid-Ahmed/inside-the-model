import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'A target network changes in steps',
  legend: [
    { label: 'Online network prediction Q', color: '--viz-cat-1', shape: 'line' },
    { label: 'Target y from the target network', color: '--viz-cat-2', shape: 'line' },
    { label: 'Value iteration Q*', color: '--text-muted', shape: 'line' },
  ],
  aspect: '720 / 420',
  narrowAspect: '360 / 600',
  lanes: 2,
  loading: 'Training the network for 300 updates in your browser…',
} satisfies ModeMeta;
