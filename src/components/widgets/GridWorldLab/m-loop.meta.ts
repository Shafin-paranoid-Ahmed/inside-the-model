import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'One episode of a random agent',
  legend: [
    { label: 'Agent', color: '--accent' },
    { label: 'Action asked for', color: '--text-muted', shape: 'line' },
    { label: 'Move made', color: '--viz-forward', shape: 'line' },
  ],
  aspect: '720 / 380',
  lanes: 2,
  narrowAspect: '360 / 560',
} satisfies ModeMeta;
