import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'REINFORCE in the grid, with and without a baseline',
  legend: [
    { label: 'Policy (arrow opacity = probability)', color: '--text', shape: 'line' },
    { label: 'No baseline', color: '--viz-cat-1', shape: 'line' },
    { label: 'With baseline', color: '--viz-cat-2', shape: 'line' },
    { label: 'Best possible', color: '--text-muted', shape: 'line' },
  ],
  aspect: '720 / 440',
  narrowAspect: '360 / 720',
  lanes: 2,
  loading: 'Training 20 runs in your browser…',
} satisfies ModeMeta;
