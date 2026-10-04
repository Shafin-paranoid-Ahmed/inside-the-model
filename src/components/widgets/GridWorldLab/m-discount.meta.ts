import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'The best moves for a short-sighted and a patient agent',
  legend: [
    { label: 'Best move', color: '--text', shape: 'line' },
    { label: 'Differs from the left grid', color: '--accent', shape: 'line' },
  ],
  aspect: '720 / 400',
  lanes: 1,
  narrowAspect: '360 / 680',
  loading: 'Solving the grid for every discount factor…',
} satisfies ModeMeta;
