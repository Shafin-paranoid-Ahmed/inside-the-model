import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'The best moves as the cost of a move changes',
  legend: [
    { label: 'Best move', color: '--text', shape: 'line' },
    { label: 'Arrow that just changed', color: '--accent', shape: 'line' },
  ],
  aspect: '720 / 400',
  lanes: 2,
  narrowAspect: '360 / 600',
  loading: 'Solving the grid for every cost of a move…',
} satisfies ModeMeta;
