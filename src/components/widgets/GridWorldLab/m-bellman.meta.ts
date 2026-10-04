import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'One Bellman backup, then the whole grid',
  legend: [
    { label: 'Cell being backed up', color: '--accent', shape: 'line' },
    { label: 'Outcome of the move (width = probability)', color: '--viz-forward', shape: 'line' },
    { label: 'Largest change per sweep', color: '--viz-gradient', shape: 'line' },
  ],
  aspect: '720 / 460',
  lanes: 2,
  narrowAspect: '360 / 760',
} satisfies ModeMeta;
