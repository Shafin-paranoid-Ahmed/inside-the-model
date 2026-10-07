import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'An arena: votes become ratings',
  legend: [
    { label: 'Vote: arc from loser to winner', color: '--text-muted' },
    { label: 'Rating with 95% interval', color: '--text-muted', shape: 'line' },
    { label: 'Rating fair voters would give', color: '--text', shape: 'line' },
  ],
  controls: [{ key: 'lengthBias', label: 'Voters favour longer answers', kind: 'toggle', then: 'end' }],
  aspect: '720 / 400',
  narrowAspect: '360 / 500',
  lanes: 2,
} satisfies ModeMeta;
