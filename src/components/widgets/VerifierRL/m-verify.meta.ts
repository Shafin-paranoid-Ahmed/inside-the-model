import type { ModeMeta } from '../_rl/kit';

export default {
  title: 'A program checks the answer',
  legend: [
    { label: 'Pass', color: '--viz-pos' },
    { label: 'Fail', color: '--viz-neg' },
    { label: 'What the verifier reads', color: '--accent' },
  ],
  aspect: '720 / 300',
  narrowAspect: '360 / 600',
  lanes: 2,
} satisfies ModeMeta;
