export const SITE_NAME = 'AI Learner';
const site = import.meta.env.SITE;

/** Absolute URL, trailing-slash consistent with build output (directory format). */
export function absUrl(pathname: string): string {
  const p = pathname.startsWith('/') ? pathname : `/${pathname}`;
  const last = p.split('/').pop() ?? '';
  return new URL(p.endsWith('/') || last.includes('.') ? p : `${p}/`, site).href;
}

const publisher = { '@type': 'Organization', name: SITE_NAME, url: absUrl('/') };
const ctx = { '@context': 'https://schema.org' };
const courseRef = { '@type': 'Course', name: 'AI Learner: how AI actually works in two weeks', url: absUrl('/path'), provider: publisher };
const LEVELS = { intro: 'Beginner', core: 'Intermediate', advanced: 'Advanced', frontier: 'Advanced' } as const;

export const websiteLd = () => ({
  ...ctx, '@type': 'WebSite', name: SITE_NAME, url: absUrl('/'), inLanguage: 'en',
  description: 'Learn how AI actually works in two weeks, by playing with interactive visualizations of the real mechanisms.',
});

export const courseLd = (days: { day: number; title: string; lessons: { id: string; title: string }[] }[]) => ({
  ...ctx, ...courseRef, inLanguage: 'en', isAccessibleForFree: true,
  description: 'A free 15-day curriculum from "what is a model?" to the frontier, taught with interactive visualizations.',
  educationalLevel: 'Beginner',
  hasCourseInstance: { '@type': 'CourseInstance', courseMode: 'online', courseWorkload: 'P14D' },
  hasPart: days.map((d) => ({
    '@type': 'CreativeWork', name: `Day ${d.day}: ${d.title}`, position: d.day,
    hasPart: d.lessons.map((l) => ({ '@type': 'LearningResource', name: l.title, url: absUrl(`/learn/${l.id}`) })),
  })),
});

export const lessonLd = (l: { id: string; title: string; summary: string; level: keyof typeof LEVELS; duration: number }) => ({
  ...ctx, '@type': 'LearningResource', name: l.title, description: l.summary, url: absUrl(`/learn/${l.id}`),
  educationalLevel: LEVELS[l.level], timeRequired: `PT${l.duration}M`, inLanguage: 'en',
  isPartOf: courseRef,
});
