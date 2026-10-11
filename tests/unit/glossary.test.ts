import { describe, it, expect } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { GLOSSARY } from '../../src/lib/glossary';
import { REFRESHER } from '../../src/lib/refresher';

const written = (id: string) => {
  const file = `src/content/lessons/${id}.mdx`;
  return existsSync(file) && !/^status: stub/m.test(readFileSync(file, 'utf8'));
};

describe('glossary and refresher', () => {
  it('every glossary term links to a written lesson', () => {
    expect(GLOSSARY.filter((t) => !written(t.lesson)).map((t) => t.term)).toEqual([]);
  });
  it('terms are unique', () => {
    const names = GLOSSARY.map((t) => t.term.toLowerCase());
    expect(names.filter((n, i) => names.indexOf(n) !== i)).toEqual([]);
  });
  it('every refresher card belongs to a written lesson', () => {
    expect(Object.keys(REFRESHER).filter((id) => !written(id))).toEqual([]);
  });
});
