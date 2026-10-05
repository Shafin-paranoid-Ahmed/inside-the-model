/** Features and labels: a toy housing dataset. Pure logic, no DOM. */

export type Feature = 'area' | 'bedrooms' | 'age';
export type Hood = 'A' | 'B' | 'C';
export interface Row { id: number; area: number; bedrooms: number; age: number; hood: Hood; price: number }

/** Ten houses. price is the label, in thousands of dollars. */
export const ROWS: Row[] = [
  { id: 1, area: 45, bedrooms: 1, age: 40, hood: 'A', price: 130 },
  { id: 2, area: 62, bedrooms: 2, age: 25, hood: 'B', price: 175 },
  { id: 3, area: 78, bedrooms: 2, age: 18, hood: 'B', price: 205 },
  { id: 4, area: 90, bedrooms: 3, age: 33, hood: 'A', price: 190 },
  { id: 5, area: 110, bedrooms: 3, age: 8, hood: 'C', price: 300 },
  { id: 6, area: 125, bedrooms: 4, age: 20, hood: 'B', price: 285 },
  { id: 7, area: 140, bedrooms: 3, age: 5, hood: 'C', price: 380 },
  { id: 8, area: 150, bedrooms: 4, age: 35, hood: 'A', price: 270 },
  { id: 9, area: 175, bedrooms: 5, age: 10, hood: 'C', price: 440 },
  { id: 10, area: 95, bedrooms: 2, age: 2, hood: 'C', price: 290 },
];
export const HOODS: Hood[] = ['A', 'B', 'C'];
export const FEATURES: { key: Feature; label: string; unit: string; min: number; max: number }[] = [
  { key: 'area', label: 'Area', unit: 'm²', min: 30, max: 190 },
  { key: 'bedrooms', label: 'Bedrooms', unit: '', min: 0, max: 6 },
  { key: 'age', label: 'Age', unit: 'yr', min: 0, max: 45 },
];
export const PRICE = { min: 80, max: 480 };
export const oneHot = (h: Hood) => HOODS.map((k) => (k === h ? 1 : 0));

export const T_ROW = 1.2; // seconds per row in the scan
/** Seconds the finished plot holds before the scan loops. */
export const T_HOLD = 6;
export const duration = () => ROWS.length * T_ROW + T_HOLD;
export const markers = () => [{ at: 0, label: 'row 1' }, { at: 5 * T_ROW, label: 'row 6' }];

export interface Frame {
  /** Row currently being read (index). */
  active: number;
  /** 0..1 per row: how far its point has flown from the table into the plot. */
  arrive: number[];
}
/** The scan reads the table row by row; each row's point flies into the plot. All rows are in by the end. */
export function frameAt(t: number): Frame {
  const u = t / T_ROW;
  return { active: Math.min(ROWS.length - 1, Math.floor(u)), arrive: ROWS.map((_, i) => Math.min(1, Math.max(0, u - i))) };
}

/**
 * Unlabeled view: group houses whose ages sit close together (sort by age, start a new group at every gap wider
 * than AGE_GAP years). Returns the group index per row, groups numbered from newest to oldest.
 */
export const AGE_GAP = 5;
export function groups(rows: Row[] = ROWS): number[] {
  const order = rows.map((_, i) => i).sort((a, b) => rows[a].age - rows[b].age);
  const g = new Array<number>(rows.length);
  let k = 0;
  order.forEach((i, j) => { if (j > 0 && rows[i].age - rows[order[j - 1]].age > AGE_GAP) k++; g[i] = k; });
  return g;
}
