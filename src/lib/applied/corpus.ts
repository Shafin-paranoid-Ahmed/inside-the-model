export interface Passage { id: string; title: string; text: string; date?: string }
// Invented teaching material, not a real organisation's policies.
const rows = [
 ['overdue','Overdue tools','Overdue tools cost 50 cents per day.'],
 ['washer-fee','Pressure washer late fee','The late fee for the pressure washer is 2 dollars per day.'],
 ['washer-loan','Pressure washer loan','The pressure washer loan lasts seven days except on public holidays, when the limit is three days.'],
 ['washer-code','PW-200','PW-200 requires a refundable deposit of 30 dollars.'],
 ['hours-old','Saturday hours (old)','Saturday opening hours are 9 am to noon.','2025-01-01'],
 ['hours-new','Saturday hours (current)','Saturday opening hours are 10 am to 4 pm.','2026-10-01'],
 ['membership','Membership','Annual membership costs 20 dollars.'],
 ['join','Joining','Bring photo identification and proof of address to join.'],
 ['drill','Drill collection','Collect a reserved drill at the front desk.'],
 ['drill-loan','Drill loan','A drill loan lasts three days.'],
 ['renew','Renewals','Renew a loan once if nobody has reserved the tool.'],
 ['reserve','Reservations','Reserve tools online up to fourteen days ahead.'],
 ['cancel','Cancellations','Cancel a reservation at least one day before collection.'],
 ['return','Returns','Return tools to the staffed counter during opening hours.'],
 ['clean','Cleaning','Clean tools before returning them.'],
 ['damage','Damage','Report damage immediately and stop using the tool.'],
 ['lost','Lost tools','Lost tools incur their replacement cost.'],
 ['safety','Safety training','Safety training is required before borrowing a chainsaw.'],
 ['goggles','Eye protection','Wear safety goggles when cutting or drilling.'],
 ['battery','Batteries','Return both drill batteries and the charger.'],
 ['saw','Saw loan','A circular saw loan lasts five days.'],
 ['ladder','Ladder loan','A ladder loan lasts two days.'],
 ['ladder-deposit','Ladder deposit','The ladder deposit is 15 dollars.'],
 ['saw-deposit','Saw deposit','The circular saw deposit is 25 dollars.'],
 ['drill-deposit','Drill deposit','The drill deposit is 10 dollars.'],
 ['sander','Sander deposit','The sander deposit is 12 dollars.'],
 ['pump','Pump deposit','The pump deposit is 18 dollars.'],
 ['welder','Welder deposit','The welder deposit is 40 dollars.'],
 ['cart','Cart deposit','The cart deposit is 8 dollars.'],
 ['monday','Monday hours','The library is closed on Monday.'],
 ['weekday','Weekday hours','Tuesday through Friday opening hours are noon to 6 pm.'],
 ['sunday','Sunday hours','Sunday opening hours are 11 am to 2 pm.'],
 ['contact','Contact','Email the desk at help@example.test for assistance.'],
 ['location','Location','The entrance is beside the Elm Street community garden.'],
 ['parking','Parking','Bicycle parking is available outside the entrance.'],
 ['volunteer','Volunteers','Volunteers attend an orientation on the first Sunday of each month.'],
 ['workshop','Workshops','Repair workshops run on the second Saturday of each month.'],
 ['age','Age limit','Borrowers must be at least eighteen years old.'],
 ['payment','Payment','Pay membership charges and deposits by card.'],
 ['refund','Refunds','Refundable deposits are returned when equipment passes inspection.'],
];
export const PASSAGES: Passage[] = rows.map(([id,title,text,date]) => ({id,title,text,...(date ? {date} : {})}));
const bridges = [
 'late overdue bring back drill penalty cost fee daily',
 'late overdue bring back drill penalty cost fee daily',
 'borrow loan duration days week renew',
 'cost fee payment charges dollars cents',
 'pressure washer wash cleaning water loan',
 'Saturday opening hours morning afternoon time',
 'deposit refundable return equipment inspection',
 'membership join identification address annual',
 'reserve reservation book cancel collection desk',
 'safety goggles cutting drilling chainsaw training',
 'battery charger drill electricity',
 'damage lost replacement report repair',
 'ladder saw sander pump welder cart tools',
 'Sunday Monday Tuesday weekday hours closed',
 'garden entrance parking bicycle location',
];
export const BACKGROUND = Array.from({length:150}, (_,i) => bridges[i % bridges.length]);
export const QA = PASSAGES.filter(p => !['hours-old','drill','ladder-deposit','saw-deposit','drill-deposit','sander','pump','welder','cart','contact','location','parking','volunteer','workshop','payment','refund'].includes(p.id)).map(p => ({q:`What does the handbook say about ${p.title.toLowerCase()}?`,goldIds:[p.id],answer:p.text}));
export function chunk(text: string, size: number, overlap = 0): string[] {
 if (!Number.isInteger(size) || size < 1 || !Number.isInteger(overlap) || overlap < 0 || overlap >= size) throw new RangeError('Require size > overlap >= 0 in whole words');
 const words=text.trim().split(/\s+/).filter(Boolean), out:string[]=[];
 for(let i=0;i<words.length;i+=size-overlap) {out.push(words.slice(i,i+size).join(' '));if(i+size>=words.length)break;} return out;
}
