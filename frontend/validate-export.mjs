import { readFile } from 'node:fs/promises';
import { validateGeometry, validatePayload } from './core.mjs';
const read = async path => JSON.parse(await readFile(path, 'utf8'));
const geometry = validateGeometry({energy:await read(new URL('data/texas-load-zones.json',import.meta.url)),outages:await read(new URL('data/texas-counties.json',import.meta.url))});
const payload = await read(process.argv[2]);
const checked = validatePayload(payload,geometry);
console.log(JSON.stringify({status:checked.status,prices:checked.priceIndex.size,counties:checked.outageIndex.size,decisions:checked.decisionIndex.size}));
