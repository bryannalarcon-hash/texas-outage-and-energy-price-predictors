"""Simulated battery plans; no device commands or verified address assignments."""
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

CAPACITY, POWER, TERMINAL = 25., 5., 15.
ETA = np.sqrt(.90)
RULE_VERSION = 'reserve-milp-25kwh-v1'
REPRESENTATIVES = {'LZ_HOUSTON': '48201', 'LZ_NORTH': '48113', 'LZ_SOUTH': '48355', 'LZ_WEST': '48329'}


def reserve_curve(probabilities, origin, boundaries):
    probabilities = np.asarray(probabilities, float)
    if probabilities.shape != (24,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or probabilities.sum() > 1 + 1e-6:
        raise ValueError('Invalid first-onset probabilities')
    offset = np.array([(b - origin).total_seconds()/3600 for b in boundaries])
    covered = (offset >= 0) & (offset + 6 <= 24)
    overlap = np.maximum(0., np.minimum(offset[:, None]+6, np.arange(1, 25)) - np.maximum(offset[:, None], np.arange(24)))
    mass = overlap @ probabilities
    mass[~covered] = 0.
    return 5 + 10*np.minimum(1., mass/.25), covered


def solve(prices, targets, initial=TERMINAL):
    prices, targets = np.asarray(prices, float), np.asarray(targets, float)
    n, step = len(prices), POWER*.25
    if not n or targets.shape != (n+1,) or not np.isfinite(initial) or not 0 <= initial <= CAPACITY or not np.isfinite(prices).all() or not np.isfinite(targets).all() or np.any(targets < 0) or np.any(targets > CAPACITY):
        raise ValueError('Invalid dispatch inputs')
    floors = np.minimum(targets, initial + ETA*step*np.arange(n+1))
    lower = np.zeros(4*n)
    upper = np.r_[np.full(2*n, step), np.full(n, CAPACITY), np.ones(n)]
    lower[2*n:3*n] = floors[1:]
    lower[3*n-1] = upper[3*n-1] = TERMINAL
    matrix = lil_matrix((3*n, 4*n))
    lo = np.r_[np.zeros(n), np.full(2*n, -np.inf)]
    hi = np.r_[np.zeros(n), np.zeros(n), np.full(n, step)]
    lo[0] = hi[0] = initial
    for t in range(n):
        matrix[t, t], matrix[t, n+t], matrix[t, 2*n+t] = -ETA, 1/ETA, 1
        if t:
            matrix[t, 2*n+t-1] = -1
        matrix[n+t, t], matrix[n+t, 3*n+t] = 1, -step
        matrix[2*n+t, n+t], matrix[2*n+t, 3*n+t] = 1, step
    result = milp(np.r_[prices/1000+1e-9, -prices/1000+1e-9, np.zeros(2*n)],
                  integrality=np.r_[np.zeros(3*n), np.ones(n)], bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix.tocsr(), lo, hi),
                  options={'mip_rel_gap': 1e-9, 'time_limit': 30})
    if result.status != 0 or result.x is None:
        raise RuntimeError('Battery optimization did not converge; no plan published')
    c, d = result.x[:n], result.x[n:2*n]
    soc = np.r_[initial, result.x[2*n:3*n]]
    if not (np.allclose(soc[1:], soc[:-1]+ETA*c-d/ETA, rtol=0, atol=1e-7) and np.all(soc >= floors-1e-7) and np.all(soc <= CAPACITY+1e-7) and np.max(c*d) < 1e-7 and abs(soc[-1]-TERMINAL)<1e-7):
        raise RuntimeError('Battery physics verification failed')
    return c, d, soc, floors


def decisions(price, outage, previous=None):
    result = {'rule_version': RULE_VERSION, 'is_demo': False, 'scope': 'simulated_household', 'records': [],
              'battery': {'capacity_kwh': CAPACITY, 'power_kw': POWER, 'round_trip_efficiency': .9, 'initial_and_terminal_kwh': TERMINAL, 'household_load_kw': 1.25},
              'limitations': 'Representative counties only; no address mapping or device telemetry. A new simulation assumes 15 kWh at its first interval; updates carry prior simulated energy. Missing six-hour risk windows use a 5 kWh reserve. No hardware control.'}
    if price['status'] != 'available' or outage['status'] != 'available':
        return result
    counties = {r['county_fips']: r for r in outage['records']}
    origin = datetime.fromisoformat(outage.get('forecast_origin_utc', outage['issued_at_utc']))
    issued = max(datetime.fromisoformat(price['issued_at_utc']), datetime.fromisoformat(outage['issued_at_utc']))
    old = {(r['region_id'],r['interval_start_utc']):r for r in (previous or {}).get('decisions',{}).get('records',[]) if r['mode']=='energy'}
    outage_starts = {datetime.fromisoformat(x['interval_start_utc']) for x in outage['intervals']}
    for zone, fips in REPRESENTATIVES.items():
        county = counties.get(fips)
        if not county or county['coverage'] == 'unknown' or county['active_outage'] is not None:
            continue
        rows = sorted((r for r in price['records'] if r['settlement_point'] == zone), key=lambda r:r['interval_start_utc'])
        if not rows or any(r['availability'] != 'available' for r in rows):
            continue
        times = [datetime.fromisoformat(r['interval_start_utc']) for r in rows]
        if any(datetime.fromisoformat(r['interval_end_utc']) != start + timedelta(minutes=15) for r, start in zip(rows, times)) or any(b-a != timedelta(minutes=15) for a,b in zip(times,times[1:])):
            continue
        boundaries = times + [datetime.fromisoformat(rows[-1]['interval_end_utc'])]
        targets, covered = reserve_curve(county['p_first_start_by_hour'], origin, boundaries)
        past = sum(t < issued for t in times)
        if past and not all((zone,r['interval_start_utc']) in old for r in rows[:past]):
            continue  # No device state or prior simulation: do not invent already-executed actions.
        initial = (old[(zone,rows[past-1]['interval_start_utc'])]['stored_energy_end_kwh'] if past else
                   old.get((zone,rows[0]['interval_start_utc']),{}).get('stored_energy_start_kwh',TERMINAL))
        if past < len(rows):
            c, d, soc, floors = solve([r['rtm_mean_usd_mwh'] for r in rows[past:]], targets[past:], initial=initial)
        for i, row in enumerate(rows):
            if i < past:
                result['records'].append({**old[(zone,row['interval_start_utc'])], 'locked': True})
                continue
            k = i-past
            action = 'charge' if c[k] > 1e-6 else 'discharge' if d[k] > 1e-6 else 'hold'
            constrained = action != 'discharge' and (soc[k] <= floors[k]+1e-6 or soc[k+1] <= floors[k+1]+1e-6)
            reasons = ['reserve_protection'] if constrained else ['low_price' if action=='charge' else 'price_opportunity' if action=='discharge' else 'no_clear_opportunity']
            relation = {'mode': 'outages', 'region_id': fips, 'method': 'representative_scenario',
                        'description': f'This simulated {zone} household uses county {fips} as its risk scenario. It is not a verified address or zone-wide risk.',
                        'evidence': 'Four representative locations from the July–December 2025 battery comparison; see rules.html#data.'}
            record = {'mode': 'energy', 'region_id': zone, 'interval_start_utc': row['interval_start_utc'],
                      'action': action, 'strength': float(min(1., max(c[k], d[k])/(POWER*.25))),
                      'reserve_constraint': bool(constrained), 'reason_codes': reasons, 'relationship': relation,
                      'charge_kwh': float(c[k]), 'discharge_kwh': float(d[k]), 'stored_energy_start_kwh': float(soc[k]),
                      'stored_energy_end_kwh': float(soc[k+1]), 'reserve_kwh': float(floors[k+1]), 'risk_window_covered': bool(covered[i]),
                      'locked': False, 'planned_at_utc':issued.isoformat().replace('+00:00','Z'),
                      'basis_outage_issued_at_utc':outage['issued_at_utc']}
            result['records'].append(record)
            # County view shows the first 15-minute action of a covered UTC hour.
            if times[i] in outage_starts:
                result['records'].append({**record, 'mode': 'outages', 'region_id': fips,
                    'relationship': {'mode': 'energy', 'region_id': zone, 'method': 'representative_scenario',
                        'description': f'Simulated household paired with {zone}; action is for this hour’s first 15 minutes, not the whole county.', 'evidence': relation['evidence']}})
    return result
