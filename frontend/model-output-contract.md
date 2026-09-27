# Model-output contract, version 1

The consumer is `validatePayload()` in `core.mjs`. Producers write `data/model-output.json`; this document specifies the accepted wire format. The shipped dummy JSON is a complete synthetic example of the same shape, with `kind: "dummy"` and demo rules. It must not be relabeled as a real model run.

## Envelope and run identity

| Field | Required value |
| --- | --- |
| `schema_version` | Integer `1`. |
| `kind` | `model` for model-output.json, `dummy` for the fixed fixture. Source and kind must agree. |
| `generated_at_utc` | Export timestamp, `YYYY-MM-DDTHH:mm:ssZ`. All timestamps use this explicit, whole-second UTC format. |
| `price`, `outage` | Run objects described below; missing models use explicit unavailable objects. Optional producer `provenance` and `source_receipts` preserve model hashes, feature versions, blend calibration, and source receipt metadata. |
| `decisions` | `{rule_version, is_demo, records}`. Nonempty version string (max 100 characters), boolean demo flag, decision array. |

Each available model run requires `status: "available"`, `model_version` (nonempty, max 100), `issued_at_utc`, `input_cutoff_utc`, `max_age_hours` (finite, 0.01–168), and `records` (array). Require cutoff ≤ issue ≤ export, and issue/export no more than five minutes ahead of the actual clock. Missing, invalid, offsetless, or impossible calendar timestamps are rejected. `max_age_hours` is elapsed hours from **forecast origin** when provided, otherwise from **issue**, not export; re-exporting old results does not refresh them.

Each unavailable model is `{ "status": "unavailable", "reason": "Nonempty explanation" }`. Its `records` must be omitted or empty. Text reasons are limited to 300 characters.

The envelope carries one price run and one outage run, which may have different issue times. Every new decision inherits these exact run identities and the envelope's rule version. Previously fixed simulated actions retain explicit planning/basis timestamps as described below. Build decisions and signals together; never splice recommendations from another run. Preserve prior bundles upstream rather than mutating a run's values in place. This static consumer validates shape and consistency, not model provenance or calibration.

A valid empty model export looks like this (replace the export time):

```json
{
  "schema_version": 1,
  "kind": "model",
  "generated_at_utc": "2026-09-27T05:00:00Z",
  "price": {"status": "unavailable", "reason": "No price forecast exported."},
  "outage": {"status": "unavailable", "reason": "No county forecast exported."},
  "decisions": {"rule_version": "not-configured", "is_demo": false, "records": []}
}
```

## Price run

Add `target_date` (`YYYY-MM-DD`, America/Chicago operating day). Each of 1–800 rows requires:

| Field | Meaning and validation |
| --- | --- |
| `settlement_point` | One of `LZ_WEST`, `LZ_NORTH`, `LZ_SOUTH`, `LZ_HOUSTON`, `LZ_AEN`, `LZ_CPS`, `LZ_LCRA`, `LZ_RAYBN`. `LZEW` is not accepted. |
| `interval_start_utc`, `interval_end_utc` | Aligned, consecutive 15-minute UTC interval; start no earlier than issue. |
| `delivery_date` | Equals `target_date` and the interval start's Central date. |
| `hour_ending`, `quarter` | Central start-hour + 1 (1–24), and start-minute / 15 + 1 (1–4). |
| `repeated_hour_flag` | `Y` for the second occurrence of a fall-back local quarter; otherwise `N`. |
| `availability` | `available` or `unavailable`. An unavailable row requires `reason`; its numeric fields are not consumed. |
| `dam_spp_usd_mwh` | Available rows: finite absolute hourly DAM price, repeated across the matching UTC hour's quarters. |
| `rtm_mean_usd_mwh` | Available rows: finite expected **absolute RTM** price, not a residual or DAM spread. |
| `rtm_p10_usd_mwh`, `rtm_p50_usd_mwh`, `rtm_p90_usd_mwh` | Available rows: finite values satisfying P10 ≤ P50 ≤ P90. The mean may lie outside the band. |

Negative prices are valid. All prices use wholesale USD/MWh, not customer profit. P10–P90 is not historical min/max, and 80% coverage requires calibration.

The union of supplied intervals must span one full Central operating day, midnight to midnight: **92, 96, or 100** aligned UTC intervals, without gaps. A zone may be absent or partially supplied; its missing values stay unavailable. Duplicate `(settlement_point, interval_start_utc)` rows are rejected, equivalent to duplicate end-time identities inside the one run. The upstream identity also includes `model_version` and `issued_at_utc`. DST labels are checked against UTC; the UI displays CST/CDT.

## County outage run

Add these required fields:

- `grain: "county_scenario"`. Home/site forecasts cannot be relabeled as county forecasts.
- `probability_kind: "first_onset"`.
- `scenario_definition`: nonempty text (max 300) stating what county event the probabilities describe, with the producer's coverage/target assumptions. The UI exposes it in metadata.
- Optional `forecast_origin_utc`: original model horizon origin. Require input cutoff ≤ origin ≤ actual issue/publication; it controls age and horizon, so a later export cannot renew freshness. The shipped county model uses 12:00 UTC.
- `intervals`: exactly 24 `{interval_start_utc, interval_end_utc}` objects. Each is one elapsed UTC hour, aligned to the hour and contiguous. The first starts at the first whole hour **at or after forecast origin** (or issue when origin is omitted; less than an hour after that origin). The summary covers these 24 intervals, independent of local DST.

There are at most 254 unique county records. Every row uses a five-character Texas `county_fips` present in the geometry, `coverage`, and all three nullable signal fields shown below. Omitted county rows mean unknown coverage.

| Branch | Required signal fields |
| --- | --- |
| `coverage: "unknown"` | `p_first_start_by_hour: null`, `p_any_next_24h: null`, `active_outage: null`. |
| `coverage: "scenario"`, assumed at risk | `at_risk_assumed: true`, `active_outage: null`, 24 first-onset probabilities and their sum. Conditional modeled coverage, not an observed active-state feed. Always marks the bundle partial. |
| `coverage: "observed"`, no active episode | `active_outage: null`; 24 finite first-onset probabilities in `[0,1]`; finite `p_any_next_24h` in `[0,1]`. |
| `coverage: "observed"`, episode active at issue | Both onset fields `null`; `active_outage` object with the conditional restoration fields below. |

`p_first_start_by_hour[h]` is the probability that the **first** onset occurs in interval h, not a conditional hazard. Require `sum(p_first_start_by_hour) = p_any_next_24h ≤ 1`, within `1e-6` numerical tolerance. An observed all-zero array means zero modeled risk and renders differently from unknown coverage.

If the model emits conditional hazards `q_h`, the producer converts them before export:

```text
p_first[h] = q_h × product(1 − q_j for j < h)
p_any = sum(p_first) = 1 − product(1 − q_h)
```

Older frontend drafts described `p_first_start_by_hour` as conditional; that interpretation is not accepted here. Historical episode counts are not probabilities, and offsetless research timestamps cannot be assigned a timezone without verification.

### Active and restored episodes

`active_outage` requires `outage_started_at` (UTC), `elapsed_minutes` (finite ≥ 0), and `p_remaining_gt_1h`, `p_remaining_gt_4h`, `p_remaining_gt_12h`, `p_remaining_gt_24h`. These finite probabilities lie in `[0,1]` and are nonincreasing. Elapsed age must match issue minus start within one minute. Issue/model metadata is inherited from the outage run.

Supply this object only after detection and while the qualifying county episode is active. Do not issue remaining-duration estimates before onset or subtract elapsed time from an initial duration prediction. The UI labels persistence as of issue time, not a home's restoration time. After restoration, a newly issued run sets `active_outage: null` and supplies new onset probabilities. An active episode is not another onset.

## Decision records and relationships

At most 7,000 decisions, each with:

| Field | Accepted value |
| --- | --- |
| `mode`, `region_id` | `energy` plus load-zone ID, or `outages` plus county FIPS. Must exist in that geometry. |
| `interval_start_utc` | Exact start in that mode's timeline. `(mode, region_id, interval_start_utc)` is unique. |
| `action` | `charge`, `hold`, or `discharge`; displayed as Charge, Hold reserve, Discharge. |
| `strength` | Finite `[0,1]`, explicitly not a calibrated probability. |
| `reserve_constraint` | Boolean; `true` forbids `discharge`. |
| `reason_codes` | Ordered array of 1–8 supported codes listed below. |
| `relationship` | Explicit `null`, or the relationship object below. |

An available primary price row or observed primary county is required. Missing decisions leave the model values visible with `No recommendation`. The browser never supplies a missing action by recomputing policy. Runs with `scope: "simulated_household"` display hypothetical battery plans, not device commands. Dummy bundles require `decisions.is_demo: true`. A model bundle may also carry demo rules, but must keep that flag true until a real policy supplies its actions.

Supported reasons: `reserve_protection`, `low_price`, `price_opportunity`, `no_clear_opportunity`, `no_price_relationship`, `active_scenario`, `device_constraint`, `missing_input`, `policy_threshold`. The renderer uses fixed text, not HTML supplied by the producer.

A relationship is `{mode, region_id, method, description, evidence?}`. It names a valid region in the **other** mode. `description` is required (max 300 characters). `method` is `verified_pair`, `documented_aggregate`, or `illustrative_pair`; the first two require nonempty `evidence` (max 1,000 characters). `illustrative_pair` is accepted only in dummy data. For simulated-household scope only, `representative_scenario` is also accepted with explicit evidence and a description stating that the pairing is a scenario, not verified address membership. Geometry overlap/centroids never invent a pairing. The description must state the actual pairing or aggregation scope; a representative county does not imply an entire zone has that county's risk.

For the secondary model, the UI finds the interval containing the selected UTC start (`start ≤ selected < end`). It does not align arrays by index or extrapolate outside a model horizon. A missing relationship, signal, or temporal overlap produces explicit unavailable content. For an hourly outage selection, any related price shown is the first overlapping 15-minute interval, not an hourly average.

### Simulated battery plans

`decisions.scope: "simulated_household"` requires each action to include finite `charge_kwh` and `discharge_kwh` (0–1.25), `stored_energy_start_kwh`, `stored_energy_end_kwh`, and `reserve_kwh` (0–25), plus boolean `risk_window_covered` and `locked`. Flows cannot be simultaneous; the action must match them. Stored energy must follow the 90%-round-trip transition and satisfy the reserve. Consecutive energy actions in each zone must carry the same stored energy across their shared boundary. Actions at the operating day's start and end must match the declared initial and terminal energy; the shipped policy declares 15 kWh at both ends. Missing actions remain missing, without inferring continuity across a gap.

`planned_at_utc` must precede the action interval and export; `basis_outage_issued_at_utc` must be no later than planning. A midday update keeps earlier simulated actions with `locked: true` and their original basis timestamps, then carries their resulting energy into a new remaining-day solve. These historical records do not claim to use the newly published outage values and are labeled as prior simulated actions. No prior plan or device state means no invented past dispatch. County-mode actions describe the first quarter of that hour only.

The live mean is adaptive DAM + E2; E2's P10/P50/P90 models remain separately identified in `price.provenance.quantiles`. Do not claim their band is calibrated to the blended mean. `price.provenance.blend.adaptation_status` states whether new delayed outcomes were admitted or calibration remains on saved history.

## Status, errors, and freshness

Validation fails closed for the entire bundle on malformed input. Explicit unavailable fields allow partial rendering. Final status precedence is **empty → stale → partial → current**:

1. No available price or observed county signal: `empty`.
2. Any available run older than its origin-based (or issue-based) age limit, or any supplied timeline ended: `stale` (retains a separate partial flag).
3. Missing model, row, county coverage, or per-region interval decision: `partial`.
4. Otherwise: `current`.

`loading` and `error` belong to the request/FSM, not the JSON. HTTP 404 is unavailable only for the model endpoint; other HTTP failures, invalid JSON, wrong source kind, and validation failures produce `error` with retry. Error text is bounded to 240 characters. Responses are capped at 10 MiB, including streamed bodies. A newer source request makes older responses ineligible. Old source data is cleared at load start.

Dummy mode evaluates age at its fixed export time. Model mode evaluates age against the real clock, refreshed each minute. This is conservative across both runs: an expired secondary run makes the bundle stale even if the selected primary remains fresh.
