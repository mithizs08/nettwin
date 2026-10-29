# Audit: Phase 3 dataset (data/datasets/v1)

Independent audit, 2026-10-30, at commit `eb50843`. Scope: `PLAN.md`, `docs/decision_log.md`, `docs/dataset_card.md`, `src/nettwin/dataset.py`, `scripts/phase3_build_dataset.py`, `results/phase3_dataset_summary.json`, and the built files under `data/datasets/v1` (1,940 scenarios, two variants each). Nothing in the repo was changed apart from adding this file.

Method: throwaway scripts in the `nettwin` env read the Parquet, JSON and manifest files directly. No simulation was run on any topology. The only computation that touched test topologies was re-running the public scenario sampler for one test scenario to prove finding P1, and its faults were compared, not printed. Numbers below are audit evidence, not project results.

Grades: **critical** = can invalidate a headline RQ result if not fixed before Phase 4; **major** = biases a result or a claim and needs a fix or a clearly reported limitation; **minor** = correctness, hygiene or reporting issue with small expected effect.

## Summary

| ID | Grade | Finding |
|---|---|---|
| P1 | major | The scenario id gives away the seed, so the "no seed in observable files" protection does not work. Ground truth can be regenerated exactly from observable `meta.json` plus the committed config |
| P2 | minor | The card's severity table uses all faults as the denominator, unlike the pre-registered T2. This is the source of the misconfig "hit / miss" contradiction |
| P3 | minor | The two variants also differ in down-link telemetry, not only in status events. This is by design and documented, but question 3's premise is not literally true |
| P4 | minor | Fault start times lean toward night because faults are placed near maintenance windows |
| P5 | minor | Unobservable faults carry an impact class (almost always "subtle"), so per-class tables mix them in |
| P6 | minor | Labels are shared by both variants. In status_blind, 17% of root-labelled cells have no telemetry at all |
| P7 | minor | File sizes carry weak signal (AUC up to 0.62), bounded by the content itself |

No critical finding. Label alignment, card counts and the observable / ground-truth split all check out.

## 1. Ten random scenarios: labels against observable data

Picked with `numpy.random.default_rng(20261030)`: train `tr0017002`, `tr0234001`, `tr0143001`; val `va0022003`, `va0055002`, `va0030000`; test `te0001022` (germany50), `te0000020`, `te0000016` (Geant2012); dev `dv0000019` (AARNet). 19 faults in total.

For all 10, in both variants:
- `meta.json` scenario id, topology and step count match the manifest.
- `links.parquet` has exactly `n_steps x n_links` rows and `labels.parquet` has the same count. The `t` and `link` columns are identical row by row, and `t` runs 0 to 2015.
- `services.parquet` has `n_steps x n_services` rows.
- Labels are identical in the two variants.

Symptoms at the root element, comparing the fault window with the same steps one day earlier:
- link_down (2) and node_down (4): root links read util exactly 0 with latency missing on 97% to 100% of window cells (the remainder are random drop-outs). Services at a dead PoP read 100% loss. Root labels cover 100% of the window.
- demand_surge (5): root util rises (for example 0.498 to 0.804, latency x1.74), or rises only slightly for the "subtle" and "moderate" cases with only `obs_link_util` set. That is consistent with their flags.
- link_degradation (2, both visible): root counter loss 0.0129% and 0.0590% against about 0.0001% the day before, util and latency unchanged. Correct for visible gray failures.
- link_flapping (4): util dips (0.287 to 0.219, 0.153 to 0.102), small service loss increases, status events where `obs_status` is set.

Every fault's observed behaviour agrees with its type and its observability flags. All 19 were flagged observable, so I also drew 8 unobservable faults from val and test at random (`default_rng(3)`). For each I ranked the fault window's mean root counter loss and its worst-service mean probe loss against 40 random windows of the same length in the same scenario. The percentiles were spread from 0.00 to 0.97 with no pattern (root counters: 0.17, 0.00, 0.65, 0.23, 0.50, 0.53, 0.57, 0.00). That is what "no signal" should look like.

## 2. Dataset card numbers recounted from disk

Recounted from the manifests and the directory tree, not from the summary JSON:

| Check | Card | Recount |
|---|---|---|
| Scenarios / faults (train, val, test, dev) | 1232/2434, 468/937, 210/431, 30/55 | same |
| Observable dirs = label dirs = manifest ids | (implied) | equal in every split |
| Concurrent-fault scenarios (train / val / test / dev) | 117, 53, 14, 3 | same |
| Faults overlapping maintenance (train / val / test / dev) | 669, 247, 111, 11 | same |
| Maintenance windows, val | 966 in 467 scenarios | 966; 467 of 468 val scenarios have maintenance (one has none), so the wording is right |
| Unobservable, with_status / status_blind (train / val / test / dev) | 97/109, 35/41, 22/26, 5/5 | same |
| val link_degradation subtle / obvious / unobservable | 28.4 / 21.9 / 7.7% | same |
| val routing_misconfig subtle / obvious / unobservable | 19.9 / 50.0 / 5.8% | same |
| test routing_misconfig | 16.4 / 43.8 / 11.0% | same |
| train link_down | 2.1 / 89.5 / 0.7% | same |
| Disk (train / val / test / dev) | 5.243 / 1.868 / 0.785 / 0.067 GB | same |

Every sampled number matches.

## 3. with_status against status_blind

- Observed events: for all 1,940 scenarios, removing `link_status` and `pop_status` from the with_status events gives exactly the status_blind events. Maintenance events are identical.
- `meta.json`: for all 1,940, only `status_blind` differs.
- `services.parquet`: identical in all 60 sampled scenarios (15 per split).
- `links.parquet`: **not identical** (P3). In the 60 sampled scenarios 2,154 link cells differ. All 2,154 are cells where with_status shows the down-link signature (util 0, loss 0, latency missing) and status_blind shows all three missing. No other cell differs.
- Labels: identical. Manifests: identical except `observability`, `status_change_observed` (1,469 faults) and `observable` (22 faults), which are recomputed for the blind view by design.

So the variants differ in status events and in how down links are reported, as the card describes, and in nothing else.

## 4. Leak scan of the observable tree

Checked and clean:
- Files: only `links.parquet`, `services.parquet`, `observed_events.json` and `meta.json` (3,880 of each JSON). No other files.
- Words: a regex for `fault|root|degrad|gray|grey|flap|surge|misconfig|silent|seed|impact|severe|mild|subtle|obvious|placement|caused|concurrent|F<digit>` over every observable JSON file found **0** hits. The same scan over the Parquet columns and categorical values (link and service names, 80 sampled files) also found 0.
- Keys: events carry only `{t, type, element, status}` or the maintenance keys `{t, type, id, target, t_start, t_end, action, description}`. Meta carries `{scenario_id, topology, n_steps, step_minutes, status_blind}`.
- Id order: the scenario's position within its topology has AUC 0.49 to 0.50 for every fault type.

Findings:
- **P1 (major): the seed is recoverable from the id.** `dataset.py` sets `sid = code + f"{index * 1000 + k:07d}"` and `seed = seed_base[split] + index * 1000 + k`. So `seed = config.dataset.seed_base[split] + int(sid[2:])` holds for 1,940 of 1,940 scenarios. With the topology from `meta.json`, re-running `sample_scenario` reproduced the exact fault list (type, root, start, end) for `te0000007` (test), `va0075000` and `tr0002003`. The card says the observable tree holds no seed "since the seed plus the config would let anyone re-run the sampler and recover the faults"; that protection is defeated by the id. No honest detector would use the id. But any tool, agent or notebook that sees a scenario id and can run repo code can read off the answer, test split included. Fix: issue ids from a keyed hash or a random permutation stored only under `truth/`, and add a test that the sampler cannot be rerun from observable data. The code change does not alter the simulated data, so a rebuild needs only new ids.
- **P7 (minor): file sizes.** Within-topology residuals of `links.parquet` size separate scenarios containing a link_degradation with AUC 0.62, and routing_misconfig 0.57. Other types are 0.42 to 0.49. Parquet output is a deterministic function of the content (visible gray loss gives non-zero counter values that compress worse), so size cannot leak more than the telemetry itself. It only matters if a pipeline ever feeds file metadata to a model.
- **P4 (minor): timing.** Fault starts per 3-hour bin of the day, 00:00 onward: 633, 689, 457, 454, 436, 422, 399, 367. The night excess comes from maintenance-linked placement (86% of change windows start 00:00 to 04:00) plus incidental overlaps after N6. This is plausible (changes cause incidents), but a detector can learn "night raises the fault prior". Faults never start on day 0 (the warmup), as documented. Report per-time-of-day results or note this in the card.

## 5. The routing_misconfig T2 contradiction

I could not find a committed statement that validation misconfigs hit T2. The decision log, the commit messages of `7fd2253` and `4dab96c`, the card's limitations section and `results/phase2_summary.json` (`routing_misconfig pass: false`) all say "miss". The contradiction comes from the card's severity table: it reads 19.9% subtle and 50.0% obvious for val misconfigs, which looks like a pass at rounding. The two statements measure different things:

| Subset (val, with_status) | Denominator | Subtle | Obvious | T2 (subtle >= 20%, obvious <= 50%) |
|---|---|---|---|---|
| All val misconfigs, as in the card (n = 156) | all faults | 19.9% | 50.0% | miss by 0.1 point on subtle; obvious at the ceiling |
| All val misconfigs, as pre-registered (n = 147) | observable faults | **15.0%** | **53.1%** | **miss on both** |
| Generated val graphs only (118 / 110) | all / observable | 23.7% / 18.2% | 45.8% / 49.1% | hit with all faults; miss with observable |
| validation_real only (38 / 37) | all / observable | 7.9% / 5.4% | 63.2% / 64.9% | miss either way |
| Isolated, no maintenance overlap (79 / 73) | all / observable | 25.3% / 19.2% | 50.6% / 54.8% | miss either way |
| Phase 2 diagnostic set (188 scenarios, `phase2_summary.json`) | observable | 17.0% | 48.9% | miss on subtle |

status_blind gives the same misconfig numbers.

Resolution:
- The pre-registered target (observable faults only) is **missed** for validation misconfigs in every subset.
- The only reading that "hits" is generated validation graphs with unobservable faults counted in the denominator. All 9 unobservable val misconfigs are classed "subtle", so including them lifts the subtle share.
- The real validation networks are far harder to make subtle (5.4%), which matters for RQ3.
- Test is similar to validation (observable: 10.8% subtle, 49.2% obvious).
- Flapping misses in every subset (observable subtle 2% to 7%).

**P2 (minor):** the card states its denominator ("of all faults"), so this is not hidden. But it differs from the pre-registered T2 definition and invites the wrong reading. Add the observable-only columns, or a T2 row with the pre-registered definition, and split generated from real networks.

**P5 (minor):** unobservable faults receive an impact class, almost always "subtle" (all 9 unobservable val misconfigs, for instance). Phase 4 per-class tables must exclude them, or "subtle" results will include faults that cannot be detected at all.

## Other checks

- **P6 (minor):** both variants share one `labels.parquet`. Across 60 sampled scenarios, 0.4% of symptomatic (label 1) and 0.5% of root (label 2) cells have no observed link telemetry in with_status (gaps). In status_blind, 17.1% of root cells are fully missing (blanked down links). A detector can still pick these up from missingness, so the labels are not wrong. RQ1 PR-AUC on status_blind should state whether all-missing cells are scored, and treat them the same way for every method.
- Split hygiene: train manifests contain generated graphs only; val contains generated validation graphs plus the five validation_real networks; test and dev topologies match `config.yaml`.

## What should happen before Phase 4

1. Fix P1 (opaque ids, mapping kept under `truth/`) and rebuild the ids. Everything else in the data can stay.
2. Fix P2 and P5 in the card and the Phase 4 reporting plan.
3. P3, P4, P6 and P7 need only a sentence in the dataset card.
