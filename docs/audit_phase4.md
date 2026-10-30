# Audit: Phase 4 (alerts, evaluation harness, baselines B1 to B3, validation results, sensitivity run)

Independent audit, 2026-10-30, at commit `d6ec794`. Scope: `PLAN.md`, `docs/decision_log.md` (Phase 4 pre-registration, Phase 4 results, parked rules), `docs/dataset_card.md`, `README.md` (Phase 4 section), `src/nettwin/observed.py`, `alerts.py`, `baselines.py`, `evaluation.py`, `scripts/phase4_baselines.py`, `phase4_build_pretune.py`, `phase4_tables.py`, `tests/test_phase4.py`, and `results/phase4_*`. Nothing in the repo was changed apart from adding this file.

Method: throwaway scripts in the `nettwin` env, kept outside the repo. They load observable files through the repo's own loader and produce raw scores with the committed calibration (`data/models/phase4/calib_*.pkl`, the same B2 model, bins and thresholds as the results), then recompute every metric with independent code: exact average precision with sklearn on unbinned scores, my own fault attribution, ranks, window classes and bootstraps. Only `data/datasets/v1/*/val`, `data/datasets/v1_pretune/*/val` and the train calibration were touched. Nothing under `test` or `dev` was read, and the dataset key was not opened. Telemetry is synthetic; numbers below are audit evidence, not project results.

Grades: **critical** = can invalidate a headline RQ result if not fixed before Phase 5; **major** = biases a result or a claim and needs a fix or a clearly reported limitation; **minor** = correctness, hygiene or reporting issue with small expected effect.

## Summary

| ID | Grade | Finding |
|---|---|---|
| Q1 | major | B3's "one-element" link paths also contain the link's two end PoPs. This departs from the pre-registered definition, is not disclosed, and is what gives B3 its node_down top-1 of 0.95 (0.45 with the pre-registered paths) |
| Q2 | major | PoP score = mean of link scores, combined with pessimistic ties, makes B1 and B2 unable to rank a PoP root first, and also unable to rank a link on a leaf spur first. node_down / demand_surge top-1 of 0.00 to 0.01 and E5 are structural, not findings |
| Q3 | major | "B2 is the best baseline on hard types", the E3 explanation and the flapping explanation all depend on the 0.9-quantile window score. With a window max, B1 ties B2 on hard top-3; with a window mean, B2 keeps link_down top-3 at 0.59 without status events |
| Q4 | minor | "Planned maintenance is the main false-alarm source" is wrong: maintenance-only hours have the highest alarm rate but produce only 18% to 27% of false alarms |
| Q5 | minor | E9 is logged as "not met, for B1 only". B3 also misses it in status_blind (0.235 against 0.182 + 0.05) |
| Q6 | minor | The E6 explanation (flags undercount missing-data evidence) fits the status_blind easy misses only. B2's with_status hard "misses" are near B2's own base rate for those elements |
| Q7 | minor | Per-type exceptions to "B2 best on hard, B3 best on easy": B1 is at least as good as B2 on demand_surge, B2 ties B3 on link_down top-3, and B3 has the worst node_down root PR-AUC |
| Q8 | minor | The sensitivity run pairs 878 of 921 faults, not every scenario. 29 scenarios with a misconfig diverge. E12 holds on matched faults; the misconfig "exception" is noise |
| Q9 | minor | Alert `value` and `t_end` use data from after the alert is raised. Harmless for B3; a leak if Phase 6 tools return them as is |
| Q10 | minor | Bins and alarm thresholds come from train scores. Fine for unsupervised baselines, biased for a supervised GNN unless calibrated on train scenarios it was not fitted on |
| Q11 | minor | Symptomatic positives and "detected at threshold" are attributed by time only, so in concurrent scenarios one fault's symptoms count for the other |
| Q12 | minor | Several met / not-met verdicts are point comparisons inside the noise (E9, E10). CIs bootstrap scenarios, not topologies |

No critical finding. No method, alert or harness input can see ground truth, other splits, clock time or file metadata, and every headline number in the decision log reproduces exactly from independent code.

## 1. Leakage

Checked in the code and confirmed by running it:
- **Method inputs.** `Method.score(obs, kg, ix)` receives only an `Observed` (fields: id, topology, variant flag, link and service arrays, observed events), the operator inventory (`inventory(topology, cfg)`: public topology plus config, no scenario data), and the element index. `load_observed` refuses any path under `truth/`. Labels and manifests are read in `_load` but are only passed to `scenario_record` after scoring. Scenario ids are opaque since the P1 fix and no method reads them.
- **Oracle window.** `localise(state, t_start, t_end)` gets the true fault window. This is pre-registered ("oracle window") and is the same for every method, but it is a truth input (see Section 6).
- **Train only.** B2 is fit on 100 train scenarios without labels; bins and thresholds come from 200 other train scenarios (disjoint, `default_rng(3003)` permutation), per variant. B1's span and persistence were picked on train. The script hard-codes `"train"` for calibration and `"val"` for evaluation.
- **Time.** No feature uses the step index, time of day or time since start. B1 is an EWMA, B2 uses trailing EWMAs, 12-step rolling stats and a one-day lag (allowed diurnal comparison). Day-0 values of the lag features are set to neutral, and B2 is fit on steps >= 288 only, so "early in the scenario" is not learnable.
- **File metadata.** Not used anywhere.
- **Day-0 exclusion.** Fair to every method: no cell before step 288 carries label 1 or 2 (recount: 0 in both variants), every method scores the same cells from 288, and all three have a full day of history by then (B2's one-day lag starts exactly at 288). Thresholds are fixed on train windows from 288 as well.
- **Causality.** Scores at t use data up to t (the repo test cuts a scenario at step 500; I also read every rolling operation, all trailing or shifted). Alerts are raised on the step that completes a 2-step run.

One forward-looking leak exists in alert metadata (Q9), unused in Phase 4.

## 2. Metric correctness

Recomputed from raw scores on all 468 val scenarios, both variants. "Harness" is the committed JSON.

| Check (with_status unless noted) | Independent | Harness |
|---|---|---|
| Root AP, B1 / B2 / B3, hard (sklearn, exact scores) | 0.0057 / 0.0498 / 0.0324 | 0.0057 / 0.0495 / 0.0324 |
| Root AP, easy | 0.0370 / 0.0689 / 0.1755 | 0.0370 / 0.0688 / 0.1754 |
| Root AP, link_degradation | 0.0019 / 0.0348 / 0.0259 | 0.0019 / 0.0349 / 0.0259 |
| Root AP, node_down | 0.0526 / 0.0833 / 0.0384 | 0.0526 / 0.0832 / 0.0385 |
| status_blind root AP, hard / easy, B2 | 0.0455 / 0.0123 | 0.0451 / 0.0123 |
| Top-1 / top-3, every type, hard, easy, both variants, all methods | identical to 4 decimals | |
| Unobservable top-3 (6 cells per variant) | identical | |
| False alarms, fault-free: alarms / windows | B1 2391 / 55801, B2 2170, B3 2875 | 0.0428, 0.0389, 0.0515 (same) |
| False alarms, maintenance-only | B1 882 / 3276, B2 610, B3 650 | 0.2692, 0.1862, 0.1984 (same) |
| Hard subtle detected at threshold | 0.25 / 0.13 / 0.06 | same |
| Pre-tuning hard top-3 | 0.344 / 0.4465 / 0.2665 | same |

- **Positive-cell attribution.** No label-2 cell lacks an owning fault, and no symptomatic cell is unowned (0 of each, matching `symptomatic_cells_without_fault`). The 5-cell gap in hard positives (81,197 against 81,202) is my simpler attribution assigning 5 overlapping cells to an unobservable fault; the harness keeps them once for the observable one, which is the pre-registered rule.
- **Window classes.** No cell labelled 1 or 2 lies in any fault-free or maintenance-only window (0 in both variants), so false-alarm rates are not contaminated by late symptoms.
- **Set-based scoring and pessimistic ties.** Implemented as specified. Pessimism matters a lot for some cells (Q2): optimistic ties would raise B2 hard top-1 from 0.22 to 0.29 and B3 demand_surge top-3 from 0.05 to 0.17 (many elements tie at score 0).
- **Binned AP.** Within 0.0004 of exact AP on val for every group I checked, confirming the train check.
- **Bootstrap.** Ratio-of-sums over multinomial scenario weights, seed 42 as pre-registered. My independent bootstraps (other seeds) give the same widths.

## 3. Fairness between methods

Same cells, same windows, same threshold procedure (5% of train fault-free windows, link cells only) for all three: confirmed. B1's tuning on train helps B1, and B1 still ranks last on most groups, so it changes no conclusion. Three method-specific choices do change stated results.

**Q1 (major): B3's link paths include their end PoPs.** The pre-registration and the docstring say "every link and every PoP as a one-element path". `B3Tomography._matrices` sets `M[S + j, L + ix.ends[j]] = 1`, so a link alert also counts as an alerted path through both end routers. Re-running B3 with true one-element link paths, everything else unchanged:

| B3, top-1 / top-3 | as committed | pre-registered paths |
|---|---|---|
| with_status node_down | 0.947 / 0.993 | 0.447 / 0.974 |
| with_status link_down | 0.667 / 0.858 | 0.722 / 0.870 |
| with_status link_flapping | 0.591 / 0.742 | 0.470 / 0.705 |
| with_status easy | 0.740 / 0.870 | 0.554 / 0.857 |
| with_status hard | 0.132 / 0.192 | 0.109 / 0.185 |
| status_blind node_down | 0.364 / 0.934 | 0.510 / 0.947 |
| status_blind link_down | 0.392 / 0.551 | 0.165 / 0.519 |

Top-3 barely moves, so E2 and E5 stay met. Top-1, which RQ4 will compare against, moves by up to 0.5. The choice is defensible (a failed link does sit between two routers) but it is a deviation from the pre-registration and must be logged as one, with both numbers reported.

**Q2 (major): PoP = mean of link scores plus pessimistic ties.** For B1 and B2 a PoP's score can never exceed its highest-scoring link, so with ties counted against the root a PoP can never be ranked first unless its links are removed by set-based scoring. That covers every node_down and demand_surge (297 observable faults with status events). A degree-1 PoP gets exactly its spur link's score, so a fault on a spur link can never be ranked first either:

| Observable link faults, with_status | n | B1 top-1 pessimistic / optimistic | B2 | B3 |
|---|---|---|---|---|
| root link touches a degree-1 PoP | 58 | 0.000 / 0.466 | 0.000 / 0.552 | 0.759 / 0.776 |
| other link roots | 523 | 0.205 / 0.205 | 0.348 / 0.348 | 0.379 / 0.409 |

56 of the 297 PoP roots are degree-1 PoPs. Other link-to-PoP aggregations do not fix this: taking the PoP score as the max of its links gives node_down top-1 0.02 (B1) and 0.01 (B2) and costs link faults their top-1 entirely. So "node_down top-1 is 0.00 for B1 and 0.01 for B2" (decision log, E5) is a property of the harness and the aggregation, not an observation about the methods, and B1 / B2 hard top-1 includes demand_surge at a structural zero. The pre-registration anticipated low numbers ("PoP score is a mean"), so E5 is not wrong, but it was met by construction and should be described that way. Fix: collapse a degree-1 PoP and its link into one candidate when ranking, or report top-1 separately for link and PoP roots.

**Q3 (major): the 0.9-quantile window score decides several claims.** It is one pre-registered rule for B1 and B2, so it is not tuning. But it is not neutral between them:

| Top-3 (with_status unless noted) | B1 q0.9 | B1 max | B1 mean | B2 q0.9 | B2 max | B2 mean |
|---|---|---|---|---|---|---|
| hard | 0.280 | 0.375 | 0.308 | 0.382 | 0.322 | 0.401 |
| link_degradation | 0.342 | 0.542 | 0.426 | 0.574 | 0.497 | 0.574 |
| link_flapping | 0.045 | 0.053 | 0.068 | 0.061 | 0.045 | 0.068 |
| status_blind link_down | 0.000 | 0.000 | 0.000 | 0.297 | 0.108 | 0.595 |
| status_blind node_down | 0.000 | 0.000 | 0.000 | 0.338 | 0.159 | 0.444 |

- "Isolation Forest (B2) is the best baseline on hard types (top-3 0.38)": true under the rule as registered (paired difference over B1 +0.102, 95% CI +0.061 to +0.143). With a window max for B1 the gap is +0.007 (CI -0.030 to +0.040); with each method's best aggregation it is +0.025 (CI -0.010 to +0.062). The ranking B2 > B1 on hard types is not robust to this choice.
- E3: "Its missing-data flags did not make blanked links stand out". With a window mean, B2 keeps status_blind link_down top-3 at 0.59 (drop 0.26, smaller than B3's 0.31). The flags do make blanked links stand out on average; the 0.9 quantile of the other links' noise is what hides them. E3 stays "not met" as registered, but the explanation should change.
- "The 0.9-quantile window score dilutes intermittent symptoms" (flapping): not supported. Max and mean give B1 / B2 flapping top-3 of 0.05 to 0.07 too. The link telemetry simply carries little flapping signal; B3 gets it from status events and end-to-end loss.

Fix: keep the registered numbers as headline, add a one-row aggregation sensitivity (max, mean), and word the B2-over-B1 and E3 statements as conditional on it.

## 4. Claims against outputs

Every number in the Phase 4 results entry and the headline table matches `results/` (checked one by one, including 432 / 446 / 43 faults, 66,732 of 95,462 root cells, 135 SLA alerts per week, 72% / 75% specificity, 2.8x to 6.3x, 4.5% to 5.0%, 2.4% / 1.7% / 5.9%, the pre-tuning mix 8.7% / 60.2%). The README Phase 4 section has no numbers and its statements are accurate. The problems are interpretive:

- **Q4 (minor):** "Planned maintenance is the main false-alarm source for every method" and the parked rule "maintenance-only hours cause most false alarms for every baseline (19% to 27%)". 19% to 27% is the alarm *rate* in maintenance-only hours. By count there are 17x more fault-free hours, so maintenance supplies only 27% (B1, 882 of 3,273), 22% (B2, 610 of 2,780) and 18% (B3, 650 of 3,525) of false alarms with status events. Maintenance is the highest-rate source, not the main one. The Phase 6 change-log measurement is still worth doing; the motivation needs rewording.
- **Q5 (minor):** E9 "not met, for B1 only". `results/phase4_expectations.json` also fails B3 in status_blind: real 0.2353 against generated 0.1816 + 0.05 = 0.2316. The JSON verdict (not met) is right; the log text is not.
- **Q6 (minor):** E6. For every fault I took a control window: the same length, shifted by whole days into a part of the same scenario with no fault, and ranked the same root element there.

| Unobservable faults, top-3 rate | B1 fault / control | B2 fault / control | B3 fault / control |
|---|---|---|---|
| hard, both variants (n = 33 with a control window) | 0.03 / 0.09 | 0.18 / 0.09 | 0.09 / 0.00 |
| easy, status_blind (n = 15) | 0.07 / 0.00 | 0.40 / 0.13 | 0.13 / 0.00 |

The status_blind easy hits mostly fit the log's explanation: B2 ranks three blanked link_down roots and one node_down root first or second, against control ranks of 32 to 87 (a fourth link_down root is ranked second in its control window too). The hard hits do not: they exist in with_status too, where nothing is blanked. 3 of B2's 6 hits are on links B2 ranks 1st to 4th in a fault-free window anyway (chronically anomalous links), and 6 of 33 against 3 of 33 is not a clear excess. Two misconfig hits (control ranks 8 and 19) may be real sub-threshold signal: "unobservable" is a threshold on noise-free deltas, not zero signal. E6's "near chance" should have been judged against this control rate, not a uniform 3 / n.
- **Q7 (minor):** "B2 best on hard types, B3 best on easy types" holds pooled with paired CIs (hard top-3 B2 minus B3 +0.19, CI +0.14 to +0.24; easy top-3 B3 minus B2 +0.31, CI +0.26 to +0.37; easy root PR-AUC B3 minus B2 +0.107, CI +0.071 to +0.144; hard root PR-AUC B2 minus B3 +0.017, CI +0.001 to +0.034, marginal). Per type it does not:
  - demand_surge top-3: B1 0.283 against B2 0.228 (B2 minus B1 -0.055, CI -0.123 to +0.007).
  - link_down top-3: B3 0.858 against B2 0.864 (B3 minus B2 -0.006, CI -0.077 to +0.064). The log's "B3 is strong for link_down" holds for top-1 and PR-AUC, not top-3.
  - node_down root PR-AUC: B3 is the worst method (0.038 against B2 0.083, difference CI -0.071 to -0.024).
  - "B3 ... the worst on hard ones" is true for top-1 / top-3 but not root PR-AUC (B1 0.006 is lower).
- E12 "hard top-3 rises by 0.06 to 0.07": the B3 rise is 0.074, so 0.06 to 0.08 is the honest rounding range. Trivial.

## 5. Sensitivity run

- **Patch.** `_patch` replaces the module global `faults._loguniform` in every worker, and `make_fault` looks it up at call time, so the surge factor is uniform: v1_pretune factors run 1.37 to 3.98 with mean 2.63 (uniform on [1.3, 4] has mean 2.65). The other overrides match `ad950f7` and the two logged adjustments: gray loss log-uniform 0.01 to 2% (observed 0.01 to 1.92), flap down fraction 0.01 to 0.5, flaps per step 0.5 to 6, convergence 0.05 to 1 s, misconfig near-miss share 0.5 (observed 0.41 against 0.67 in v1). It keeps v1's later physics fixes (ramp-downs, link_down convergence), which is the right counterfactual for "ranges only", though not literally the `ad950f7` sampler.
- **Q8 (minor): pairing is partial.** Ids, slots, seeds, topologies and maintenance logs are identical for all 468 scenarios. But changing `minimal_prob` changes how many random draws a misconfig consumes (1 against 100), so every draw after it shifts: 29 scenarios, all with a misconfig, have a different fault list; 46 have different telemetry gaps; 878 of 921 faults match on (type, root, start, end). "The two builds pair up scenario by scenario" overstates it.
- **Calibration** is the v1 train calibration for both runs (`--calib`), as stated. Fair.
- **E12 on matched faults only** (405 hard faults observable in both builds), paired: B1 +0.067 (CI +0.034 to +0.100), B2 +0.074 (+0.047 to +0.103), B3 +0.091 (+0.061 to +0.122). E12 stands. The "routing_misconfig exception" (B3 0.10 to 0.09) becomes +0.009 on matched faults: no change, not a fall.

## 6. What would bias Phases 5 and 6 if the harness is reused unchanged

- **Q2 again.** A GNN with its own PoP head can rank PoPs first; B1 / B2 cannot. Any "GNN beats B1 / B2 on node_down or demand_surge top-1" result would be a harness artefact. If the GNN scores links only and reuses `pop_mean`, it inherits the cap.
- **Q10 (minor): calibration from train scores.** `make_edges` and `threshold_for` use train scores. For a supervised GNN those are fitted scores, typically more extreme than on val, so the 5% train alarm rate will not carry over and bins may be coarse where val scores land. Keep the Phase 4 pattern (fit and calibrate on disjoint train scenarios), and prefer exact AP: sklearn on all of a variant's roughly 32 million scored val cells is cheap enough to run per group (this audit did it for 12 groups in one short script).
- **Q9 (minor): alert metadata.** In `generate_alerts`, `value = nanmax(val[s:e])` runs over the whole episode, from before the raise step to the clear step, and `t_end` is the clear time. `add_to_kg` stores both. A Phase 6 `get_active_alerts(t0, t1)` that returns them at query time hands the agent the future peak and the clear time. Truncate both to the query time.
- **Oracle window.** Baselines and the GNN get the true [t_start, t_end). RQ4 compares the agent with "the best non-LLM localiser": give the agent the same window, or rerun the localisers on whatever window the agent gets.
- **Q11 (minor): attribution by time.** A symptomatic cell inside an overlapping window is credited to every overlapping fault, and `hit_max` takes any labelled cell in the window, including the other fault's. This touches 9% of scenarios and stays small, but per-type symptomatic PR-AUC and "detected at threshold" for concurrent faults mix types.
- **Q12 (minor): uncertainty.** Verdicts are point comparisons. E10 (B3 easy top-3, concurrent minus not, -0.10, CI -0.22 to +0.02) and E9 for B1 (+0.09, CI -0.02 to +0.21) are inside the noise either way. CIs resample scenarios, not topologies. On val a topology-cluster bootstrap gives similar widths (B2 real hard top-3: scenario CI 0.29 to 0.51, topology CI 0.26 to 0.52), but with 5 real val and 7 test networks, RQ3 should report per-network numbers next to pooled CIs.
- B3 ignores the maintenance log by design, so Phase 6 can measure the change-log tool on maintenance-only hours as planned (with Q4's wording fixed).

## What should happen before Phase 5

1. Q1: log the B3 path definition as a deviation from the pre-registration, and report B3 top-1 with both path definitions.
2. Q2: collapse degree-1 PoP / spur-link duplicates in ranking (or report link and PoP roots separately), and describe E5 as met by construction.
3. Q3: add a window-aggregation sensitivity row and reword the B2-over-B1 claim and the E3 and flapping explanations.
4. Q4 to Q8: fix the wording in the decision log results entry and the parked rules.
5. Q9 and Q10 before Phases 6 and 5 respectively; Q11 and Q12 when reporting.
