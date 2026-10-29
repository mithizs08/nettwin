# Re-verification: Phase 0 to 2 audit fixes

Independent re-check, 2026-10-29, at commit `54c8ad3`, of the fixes claimed in commits `6e5491e` (pre-registration), `94e1bcb` (fixes) and `54c8ad3` (fresh-clone log) against `docs/audit_phases_0_2.md`. Nothing in the repo was changed apart from adding this file.

Method: I read the code, not the decision log, and ran small throwaway scripts in the `nettwin` env. All simulation checks ran on validation networks only (generated validation graphs `generated_split(cfg)["val"]` and `validation_real`) with scenario seeds 8700 to 8961, inside the already-reserved range 8000 to 8999. No test network was simulated. For germany50 I only checked that its demand-matrix names are valid topology nodes (no simulation, no metrics). AARNet was not simulated. The numbers below are audit evidence, not project results.

`pytest`: 80 passed (matches the log).

## Verdicts by audit ID

| ID | Claimed fix | Verified | Evidence |
|---|---|---|---|
| C1 | AARNet becomes dev / demo; test is Geant2012, germany50 plus 5 rule-picked Zoo networks; 5 validation_real; L13 near-duplicates skipped; simulator tests off AARNet; README lists test-driven decisions | **Yes** | Re-ran the selection logic from `scripts/select_topologies.py` without writing: 79 representatives, test picks, validation_real picks and the 69-network reference pool are identical to `config.yaml` and `results/topology_selection.json`. No reference-pool network shares a family with dev, test or validation_real. Pre-registration commit `6e5491e` (23:02) predates the fix commit (23:37). `tests/test_simulator.py` uses `gabriel/20/1` and toy graphs only. Only remaining test-network touch is `tests/test_env.py` loading Geant2012 / germany50 for size and connectivity. README lists L2 to L6 and L9. All L13 names are in `topologies.skip`. See new finding N5 for siblings of the five new test picks. |
| C2 | Background end-to-end probe loss (per-service base rate plus bursty episodes) from the `measure` stream | **Yes** | Code: `simulator.py:486-496`, fixed-shape draws, shared by the counterfactual. On a fault-free week of `gabaug/17/1000232`, 3.3% of post-warmup service-steps lost at least one probe (was 0.0004% at audit). "Any lost probe" is useless. A stronger, legitimate per-service Poisson test against the warmup rate still fires on 14 of 40 fault-free 4 h windows, and catches silent gray failures at 0.005% / 0.02% / 0.1% loss on 6 / 8 / 15 of 15 links. Small failures are no longer given away; larger ones remain detectable, which is correct. |
| M1 | Down interface reads util 0 / discards 0 with latency missing; healthy probe timeouts and probe-responder gaps; `pop_status` from node_down and PoP gaps, after a timeout, with p = 0.8; status-blind variant blanks down links and drops status events | **Partly** | Status-blind blanking and the `pop_status` sources are as claimed (`simulator.py:502-510, 595-607`). The literal audit pattern "util present, latency missing" is no longer unique: on 55 validation scenarios, 12.3% of such cells were down links. But adding `util == 0` restores the shortcut: 2,574 of 2,581 cells (99.7%) with util exactly 0 and latency missing were down links, and every visible down-link cell has util exactly 0 and loss exactly 0. See N2. |
| M2 | Maintenance placed first; 15% of faults maintenance-linked (half on the maintained element); 10% concurrent; 15% daytime maintenance | **Yes** | Sampling 100 scenarios on one validation graph: 16.5% maintenance-linked, 45% of those caused by maintenance, 12.8% daytime maintenance, 7 concurrent of 200 faults. On 55 simulated validation scenarios, "blame the maintained element when a fault overlaps a change window" is right on 8 of 21 faults, so it is no longer a perfect rule. Residual timing gap: see N6. |
| M3 | Per-path carried-load fixed point (4 passes); chain test; time-shared routing states; convergence loss at hard down and restore | **Yes** | Code `_carried` (`simulator.py:302-323`) is correct: survival is accumulated over earlier hops of the same path only. The toy-chain test passes. On a link_down that pushed rho to 2.80, 4 passes and 20 passes give identical util (max difference 0.0) and service loss (max 3e-14 pp), so 4 passes is enough. |
| M4 | 400 generated graphs, every n 15 to 60, seeds outside the shipped scheme, augmented from the reference pool; split 308 / 92 by graph; relative impact thresholds | **Yes** | Split is 308 / 92, disjoint, all 46 sizes in validation. On 30 validation graphs: median average degree 2.49, median mean link length 390 km, 8 of 30 have 0 km dual-router links, median pseudo-longitude span 40.8 degrees. Impact thresholds are relative (`subtle_latency_rel`, `obvious_latency_rel`). The augmentation reads only `reference_pool` from the selection file. |
| M5 | Per-fault `obs_status`, `obs_link_util`, `obs_link_counters`, `obs_e2e`, `observable`, as pre-registered | **Partly** | Definitions in `_impact` match the pre-registration. A silent 0.3% gray failure on the one service-free link of `gabaug/17/1000232` is correctly `observable = False`, `subtle`. But the same fault with a concurrent link_down elsewhere becomes `observable = True`, `obvious`, 18 services affected. Impact is measured against a counterfactual with every fault removed. See N1. |
| M6 | Status lag 0 / 1 / 2 steps (0.5 / 0.35 / 0.15) plus a 10% missed poll; twin rebuilt from lagged events | **Yes** | Code `simulator.py:572-594`. On 26 link_down faults in validation scenarios, the first root status event lagged 0 / 1 / 2 steps 14 / 8 / 4 times (54% zero lag). `observed_kg` is built only from `res.observed_events`, so it inherits the lag. The down / up cancel logic gives the right net state when lags reorder events. |
| M7 | Per-link per-step labels (2 root, 1 symptomatic, 0 unaffected) in `truth/<sid>/labels.parquet` | **Yes** | `_labels` matches the pre-registered definition. Over 100,224 steps outside every fault window plus 3 steps, no link had a non-zero label. `write_scenario` puts labels and the manifest under `truth/`; `observable/<sid>/` holds only the two Parquet files and `observed_events.json`. |
| M8 | Pre-registered targets; `--log-tuning` appends targets with the config hash; at most two adjustments | **Yes**, with caveat N4 | `results/tuning_log.jsonl` has three entries with distinct hashes. The final hash `425a5f41a9cef7af` matches `config.yaml` now and `phase2_summary.json`. I rebuilt both earlier configs by reverting the logged changes in text: they hash to exactly `a7ec203211ccafb0` (adjustment 1) and `4adbffecb57b013b` (run 0). So the config history is as described and nothing else in `config.yaml` was changed between runs. The sensitivity run is parked for Phase 4, as logged. |
| m1 | Gray visibility per fault window | **Yes** | Silent then visible gray failure on the same link in separate windows: counter loss 0.000% in the silent window, 0.80% in the visible one. |
| m2 | Flapping lower bound `min(0.02, max_down_frac)` | **Yes** | `simulator.py:378`. |
| m3 | Change review includes event-day factor and a 1.15 margin | **Yes** | Of 91 non-noop maintenance windows, none exceeded 0.85 realised util with faults removed. The one window that hit 1.0 did so only because of an overlapping fault. |
| m4 | Docstring rewritten | **Yes** | `simulator.py:1-36` describes the noisy history week, hot links, N-1 and the 0 km rule. |
| m5 | PoP gaps hit services at either endpoint | **Yes** | `simulator.py:526`. |
| m6 | Figure examples inside config ranges | **Yes** | All 12 AARNet examples in `phase2_summary.json` are inside the current ranges, durations included. |
| m7 | link_down / node_down impact class ignores the root's own drop | **Yes** | `simulator.py:679`. |
| m8 | Hot links never zero capacity | **Yes** | `hot` requires `peak > 0`; final `np.where(cap > 0, cap, floor)`. |
| m9 | Real-demand branch validates names, tested on a toy matrix | **Yes** | `test_real_demand_matrix_branch` passes. germany50's 47 demand rows contain no unknown names, so the branch will not raise on it (names only, no simulation). |
| m10 | Wilson intervals, 188 validation scenarios | **Yes** | `wilson()` in `phase2_simulate.py`; summary reports 188 scenarios, 370 faults. |
| m11 | Fault-free windows split by maintenance | **Yes** | `fault_free_windows.non_maintenance` / `.maintenance` in the summary. |
| m12 | `seeds.reserved_ranges` and `check_fresh_seed` | **Yes** (tool only) | Function exists and is tested. It is not called anywhere yet; the log assigns that to Phase 3. |
| Other minors | Event-day ramp, gray and surge ramp-down, truth apart from observable, convergence loss at onset, TCP goodput, flap drops kept out of counters on purpose | **Yes** | `event_factor` ramps over an hour (test passes). Ramp-downs in `_run`. Convergence loss in `simulator.py:456-466`, services only. Goodput factor `simulator.py:430-432`. Dead router counters read 0 via neighbours rather than NaN: a documented deviation from the audit suggestion, folded into N2. |

## New findings

### N1 (major): impact and observability of concurrent faults include the other fault's effects

`simulate()` compares each fault against one counterfactual with every fault removed (`simulator.py:330-331`). For a concurrent fault, deltas in its window include the other fault's effects. Check: a silent 0.3% gray failure on a service-free link is `observable = False`, `subtle` alone. With a link_down elsewhere overlapping it, the same fault is `observable = True`, all four channels except status true, `obvious`, 18 services affected. All 12 concurrent faults in my 55 validation scenarios were flagged observable. By code reading, `obs_status` (`simulator.py:650-653`) also credits any status event on the root's links, including planned-outage events from a maintenance "down" on the same element. So a caused-by-maintenance gray failure on a link under planned outage counts as observable.

Effect: the M5 rule "exclude unobservable faults from headline metrics" does not work for concurrent faults (about 4 to 7% of faults), impact classes for them are inflated, and T1 / T2 numbers are slightly off. Fix: compute each fault's impact against a counterfactual with only that fault removed (keep the others and maintenance), and count only fault-caused status changes in `obs_status`.

### N2 (major): the down-link shortcut survives as "util exactly 0 with latency missing"

T4 tests the literal pattern from the audit, so it passes (7.6%). Adding `util == 0` makes the pattern 99.7% specific to down links (2,574 of 2,581 cells). Healthy links never read exactly 0 because util noise is added before clipping, and loaded links sit well above 0. A node_down shows the same on every incident link of one PoP, while a PoP gap blanks them, so node_down vs PoP gap is also separable from link telemetry alone. Telemetry also changes at exactly `t_start`, so the M6 status lag does not delay onset for a telemetry-based detector.

Physically this is realistic (a down interface does read 0 traffic), so the fix is mostly about claims and ablations, not physics. The decision log should not say the M1 signature is gone. The status-blind variant is the only setting where it is gone, and Phase 4 / 5 results for link_down and node_down should be reported both ways. If a non-blind variant must be hard, add a small chance that a down interface's counters are unreadable, and replace T4 with a check on the best single-cell rule, not one fixed pattern.

### N3 (minor): T3 fault-free windows are not length-matched as pre-registered

The pre-registration says "fault-free windows of matching length". `diagnostics()` draws fault-free windows of 12 to 72 steps, while fault windows run up to 288 (gray) and 576 (misconfig) steps. This raises the fault hit rate relative to the fault-free one, so J is overstated and the pass is conservative. The result stands, but the implementation deviates from what was registered and should be logged or fixed.

### N4 (minor): the tuning hash covers `config.yaml` only

Adjustment 1 also changed the demand-surge factor from uniform to log-uniform. That is a code change. My run-0 reconstruction matches the logged hash with the old `[1.3, 4.0]` range, which confirms that the hash did not see the sampling change. Hash the relevant source files too, or record the git commit with each tuning entry.

### N5 (minor): siblings of the five new test networks are not on the skip list

`Garr199901` to `Garr201110` (12 snapshots of test Garr201201) and `Belnet2003` to `Belnet2009` (7 snapshots of test Belnet2010) are not in `topologies.skip`. Nothing uses them today (siblings never reach the reference pool, which takes one representative per family, and training uses only generated graphs), so there is no leak now. PLAN.md section 3 calls `topologies.skip` the full list, though, so add them before any code draws Zoo networks directly.

### N6 (minor): residual timing gap around maintenance

Isolated faults are still kept at least 36 steps from every change window (minimum seen: 44 steps), and maintenance-linked faults start no later than 12 steps after a window ends. So the 13 to 36 steps after any change window are fault-free by construction. This is a weak rule, but an agent with `get_change_log` could learn it. Either let isolated placement ignore maintenance, or widen the "follows a window" range to cover the gap.

### N7 (minor): demo scenario folder names encode the fault type

`data/scenarios/phase2_demo/observable/` has folders like `aarnet_link_down_severe` and `aarnet_routing_misconfig_mild`. This is demo data only, but the Phase 3 builder must use opaque scenario IDs, since the ID is part of the observable path.

### N8 (minor): observability ignores telemetry gaps

The flags come from noise-free truth, so a fault fully covered by a link gap or probe gap (with no crossing services) is still `observable = True` even though no observed data shows it. Gaps last up to 24 steps and faults as few as 2, so this is rare. Either mask the noise-free deltas with the gap masks, or document it.

## Summary

- **Verified:** C1, C2, M2, M3, M4, M6, M7, M8 and all of m1 to m12 and the other minors. Highlights: the selection rule reproduces exactly; the tuning history in `results/tuning_log.jsonl` rebuilds byte-exactly from its descriptions; the carried-load fixed point is correct and converged; ground truth is separated from observable files.
- **Partly verified:** M1 (the status-blind variant and the event-source changes work, but a one-condition refinement restores the down-link signature, N2) and M5 (correct for isolated faults, wrong for concurrent ones, N1).
- **New:** two major findings (N1, N2) and six minor ones (N3 to N8). None of them involves test-set leakage. N1 should be fixed before Phase 3 builds datasets, since it decides which faults are excluded from headline metrics.
