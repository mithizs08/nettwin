# Audit: Phase 5 (line-graph GNN, ablations, validation results)

Independent audit, 2026-10-31, at commit `812f12f`. Scope: `PLAN.md`, `docs/decision_log.md` (Phase 4 audit fixes, Phase 5 pre-registration, Phase 5 results, parked rules), `docs/audit_phase4.md`, `results/phase4_tables.md`, `results/phase5_tables.md`, `results/phase5_gnn_val.json`, `src/nettwin/gnn/`, `scripts/phase5_gnn.py`, the parts of `scripts/phase4_baselines.py`, `src/nettwin/baselines.py` and `src/nettwin/evaluation.py` that Phase 5 calls, `tests/test_phase5.py`, and the checkpoints in `data/models/phase5/`. Nothing in the repo was changed apart from adding this file.

Method: throwaway scripts in the `nettwin` env, outside the repo. They load the saved checkpoints (no retraining), refit B2 on train exactly as the pipeline does, rescore all 468 validation scenarios in both variants, and recompute every metric with independent code (expected tie credit, sklearn exact AP, window classes, thresholds, paired bootstrap). A second script edits observable telemetry and re-runs the GNN to test what drives its link_down / node_down ranking. Only train (for the B2 fit) and val were read. Nothing under `test` or `dev` was opened, and the dataset key was not read. Telemetry is synthetic; numbers below are audit evidence, not project results.

Grades: **critical** = can invalidate a headline RQ result if not fixed before the test run; **major** = biases a result or a claim, or leaves a path to test leakage, and needs a fix before the test run; **minor** = reporting, wording or hygiene with small expected effect.

## Summary

| ID | Grade | Finding | Fix |
|---|---|---|---|
| R1 | major | There is no test scoring path. `phase5_gnn.py` always retrains. Its only scoring code calibrates bins and thresholds on whatever split it scores, so reusing it for test would calibrate on test. Baseline calibration is not saved, and checkpoints are not hashed | Persist the val calibration and checkpoint hashes now, and add a separate test script that only loads them |
| R2 | minor | The 60-scenario selection subset is inside the 468 scored scenarios, and the GNN does much better on it (hard top-3 0.78 against 0.60 elsewhere). Excluding it changes no verdict | Report the headline on the 408 held-out scenarios next to all 468 |
| R3 | minor | The 1.000 top-3 on link_down / node_down comes from a telemetry signature. Status events play no part, since the GNN never sees them. Painting the signature on any healthy link makes the GNN rank that link first 97% to 98% of the time | Describe it as a rendering shortcut, and keep it out of any "hard detection" claim |
| R4 | minor | The MLP ablation is fair on budget, but it inherits the GAT's selected hidden size and keeps PoP pooling plus graph-derived static inputs | Add one sentence to the X6 wording |
| R5 | minor | Hygiene: the test-split guard is an argv string match, the script test cannot see the `split` variable, and `data/models/phase4/` still holds stale pre-Q10 calibration pickles | Fix or delete as listed in R5 |

No critical finding. Leakage checks pass. Every headline number spot-checked reproduces exactly from the saved checkpoints.

## 1. Ground truth and banned features at inference; training labels

From the code:
- At inference the GNN gets only `b2_features(obs)` (the 10 Phase 4 B2 features: util, latency and discard levels, one-day lags, short rolling stats, two missing flags) and `graph_for(topology)`. The graph comes from the operator inventory: line-graph edges, PoP degree, link length, normal-path and service counts at t = 0. `GNNMethod.score(obs, kg, ix)` never receives the manifest, labels, events or service probes. `predict` sees only `feat` and the graph.
- No clock or time-since-start input. Features are causal: windows end at t, and `Temporal.sequence` matches the window model (`test_fast_prediction_equals_window_model`, `test_prediction_is_causal`). Day-0 lag features are neutral, but training samples come from steps >= 288 only, and day 0 is never scored.
- Training labels: `_samples_one` reads `truth/train/<variant>` only. Targets are label >= 1 for detection and the roots of observable faults active at the step (train manifest) for localisation. Normalisation statistics come from train samples.
- Validation labels are used in one place outside evaluation: the selection score on the 60-scenario subset (pre-registered; see R2).

## 2. Same window and same calibration for every method

- **Window.** Every method goes through `scenario_record` with the same oracle [t_start, t_end) and the same q90 / max / mean aggregations. The GNN aggregates its per-step element probabilities; B1 / B2 aggregate link and PoP scores. Confirmed identical by recomputation.
- **Threshold.** Same procedure for all: the smallest score at which at most 5% of the method's own validation fault-free windows alarm. My recomputation gives the committed thresholds exactly (gat-h32-s0 0.8372, s1 0.7701, s2 0.7426, MLP s0 0.8850, B2 0.6698 with status), and fault-free rate 0.0500 for each.
- **Bins.** Different but both label-free: baselines use quantiles of a 2% sample of their own val scores, and the GNN uses a fixed logit grid. Binned AP is within 0.0005 of exact AP for every value I checked (Section 7), so this does not matter.
- **Asymmetry.** The GNN's architecture and hidden size were chosen on validation labels (the 60-scenario subset). The baselines' only choice, the PoP aggregation, was made on train. That is pre-registered and disclosed; R2 measures its effect.

## 3. The 1.000 top-3 on link_down and node_down

This comes from the observable telemetry, not from a leak. The GNN never sees status events (they are not among its inputs), so the with_status result also does not come from status events. What it sees is the down-link rendering in the link telemetry.

Signature frequency on val (observable files only):

| | link_down root cells with the signature | other links in the same fault windows | link_down windows where another link also shows it |
|---|---|---|---|
| with_status (util exactly 0, latency missing) | 6,187 / 6,218 (99.5%) | 163 / 274,283 (0.06%) | 10 / 162 |
| status_blind (all three metrics missing) | 6,195 / 6,195 (100%) | 1,126 / 273,551 (0.41%) | 143 / 160 |

Counterfactual edits, GAT-h32 seed 0. Only observable arrays were edited; the model and the window stayed the same:
- **A:** replace the true root's telemetry in [t_start, t_end + 3) with the same steps one day earlier.
- **B:** paint the signature (util 0 / discards 0 / latency missing with status; all missing status-blind) on a random link, or on every link of a random PoP, in a same-length window with no fault and no maintenance.

| | true root top-1 / top-3 | A: signature removed | A: median rank | B: painted element before | B: after painting |
|---|---|---|---|---|---|
| with_status link_down (n = 162) | 0.944 / 1.000 | 0.025 / 0.093 | 14 | 0.025 / 0.062 | 0.981 / 1.000 |
| with_status node_down (n = 152) | 0.875 / 1.000 | 0.033 / 0.039 | 41 | 0.000 / 0.007 | 0.849 / 1.000 |
| status_blind link_down (n = 160) | 0.944 / 1.000 | 0.019 / 0.075 | 14 | 0.031 / 0.062 | 0.969 / 1.000 |
| status_blind node_down (n = 151) | 0.881 / 0.993 | 0.033 / 0.040 | 43 | 0.000 / 0.007 | 0.848 / 0.993 |

Ranks here count ties in the root's favour; with continuous probabilities ties are rare.

Removing the signature destroys the result, and painting it on a healthy element reproduces it. The ranking is fully explained by the signature, as expected from the dataset card (audit re-verify N2).

**R3 (minor).** Two consequences:
- The model localises *any* whole-window blanking as a root. In status_blind most link_down windows also contain gaps on other links (143 of 160), so the result relies on the root being blanked for the whole window while gaps are partial. It does not show an ability to tell a down link from a long collector outage.
- It is a plausible contributor to the GNN's 48% maintenance-hour alarm rate, since planned "down" actions render the same way. I did not test this.

Fix: in the write-up, describe the link_down / node_down numbers as "recognises the simulator's down-link rendering". Keep the parked rule that they are never evidence of hard detection. Optionally add one row scoring these types with the signature removed (edit A) as a stress test.

## 4. The selection subset (R2, minor)

The subset is `default_rng(3003 + 5).permutation(468)[:60]`, the same 60 ids in both variants (the manifests are in the same order). All 60 are among the 468 scored scenarios. They hold 57 observable hard and 51 easy faults per variant, and 16 of the 60 are real networks.

GAT-h32 seed mean against B2, q90, paired scenario bootstrap (2,000 draws, my own):

| with_status | all 468 | excluding the 60 (408) | the 60 only |
|---|---|---|---|
| hard top-3, GNN / B2 | 0.629 / 0.383 | 0.606 / 0.377 | 0.778 / 0.421 |
| hard top-3, GNN - B2 [CI] | +0.246 [+0.198, +0.294] | +0.229 [+0.179, +0.279] | +0.357 [+0.233, +0.493] |
| hard top-1, GNN - B2 | +0.233 [+0.194, +0.274] | +0.217 [+0.171, +0.262] | +0.342 |
| easy top-3, GNN - B2 | +0.313 [+0.272, +0.355] | +0.309 [+0.266, +0.355] | +0.343 |
| hard root PR-AUC, GNN / B2 | 0.0757 / 0.0498 | 0.0742 / 0.0502 | 0.0907 / 0.0479 |

| status_blind | all 468 | excluding the 60 |
|---|---|---|
| hard top-3, GNN - B2 | +0.235 [+0.187, +0.283] | +0.215 [+0.165, +0.265] |
| easy top-3, GNN - B2 | +0.622 [+0.575, +0.668] | +0.621 [+0.574, +0.669] |
| hard root PR-AUC, GNN / B2 | 0.0775 / 0.0455 | 0.0759 / 0.0457 |

No verdict changes: every GNN-over-B2 difference stays supported with its CI well above 0. But the subset is unusually favourable to the GNN. Its hard top-3 there is 0.78, against 0.60 on the other 408. B2's is 0.42 against 0.38, so the subset is only slightly easier for B2. Seeds 1 and 2 were never used for selection and show the same gap, so this is mostly subset composition, not overfitting to seed 0. Including the subset inflates the GNN headline by about 0.02 on hard types.

Fix: report the headline and the X1 / X2 verdicts on the 408 held-out scenarios beside the all-468 numbers, or say in the results entry that the selection subset is included. (Test is unaffected; selection never touched it.)

## 5. The MLP ablation (R4, minor)

Checked and fair:
- Same training samples (the one 29,568-sample list in memory), features, static inputs, normalisation, heads, loss, optimiser, lr 2e-3, batch 32, 6 epochs and seeds 0 to 2 (`_train_one` passes the same arguments; `train_history` has 6 epochs for every MLP run).
- Parameters: 11,843 against 12,099 for GAT-h32 (2% fewer). The MLP replaces each conv by a `Linear(hidden, hidden)` with the same residual and LayerNorm.
- The MLP checkpoint s0 reproduces its committed numbers exactly (hard top-3 0.6204, easy top-1 0.7646).

Two caveats belong in the X6 wording:
- The MLP inherits hidden = 32, which was chosen for the GAT on validation. It got no selection of its own. That biases X6 toward the GNN, so the "MLP is as good" conclusion is if anything conservative.
- "No message passing" is not "no graph". The MLP keeps the link-to-PoP mean pooling for the PoP head and the graph-derived static inputs (end-PoP degree, normal-path and service counts). The decision log already says the graph adds little "beyond PoP pooling". The README sentence "most of the gain ... comes from supervised training, not from message passing" is accurate. A future "graph-free" claim would not be.

## 6. Checkpoints and the test path (R1, major)

What exists:
- All 21 checkpoints are in `data/models/phase5/` (gitignored). They are state dicts that include the normalisation buffers. Architecture, hidden size and ablation are encoded in the filename (`static_mask` is rebuilt from the ablation name). They load into `NetTwinGNN` without retraining, and the GAT-h32 seeds 0 to 2 and MLP seed 0 reproduce `phase5_gnn_val.json` to 4 decimals (Section 7). So test can be scored without retraining.
- The GNN val thresholds are stored in `results/phase5_gnn_val.json` (`variants.<v>.thresholds`). Its bins are a fixed grid.

What is missing, and why it is a risk:
- No code scores test from the checkpoints. `phase5_gnn.py` has no load option and always runs all 21 trainings.
- The only scoring code calibrates on the split it scores:
  - line 330 calls `P4.calibrate(root, variant, entries, ...)` with the evaluated entries, which sets baseline bins and thresholds from those scenarios;
  - line 344 sets each GNN threshold from the evaluated records' fault-free windows.
  
  `_evaluate_one` already takes `split` from the context. Pointing this block at test is the obvious shortcut, and it would calibrate every method on test.
- Baseline calibration (B2 model, bin edges, thresholds) is not saved by the current pipeline. `data/models/phase4/` holds only the old `calib_*.pkl` from 2026-10-30 12:36. Those are train-calibrated, from before the Q10 fix, and do not match the current results. Loading them for test would silently use the wrong calibration.
- No checkpoint hash is recorded in any committed result, so nothing ties a later test run to the evaluated weights. For reference, sha256 prefixes now: gat-h32-s0 `d81615b989d2503c`, s1 `8f5c8f320932101f`, s2 `aaaab0be00f602e6`.

Fix, before any test run:
1. Write a `data/models/phase5/calibration.json`, and commit its hash plus the checkpoint sha256 values in a results file. It holds the val thresholds and bin edges for every method and variant, and the B2 fit ids and seed (or a pickled B2).
2. Add `scripts/phase5_test.py`. It loads checkpoints by name, verifies their hashes, loads the frozen val calibration, refits nothing and calibrates nothing. It never calls `calibrate`, `threshold_for` or `make_edges`, and writes only test results.
3. Add a test that the test script never calls those functions and never reads `truth/test` before scoring is complete.
4. Delete the stale `data/models/phase4/calib_*.pkl`.

## 7. Spot-checks with independent code

All on validation, all 468 scenarios, recomputed from raw scores.

| Quantity | Mine | Committed |
|---|---|---|
| gat-h32 s0 / s1 / s2 hard top-3, q90 (with_status) | 0.6273 / 0.6366 / 0.6227 | 0.6273 / 0.6366 / 0.6227 |
| gat-h32 s0 / s1 / s2 easy top-1 | 0.7534 / 0.7601 / 0.7422 | same |
| gat-h32 seed-mean hard top-3, GNN - B2 | +0.246 [+0.198, +0.294] | +0.246 [+0.201, +0.294] |
| gat-h32 s0 hard / easy root AP, exact sklearn | 0.0843 / 0.2341 | 0.0841 / 0.2336 (binned) |
| gat-h32 s1 / s2 hard root AP (fine grid) | 0.0598 / 0.0831 | 0.0597 / 0.0830 |
| seed-mean hard root PR-AUC | 0.0757 | 0.0756 |
| MLP s0 hard top-3 / easy top-1 | 0.6204 / 0.7646 | same |
| B2 hard / easy top-3; hard root AP | 0.3831 / 0.5667; 0.0498 | 0.3831 / 0.5667; 0.0497 |
| gat-h32 s0 / s1 / s2 maintenance-only alarm rate | 0.4539 / 0.4817 / 0.5165 (mean 0.484) | same (table 0.484) |
| status_blind gat-h32 s0 hard / easy top-3 | 0.6250 / 0.8801 | same |
| status_blind gat-h32 s0 hard root AP | 0.0860 | 0.0858 |
| status_blind B2 easy top-3 | 0.2560 | 0.256 |
| gat-h32 s0 on the selection subset, hard / easy top-3 | 0.7895 / 0.8235 | selection log 0.7895 / 0.8235 |

My exact AP counts 81,197 hard positives against 81,202. The 5 cells are the overlap case seen in the Phase 4 audit; I attribute them to the first fault only.

## R5 (minor): hygiene

- The test-split guard in `phase5_gnn.py` checks only for the substring "test" in argv. The split is a hard-coded variable (`split = "val"`, or "train" for `--smoke`). `test_phase5_script_reads_train_and_val_only` searches for `/ "x" /` literals, so it cannot see the `split` variable or `CTX.get("split", ...)`. Fix: assert `split in {"train", "val"}` inside `_load` / `_evaluate_one`, and test that.
- Stale `data/models/phase4/calib_*.pkl` (see R1).
- The decision log's Phase 5 entry should name the checkpoint hashes once R1 is done.

## What should happen before the test run

1. R1: persist the val calibration and checkpoint hashes, and write a load-only test script that cannot calibrate on test.
2. R2: add the 408-scenario headline.
3. R3 and R4: wording in the decision log and later the README.
4. R5: tighten the split guard and delete the stale pickles.
