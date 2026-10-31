# Final audit (before the write-up)

Independent audit at commit `7cba229`, 2026-10-31. Earlier phases had their own audits (`docs/audit_*.md`); this one does not redo them. Scope:

1. Phases 7 and 8 (not audited before): leakage, the test-once guarantee, the Phase 8 breach labels and window classes, and whether the app shows only committed results with the right test / dev labels.
2. Cross-phase consistency: headline test numbers in `docs/decision_log.md` and the app against the committed results files.
3. Proof that each test set was scored once per phase.
4. Claims against evidence.
5. A fresh-clone check of Phases 7 and 8 (train and tune only, not test) with the dataset key, and the full test suite.

Nothing was fixed. Every number below was read from a committed results file or from a command run for this audit. Telemetry is synthetic, topologies are public, and none of this is Telstra's network.

**Summary.** No critical findings. No leakage was found in Phase 7 or Phase 8. Every headline test number checked in the decision log and the app matches its results file. Git history, LOCK files, frozen hashes and file timestamps are consistent with each test set being scored exactly once per phase. Two major findings are about claims, not numbers: the not-pre-registered Opus 5.5 versus GNN result is presented next to pre-registered results without that label, and the app's limitations page words two inconclusive results as equivalence. The rest are minor.

The fresh-clone check passed:
- Phase 8 tuning in a clean clone and a new env reproduced both frozen model files byte for byte. Every committed result matched except one wall-clock field.
- `--test` refused without scoring.
- 141 tests passed.

---

## 1. Phases 7 and 8

### Phase 7 (AARNet dev traces and demo)

Checked:
- `scripts/phase7_aarnet_traces.py` reads only the dev split, through `guard(p, {"dev"})`. Every record in `results/phase7/aarnet_dev_records.jsonl` has `split: dev` and agent `code_sha256` 87a0082db4b3, the same as `results/phase6_frozen.json`. The agent hash recomputed now also matches, so no frozen agent file changed after the freeze.
- The agent sees only `Incident(obs, okg, ...)` (observable telemetry and the observable KG). Ground truth (`inc["root"]`, `change_target`) is used only for scoring and for choosing which links the demo plots. No leakage.
- 7 incidents x 2 variants x 2 models = 28 agent runs, one log line each and no duplicates. Cost in the logs is $0.9169, matching "$0.92". Project spend: Phase 6 logs $40.44 + Phase 7 $0.92 + $0.30 unlogged allowance = $41.65, matching the log.
- Every number in the Phase 7 decision-log entry (faults top-1 0.500 for both agents and the GNN; fault recall Haiku 0.667 / 0.833, Opus 0.833 / 0.833, GNN 0.611 / 0.556; both agents "no fault" on D04 demand_surge; both name planned maintenance on D06) matches `results/phase7/aarnet_dev_summary.json` and the records.

**App** (`app/streamlit_app.py`):
- It reads only `results/` files that are committed: `phase7/demo_bundle.json`, `phase7/aarnet_dev_summary.json`, the two Phase 6 test `summary.json` files, `phase6/test_haiku_vs_opus.json` and `phase8_test.json`. No data folder and no API key are needed (`tests/test_app.py` checks this).
- Test and dev are labelled correctly everywhere: the explorer is "AARNet (dev)" with the "not held-out evidence" caption; the AARNet dev table on the test page is labelled "reported separately; seen during development"; the forecasting page's split selector names test and "AARNet dev ... reported separately". Haiku against Opus is labelled "descriptive, not pre-registered". The honesty line is on every page and in the sidebar.

### Phase 8 (capacity forecasting)

**Leakage: none found.**
- Labels: `targets` takes `util[t+1 : t+13]`, so the window is (t, t+12]. The grid ends at T - 13, so no window runs past the scenario. Pairs whose 12 future steps are all missing are dropped, as pre-registered.
- Inputs, all at or before t:
  - forward-fill is causal;
  - rolling windows end at t;
  - yesterday's horizon is `util[t-287 : t-275]`, which is (t-288, t-276];
  - `u_288` is `u[t-288]`;
  - the latency EWM is shifted one step;
  - the GNN features (`b2_features`) use past EWMs and lags only;
  - `Temporal.sequence` uses unpadded convolutions over a front-padded sequence, so step t sees only the W steps up to t.
  - No clock, step index or file metadata is used. The first prediction time is 288, so every lag index is valid.
- Splits:
  - Train uses 300 train scenarios (`default_rng(8008)`).
  - Tuning, early stopping, model selection and both thresholds use 120 validation scenarios (`default_rng(8009)`).
  - Test and dev are read only in `--test`.
  - The encoder is the frozen Phase 5 `gat-h32-s0`. Its hash is checked, it gets no gradient, and it was trained on train only.
- Head normalisation statistics come from train only.

**Breach labels and window classes:** they match the pre-registration.
- `window_class` uses `[t - 11, t + 13)`, the integer form of (t - 12, t + 12].
- A fault counts over [t_start, t_end + 3), the same LAG as the Phase 4 harness. Fault takes precedence over maintenance.
- Maintenance covers all logged actions (drain, down, no-op). In the validation truth these are 480 drains, 266 downs and 202 no-ops, and none of them is unlogged.
- Ground truth is used only to stratify the scoring.

**Test once:** see section 3.

---

## 2. Cross-phase consistency of headline test numbers

Each headline number below was checked against its results file. Where the decision log rounds, it was checked to the shown precision.

| Source | Checked against | Result |
|---|---|---|
| Phase 5 test headline table (both variants, all six methods, hard / easy top-1 / top-3, root PR-AUC, maintenance and fault-free alarm rates) | `results/phase5_tables.md`, `results/phase5_test.json` | all match. Example: GNN hard top-3 0.587 (val 0.606), B2 0.356, GNN - B2 +0.231 [+0.168, +0.297]; GNN seed fault-free rates 0.082 / 0.102 / 0.081, so the table's 8.8% mean and the text's "8.1% to 10.2%" are both right |
| Phase 5 verdicts (X1 to X9, GNN - MLP +0.044, easy top-3 margin +0.086, maintenance +0.19 to +0.34) | `results/phase5_test.json` comparisons | all match |
| Phase 6 test table (9 methods x 2 variants x 5 columns, validation in brackets) | `results/phase6/test_claude-{haiku,opus}-5-5/summary.json`, `results/phase6/val_haiku_v2/summary.json` | all 90 test cells match, and all bracketed validation values match |
| Phase 6 paired verdicts (agent vs GNN+CL, G1 to G13, Opus A2 - A0) | same summaries, `comparisons` | all quoted points, CIs and verdicts match |
| Phase 6 cost, forced submissions, groundedness | `agent_cost_latency_groundedness`, the agent logs | all match (Haiku $1.96 / 1002 runs, Opus $32.68 / 668 runs; forced 85 / 119 / 60; lenient and strict ungrounded rates) |
| Haiku against Opus | `results/phase6/test_haiku_vs_opus.json` | all quoted numbers match |
| Phase 7 dev | `results/phase7/aarnet_dev_summary.json` | all match (section 1) |
| Phase 8 validation (pairs, base rates, AP per configuration, thresholds, validation P / R) | `results/phase8_val.json` | all match |
| Phase 8 test and dev table, paired verdicts, F1 to F6, "83%" | `results/phase8_test.json` | all match (26,456 / 31,725 = 0.834) |
| App | reads the same files directly | the app computes nothing; it formats committed values, so it cannot drift from them |
| README Phase 5 line (0.59 vs 0.36, 48%) | `results/phase5_tables.md` | matches |

**Mismatches found: none.**

---

## 3. Each test set scored exactly once

| Phase | Evidence |
|---|---|
| 5 | `results/phase5_test.json` has one commit (`b0c0957`). The pre-registration commit `f0a4461` is at 11:37:42. The file's mtime is 11:52:41, and the logged run time is 15 min, so it started just after the pre-registration commit and there is room for exactly one run. All 9 hashes in `results/phase5_frozen.json` match the current files. The earlier fresh-clone check regenerated it byte-identically after a deliberate delete in the clone, not in this repo. |
| 6 | One commit (`8461831`) for both test folders. Each folder has a `LOCK.json`: Haiku started 19:22:11 and its summary mtime is 20:20:13 (58 min, as logged); Opus started 20:20:38 and its summary mtime is 21:37:49 (77 min). Haiku has 1002 log lines (167 x 6) and Opus has 668 (167 x 4), with no duplicate (incident, condition) pairs. Every record has `code_sha256` 87a0082db4b3, which equals the frozen value and today's recomputed hash. The validation rerun commit `5a3829e` (19:13) comes before the Haiku LOCK. |
| 7 | Dev only, so there is no test set. See minor finding m3 on reruns. |
| 8 | `results/phase8_test.json` has one commit (`7cba229`). Sequence: pre-registration commit 21:56:37; `phase8_val.json` and `phase8_frozen.json` written 22:02:09 (164 s tuning run, so it started about 21:59:25, after the pre-registration); freeze commit `0d3d375` at 22:02:55; `phase8_test.json` written 22:04:11, with a logged run time of 63.2 s. `--test` needs `phase8_frozen.json`, which did not exist before 22:02:09. That leaves 122 s, enough for only one 63 s run. The test run therefore started after the freeze was committed. All 5 frozen hashes match the current files, and the `frozen` block inside `phase8_test.json` equals `results/phase8_frozen.json`. `--test` refuses while the output exists (confirmed in the fresh clone, section 5). |

**Verdict:** the evidence fits exactly one test scoring per phase, after a committed pre-registration and freeze. The model files under `data/models/` are gitignored, so their hashes can only be checked on this machine. They do match here.

---

## 4. Findings

### Critical

None.

### Major

**M1. The Opus 5.5 versus GNN result is not pre-registered, but it is presented next to pre-registered results without that label.**
- The Phase 6 design change (decision log, "pre-test design change") pre-registers only Opus A2 - A0, and says "G1 to G8 are not pre-registered for Opus".
- The claim appears in three places:
  - The decision-log bullet "Opus A2: localisation is better than the rule on faults top-3 in both variants ..." sits in the "Agent against the GNN + change-log rule" block, among the pre-registered Haiku verdicts.
  - The app's Test results page lists "Opus A2 vs GNN+CL" and "Opus A2 vs GNN" in the same paired-verdict table as the pre-registered pairs, with no label.
  - The Limitations page says "Opus 5.5 with tools beats it on faults top-3".
- The result is also fragile. Against GNN seed 0, the GNN the agent's tool actually uses, Opus A2 faults top-3 is supported with status (+0.078 [+0.021, +0.142]) but inconclusive status-blind (+0.053 [-0.005, +0.112]). Faults top-1 is inconclusive in both variants.
- The numbers are correct; the presentation is stronger than the design. The write-up should label this result post hoc and give the seed-0 comparison next to it.

**M2. The app's limitations page states two inconclusive results as equivalence.**
- "Haiku 5.5 with tools matches it (inconclusive)": Haiku A2 - GNN faults top-3 is +0.044 [-0.006, +0.095] with status and -0.022 [-0.079, +0.034] blind. An inconclusive CI does not show a match. The CIs allow up to about 0.08 worse or 0.10 better.
- "A one-line rule that silences the GNN on logged changes does as well": distractor claims are 1/30 or 0/30 for both. That is fine as a description, but it is not a tested equivalence.
- Suggested wording: "is not distinguishable from it at this sample size (CI ...)".

### Minor

**m1. Phase 8 tuning selected on rounded AP with a tie rule that was not pre-registered.** `tune()` rounds validation AP to 4 decimals before `max()`, so 31 / 0.05 and 63 / 0.05 tie at 0.973, and the first in grid order wins. The decision log states the tie rule, but the pre-registration says only "selected by validation average precision". The unrounded APs may differ. The practical effect is probably negligible, but the rule should be disclosed as a deviation.

**m2. The Phase 8 "Deviation, logged before the test run" was not committed before the test run.**
- The `torch.set_num_threads(1)` change is in the frozen code (`0d3d375`), so it did come before the test.
- The decision-log text describing it was first committed in `7cba229`, together with the test results. So were the validation-tuning prose and the test-suite fix.
- Suggested wording: "made before the test run, logged with the results".

**m3. Phase 7 has no run-once or rerun record.**
- `phase7_aarnet_traces.py` opens its agent logs and records with `"w"`. A rerun would erase the earlier run's traces and its cost from the spend total, which counts only logs on disk.
- The incident rule is described as "fixed before any run", but no commit records it before the run. The Phase 6 test commit (21:39:35) and the first Phase 7 output (21:44:44) are about 5 minutes apart, enough for an earlier run of about 3 minutes.
- No evidence of a rerun was found, and these are dev numbers. But neither the cost total nor the claim that the demo traces were not hand-picked can be proven from the repo.

**m4. Phase 8 maintenance window class: few breaches, and NaN resamples are dropped.**
- Only 114 test breaches (0.4%) fall in maintenance windows, apparently in few scenarios.
- Persistence maintenance recall is 0.474 with CI [0.062, 0.615]. The point estimate sits near the top of a very skewed interval.
- `ci()` silently drops bootstrap resamples where the ratio is undefined (no maintenance breach drawn), so these CIs are conditional on drawing at least one breach.
- F6 is reported as inconclusive, which is right. This class should not carry any conclusion.
- The window has no `MAINT_TAIL`, unlike the Phase 4 harness, so the end of a drain may count as fault-free. This matches the Phase 8 pre-registration, so it is a difference of definition, not an error.

**m5. Phase 8 F4 interpretation is stronger than the design.**
- "An encoder trained for anomaly detection adds nothing over plain lag features here" goes beyond what the comparison tests.
- The comparison is LightGBM on 19 lag features against an MLP on the embedding plus 3 scalars. That mixes model class with feature set.
- Also, the encoder was trained on train scenarios, so the head learns on embeddings of scenarios the encoder has seen. That may make the head's train inputs unrepresentative of test.
- What the result supports: "the frozen encoder plus a small head is worse than LightGBM on lag features (-0.027 F1)". Whether the embedding adds anything to the lag features is untested, for example LightGBM on lags plus embedding.

**m6. Phase 8 attribution was not computed.** "Most test breaches (83%) fall in fault-free windows: diurnal peaks and event days, not faults." The 83% is computed, but no script splits fault-free breaches into diurnal peaks and event days.

**m7. Phase 8 per-scenario counts are not saved.** `phase8_test.json` stores only aggregates. The bootstrap CIs and verdicts cannot be recomputed from committed files without scoring test again. Saving the per-scenario (tp, fp, fn) rows that `score_split` already builds would make them checkable.

**m8. Many comparisons, and the "mostly differs little" summary is selective.**
- `test_haiku_vs_opus.json` holds 68 paired comparisons (27 supported, 41 inconclusive) with no multiplicity correction, so some "supported" verdicts are expected by chance.
- For Opus A0 against Haiku A0, the decision log says the two "mostly differ little" and names two supported differences. The file has 8 supported A0 differences across the two variants, for example status-blind change-overlap top-1 +0.100 and type accuracy with status +0.067.
- The whole comparison is labelled descriptive, so this is minor. The write-up should say how many comparisons were made.

**m9. Smaller app issues.**
- Explorer: the "root cause" column for GNN+CL shows the GNN's top element even when the rule has silenced the GNN (`claims fault` 0).
- Test results caption: "167 test incidents ...: 120 faults, 30 maintenance distractors, and 40 faults whose window overlaps a logged drain or outage". 120 + 30 + 40 does not give 167. The 40 includes 23 of the 120, plus 17 stratum faults; say so.
- Forecasting page: the dev option says "reported separately" but not "seen during development", unlike the other pages.

**m10. README is behind.** The README results stop at Phase 5. Phases 6 to 8 are not there yet. That is expected before Phase 10, but the number checker should cover the new sections.

---

## 5. Fresh-clone check (Phases 7 and 8, train and tune only) and the test suite

**Passed.** Script and full log: in this session's scratchpad, not committed. The clone and env were deleted afterwards.

- **Setup:**
  - `git clone` of this repo at `7cba229`. No `.env`, no `data/`, no key.
  - New conda env `nettwin-finalaudit` from `environment.yml`, with `pip install --no-deps -e` of the clone.
  - `nettwin` imports from the clone's `src/`, not from the working repo. The main env's editable install points at the working repo, so a new env was needed for the check to mean anything.
- **Without the key:** `phase3_build_dataset.py --estimate` refuses (exit 1, "the dataset key is local only").
- **With the key:**
  - `phase0_check_env.py`, `phase1_build_kg.py`, `phase2_simulate.py` and `phase3_build_dataset.py --workers 6` all exit 0. The Phase 3 build took 21.8 min, the same as the logged Phase 3 check.
  - All 8 truth manifests (`data/datasets/v1/truth/*/*/manifest.json`) are byte-identical to the local build.
  - The observable tree has 15,520 files in both builds. A sample of 200 files (every 50th in sorted order, first 200) shows no difference.
- **Phase 8 tune (train and validation only):**
  - The frozen Phase 5 encoder `gat-h32-s0.pt` was copied in from the local build, because a full Phase 5 retrain takes about 4.3 h. The script checks it against `results/phase5_frozen.json`, and the Phase 5 fresh-clone checks already showed that training reproduces it.
  - `phase8_forecast.py` exits 0 in 156.7 s.
  - `data/models/phase8/lightgbm.txt` and `gnn_head.pt` are byte-identical to the local models (sha256 77175653... and eff386cd..., equal to `results/phase8_frozen.json`).
  - `git status` shows one changed file, `results/phase8_val.json`, and the only changed line is `"seconds"` (164.1 to 156.7, wall-clock). `results/phase8_frozen.json` is byte-identical, so every validation number, both thresholds and every hash reproduce exactly.
- **Phase 8 `--test`** refuses with "results/phase8_test.json exists; the test set is scored once" (exit 1). Test was not scored.
- **Phase 7:** the agent run was not repeated. It spends API budget and LLM output is not deterministic, so a rerun could not be compared byte for byte. The Phase 7 deliverable that must work from a clone is the offline app, and `tests/test_app.py` renders every incident, variant and page there with no data from Phase 7 and no API key. The frozen-hash guard that the Phase 7 script relies on was checked in section 1.
- **Full test suite in the clone:** 141 passed, 1 warning (torch.jit deprecation), in 31 s.

One small inconsistency: the Phase 8 decision-log entry says "all 140 tests pass". The suite at `7cba229` has 141 tests, because the same commit changed `tests/test_app.py`. See m11.

### m11 (added after the clone check)

The test count in the decision log (140) is behind the committed suite (141). The log entry was probably written before the final `test_app.py` change. Update the count, or say which commit it refers to.
