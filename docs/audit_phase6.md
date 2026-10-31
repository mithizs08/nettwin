# Audit: Phase 6 (LLM root-cause agent, validation eval)

Independent audit, 2026-10-31, at commit `7520e88`. Scope: `PLAN.md` section 7, `docs/decision_log.md` (Phase 6 pre-registration, smoke test, pre-test design change, Haiku validation eval, parked rules), the earlier audit files, `results/phase6/` (all four run folders, mainly `val_haiku/`), `src/nettwin/agent/` (`tools.py`, `loop.py`, `eval.py`, `rag.py`, the 9 runbooks), `src/nettwin/gnn/explain.py`, `src/nettwin/gnn/method.py`, `src/nettwin/alerts.py`, `src/nettwin/observed.py`, `scripts/phase6_agent.py` and `tests/test_phase6.py`. Nothing in the repo was changed apart from adding this file.

Method: throwaway scripts in the `nettwin` env, outside the repo. No Anthropic API call was made. The scripts:
- call every tool on real validation incidents with adversarial arguments;
- scramble all telemetry and events at and after each incident's end and check that every tool output stays byte-identical;
- rescore all 900 Haiku runs from the raw diagnoses in `agent_log.jsonl` with my own expected-rank code;
- rebuild each run's groundedness corpus and recompute the check;
- recompute the headline means and the paired verdicts with my own bootstrap;
- rerun the frozen GNN seeds on the 30 distractor windows to see what drives its fault claims.

Only validation (and the frozen Phase 5 assets) was read. Nothing under `test` or `dev` was opened, and the dataset key was not read. Telemetry is synthetic, the topologies are public, and none of this is Telstra's network. Numbers below are audit evidence, not project results.

Grades: **critical** = can invalidate a headline RQ4 result if not fixed before the test run; **major** = biases a result or a claim, or leaves a path to test contamination, and needs a fix before the test run; **minor** = reporting, wording or hygiene with small expected effect.

## Summary

| ID | Grade | Finding | Fix |
|---|---|---|---|
| S1 | major | The test path does not freeze the agent code and can be rerun. `results/phase6_frozen.json` does not exist and no agent hash is checked. The "run once" rule only blocks reusing the same `--name`. Runs record no code hash | Write the freeze file now, check it on test, lock test per model, and record the hashes in every run |
| S2 | major | The distractor headline "0-3% vs 83-87%" overstates. Every distractor window equals a logged change exactly (30 of 30; 0 of 120 fault windows do), so the test is mostly "read the change log". A one-line change-log rule on top of the GNN also gives 0%. The agent pays for it on real faults that overlap a change (it names a fault in 40% to 53% of them, against 96% for the GNN) | Use the wording in section 3. Add a "GNN + change-log rule" reference and a fault-claim row for faults that overlap a change, both computable without API calls |
| S3 | minor | Absolute time leaks through window clipping. A start before the scenario start is clipped to step 0 and echoed back, so `range` shows minus the window's absolute start (for example `[-1481, 38]`). No validation run did this | Cap the lookback so a range never reaches step 0, and add a shift-invariance test |
| S4 | minor | Two runbooks lean on the evaluation design: "blanked for the whole window" and "dark for the whole incident" are only reliable because each window is exactly the fault's duration. The rest reads as general domain knowledge | Disclose with the A2 results. Do not edit the runbooks after the freeze unless logged as a design change |
| S5 | minor | The groundedness check is implemented as pre-registered (my recompute matches all 900 runs), but it is lenient. About 44% of cited numbers are 1 to 2 digit integers, rounding grounds approximate values, and bare names are not checked. A verbatim-number variant flags 329 of 5567 items (5.9%), against 94 (1.7%) | Report both rates and call it a check on ids and numbers, not on explanations |
| S6 | minor | The 15-call budget is checked only before each request, so parallel calls in the last turn overshoot it: 91 of 900 runs used 16 or 17 calls. Forced submission otherwise works | Refuse calls beyond the budget inside a turn, or report the budget as "15, up to 17" |
| S7 | minor | Smaller scoring asymmetries: what counts as a fault claim differs in kind between agent and references; the GNN reference averages 3 seeds while A2's tool is seed 0; 2 reversed link names scored as misses; 16 of 150 incidents come from the Phase 5 selection scenarios | Wording, as listed in section 4 |

No critical finding. No tool returns ground truth, test data, seeds, or any value at or after the window end. Every validation headline number and verdict I recomputed matches `summary.json`.

## 1. Tool leakage and window clipping

From the code:
- `Incident` gets an `Observed` (observable tree only; `load_observed` refuses any path with `truth` in it), the static inventory KG (built from the topology and config, no scenario data), the frozen GNN and the runbooks. No tool opens a manifest, label or truth file. `meta.json` carries no seed. The split guard (`guard(p, {split})`) limits the run to one split.
- Every time range goes through `abs_range`, which clips the end to `now - 1` (now = window end, exclusive). Alerts are read through `alerts_at(alerts, now - 1)`, so a clear time after now is hidden (audit Q9). The alert `value` is the reading at the raise step. The change log lists only changes with `t_start < now`. The GNN is fed `b2_features(obs)[:now]`, and `explain_gnn` uses `windows(feat, [t])` with `t <= now - 1`.

Attempts to break it, on three validation incidents (link_down, drain distractor, flapping):

| Attempt | Result |
|---|---|
| `t0 = -10^7`, `t1 = +10^7` on `get_active_alerts`, `get_metric_series`, `list_links_by_change`, `get_change_log` | Range ends at w - 1. Latest alert start 0 or later inside the window, no clear time at or after w |
| `get_gnn_scores(t = 10^7)`, also with `over_window = false` | Window mode clips to [0, w - 1]. Single-step mode returns an "empty range" error |
| `explain_gnn(t = 10^7)`, `t = -10^7` | "empty range" error |
| String `"-5"` and float `1e9` as times | Cast and clipped normally |
| Extra keyword `max_alerts = 10^6` (schemas are not strict) | Accepted, but it only lifts the 60-row display cap on data already up to now |
| Tool name outside the condition | Refused in the loop before `call_tool` |
| All telemetry from `now` onward replaced by random values, events from `now` onward removed | All 7 tool outputs tested were byte-identical to the originals |
| GNN on full features against features cut at now | Same probabilities before now (max difference 2.4e-7, float noise) |

**S3 (minor): absolute time leaks through the clip at step 0.** A range that starts before the scenario start is clipped to step 0, and the tools echo the clipped range relative to the window start. For incident I000 every tool returned `"range": [-1481, 38]`, which says the window starts at absolute step 1481. `get_gnn_scores(t = -t0, over_window = false)` does the same, and alert start values do as well when the agent asks for all history. That gives time since scenario start and, modulo 288, time of day. The parked rule from audit P4 / P7 forbids exactly that. Impact on validation: none observed. No call in the 900 runs reached before the scenario start, and 36 calls looked back more than one day. Fix:
- cap the lookback at one day before the window start (every window starts at step 288 or later, so the clip at 0 then never triggers);
- or return an error instead of clipping;
- add a test that tool outputs are identical when the scenario is shifted later in time with identical relative data.

The change log shows the planned end of a change still running at now. That is a real schedule, not future data. Section 3 covers how the distractor windows interact with it.

## 2. Runbooks

The 9 runbooks contain no labels, scenario ids, element names, thresholds that were tuned on any split, or timing priors (time of day, day, how long into the week). Most of it is ordinary NOC knowledge:
- check the change log first;
- shared elements of alerted service paths;
- all links of one router dark points to the router;
- a degraded but up link keeps its traffic;
- a metric change moves traffic at one step with no log entry.

Several lines read like the simulator's rendering, but are also plausible general knowledge:
- RB02: "utilisation at 0 with the one-hop latency probe missing";
- RB06: "ramping up over some steps rather than jumping";
- RB09: "for every link of one router for a few steps".

**S4 (minor): two lines depend on the evaluation design, not on the domain.**
- RB02: "Is the link the only one blanked for the whole window? That supports link_down."
- RB09: "A down link is dark for the whole incident."

These are reliable only because the incident window is exactly the fault's [t_start, t_end). A real operator's window does not line up with the fault. They also describe the down-link signature that gives the GNN its 1.000 top-3 on link_down (Phase 5 audit R3).

Effect is limited:
- Runbooks are in A2 only.
- 98 of 300 A2 runs searched them (RB07 33 hits, RB02 30, RB05 29, RB03 25, RB08 25).
- Every A2 run also called `get_gnn_scores`, which already carries the signature.

The pre-registration already says the runbooks favour the agent. Fix: add these two lines to that disclosure. Do not edit the runbooks after the freeze; if they are changed, log it as a design change before test.

## 3. Distractor comparison

How each side is scored on the 30 distractor windows (15 drain-only, 15 planned-outage-only, no fault within the window plus the 1-hour tail):
- **Agent:** it claims a fault if `fault_type` is one of the six fault types. `planned_maintenance` and `no_fault` count as no claim, and so would a missing submission (none occurred).
- **References** (B1 to B3+, GNN): they claim a fault if the detection score reaches the frozen threshold on any step of the window. The threshold makes 5% of 1-hour fault-free validation windows alarm. The GNN value is the mean over 3 seeds.

These are different questions. The references have no "planned maintenance" output and no change-log input, so the comparison measures what the change log adds, not a like-for-like false-alarm rate.

What drives the numbers (recomputed; with_status / status_blind):

| Method | Distractor fault claims | Drain | Planned outage |
|---|---|---|---|
| A0 (alerts only) | 0.533 / 0.400 | 0.067 / 0.267 | 1.000 / 0.533 |
| A1 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| A2 | 0.033 / 0.000 | 0.067 / 0.000 | 0.000 / 0.000 |
| GNN | 0.867 / 0.833 | 0.733 / 0.733 | 1.000 / 0.933 |
| B1 | 0.867 / 0.767 | | |
| B2 | 0.733 / 0.500 | | |
| B3, B3+ | 0.233 / 0.167 | | |

- **Window length does not explain the GNN rate.** Distractor windows have a median of 29 steps, longer than the 12-step calibration windows. But the GNN alarms in the first hour alone just as often (0.867 / 0.833). In clean windows of the same length and time of day, one to three days earlier, it alarms in 0.028 / 0.042 (24 controls). So it reacts to the drains and outages themselves, which really move traffic.
- **The window gives the answer away.** Every distractor window is exactly a logged change. A change with start 0 and end w appears in `get_change_log` for 30 of 30 distractors and for 0 of 120 fault incidents. With that tool, telling a distractor from a fault is a lookup.
- **A trivial rule matches the agent on distractors.** "GNN claim, unless the change log shows a drain or planned outage overlapping the window" gives 0.000 distractor claims in both variants. On fault incidents it lowers GNN detection from 0.719 to 0.600 (with_status) and from 0.692 to 0.572 (status_blind). 15 of 120 fault incidents overlap a drain or outage.
- **The agent pays the same price.** On those 15 fault incidents, A1 named a fault in 0.47 (both variants) and A2 in 0.40 / 0.53, against 0.96 for the GNN.

**S2 (major).** The pre-registered G4 ("A1 and A2 claim a fault less often than the GNN on distractors") is supported. But "0-3% vs 83-87%" without context reads as a diagnostic win and as a false-alarm rate comparable to Phase 4 / 5. Neither is right. Suggested wording:

> On 30 validation windows that coincide with a logged drain or planned outage and contain no fault, Haiku 5.5 with the change-log tool (A1, A2) named a fault in 0 to 1 of 30 windows (0% to 3%). The frozen GNN, which has no change-log input, crossed its 5% fault-free alarm threshold in 83% to 87% of them (mean of 3 seeds). B3 and B3+ did in 17% to 23%, and A0 (alerts only, no change log) named a fault in 40% to 53%. A one-line rule that silences the GNN whenever the change log shows a drain or outage overlapping the window also reaches 0%. So the gap measures use of the change log, not diagnostic skill. The agent pays for it on real faults: of the 15 fault incidents that overlap a drain or outage, A1 and A2 named a fault in 40% to 53%, against 96% for the GNN. Telemetry is synthetic and n = 30.

Fix, before test:
- add a "GNN + change-log rule" reference and a "faults overlapping a change" group to the scoring (no API cost; observable events only);
- report both next to G4, as a disclosed addition to the pre-registration;
- keep the 1-hour Phase 4 / 5 false-alarm rate separate from this window-level claim rate.

## 4. Agent scoring against the references

Checked and equivalent:
- **Same windows.** The agent's horizon is the window end, the same [t0, t1) the references score, and the same incidents serve both variants.
- **Same tie credit.** `ranking_scores` gives the root 3, the alternatives 2 and 1, and every other element 0. `localisation_credit` calls `evaluation.tie_credit` with only the true root kept out. Unranked elements are tied, so "none" earns the random-guess expectation. That is the harness's rule for ties.
- **Same top-k.** Each reference is scored with `method.localise(state, t0, t1, "q90")`, the Phase 5 headline aggregation.
- **Independent rescore matches.** My own code computes top-1 and top-3 from the raw diagnoses (rank r of the root among distinct valid candidates; otherwise expected credit among the n - m tied elements). It agrees with `records.jsonl` on every one of the 720 fault runs.

**S7 (minor) asymmetries, each small:**
- Fault claim: the agent classifies, the references cross a threshold (section 3).
- The GNN reference is the mean of 3 seeds. A2's `get_gnn_scores` uses seed 0 only. Say so where A2 is compared with the GNN.
- Ids outside the inventory get no credit. 192 candidates (170 in A0) were not valid element ids, mostly free text such as "link:unidentified" or a service id. Two were reversed link names (`link:B--A` for an existing `link:A--B`) and were scored as misses. Negligible, but normalising link direction is a cheap fix if logged as a bug fix before test.
- 16 of the 150 incidents come from scenarios in the Phase 5 60-scenario selection subset. Excluding them, the GNN faults top-3 is 0.756 (against 0.764) and A2 is 0.743 (against 0.743), so this has no effect. It does not apply on test.

## 5. Groundedness and forced submission

**Groundedness.** The code does what the pre-registration says:
- Ids are matched against the incident's inventory, longest first, so names with spaces work.
- Id-like tokens outside the inventory count as ungrounded.
- A number counts as grounded if some corpus number equals it after rounding to the cited precision.

I rebuilt each run's corpus (the regenerated prompt plus logged tool outputs) and reran the check. It agrees with the logged count in all 900 runs: 94 of 5567 items are ungrounded. No run had empty evidence.

**S5 (minor): the check is weak, so the rate should be worded carefully.**
- 8016 of 18239 cited numbers (44%) are 1 to 2 digit integers. These are almost always somewhere in the corpus.
- Rounding means "0.5" is grounded by any value from 0.45 to 0.55 anywhere in the outputs, not only by the value for that element.
- Names cited without a `link:` / `pop:` prefix, the root and the alternatives are not checked.
- Requiring each cited number to appear verbatim as a token flags 329 items (5.9%), against 94 (1.7%).

Fix: report the pre-registered rate and the verbatim rate. Describe the result as "cited ids and numbers appear in the tool outputs", not as checked explanations.

**Forced submission.** It works as intended:
- 183 runs were forced (A1 78, A2 105) and all 183 then submitted.
- No call made after a force was executed. There were 0 "budget reached" refusals, because every forced run submitted on its next turn.
- No run hit the token budget, and no refusal or max-token stop occurred.
- The tool list never changes within a conversation.
- The 92 tool errors are argument errors, not forced refusals or calls to tools outside the condition (0 of each).

Two edge cases:
- **S6 (minor).** The budget is checked only at the top of the loop, so all parallel calls in a turn run. 91 runs used 16 or 17 calls (max 17). Fix: refuse calls beyond the 15th inside a turn, or report the budget as it was enforced.
- The token budget sums cache reads on every request, so it counts the cached prefix again each turn. It never triggered, so this has no effect here. A no-tool reply after a force ends the run at once as "no_submission"; this never happened.

## 6. Test path

**S1 (major).** The pre-registration says that after the validation eval the prompt, tool code and runbooks are frozen in a committed `results/phase6_frozen.json`, and that `--split test` checks them. Neither exists yet:
- `results/phase6_frozen.json` is missing, and `phase6_agent.py` has no agent-code hash check. Only the Phase 5 assets are checked (`frozen_assets`).
- "Run once" is enforced only by refusing an existing `results/phase6/test_<name>` folder. A second test run under another `--name` goes through, for either model.
- `--limit`, `--model dev` and any `--conditions` subset are accepted on test.
- `config.yaml`'s `agent` section (model ids, budgets, effort) is not hashed.
- Neither `agent_log.jsonl` nor `summary.json` records a code hash or commit. So there is no machine check that `val_haiku` ran with the committed code (the git history is consistent with it: no agent file changed between `2958e46` and `7520e88`).
- A crashed test run cannot be resumed except under a new name, which is the same as a rerun.

Feeding back: nothing reads test outputs except `logged_spend`, which only sums cost. Incident sampling reads the test manifest, as it must for oracle windows and scoring. No selection, calibration or threshold step runs in Phase 6; the GNN and baseline thresholds are frozen from Phase 5 and hash-checked. So the feedback risk is the rerun, not code. Fix, before any test run:
1. Write `results/phase6_frozen.json` with the sha256 of:
   - `src/nettwin/agent/*.py` and every runbook;
   - `src/nettwin/gnn/explain.py`, `scripts/phase6_agent.py`;
   - the `agent` section of `config.yaml`;
   - the commit.
   Commit it.
2. On `--split test`, refuse on any hash mismatch, on `--limit`, and on the dev model.
3. Fix the folder name to `test_<model>` and write a lock file per model when a test run starts. Allow resuming into that folder only with identical hashes.
4. Write the hashes into each `agent_log.jsonl` line and into `summary.json`.
5. Add a test that a second test run is refused.

## 7. Recomputed validation headlines

All from `results/phase6/val_haiku/` with my own code. "Committed" is `summary.json` or the decision log.

| Quantity | Recomputed | Committed |
|---|---|---|
| Runs submitted / forced (A1, A2) | 900 / 78, 105 | 900 / 78, 105 |
| Tool calls / errors | 7002 / 92 | 7002 / 92 |
| Ungrounded items / evidence items | 94 / 5567 | 94 / 5567 |
| Actual cost | $1.7584 | $1.76 |
| with_status faults top-3: A0, A1, A2, GNN, B3+ | 0.4838, 0.6285, 0.7434, 0.7639, 0.5106 | 0.48, 0.63, 0.74, 0.76, 0.51 |
| with_status faults top-1: A0, A1, A2, GNN | 0.4290, 0.5167, 0.5833, 0.6250 | same to 4 dp |
| status_blind faults top-3: A0, A1, A2, GNN | 0.3458, 0.4881, 0.7436, 0.7639 | same to 4 dp |
| Distractor claims A1 / A2 / GNN, with_status | 0.000 / 0.033 / 0.867 | same |
| Distractor claims A1 / A2 / GNN, status_blind | 0.000 / 0.000 / 0.833 | same |
| with_status A2 - A0 faults top-3 | +0.260 [+0.181, +0.341] supported | +0.260 [+0.183, +0.345] supported |
| with_status A2 - GNN faults top-1 | -0.042 [-0.103, +0.019] inconclusive | -0.042 [-0.105, +0.018] inconclusive |
| with_status A1 - GNN faults top-1 | -0.108 [-0.181, -0.033] contradicted | -0.108 [-0.181, -0.035] contradicted |
| status_blind A2 - A1 hard top-1 | +0.217 [+0.100, +0.333] supported | +0.217 [+0.108, +0.338] supported |
| with_status A2 - A1 hard top-1 | +0.100 [-0.033, +0.233] inconclusive | +0.100 [-0.018, +0.232] inconclusive |

The CIs differ only in the third decimal because I used my own resampling (4000 percentile resamples, seed 7). Every verdict matches. One reading note for the write-up: G2 (A2 beats A1 on hard top-1 in both variants) holds only in status_blind on validation. With status events it is inconclusive. That is in `summary.json`, but the decision-log summary does not mention it.
