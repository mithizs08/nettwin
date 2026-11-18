# Audit: Phases 11 and 12

Independent audit at commit `46a96d5`, 2026-11-18.

Scope:
- the Phase 11 and Phase 12 decision-log entries;
- their code, tests and results;
- the README and report sections for both phases;
- the two pages on the live app.

No Anthropic API call was made and nothing was fixed. Every check below was run for this audit with my own scripts, which are kept in the session scratchpad and not committed. Telemetry is simulated, topologies are public, and no real operator data is used.

**Summary.** No critical findings.

What holds:
- All 429 Phase 11 gold answers match an independent recomputation, and the neutral ids hide every real PoP name.
- Phase 12 is causal: 1,520 tool outputs, the live KG and the episode lists do not change when everything after the decision step is replaced with random data.
- My own episode merging and page scoring reproduce every Phase 12 test metric exactly. The three policies are scored on the same episodes.
- Each test set was scored once.
- The mid-phase changes touched only what the log says, and no decision already made was altered.
- Every Phase 11 and Phase 12 number in the README, report and app matches its results file.
- The live app renders both pages without errors.

Two major findings:
- The README and report present the agent's "better top-3 at escalation" without noting that the two policies are scored on different faults.
- The decision log gives recovery counts for the resumed runs that the committed files contradict.

---

## 1. Phase 11

**Gold answers: correct (429 of 429).**
- My script recomputes every answer with plain networkx:
  - ECMP lowest-cost routing for every PoP pair;
  - services on links and PoPs;
  - what-if failures, recomputed by removing the element and rerouting;
  - lowest and second-lowest cost paths;
  - counts;
  - the time questions, by replaying the change schedule and diffing routing states.
- Its only inputs are the raw topology loader, the service generator and the scheduled change events. It does not use `kgqa.py`'s answer functions or the KG query API.
- All 429 questions (177 validation, 252 test) match. No pair needed more than 4 ECMP paths, so the ECMP cap never mattered.
- Every lowest-cost path asked about is unique, and every second-lowest path has strict cost gaps.

**Neutral ids: effective.**
- I re-implemented the seeded relabelling, and it reproduces the committed ids.
- 0 to 2 PoPs per network keep their alphabetical position, which is what chance gives.
- No real PoP name appears in any question or in the BM25 fact text of any network.
- No Q0 answer contains a non-neutral id.
- Q0's 8 credited test answers are guesses: four times "2" on gold-SLA counts, two partial link guesses, one partial path, and one empty-set answer to a question whose gold is empty.
- See m6 for what is still real in the graph text.

**Test scored once.**
- The test log has 756 lines (252 questions x 3 conditions), with no duplicate (question, condition) pairs, and every run submitted.
- `--split test` refuses if `results/phase11/test` exists, and it never appends to a test folder. The folder is created only after the question check passes. So the first refused call (the seed bug) stopped before any API call and created nothing.
- All 4 hashes in `results/phase11_frozen.json` match the current files.
- Order of events:
  - pre-registration commit 16:47:02;
  - validation runs and freeze 16:52:10;
  - fix and re-freeze 16:54:20;
  - test log last written 17:00:32;
  - results commit 17:03:38.
- The fix commit changes only `scripts/phase11_kgqa.py`, as logged.

## 2. Phase 12

**Causality: holds.**
- **Truncation test.**
  - For real episodes from 4 validation and 10 test scenarios, I built the agent's view at each decision step (round 1 and the round-2 re-check) twice: once from the real scenario, and once with all telemetry after the step replaced by random values and all later events removed.
  - Both views were built for every tool in the agent's list: alerts, metric series, links by change, change log, GNN scores and explanations, KG queries, runbooks and `kg_current_state`.
  - 1,520 tool outputs compared: 0 differences, 0 tool errors.
  - The list of episodes opened up to each decision step is also identical.
- **By code reading:**
  - The live KG applies status events at or before the decision step, and those events already include the simulator's polling lag.
  - Alerts go through `alerts_at` (open alerts have no end time).
  - GNN features are cut at `now`, and the encoder and features are causal (past EWMs, lags, unpadded convolutions).
  - The GNN + change-log rule reads only changes that have started.
  - Status events are sorted by step in every checked scenario, which `kg_at`'s early `break` relies on.
- See m9 for the planned end times of logged changes.

**Episode merging: correct.**
- My own implementation (top-1 element at each alarm step; extend if the same element alarmed within 12 steps of the episode's last alarm) reproduces the committed `alarm_steps` and every episode (element, open, last) in 21 test scenarios, one in ten.
- This also shows that the replay rebuilt on the resume equals a fresh run.

**Page counting and scoring: correct.**
- My own scorer reproduces every value in `results/phase12/test/summary.json` to 4 decimals for all three policies:
  - pages and false pages per day;
  - detection rate;
  - time to detect;
  - missed faults during maintenance.
- It works from `replay.jsonl`, `decisions.jsonl` and the truth manifest. Totals: 1,260 days, 390 faults, 56 change-overlapping faults.
- GNN + change-log pages recomputed from the logged changes match the committed list in all 210 scenarios.

**Same episodes for all three policies: yes.**
- GNN and GNN + change-log pages come from the same replayed episodes, and the agent's decisions are keyed to them.
- Every episode has a round-1 decision. Round 2 runs exactly for the waits: 1,543 waits, 1,543 re-checks.
- All 5,427 decisions sit at their episode's opening step, or opening + 6 for a re-check, in the final `replay.jsonl`. The resume regenerated that file, so this check matters.
- Decision files have no duplicate keys. Every decision equals its agent-log entry. The agent log has no duplicate wakes, so no wake was paid twice.

**Mid-phase changes: no decision altered.** The git diffs show only what the log says.
- `cb546a5`: test sample 10 to 30 per network, guard $4 to $8. Before any validation or test run.
- `833b6ee`: the unknown-tool branch in `replay_agent.py`. It only fires where the old code crashed, so it cannot change a decision already made.
- `833b6ee`: the log-recovery block.
- `c4a8f70`: `--resume`. This was committed before the test started (the test's first output came after 18:34).
- `9976734`: guard $8 to $10.50, and `--freeze` now refuses only after `summary.json` exists. These two lines are the whole diff to the script.
- Across the three freezes only the script's hash changed. The other 10 frozen files kept the same hashes, and all match the current files, as does `results/phase5_frozen.json`.
- Decisions are append-only, round 1 was complete before the stop, and every re-check is keyed to an existing wait.
- What I cannot prove from the repo: the timing of the stop itself (201 re-checks done, $7.03). The agent log has no timestamps. Summing the log in order (all round 1, then the first 201 re-checks) gives $7.04, which is consistent.

## 3. README, report and app against the results files

- **Phase 11:**
  - all 21 cells of the README table (value and CI) match `results/phase11/test/summary.json`;
  - the report's overall, per-type and per-template values match;
  - so do the exact-match rates, the paired verdicts, the cost and the expectations.
- **Phase 12:**
  - every value and CI in the README table matches `results/phase12/test/summary.json`;
  - so do the report's figures: pages, false pages, detection, time to detect, missed faults during maintenance, the paired differences and the expectations;
  - the 60% and 71% reductions, the "about a third" ratio and the 40% wait rate are correct;
  - so are the cost figures: test $9.12, Phase 12 $9.79, project $52.29. My own sum of every logged cost gives $52.29.
- `scripts/check_numbers.py` passes (372 numbers), and the full test suite passes (146 tests).
- **App:** the Phase 11 and Phase 12 pages read the committed summaries directly and label everything test, scored once. The Phase 12 table also shows top-3 over all faults, which the README omits (M1).
- **Live app** (https://nettwin17.streamlit.app, checked in the built-in browser):
  - Both the "KG question answering (Phase 11)" and "Real-time replay (Phase 12)" pages render with no exception elements.
  - Phase 11: table, cost, expectations, comparisons and the question browser all render.
  - Phase 12: 3 tables and the scenario timeline chart render.
  - The visible values match the committed files.
  - Local `main` is level with the last known `origin/main`. A fresh `git fetch` failed here, so I could not confirm the deployed commit independently.

## 4. Findings

### Critical

None.

### Major

**M1. "Top-3 at escalation is better than the GNN's" is confounded by which faults each policy detects.**
- The metric is averaged over each policy's own detected faults: 280 for the GNN, 249 for the agent. The 32 faults only the GNN detects have a GNN top-3 of 0.375, which pulls the GNN's average down.
- My post-hoc check on the 248 faults both detect gives:
  - agent top-3 0.835 against GNN 0.786, a difference of +0.049 [0.005, 0.092] (scenario bootstrap);
  - top-1 0.585 against 0.625, so the agent is lower on top-1.
- The fair pre-registered measure, top-3 over all faults, is level: +0.000 [-0.029, 0.031].
- Where it shows:
  - README: "when it escalates, its top-3 root cause is better than the GNN's ranking (+0.092 [0.046, 0.146])";
  - report sections 4.7 and 6 ("the agent's top-3 was better instead").
  - The decision log mentions the level all-faults result. The README and report do not.
- The numbers are right; the reading is stronger than the design supports. About half the gap is selection.

**M2. The decision log's counts of recovered wakes contradict the committed files.**
- The test entry says "240 wakes were recovered from the agent log on the resume", but `results/phase12/test/decisions.jsonl` has no record with `recovered_from_log`.
- That fits how the run stopped. The guard stop is graceful: `pool.map` writes every finished wake before exit, so there was nothing to recover.
- The validation entry says 36 such wakes were recovered, but `val_dev/decisions.jsonl` flags 22.
- The integrity of both runs is verified (section 2), so no result is affected. But the log is the record of how the test ran, and these two numbers have no source.

### Minor

**m1. Detection counts any page in the fault's window, whatever element it names.**
- This is pre-registered, but it rewards a high page rate. A policy paging 3 times a day gets some detections by coincidence.
- The README headline ("catches 64% of faults against the GNN's 72%") does not say this.
- On top-3 over all faults (a page that names the root among 3), the agent and the GNN are level: 0.531 and 0.531.

**m2. Equivalence wording from inconclusive results.**
- README: "detects about as many faults (-0.005 [-0.036, 0.028])". The CI is given, so this is acceptable.
- Report: "while keeping the change-log rule's detection" states equivalence from an inconclusive difference. The CI is narrow (about 3 points), so the claim is defensible, but it should say "within about 3 points".

**m3. "Both logged before they could affect a result" is too strong for the guard raise.**
- It was decided after the test had run for about 45 minutes, on counts (wait rate, spend) that the run itself produced, though no metric had been computed.
- The new `--freeze` rule also allows re-freezing while a test run is unfinished. The guarantee that nothing else changed now rests on reading the diff, which I did (section 2). The tooling no longer enforces it.

**m4. Phase 11 CIs ignore clustering by network.**
- The bootstrap resamples questions, but the 252 questions come from 7 networks, and the "medium" size group is one network (36 questions). So the CIs are too narrow.
- The verdicts are large enough (Q2 - Q1 +0.334) that this does not change them. But "Q2 beats Q1 in every graph-size group" rests on one network for "medium".

**m5. Report section 4.6 overstates what Phase 11 shows about the agent.**
- It says "the agent's KG tools give exact answers a text retrieval cannot".
- The Q2 tools are new wrappers in `kgqa.py` (what-if, paths between, changes between), not the Phase 6 agent's tool set.
- Retrieval did give exact answers on one-hop questions (1.000).

**m6. The neutral ids hide names, not geography.**
- Q1 facts and Q2 tools still carry real link lengths (km) and routing metrics. That is no practical help without the names.
- "So no answer can come from knowing the real network" (README) is untested, because there is no real-id control. The Q0 result (0.024) is consistent with it.
- One test question has an empty gold set (services_rerouted, `test` split), so an empty answer scores 1.0. The generator's docstring says answers are non-trivial.

**m7. No unit tests for Phase 11.** No test covers gold generation, `normalise` or `score`, so path scoring in either direction and the number handling are unchecked. The independent recomputation in section 1 covers the gold answers, but not the scorer.

**m8. Some logged numbers have no results file.** These cannot be traced by the number checker:
- the stop point ($7.03, 201 re-checks), reconstructable only from log order;
- the estimate of "about 2,700" episodes;
- the validation crash point ("23 written decisions").

They are in the decision log only, not the README or report. Wakes in flight at the validation crash may have been paid without being logged; the amount is unknown and small.

**m9. Planned end times are visible early.**
- The change log shows a started change's planned end step, even when it is in the future.
- The GNN + change-log rule also uses it ("ended within the hour").
- This is fair if the logged plan is what happened, which the simulator guarantees. Real change windows overrun, so this flatters both the agent and the rule a little.

**m10. App comparison table direction.**
- On the Phase 12 page, "supported means the CI is above 0" applies to every row.
- For pages, false pages and missed faults, "contradicted" is the good outcome for the agent.
- A short note on direction per metric would prevent misreading.
