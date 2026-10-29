# Audit: Phases 0 to 2 (simulator and faults)

Independent audit, 2026-10-29, at commit `329f96f`. Scope: `PLAN.md`, `docs/decision_log.md`, `src/nettwin/{topology,kg,simulator,faults}.py`, `scripts/phase{0,1,2}_*.py`, `tests/`, `results/`. Nothing was changed in the repo. All numbers below come from throwaway scripts run in the `nettwin` env against the committed code. They are audit evidence, not project results.

Grades: **critical** = can invalidate a headline RQ result if not fixed before Phase 3; **major** = biases a result or a comparison and needs a fix or a clearly reported limitation; **minor** = correctness, hygiene or realism issue with small expected effect.

## Summary

| ID | Grade | Finding |
|---|---|---|
| C1 | critical | The simulator was developed and tested against AARNet, a test topology |
| C2 | critical | Healthy services never lose a probe, so one lost probe is a near-perfect gray-failure detector |
| M1 | major | A down link has a unique telemetry signature, and node_down has a unique event type |
| M2 | major | Faults never overlap maintenance or each other, so "overlaps a change window = benign" is a perfect rule |
| M3 | major | Traffic dropped upstream is still counted downstream; the effect is not "slight" |
| M4 | major | Gabriel training graphs differ from the test topologies on scale and structure, and some code paths only run on test |
| M5 | major | No observability label: physically undetectable faults are mixed into "subtle" |
| M6 | major | Status events land exactly at fault start, with no polling lag |
| M7 | major | No per-link, per-step anomaly labels for RQ1 detection |
| M8 | major | Difficulty tuning had no target set in advance and left no audit trail |
| m1 to m12 | minor | Listed at the end |

## Question 1: test-set leakage

These are every place I found where AARNet, Geant2012 or germany50 was looked at, ordered by risk.

| # | Where | What was seen or decided | Disclosed? | Risk |
|---|---|---|---|---|
| L1 | `tests/test_simulator.py` (all 19 tests) | Every simulator test uses an AARNet fixture and asserts AARNet-specific behaviour: baseline peak median in (0.3, 0.7), non-hot peaks < 0.85, the demand-surge rise "is modest", subtle gray failure "> 80% zero-loss samples", flapping seen-event counts, recovery after the window. Each tuning iteration had to keep these green, so AARNet acted as an acceptance check for the simulator design. | No | High (see C1) |
| L2 | Decision log, Phase 2, "Changes made after looking at outputs" | Capacity redesign triggered by AARNet peaks (median 0.78) | Yes | Medium. The fix is principled, but it was found on test. |
| L3 | Decision log, Phase 2, "Routing" | "cap 4 vs 16 changes no AARNet link load by more than 3.3%, so the cap stays at 4". A design decision checked only on a test topology. | Logged, but not marked as test-driven | Medium |
| L4 | Decision log, Phase 1, ECMP | ECMP cap set with AARNet, Geant2012 and germany50 ECMP pair counts in view ("5 AARNet pairs have 5 equal-cost paths") | Logged | Low to medium |
| L5 | `topology.link_metric` | `max(1, round(dist_km))` floor motivated by AARNet's 6 zero-length links | Logged | Low. Needed so the graph can be routed at all. |
| L6 | `Simulator._design_capacity`, zero-length rule | "0 km links get the largest adjacent capacity". Gabriel graphs have no zero-length links (0 of 6,512 links across all 100 graphs), so this rule only ever fires on AARNet. It was designed from the test topology and is never exercised on train or validation. | Logged, not marked as test-driven | Medium |
| L7 | `scripts/phase2_simulate.py` `choose_demo_link` | Demo selection rule changed after viewing the AARNet result ("most extreme collapse") | Yes | Low (presentation only) |
| L8 | `phase2_fault_<type>.png` (12 AARNet incidents), decision log shortcut 9 | A mild and a severe AARNet instance of every fault type was inspected. The ECMP-tie misconfiguration (28% loss) was observed on AARNet. No change is recorded, but because the tuning history is not in git (see M8) it cannot be ruled out that these plots informed the ranges. | Partly | Medium |
| L9 | `Simulator.local_offset_h` | Per-PoP time zones from longitude apply only to geo topologies. Gabriel graphs all get offset 0. The feature was built for test geography and is never exercised in train or validation. | No | Medium (also M4) |
| L10 | `Simulator._base_demand` real-matrix branch | germany50's real demand path is test-only and has never run (no test or script calls it). | Plan-mandated | Low as leakage, but untested code on test (m9) |
| L11 | `scripts/phase1_build_kg.py` | KG built, summarised and failure-checked on all three test topologies | Yes | Low (plumbing only) |
| L12 | `data/scenarios/phase2_demo/` | 18 AARNet scenarios at seeds 2026, 3000, 3001, 4040, 5000 to 5002 have been viewed | Yes | Low if Phase 3 avoids these seeds |
| L13 | topohub contents | Near-duplicates of the test networks ship with topohub: `topozoo/Geant2001`, `Geant2009`, `Geant2010`, `sndlib/geant` (GEANT) and `sndlib/nobel-germany`, `topozoo/Dfn`, `sndlib/dfn-*` (German research networks). None is used yet. | n/a | High if any is later used for validation or augmentation |

### C1 (critical): the simulator was developed and tested against AARNet

PLAN.md section 3 says test topologies are "never touched for design decisions". In practice AARNet was the development topology for Phase 2: every simulator test runs on it (L1), and the capacity, ECMP and zero-length decisions were made on it (L2, L3, L6). AARNet is also the headline demo network. That is a conflict built into the plan, since Phase 2 asks for "sanity plots (one incident end-to-end on AARNet)". As it stands, AARNet results for RQ3 and RQ4 cannot be called held-out.

Fix:
1. Reclassify AARNet as the **demo / development topology** in `PLAN.md` and `config.yaml`. Report AARNet numbers in their own column marked "seen during development". They should not count as RQ3 generalisation evidence.
2. Keep Geant2012 and germany50 as the clean test set. Before running anything on them, pick one or two extra clean test networks from topohub's 203 Zoo networks with a fixed rule (for example, the first two alphabetically with 20 to 60 nodes, connected, not a near-duplicate under L13). Write the rule in the decision log first.
3. Move `tests/test_simulator.py` onto a fixed Gabriel graph (for example `gabriel/20/1`) or a hand-built toy topology, so simulator acceptance criteria no longer depend on a test network.
4. Record L3, L6 and L9 under "test-driven decisions" in the README, as PLAN.md section 3 requires.
5. Add every L13 near-duplicate to `topologies.skip` with a comment, so they can never enter train or validation.

## Question 2: shortcuts that let a detector cheat

### C2 (critical): healthy services never lose a probe

Service loss is binomial on the true drop probability. In healthy state that probability is exactly 0 below the congestion knee, and there is no background end-to-end loss (link counters get background discards, services do not). Measured on one fault-free week on each of the six diagnostic Gabriel graphs: **3 of 688,935 service-steps had any lost probe (0.0004%)**. Then I injected the mildest silent gray failure (0.02% loss) on a link with services: on **42% of steps** at least one crossing service lost a probe. So "any lost probe" detects gray failures almost perfectly, and intersecting the paths of services with lost probes localises them. This defeats the main point of silent gray failures (shortcut 5 in the log), and it makes `test_subtle_gray_failure_is_buried_in_probe_noise` misleading: the per-sample loss is mostly zero, but any non-zero sample is decisive.

Fix: add background end-to-end loss per service. For example, a per-service base rate drawn log-uniform from 0.001% to 0.02%, plus rare bursty episodes (probe-host or CPE artefacts), drawn from the `measure` RNG with fixed shapes so the counterfactual shares them. Then re-run the Phase 2 diagnostics with an "any lost probe in window" rule, and an "any lost probe on a service that has none in the warmup day" rule, and report their hit rate on fault-free windows next to the fault hit rates.

### M1 (major): down links and down nodes have unique signatures

- A fully down link reports `util = 0.0` exactly (no noise), with latency and loss NaN. A telemetry gap blanks all three. Across the 48 diagnostic scenarios, 1,770 cells had "util present, latency missing", and **0** of them were on a link that was not down. The planned status-blind ablation (parked idea, Phase 3) would therefore still give down links away.
- node_down is the only source of `pop_status` events, so the event type maps one-to-one to the fault type. Unreachable services show `loss = 100%` with `latency = NaN`, a pattern no other condition produces.

Fix: for the status-blind variant, report a down link as fully missing (all three NaN), the same as a gap. For node_down, make realistic: a dead router cannot be polled, so its side of each incident link goes missing, and neighbours emit `link_status down` for their links after a polling lag. Emit `pop_status` only after a reachability timeout (for example 1 to 3 polls) and only sometimes. Report unreachable service probes as 100% loss with latency NaN, but also give gaps on the destination side the same latency-NaN pattern some of the time.

### M2 (major): faults never overlap maintenance or each other

`sample_scenario` keeps every fault at least 36 steps away from every maintenance window and every other fault. Checked: 0 overlaps in 48 scenarios. Consequences:
- "Anomaly overlaps a logged change window, so it is benign" is a perfect rule. That inflates RQ4 false-alarm results on maintenance distractors, and the agent's `get_change_log` tool will look more useful than it is. In real networks, changes are a leading cause of incidents.
- At most one fault is active at any time, so top-1 localisation can assume a single root cause.
- Maintenance only starts between 00:00 and 04:00 sim time, and faults never fall in that window around a maintenance, so time of day plus the change log separates the two cleanly.

Fix: allow a configured fraction of faults (for example 10 to 20%) to overlap or immediately follow a maintenance window, including "maintenance caused the fault" cases where the root cause is the maintained element. Allow a small fraction of scenarios with two concurrent faults, and score them with set-based metrics. Report results for overlapping and isolated faults separately.

### M6 (major): status events have no lag

For 100% of link_down and node_down faults, the status event lands at exactly `t_start`, the same step the telemetry changes. Polling in reality has a phase offset and SNMP or trap delays. If Phase 4 or 5 reports time-to-detect, or labels detection by onset step, status-driven methods get a perfect onset for free.

Fix: draw a per-event detection lag (for example 0 to 2 steps, plus an occasional missed poll) from the `status` RNG, and apply the same lag to the `observed_kg` rebuild. Report time-to-detect per fault type only.

### Other shortcuts (minor)

- **Exact-zero utilisation.** A down link reports 0.0 with no noise. 3,525 of 5.6 million healthy cells also read exactly 0 (low-load links clipped at 0), so this is weak on its own, but it is part of M1.
- **Event days** switch on and off at sim-time midnight for the whole network at once (a visible step). Ramp them over an hour.
- **Abrupt fault ends.** demand_surge ramps up but stops in one step, and gray failures stop in one step. Add an optional ramp-down.
- **Faults never in day 1.** This is already noted in the log. Phase 3 must not give absolute time or "time since start" as a feature.
- **Manifest next to telemetry.** `scenarios.json` (ground truth, including `impact` and `status_change_observed`) is written next to the observable files. Phase 3 should put ground truth in a separate directory that the dataset loader for detectors and the agent tools cannot reach.

## Question 3: are the "hot links" and other difficulty changes justified, or rigging?

Verdict: **justified, not rigging, but under-documented (M8)**. Each change was triggered by Gabriel diagnostics (not by a test result, apart from the capacity fix in L2), and each has a realism argument:

| Change | Realism anchor | Assessment |
|---|---|---|
| Hot links (10% at 0.8 to 0.95 peak) | Real backbones do have links that run hot at evening peak while upgrades wait | Defensible. They only defeat a global threshold, though. A per-link rule ("util above the link's own day-1 peak + 0.05") fires on 5.6% of fault-free windows against 7.0% for the global `util >= 0.9` rule, so hot links barely affect per-link baselines (B1). The remaining false positives come mostly from event days. Hot links do not favour any planned method. |
| Flapping convergence 1 to 3 s lowered to 0.05 to 1 s | BFD and fast reroute give sub-second convergence. 1 to 3 s is legacy IGP timing. | Defensible |
| Misconfig factors 3 to 20 replaced by 1.3 to 10 plus minimal step | Real typos and partial changes are often small | Defensible |
| Partial N-1 planning | Real operators plan for single failures on part of the network | Defensible |
| Overnight maintenance plus change review | Standard practice | Defensible, but it creates M2 |

What is missing is a stopping rule. The changes were made until the naive-rule hit rates "looked right", with 11 to 21 faults per type and no target written down first. Fix (M8): write the intended difficulty mix in the decision log before any further change (for example "at least 20% of each non-status fault type subtle on validation Gabriel graphs"). Freeze `config.yaml` with its hash in the decision log at the start of Phase 3. Add a sensitivity run that regenerates validation with the pre-tuning ranges and reports B1 to B3 on both, so a reader can see the tuning did not decide the RQ1 ranking. Also note that impact classes use absolute thresholds (0.5 ms, 5 ms), calibrated at Gabriel's latency scale, so the difficulty mix will not transfer to test topologies (M4). Report the impact-class mix per test topology.

### M8 (major): tuning is not auditable

All Phase 2 work, including every tuning iteration, is one commit (`ad950f7`). The "before" numbers survive only as prose ("18/19 obvious"). Fix: from Phase 3 on, commit the config and the diagnostics JSON at every tuning step, or append each diagnostics run to a `results/tuning_log.jsonl` with the config hash.

## Question 4: unobservable faults

### M5 (major): no observability label

The manifest has `services_affected`, `status_change_observed` and `root_counters_show_loss`, but no flag that says whether a fault was detectable at all. `impact_class = "subtle"` mixes "hard but possible" with "no signal exists":
- 7.9% of Gabriel links (517 of 6,512, up to 20% in one graph) carry no service on intended routing. AARNet has 1 of 24, Geant2012 9 of 58, germany50 6 of 88. A silent gray failure on such a link produces no observable signal of any kind. A visible one only shows in that link's counters.
- In the diagnostics, 2 of 12 gray failures had 0 services affected.
- About 30% of gray failures draw a loss below 0.05% (log-uniform 0.01 to 2%), which is below the impact threshold. Today C2 makes them detectable anyway. Once C2 is fixed, many will not be.
- A minimal misconfiguration can move only pairs with little or no service traffic.

Fix: compute per fault, from noise-free truth against the counterfactual, one flag per observation channel: `obs_status` (status event in window), `obs_link_util` (root or other link util delta above 3 sigma of util noise for at least 2 steps), `obs_link_counters` (counter delta above the background rate), and `obs_e2e` (expected extra lost probes on some service above a threshold that you pick once C2's background loss exists). Store `observable = any(...)` in the manifest. Exclude unobservable faults from headline detection and localisation metrics, report their count and rate separately per fault type and topology, and give the agent eval a "not diagnosable" outcome for them. Optionally resample roots at generation time so the observable fraction is controlled.

## Question 5: physical realism bugs

### M3 (major): dropped traffic is still counted downstream

Offered load on every link is `demand @ rf` with no reduction for drops on earlier links in the path. In the diagnostics, **29% of link_down faults and 35% of routing_misconfig faults pushed some link above capacity**, with a maximum rho of 3.80 (74% of offered traffic dropped at that link). In those cases, downstream links show utilisation, queueing and congestion loss from traffic that never arrived. The decision log calls this "slightly overstated". The numbers show it is not slight for a third of these faults. It also bears directly on RQ1, which asks about symptoms away from the faulty element, because it inflates exactly those remote symptoms.

Fix: route per path rather than per aggregate. For each routing state, iterate a fixed point (3 to 5 passes): carried flow on hop k = offered flow × the product of survival on hops 1 to k-1, with survival from `congestion_loss` on that pass's loads. Keep service loss as the end-to-end product, which is already consistent. Add a test: a 3-link chain with the first hop at rho = 2 must show about half the load on the last hop.

### Other realism issues

- **(minor, disclosed) No convergence loss at link_down or node_down onset.** Only flapping has it, so the presence of brief loss separates flapping from a hard down. Apply one step of convergence loss at onset and restore for link_down, node_down and unplanned maintenance "down". Planned drains stay lossless.
- **(minor) Flap loss is invisible in link counters.** `counter_true` includes congestion and gray loss but not `conv`, so flap drops only show end to end. Add them to the input-error counter, scaled, or document it as an intentional choice.
- **(minor) A dead router's counters read 0.** They should be unreachable (NaN). Covered in M1.
- **(minor) Gray loss does not reduce goodput.** At 1 to 2% loss, TCP throughput drops sharply, so utilisation would not "look normal". Scale the affected demands by a Mathis-style factor for loss above about 0.5%, or state the assumption in the dataset card.

## Question 6: are 100 Gabriel graphs enough, and can topohub make more?

**topohub can make more.** `topohub.providers.gabriel.GabrielGenerator.generate_topo(nnodes, seed)` works for any n up to 4096. I reproduced the shipped graphs exactly with seed = `i * 4096 + n` (checked `gabriel/15/0`, `20/1`, `25/3`, `60/9`). I also generated 138 new graphs covering every n from 15 to 60 (not just multiples of 5), and all passed `topology.from_node_link` (connected, simple). Caveats:
- It prints to stdout and calls the global `random.seed`. Wrap the call and do not rely on Python's global `random` elsewhere.
- Generated coordinates and distances are not rounded to 2 decimals as the shipped JSON is, so metric ties can differ slightly from a shipped graph with the same seed. Round them the same way, or always generate.
- It takes about a second per graph at n = 60 (pure Python, O(n^3)). That is fine.

**Is 100 enough?** For training-set size, probably yes for a CPU GNN, since each graph yields many scenarios. For validation and for structural diversity, no:
- Validation would get about 20 graphs, too few for per-fault-type, per-impact-class estimates, and the 10 sizes in steps of 5 give little size coverage.
- The structural range is narrow and differs from the test networks (M4).

Fix: generate 300 to 500 graphs across all n in 15 to 60, with seeds outside the shipped scheme. Split by graph and stratify by size. Use at least 60 for validation. For diversity, add augmentations (random non-bridge edge removal down to average degree 2.5 to 3, leaf spurs, dual-router PoPs joined by short links, rescaling distances to backbone km), and set their parameters from **non-test** topohub networks (any Zoo or SNDlib network outside the test set and L13), never from AARNet, Geant2012 or germany50. Better still, keep a "real-topology validation" set of 5 to 10 non-test Zoo networks. That gives RQ3 a validation signal for synthetic-to-real transfer without touching test.

### M4 (major): Gabriel graphs differ from the test topologies

| | Gabriel (100 shipped) | AARNet | Geant2012 | germany50 |
|---|---|---|---|---|
| avg link length (dist units, km for real) | 70 to 112 (mean 97) | 695 | 824 | 101 |
| zero-length links | 0 | 6 | 0 | 0 |
| avg degree | 2.53 to 3.96 (mean 3.40) | 2.53 | 3.14 | 3.52 |
| leaf-node fraction | mean 4% | 10.5% | 13.5% | 0% |
| bridges | mean 1.3 | 4 | 5 | 0 |
| time-zone spread | none | yes | yes | small |

Propagation per link is about 0.5 ms on Gabriel and 3.5 to 4 ms on AARNet and Geant. Absolute latency noise (0.05 ms), impact thresholds (0.5 ms, 5 ms) and SLA margins therefore mean very different things on train and test. AARNet's dual-router ECMP, the zero-length capacity rule and time zones never appear in training (L6, L9). RQ3 results will mix "unseen topology" with "unseen latency scale and features".

Fix: rescale Gabriel distances to a backbone-km range, as part of the augmentation in Question 6. Express impact thresholds and the features you feed detectors relative to each path's propagation latency, not in absolute ms. Add time-zone offsets to Gabriel graphs by assigning a pseudo-longitude span. Report the table above in the dataset card.

### M7 (major): no detection labels for RQ1

RQ1's headline metric is link-level PR-AUC, and it singles out "faults whose symptoms appear away from the faulty element". The manifest only stores the root cause and the window. Fix: in Phase 3, store per fault a per-link, per-step mask from noise-free truth against the counterfactual (`root`, `symptomatic` when the util, latency or loss delta is above the M5 thresholds, `unaffected`). Fix the definition in the decision log before any baseline runs, and report PR-AUC for root links and for symptomatic links separately.

## Minor findings

- **m1.** `gray_visible[li] |= not silent` is set per link for the whole scenario. If a silent and a visible degradation hit the same link in one scenario, both become visible. Make visibility per fault window.
- **m2.** Flapping draws `uniform(0.02, max_down_frac)`, but `max_down_frac` can be as low as 0.01, which inverts the bounds. Make the lower bound `min(0.02, max_down_frac)` or raise the config floor to 0.02.
- **m3.** `planned_peak_util` (the change-review check) uses the noise-free profile with no day factors, AR noise or event days, so approved maintenance can still congest on event days. Include the event-day factor and a margin, and assert on realised truth in a test.
- **m4.** The module docstring in `simulator.py` says capacity comes from the "peak noise-free baseline load". The code uses a noisy history week, plus hot links, N-1 planning and the zero-length rule. Update it.
- **m5.** A PoP-scope gap blanks only services whose `src` is that PoP. `src` is the lexicographically smaller endpoint, so which services go missing is arbitrary. Blank services at either endpoint, or use a per-service collector PoP.
- **m6.** The hand-picked AARNet flapping example uses `convergence_s = 2.5` and `flaps_per_step = 7`, outside the config ranges ([0.05, 1.0] and [0.5, 6]). The "severe" figure therefore shows behaviour the sampler never produces. Keep figure examples inside the ranges or label them.
- **m7.** Impact class for link_down is mostly driven by the root link dropping to 0 utilisation (`max_root_util_delta`), so almost any loaded link counts as "obvious". Classify on the other links and on services only.
- **m8.** `_design_capacity` leaves hot links with zero peak at capacity 0 (rho would be NaN). It did not happen on the six diagnostic graphs (0 zero-capacity links), but nothing guards it. Exclude zero-peak links from `hot`, or apply the floor to them.
- **m9.** The germany50 real-demand branch of `_base_demand` has no test and has never run. It will raise `KeyError` if any matrix name is not a PoP. Add a unit test with a toy demand matrix (not germany50).
- **m10.** Phase 2 diagnostics use 48 scenarios (11 to 21 faults per type). Rates such as "79% status seen" have wide intervals (roughly ±18 points). Report counts with intervals, or run more scenarios on validation graphs.
- **m11.** The fault-free comparison windows exclude ±36 steps around faults but not maintenance windows, so the 1.4% "status event" false-positive rate is planned maintenance. Report that split so the number is not read as noise.
- **m12.** Phase 3 must keep scenario seeds disjoint from the viewed demo seeds (L12) and from the diagnostic seeds 7000 to 7507. Assert this in the dataset builder.

## Things checked and found sound

- Observed events never carry fault fields, and `observed_kg` is rebuilt from observable data only. A misconfiguration correctly leaves the twin on the wrong paths.
- Counterfactual runs share demand, measurement, missing-data and status noise exactly (fixed-shape draws from separate RNG streams), so impact deltas are clean.
- Capacity is fixed per topology from `seeds.network`, independent of scenario seeds.
- Telemetry gaps are placed independently of faults and may overlap them.
- The `.gitignore` does not hide source code.
