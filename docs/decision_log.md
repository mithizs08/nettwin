# Decision log

Every change to the plan, research questions, metrics or data splits goes here with a date and a reason. Newest entries at the bottom.

## Entries

### 2026-10-29: Phase 0 environment

- Conda env `nettwin`, Python 3.11, all packages installed with pip inside the env (not base). Versions pinned in `requirements.txt`.
- Added `PyYAML` to the package list so scripts can read `config.yaml`.
- `topohub.get(name, use_names=True)` returns a node-link dict (keys `nodes` and `edges`), not a NetworkX graph. Convert with `nx.node_link_graph(g, edges="edges")`.
- Verified: `topozoo/Aarnet` 19 nodes / 24 edges, `topozoo/Geant2012` 37 / 58, `sndlib/germany50` 50 / 88, all connected. Matches PLAN.md section 3.
- The LightGBM pip wheel needs an OpenMP runtime on macOS. Installed `llvm-openmp` from conda-forge into the env (also listed in `environment.yml`).
- Known issue: torch bundles its own `libomp.dylib`. If `torch` is imported before `lightgbm` and LightGBM trains multithreaded, the process segfaults (exit 139). `KMP_DUPLICATE_LIB_OK=TRUE` does not help. Either import `lightgbm` before `torch`, or pass `n_jobs=1` to LightGBM. Only matters for Phase 8.
- Agent model IDs in `config.yaml` left as `null` on purpose; fill from current docs before Phase 6 (PLAN.md section 7).

### 2026-10-29: Phase 0 fresh-clone check: passed

- Cloned `origin/main` at commit `1a57a5d` into a temp folder. No `.env` in the clone.
- Built a new throwaway env `nettwin-freshcheck` from the clone's `environment.yml` (did not reuse `nettwin`). Took about 1 minute. Python 3.11.18, same pinned versions, `libomp.dylib` present.
- `pip install -e .`, then `python scripts/phase0_check_env.py` exited 0 and regenerated `results/phase0_env_check.json` byte-identical to the committed file (19/24, 37/58, 50/88, all connected).
- `pytest`: 5 passed. LightGBM also imports and trains in the fresh env.
- Temp clone and throwaway env deleted afterwards.

### 2026-10-29: Phase 1 topology loader and knowledge graph

Topology (`src/nettwin/topology.py`)
- `load(name)` returns an undirected `nx.Graph`. Node attr `pos`; edge attrs `link_id`, `dist_km`, `metric`; graph attrs `name`, `geo`, `demands`. It rejects directed, multi, self-loop, duplicate-link and disconnected inputs.
- `geo` is True for `topozoo/` and `sndlib/` (pos is lon, lat) and False for `gabriel/` (planar). Gabriel `dist` is taken as km as topohub gives it.
- germany50's real demand matrix is kept in `graph["demands"]` for Phase 2. topohub's precomputed `ecmp_fwd` / `ecmp_bwd` loads are dropped (we compute our own routing).
- Link ids are canonical `A--B` with endpoints sorted.

Routing metric
- AARNet has 6 zero-length links between co-located PoP pairs (Sydney1 and Sydney2, and so on: the Zoo models two routers per major city). Routing weight is an integer IGP-style metric: `max(1, round(dist_km))`. Why: the floor stops intra-city hops being free, and integers make equal-cost ties exact rather than float coincidences. Propagation latency still uses the real `dist_km` (so 0 km gives 0 ms).
- ECMP: all equal-cost shortest paths are kept, capped at `kg.ecmp_max_paths = 4`. On AARNet this gives ECMP on 59 of 171 pairs (Geant2012: 0 of 666, germany50: 3 of 1225), because of the dual-router design. Uncapped, 5 AARNet pairs have 5 equal-cost paths, so the cap drops one path for those 5 pairs. Kept at 4 for now; revisit in Phase 2 if it distorts load.

Knowledge graph (`src/nettwin/kg.py`)
- Node ids: `pop:`, `link:`, `path:<a>><b>...`, `svc:`, `alert:`, `mnt:`, `rb:`, `fault:`. A path's id is its PoP sequence, so the same physical path keeps one node across time.
- One set of RoutingPaths per unordered PoP pair, written from the lexicographically smaller PoP. Shortest paths on an undirected graph are symmetric; the Phase 2 simulator will route both directions of a demand over the same paths.
- Snapshots: RoutingPath nodes are never deleted. Each has `intervals` of [start, end) steps during which it was active; Link and PoP nodes have `intervals_down`. Every query takes an optional `t` (default: current time). This is cheaper than storing a full KG copy per window and answers "what did the path look like at t" exactly.
- `apply_change(t, links_down, links_up, pops_down, pops_up, metrics)` re-derives routing and logs `link_down/up`, `pop_down/up`, `metric_change` and per-pair `reroute` events (with removed/added paths and an `unreachable` flag). Time must not go backwards. A PoP going down removes its links from routing but does not mark those links down; the root cause stays the PoP.
- Added a `FaultType` node kind (the six fault types in PLAN.md section 4). PLAN.md section 5 has `Runbook -APPLIES_TO-> fault type` but no node kind for fault types, so this is a small schema addition.
- Query API as in PLAN.md section 5, with these semantics fixed now:
  - `paths_through(element, t)`: Link or PoP; for a PoP, endpoints count.
  - `services_on(element, t)`: Link, PoP or RoutingPath.
  - `neighbours(element, hops)`: same-kind physical neighbours (PoP: adjacent PoPs; Link: links sharing a PoP, i.e. line-graph adjacency, matching the GNN in section 6). Static topology, ignores failures.
  - `shared_elements(paths, kinds)`: raw counts of how many given paths contain each Link / PoP. The healthy-path penalty for B3 is left to Phase 4.
  - `changes_between(t0, t1)`: inclusive on both ends.
  - Extra helpers: `paths_between(src, dst, t)`, `service_paths(svc, t)`, `links_of_path(path)`, `is_up(element, t)`, `validate()`, `summary()`.
- Alert, Maintenance and Runbook: schema plus `add_alert`, `add_maintenance`, `add_runbook`. These are schema-checked and tested but nothing populates them yet. Maintenance windows also go into the change log, so `changes_between` will show them.
- Services are synthetic: `services_per_pop x n_pops` distinct PoP pairs (AARNet 38), seeded by `seeds.kg_services` plus a CRC of the topology name. Classes gold/silver/bronze with SLA latency = propagation latency x factor + margin, and fixed loss limits (values in `config.yaml`). These SLA numbers are invented for the simulation, not taken from any real operator.
- Neo4j-style CSV export: `nodes_<Kind>.csv` (`id:ID`, typed properties, `:LABEL`) and `rels_<REL>.csv` (`:START_ID`, `:END_ID`, `:TYPE`, plus `order:long` on TRAVERSES), in neo4j-admin import format. Intervals are flattened to `active_from/active_to` long arrays with -1 for an open end. Written to `data/kg/` (gitignored). Not test-imported into a real Neo4j.

Figures and outputs (`scripts/phase1_build_kg.py`)
- Added matplotlib 3.11.2 for static figures.
- `results/figures/aarnet_kg.png`: schema with counts, plus a layered slice of three services (one chosen to show ECMP).
- `results/figures/aarnet_map.png`: plain lon/lat axes with aspect corrected for latitude, no basemap (avoids an extra data download). Co-located PoPs are nudged 0.6 deg apart (NW/SE, across the east-coast corridor) and the dense south-east has a zoom inset.
- `results/phase1_kg_summary.json`: counts for the three test topologies plus a KG sanity check on AARNet. Cutting the busiest link (Canberra1--Melbourne1, 105 paths, 16 services) reroutes 75 pairs, strands no service, and the same paths return after link up. This is a KG plumbing check, not a research result.
- Outputs are byte-deterministic across runs. Two set-iteration orders in the figure code (string hashing is randomised per process) made the PNG change between runs; fixed by sorting.
- Tests: `tests/test_topology.py` and `tests/test_kg.py` (47 tests in total with Phase 0) cover every query function on a 5-node toy topology with hand-computed answers, plus failure, restore, metric override, unreachable pairs, schema rejection, CSV export and the three real topologies. A mutation check (forcing every path active at all times) fails 7 tests, so the time filtering is actually tested.

### 2026-10-29: Phase 1 fresh-clone check: passed

- Cloned `origin/main` at commit `4fcb91e` into a temp folder. No `.env` and no `data/` in the clone.
- New throwaway env `nettwin-freshcheck` from the clone's `environment.yml` (did not reuse `nettwin`); matplotlib 3.11.2 installed from the new pin.
- `pip install -e .`, then `scripts/phase0_check_env.py` and `scripts/phase1_build_kg.py` both exited 0.
- After regeneration `git status` in the clone was clean: `results/phase1_kg_summary.json`, `results/figures/aarnet_kg.png` and `results/figures/aarnet_map.png` are byte-identical to the committed files.
- All 24 CSVs under `data/kg/` are byte-identical to the ones built locally.
- `pytest`: 47 passed.
- Temp clone and throwaway env deleted afterwards.

### 2026-10-29: Phase 2 simulator, faults and distractors

What is observable and what is ground truth (the most important decision this phase)
- Observable: link telemetry (util from interface counters, latency from a one-hop probe, loss from discard and error counters), service telemetry (end-to-end latency and loss from 600 probes per 5 min, so loss is binomial), and observed events. Observed events are only link / PoP status from 5-minute polling plus the planned-maintenance log.
- Ground truth only: the fault list and params, noise-free metrics, the true routing, per-fault impact stats.
- The twin's KG (`observed_kg`) is rebuilt from observed events only, using intended metrics. So a routing misconfiguration is invisible to the twin, which holds the wrong paths for the whole incident ("twin drift"). Gray failures and demand surges never appear as events. Tested: observed events never contain fault fields.
- Why: if the KG change log carried fault labels or the misconfigured metric, localisation would be trivial.

Model (`src/nettwin/simulator.py`, `src/nettwin/faults.py`, params in `config.yaml`)
- Demand: gravity model (degree x lognormal weights; germany50 uses its real matrix, both directions), diurnal profile in each PoP's local time from longitude, weekend dip 0.85, per-PoP day factors, AR(1) log-noise per directed demand.
- Routing: the Phase 1 KG (`NetTwinKG.apply_change`) per distinct network state, cached; ECMP split evenly. A flapping link mixes its up and down routing by the step's down fraction. ECMP cap check (parked in Phase 1): cap 4 vs 16 changes no AARNet link load by more than 3.3%, so the cap stays at 4.
- Capacity, fixed per topology from `seeds.network`: peak of a noisy fault-free "history" week / U(0.4, 0.7), floored at 0.25 x median. Half the links (not hot ones) are also sized for the worst single-link failure at 0.75 to 1.0 (partial N-1 planning). Zero-length intra-city links get the largest adjacent capacity. 10% of links are "hot" with normal peaks of 0.8 to 0.95.
- Latency: propagation + 0.02 ms per hop + queueing 0.05 ms x rho/(1-rho), capped at a 25 ms buffer. Loss: ~0 below rho 0.9, cubic to 0.2% at capacity, plus all excess above capacity. Utilisation is the busier direction, carried load capped at 1.0.
- Noise: util N(0, 0.003); latency 2% relative plus abs jitter plus 0.1% probe outliers of 1 to 20 ms; counters 10% relative plus background discards on 2% of samples; random missing 0.3%. Measurement noise is drawn with fixed shapes, so the no-fault counterfactual shares it exactly.
- Faults, with severities sampled per fault:
  - link_down: 15 min to 6 h.
  - node_down: 10 min to 2 h.
  - link_degradation (gray): loss log-uniform 0.01% to 2%, bursty, optional ramp. 50% "silent": the drops never show in any link counter, only in end-to-end probes. Utilisation and latency are unchanged.
  - link_flapping: per-step activity, down fraction 1% to 50%, convergence loss of 0.05 to 1 s per flap.
  - demand_surge: x1.3 to x4 on all demands to and from a PoP, ramped.
  - routing_misconfig: metric x or / a log-uniform 1.3 to 10, or (50%) the smallest 1.1^k step that shifts any PoP pair; it must change routing, and bridge links are skipped.
- Distractors:
  - Planned maintenance: drain / down / noop, overnight (00:00 to 04:00 start), and rejected if it would push any link past 0.85 (a change review).
  - Telemetry gaps: PoP collector or a single link, which may overlap faults.
  - Event days: x1.1 to x1.25.
  - The hot links.
- Faults never overlap each other or maintenance (36-step gap), and never fall in the 1-day warmup.
- Per-fault impact vs the no-fault counterfactual: max service latency and loss deltas, services affected / unreachable, utilisation deltas, whether a status change was observed. Impact class: subtle / moderate / obvious.

Changes made after looking at outputs (with what triggered them)
- Capacity was first designed from noise-free traffic, which gave fault-free AARNet peaks with median 0.78 (plan says 40 to 70%). This was noticed on a test topology (AARNet), so it is disclosed here. The fix (plan from a noisy history week, like an operator) is principled and was confirmed on gabriel/40/3 too. Later tuning used Gabriel graphs only.
- Diagnostics on 48 sampled scenarios on six Gabriel graphs (never the test topologies) showed the simulator was too easy. Fixes:
  - flapping 18/19 obvious: convergence lowered from 1 to 3 s to 0.05 to 1 s, wider activity range.
  - misconfig 18/19 obvious: factors 3 to 20 replaced by log-uniform 1.3 to 10 plus the minimal-change variant.
  - Single failures overloading thin links 25x (96% loss): partial N-1 planning.
  - Maintenance drains at peak congesting links: overnight windows plus the change-review check.
  - With those, fault-free windows had zero congestion, so "any congestion = fault" was a perfect detector: added hot links.
- The AARNet demo link was first chosen as "the failure that pushes another link highest", which selects the most extreme collapse. Now it is "closest to capacity without exceeding it", to show congestion onset. Chosen link: Hobart--Melbourne2.

Final diagnostics (`results/phase2_summary.json`, 48 Gabriel scenarios)

| Fault type | n | subtle / moderate / obvious | status seen | util >= 0.9 | svc loss >= 1% |
|---|---|---|---|---|---|
| link_down | 21 | 0 / 8 / 13 | 100% | 33% | 29% |
| node_down | 11 | 0 / 0 / 11 | 100% | 18% | 100% |
| link_degradation | 12 | 3 / 6 / 3 | 0% | 17% | 33% |
| link_flapping | 19 | 1 / 12 / 6 | 79% | 16% | 58% |
| demand_surge | 18 | 5 / 11 / 2 | 0% | 22% | 6% |
| routing_misconfig | 20 | 4 / 6 / 10 | 0% | 55% | 35% |
| fault-free windows | 215 | | 1.4% | 7% | 0.5% |

Shortcuts and unrealistic parts, flagged for Phases 3 to 6
1. Status events give away link_down, node_down and most flapping, for both detection and localisation (the event names the element). This is realistic (operators get link-down alarms), so it stays, but results must be reported per fault type, and these three must not be averaged with the hard ones. Consider a status-blind variant in Phase 3 as an ablation.
2. Telemetry of a down link: util exactly 0 with latency and loss missing. A telemetry gap blanks all three. So "util present, latency missing" identifies a down link even without status events. Realistic, but a shortcut.
3. node_down always makes its services unreachable (100% loss), so it is always obvious. Inherent.
4. Healthy services almost never lose 1% end to end (0.5% of fault-free windows). Real signal, but a strong one for detection.
5. Gray failures: 50% are silent and invisible in link telemetry; only end-to-end probes on services crossing the link see them. Services cover 2 per PoP, so a gray failure on a link no service crosses is undetectable by design unless optical. The impact stats record this (services_affected = 0).
6. Faults never overlap each other or maintenance, and never occur on day 1. Detectors must not get absolute time or "time since start" as a feature.
7. Every link-level fault effect is applied at step boundaries; fault starts are uniform over the week, including nights when impact is small.
8. Simplifications that are not realistic:
   - Traffic dropped by an overloaded link is not removed downstream, so congestion propagation is slightly overstated.
   - No convergence loss at link_down / node_down onset (only flapping has it), which is a small tell between the two.
   - Polling sees flaps only at 5-minute samples (real routers also syslog each flap, so real flapping is easier than here).
   - One-hop latency probes do not report their own loss (contrived, so that silent drops stay silent).
   - The diurnal shape is identical every day apart from noise and day factors.
   - Capacities are continuous, not 10G / 100G steps.
9. A "minimal" misconfiguration is not necessarily mild. On AARNet a +10% metric change broke an exact dual-plane ECMP tie and saturated a link (28% loss). Real behaviour, worth knowing when reading results by severity.

Outputs
- `scripts/phase2_simulate.py` writes:
  - `results/figures/phase2_link_down_aarnet.png` (end to end), `phase2_fault_<type>.png` (a mild and a severe AARNet example each) and `phase2_distractors_aarnet.png`
  - `results/phase2_summary.json`
  - 18 demo scenarios (Parquet, observed events, `scenarios.json`) under `data/scenarios/phase2_demo/` (gitignored, 19 MB)
- Outputs are byte-deterministic across runs.
- Tests: `tests/test_simulator.py` (19 tests) covers physics, determinism, baseline band and diurnal shape, propagation for every fault type, distractors, sampler rules, no ground-truth leakage, and Parquet round trip. Mutation checks: removing gray loss from the path drop fails 2 tests; freezing routing to the intended state fails 4.

### 2026-10-29: Phase 2 fresh-clone check: passed

- Cloned `origin/main` at commit `ad950f7` into a temp folder. No `.env` and no `data/` in the clone.
- New throwaway env `nettwin-freshcheck` from the clone's `environment.yml` (did not reuse `nettwin`).
- `pip install -e .`, then `scripts/phase0_check_env.py`, `scripts/phase1_build_kg.py` and `scripts/phase2_simulate.py` all exited 0.
- After regeneration `git status` in the clone was clean: every committed result and figure (Phases 0 to 2) is byte-identical.
- All 76 files under `data/kg/` and `data/scenarios/` (Parquet, observed events, manifest, CSV) are byte-identical to the ones built locally.
- `pytest`: 66 passed.
- Temp clone and throwaway env deleted afterwards.

### 2026-10-29: Pre-registration for the Phase 0 to 2 audit fixes (written before running anything)

This entry is committed before the split selection is run and before any diagnostics are rerun. Audit: `docs/audit_phases_0_2.md`.

**Split selection rule (C1, M4).** Applied mechanically by `scripts/select_topologies.py`; I look only at its printed output.
1. Universe: every `topozoo/*` key in topohub 1.5.1 (203 names).
2. Family of a name: lowercase it, strip trailing digits. Two names share a family if their first CamelCase tokens are equal, or if their lowercased stripped names share a prefix of at least 5 characters. Families are closed under this relation (union-find). This over-merges a little (for example Internetmci and Internode); that only makes the rule more conservative.
3. Excluded: the families of the dev network (Aarnet) and the test networks (Geant2012 gives the Geant family), the family `dfn` (German research network, a near-duplicate of germany50), and everything on `topologies.skip`.
4. A candidate must load with `topology.load(name)` (connected, simple, coordinates present) and have 15 to 60 nodes.
5. One representative per family: the eligible member whose name sorts last (usually the latest snapshot).
6. Sort the representatives by name, permute them with `numpy.random.default_rng(20261029)`. The first 5 go to test (alongside Geant2012 and germany50), the next 5 to `validation_real`, and all remaining representatives form the reference pool used only to set augmentation parameters.
7. Any `sndlib/*` network whose first `-` token equals a picked family name is added to `topologies.skip`.
AARNet becomes the dev / demo network: it may be used for development and plots, and its results are reported in their own column marked "seen during development".

**Difficulty targets (M8).** Checked on validation networks only (generated Gabriel validation graphs plus `validation_real`), never on dev or test.
- T1 observability: for gray failure, demand surge and routing misconfig, the unobservable fraction is at most 25% per type.
- T2 difficulty mix: among observable faults of each of gray failure, demand surge, routing misconfig and flapping, at least 20% are subtle and at most 50% are obvious.
- T3 no naive rule detects the hard types. For gray failure, demand surge and routing misconfig, each naive rule's Youden J (hit rate on fault windows minus hit rate on fault-free, non-maintenance windows of matching length) is at most 0.5. The rules: any lost probe on any service; any lost probe on a service with none on the warmup day; any link util >= 0.9; any link counter loss >= 0.05%; any status event.
- T4 down-link signature: of all link-step cells with util present and latency missing, at most 80% belong to links that are actually down.
- T5 status lag: for link_down, the fraction of status events with zero lag is between 30% and 70%, and the median lag is at least 1 step on 5-minute polling or explained.
- T6 overlap: 10% to 20% of faults overlap or immediately follow a maintenance window; at least 5% of scenarios have two concurrent faults.
- T7 baseline: the fault-free median link peak utilisation is in 0.4 to 0.7 on each validation_real network and on the pooled Gabriel validation graphs.
- T8 background end-to-end loss: in fault-free weeks, the fraction of service-steps with at least one lost probe is 1% to 10%.
Stopping rule: if a target is missed, I report the miss. I may make at most two adjustments in total, each logged with before / after numbers in `results/tuning_log.jsonl` with the config hash; after that the result stands as it is.

**Observability flags (M5).** Computed per fault from noise-free truth against the no-fault counterfactual, over the fault window plus the 3-step maximum status lag.
- `obs_status`: a status event for the root, or for any link incident to a root PoP.
- `obs_link_util`: some link has |delta util| > 0.009 (3 sigma of util noise) for at least 2 steps.
- `obs_link_counters`: some link has a counter-loss delta > 0.01 percentage points for at least 2 steps.
- `obs_e2e`: some service has expected extra lost probes over the window of at least max(3, 3 sqrt(expected background lost probes)), or a relative latency increase above 3x jitter (6%) for at least 2 steps, or becomes unreachable.
- `observable` = any of the four. Unobservable faults are reported separately and excluded from headline detection and localisation metrics.

**Per-link detection labels (M7), fixed before any baseline.** Per scenario, a (T, L) int8 mask: 2 = root (the root link, or every link incident to a root PoP, during the fault window); 1 = symptomatic (not root, and noise-free |delta util| > 0.009, or link-latency increase > 6%, or counter-loss delta > 0.01 pp vs the counterfactual); 0 = unaffected. RQ1 PR-AUC is reported for root and for symptomatic links separately. Stored with ground truth, never next to observable telemetry.

### 2026-10-29: Audit fixes (Phases 0 to 2), by audit ID

Selection result (rule pre-registered above in commit `6e5491e`, run by `scripts/select_topologies.py`, output `results/topology_selection.json`): 79 family representatives with 15 to 60 nodes.
- Test added: Garr201201 (48/62), Peer1 (16/20), York (23/24), Belnet2010 (19/20), Ibm (18/24).
- validation_real: Aconet (17/24), BeyondTheNetwork (29/41), Fccn (23/25), Quest (20/31), Nextgen (16/16).
- Reference pool: the other 69.
- No sndlib network matched a picked family.
- Several picks are sparse and tree-like (York 24 links for 23 PoPs, Nextgen a 16-node ring). They were kept as the rule produced them; swapping them would be cherry-picking.

- **C1 (critical):**
  - AARNet is now dev / demo (`topologies.dev`), with its results in their own column.
  - Test is Geant2012, germany50 and the 5 rule picks; `PLAN.md` section 3 is updated.
  - All L13 near-duplicates are in `topologies.skip`.
  - Simulator tests now run on topohub's shipped `gabriel/20/1` (in no split) and toy graphs, with elements chosen by rule.
  - KG and topology tests no longer touch Geant2012 or germany50; they use dev / validation / toy graphs.
  - Phase 1 builds KGs on dev + validation instead of test. Phase 0 only loads every configured network (sizes and connectivity).
  - The README now lists the test-driven decisions L2 to L6 and L9.
- **C2 (critical):** background end-to-end probe loss.
  - A per-service base rate, log-uniform 0.001% to 0.02% of probes.
  - Plus bursty probe-host episodes: 0.3 per service per day, mean 3 steps, 0.05% to 1% loss.
  - Drawn from the `measure` stream with fixed shapes, so the counterfactual shares it.
  - Justification: the base rate sits well below the ITU-T Y.1541 IP loss objective of 1e-3 (0.1%) for a healthy managed backbone [VERIFY]. Lost probes on healthy paths also come from probe hosts and from router control-plane policing that rate-limits ICMP / TWAMP replies, independent of the data plane [VERIFY].
  - The result is that one lost probe is no longer a fault signature (see T3 and T8 below).
- **M1 (major):**
  - A down interface on a reachable router now reads util 0 and discards 0 (counters readable), with latency missing (the probe fails), as SNMP would show it.
  - A dead router's links are reported by the neighbours the same way.
  - Healthy links also lose one-hop latency samples: random probe timeouts at 0.5%, plus 1 to 4 probe-responder gaps per scenario. So "util present, latency missing" is no longer unique to down links.
  - `pop_status` is now NMS reachability ("unreachable" / "reachable"). It is raised with probability 0.8 after 1 to 3 failed polls, by both node_down and PoP collector outages.
  - A PoP outage kills the probe responder half the time, so its services show 100% loss with latency missing, exactly like an unreachable PoP.
  - The twin logs PoP reachability but does not route on it.
  - The status-blind variant (`spec["status_blind"]`) blanks down links completely and drops all status events.
- **M2 (major):**
  - Maintenance is placed first.
  - 15% of faults are maintenance-linked (during or within an hour after a change window), half of them on the maintained element (`caused_by_maintenance`).
  - 10% are concurrent with another fault.
  - 15% of maintenance happens in business hours.
  - Ground truth is always the fault, and each fault records `placement`.
  - Set-based scoring for concurrent faults is a Phase 4 task.
- **M3 (major):**
  - Carried load now uses a per-path fixed point (4 passes) on any step where some link drops more than 0.1%.
  - A test checks that a chain with the first hop at rho 2 shows 0.499 of the load on the last hop (the old model gave 1.0).
  - Each routing state is simulated at full demand and time-shared within the step, so flapping mixtures no longer average loads across states.
  - Convergence loss (0.05 to 1 s) now hits services crossing a link at link_down / node_down onset and restore. Planned outages stay lossless, because traffic is drained first.
- **M4 (major):**
  - 400 generated graphs `gabaug/<n>/<seed>` cover every n from 15 to 60, with seeds from 1,000,000 (outside topohub's shipped scheme).
  - They are split by graph, stratified by size: 308 train, 92 validation.
  - Augmentation targets come from a randomly drawn reference-pool network (`src/nettwin/synth.py`): thinning without creating leaves, backbone-km rescaling, leaf spurs, dual-router PoPs and pseudo-longitude time zones.
  - On a sample of 123 graphs, the generated median average degree is 2.44 against 2.32 in the reference pool, leaf fraction 15.6% against 12.5%, and mean link 349 km against 355 km; 31 of 123 graphs have dual-router PoPs.
  - Impact thresholds are now relative to each service's latency (subtle < 2%, obvious >= 25%).
- **M5 (major):** per-fault observability flags `obs_status`, `obs_link_util`, `obs_link_counters`, `obs_e2e` and `observable`, exactly as pre-registered, are stored in each fault's impact.
- **M6 (major):**
  - Status changes are reported after a lag of 0 / 1 / 2 steps (p = 0.5 / 0.35 / 0.15), plus a missed poll with probability 0.1. A down and an up that land on the same step cancel.
  - The twin is rebuilt from the lagged events.
- **M7 (major):** per-link per-step labels (root / symptomatic / unaffected), exactly as pre-registered, written to `truth/<sid>/labels.parquet`.
- **M8 (major):**
  - Targets were pre-registered before this run.
  - `scripts/phase2_simulate.py --log-tuning <label>` appends the target results with the config hash to `results/tuning_log.jsonl`.
  - Not done here (parked): the sensitivity run with pre-tuning ranges needs the Phase 4 baselines, and the config hash freeze happens at the start of Phase 3.
- Minor fixes:
  - **m1:** gray-failure visibility is per fault window.
  - **m2:** the flapping lower bound is `min(0.02, max_down_frac)`.
  - **m3:** the change review includes the event-day factor and a 1.15 margin, and a test asserts it.
  - **m4:** the simulator docstring was rewritten.
  - **m5:** PoP-scope gaps hit services at either endpoint.
  - **m6:** the figure examples are inside the config ranges.
  - **m7:** impact class for link_down / node_down ignores the root's own drop.
  - **m8:** a hot link can never have zero capacity.
  - **m9:** the real-demand branch validates names and is tested on a toy matrix. This found a real bug: an operator-precedence error in the name check.
  - **m10:** every rate is reported with a 95% Wilson interval, on 188 validation scenarios.
  - **m11:** fault-free windows are split into maintenance and non-maintenance.
  - **m12:** `seeds.reserved_ranges` plus `faults.check_fresh_seed` for Phase 3.
  - Event days ramp over an hour.
  - Gray failures and surges have optional ramp-downs.
  - Ground truth (manifest, labels) is written under `truth/`, apart from `observable/`.
  - Gray loss above 0.5% cuts TCP goodput (1/sqrt(p)), so heavy gray failures show in utilisation.
  - Flap drops stay out of the interface counters on purpose: a flapping link's drops happen at the upstream router while the link is down, not as discards on the link.

### 2026-10-29: Pre-registered targets, results (validation networks only)

188 scenarios, 370 faults:
- Networks: 46 generated validation graphs x 3 scenarios, plus 5 validation_real x 10.
- Seeds: 8601 to 8788, all in the reserved range 8000 to 8999.
- Placements: 304 isolated, 53 maintenance-linked, 13 concurrent.
- Every run is in `results/tuning_log.jsonl` with its config hash. The full numbers with 95% Wilson intervals are in `results/phase2_summary.json`.

| Target | Run 0 (no changes) | After adjustment 1 | After adjustment 2 (final) | Final status |
|---|---|---|---|---|
| T1 unobservable <= 25% (gray / surge / misconfig) | 5.5% / 4.9% / 1.9% | 11.1% / 8.1% / 5.7% | same | pass |
| T2 subtle >= 20%, obvious <= 50%: gray | 13% / 46% | 25% / 27% | same | pass |
| T2: demand surge | 12% / 55% | 21% / 33% | same | pass |
| T2: routing misconfig | 6% / 62% | 12% / 52% | same | **miss** |
| T2: flapping | 2% / 80% | 3% / 74% | 0% / 69% | **miss** |
| T3 max Youden J of any naive rule (gray / surge / misconfig) | 0.40 / 0.32 / 0.43 | 0.37 / 0.10 / 0.35 | same | pass |
| T4 down share of "util present, latency missing" | 7.7% | 7.6% | 7.6% | pass |
| T5 link_down status lag: zero-lag fraction, median | 45%, 1 step | 46%, 1 | 46%, 1 | pass |
| T6 maintenance-linked faults; scenarios with concurrent faults | 14.9%; 5.3% | 14.3%; 6.9% | same | pass |
| T7 fault-free median peak (5 validation_real, generated pooled) | 0.43 to 0.52 | same | same | pass |
| T8 service-steps with a lost probe, fault-free | 3.95% | 3.95% | 3.95% | pass |

The two adjustments allowed by the stopping rule:
1. Adjustment 1, aimed at T2, which every hard type missed in run 0. A breakdown on validation graphs showed which criterion made faults obvious: single-step loss >= 1 pp for gray failures and flapping, util shift >= 0.3 for surges and misconfigs, and latency >= 25% for misconfigs and flapping. Changes:
   - gray loss 0.01-2% became 0.005-1%. Above about 1% a failure is no longer gray; gray-failure reports describe low-rate silent drops [VERIFY].
   - flap convergence 0.05-1 s became 0.02-0.5 s, and flaps per step 0.5-6 became 0.2-4. Fast reroute with BFD is typical [VERIFY].
   - surge factor uniform 1.3-4 became log-uniform 1.1-3. A 4x surge on a whole PoP is extreme.
   - misconfig near-miss share 0.5 became 0.7.
   These were tuned to the target and are disclosed as such.
2. Adjustment 2: flapping down fraction 0.01-0.5 became 0.005-0.15 per step. Up to 50% of a 5-minute step (2.5 min down) is a series of outages, not a flap; real flaps last seconds. It did not reach the target.

No further change is allowed, so the final result has **two misses**:
- **routing_misconfig**: 12% subtle (CI 6 to 24%), 52% obvious (CI 39 to 65%). Further moves had no realism argument, so I did not make one.
- **link_flapping**: 0% subtle (CI 0 to 6%), 69% obvious (CI 57 to 79%).
- Likely cause: the realistic, sparse topologies (median degree about 2.4, many bridges). Flapping or misrouting a link forces long detours (latency >= 25%) or partial loss of reachability, and the obvious class takes the maximum over single steps.
- Consequence for later phases: report flapping and misconfig difficulty as found. Do not claim a balanced difficulty mix for them.
- The mix also differs by network. The impact-class counts per validation_real network are in the summary (`by_group`), and the real networks lean more obvious than the generated ones.

Other facts from the same run:
- On fault-free windows outside maintenance (n = 839), "any lost probe" fires 100% of the time. That rule is dead, as C2 intended.
- On the same windows, "lost probe on a service clean during warmup" fires 27%, util >= 0.9 fires 8.5%, counters >= 0.05% fire 1.7%, and status events fire 1.3%.
- Inside maintenance windows (n = 60), status events fire 35% (planned outages) (audit m11).

### 2026-10-29: Audit-fix fresh-clone check: passed

- Cloned `origin/main` at `94e1bcb`. No `.env` and no `data/`.
- New throwaway env `nettwin-freshcheck` from `environment.yml`, including the new `scipy` pin.
- `scripts/phase0_check_env.py`, `phase1_build_kg.py` and `phase2_simulate.py` all exited 0.
- `git status` was clean after regeneration: every committed result and figure is byte-identical.
- All 117 files under `data/` are byte-identical to the local build.
- `pytest`: 80 passed.
- Clone and env deleted afterwards.
- Correction: the `94e1bcb` commit message says "6 of 8 pass". The correct count is 7 of 8 targets pass; T2 misses for two of its four fault types (routing_misconfig, link_flapping). The table above is authoritative.

### 2026-10-30: Re-verification fixes, by ID (`docs/audit_phases_0_2_reverify.md`)

No difficulty parameter was changed. The rerun below reflects measurement and placement fixes only.

- **N1 (major):**
  - Each fault's impact and observability now use a counterfactual with only that fault removed. All other faults and all maintenance stay in, and each fault's own RNG stream keeps the others identical. Labels still use the all-faults-removed run.
  - `obs_status` and `status_change_observed` count only status events that are absent from that leave-one-out run. So a planned outage on the same link, or another fault's events, are no longer credited to the fault.
  - Tests:
    - `test_concurrent_fault_does_not_make_undetectable_one_observable` reproduces the auditor's case: a silent 0.3% gray failure on a service-free link stays unobservable, subtle, with 0 services affected, when a link_down overlaps it. Reverting to the old counterfactual fails this test.
    - `test_planned_outage_status_is_not_credited_to_a_fault`.
- **N2 (major): correction of an earlier claim.**
  - The M1 entry above said "util present, latency missing" is no longer unique to down links. That is literally true, but the down-link signature is not gone: util exactly 0 with latency missing is 98.7% down-link cells in the rerun (7,476 of 7,574).
  - This is realistic SNMP behaviour (a down interface reads 0 traffic and its probe fails), so the physics stays. It is documented in the README and now reported next to T4 in the summary.
  - link_down and node_down are easy to detect from default telemetry even without status events. Telemetry changes at the start step, so the status lag does not delay onset for telemetry-based detectors.
  - Parked for Phases 4 to 6: report link_down and node_down both with status events and with the status-blind variant.
- **N3 (minor):** T3 now uses, for each fault, a fault-free non-maintenance window of the same length in the same scenario, as pre-registered.
  - Matched windows: 54 of 55 gray, 63 of 63 surge, 43 of 51 misconfig (the longest misconfigs sometimes had no free window).
  - Result: **pass** for all three types. Max J: gray 0.36 (counters >= 0.05%), surge 0.14 (counters), misconfig 0.33 (counters).
  - Same-scope correction to T6: it previously counted the deliberate `maintenance_linked` placement. The pre-registration says "overlap or immediately follow", so T6 now counts actual overlap from each fault's new `overlaps_maintenance` flag (window intersects, or starts within 12 steps after, a change window).
- **N4 (minor):**
  - The tuning hash is now sha256 over `config.yaml`, `src/nettwin/simulator.py` and `src/nettwin/faults.py`, with the per-file hashes stored. Tuning-log entries also record the git HEAD before the run.
  - Correction to the tuning history: adjustment 1 also changed `demand_surge.factor` sampling from uniform to log-uniform in `faults.py`. That was code tuning, which the earlier config-only hash did not capture.
  - The three earlier `tuning_log.jsonl` entries are left as written; their `config_hash` covers config only.
- **N5 (minor):** all 37 older snapshots of test networks are on `topologies.skip`: 30 Garr snapshots older than Garr201201 (the audit counted 12; topohub ships 30) and Belnet2003 to Belnet2009.
- **N6 (minor):**
  - Isolated faults now avoid other faults only, not change windows, so there is no fault-free buffer around maintenance. Incidental overlaps are flagged with `overlaps_maintenance`.
  - Test: `test_no_fault_free_buffer_around_maintenance`.
  - Consequence: more faults now overlap or follow a change window (see T6 below).
- **N7 (minor):**
  - Demo scenarios use neutral ids `demo_0001`..; their descriptions are in `truth/scenario_index.json` and in the manifest only.
  - `simulator.check_neutral_id` makes `write_scenario` refuse ids containing a fault-type or severity word.
  - Test: `test_observable_paths_are_neutral` checks every observable path and the observed-events text.
- **N8 (minor):**
  - A channel's delta counts only on cells whose observed value exists. So a channel fully covered by a gap (or blanked by the status-blind variant) is unobservable.
  - Test: `test_gap_covered_channels_are_unobservable`. Removing the mask fails it.

Rerun (same 188 validation scenarios and seeds, 369 faults; `results/tuning_log.jsonl` entry 4, label "re-verify fixes N1 N3 N6 N8"):

| Target | Before (54c8ad3) | After | Status |
|---|---|---|---|
| T1 unobservable (gray / surge / misconfig) | 11.1% / 8.1% / 5.7% | 10.9% / 6.3% / 7.8% | pass |
| T2 gray (subtle / obvious) | 25% / 27% | 29% / 22% | pass |
| T2 surge | 21% / 33% | 25% / 25% | pass |
| T2 routing misconfig | 12% / 52% | 17% / 49% | **miss** (subtle < 20%) |
| T2 flapping | 0% / 69% | 2% / 63% | **miss** |
| T3 (now length-matched) | pass (unmatched windows) | pass, max J 0.36 | pass |
| T4 literal pattern | 7.6% | 8.0% | pass (see N2: the zero-util refinement is 98.7%) |
| T5 status lag | 46% zero, median 1 | 47% zero, median 1 | pass |
| T6 overlap or follow a change window; concurrent scenarios | 14.3% (deliberate only); 6.9% | **25.5%** actual (14.6% deliberate); 6.9% | **miss** (above 20%) |
| T7 baseline | pass | unchanged | pass |
| T8 lost-probe steps | 3.95% | 3.95% | pass |

- The T6 miss is a direct consequence of N6. Isolated faults now land on change windows by chance, on top of the 15% placed there deliberately.
- I did not lower `overlap_maintenance_prob` to bring it back into range, because that would be tuning to target.
- Phase 4 should report isolated, maintenance-overlapping and concurrent faults separately anyway.

### 2026-10-30: Re-verification fresh-clone check: passed

- Cloned `origin/main` at `7fd2253`. No `.env` and no `data/`.
- New throwaway env `nettwin-freshcheck` from `environment.yml`.
- Phase 0, 1 and 2 scripts all exited 0.
- `git status` was clean after regeneration: every committed result and figure is byte-identical.
- All 118 files under `data/` are byte-identical to the local build.
- `pytest`: 85 passed.
- Clone and env deleted afterwards.

### 2026-10-30: Simulator freeze (audit M8 parked item)

- Frozen before any Phase 3 dataset was generated. Combined sha256 `dcdea81026a02ad67ab0c1488a84c028d0a16d8fd5cf330fd9b1dbc82c13077e` over `config.yaml`, `src/nettwin/simulator.py`, `faults.py`, `synth.py`, `topology.py` and `kg.py`. Per-file hashes are in `results/simulator_freeze.json`.
- The freeze also covers `topology.py`, `synth.py` and `kg.py`, wider than the Phase 2 tuning hash, because they decide which graphs and routes the data comes from.
- **Rule from here on:** no change to any frozen file without an entry in this log giving the reason. The test `test_simulator_is_frozen` fails on any unlogged change. The builder refuses to run unless `--unfrozen-reason` is passed, and records that reason in the summary.
- Changes made just before the freeze, as part of Phase 3 plumbing:
  - `Simulator.simulate(..., with_blind=True)` plus `blind_view`. The status-blind variant comes from the same run, and a test checks it equals simulating with `status_blind` (telemetry, events, impacts, labels).
  - `write_scenario` takes separate observable / truth roots and writes an observable `meta.json` without the seed. The seed plus the config would let anyone re-run the sampler and recover the faults.
  - New `dataset` config section.
  - None of these changes simulated physics or difficulty.

### 2026-10-30: Phase 3 datasets

- Plan (`config.yaml` `dataset`, `src/nettwin/dataset.py`):
  - train: 4 scenarios x 308 generated train graphs.
  - val: 4 x 92 generated validation graphs plus 20 x 5 validation_real.
  - test: 30 x 7 test networks.
  - dev: 30 x AARNet.
  - Total 1,940 scenarios, each in two variants (`with_status`, `status_blind`) from the same run.
- Seeds: one block per split (train 20,000,000+, val 21,000,000+, test 22,000,000+, dev 23,000,000+, 1,000 per topology). Every seed is checked with `check_fresh_seed`, and a test proves they are disjoint.
- Scenario ids are neutral and stable: `tr` / `va` / `te` / `dv` plus the topology's plan index and scenario number.
- The variant was first named `default`; the observable-path scan caught that it contains "fault", so it was renamed `with_status`.
- Observable `meta.json` carries no seed: the seed plus the frozen config would let anyone re-run the sampler and recover the faults.
- Disk check before building:
  - `--estimate` on 6 train graphs projected 7.95 GB and 3.6 CPU-hours. That was reported to Zareen before generating.
  - Free space was 21 GB at first, later 26 GB and then 45 GB after Zareen freed space.
  - Choice made by Zareen: full build, and a full fresh-clone rebuild.
  - Actual: 7.5 GB on disk, 21.4 minutes on 6 workers.
- Result: 1,232 / 468 / 210 / 30 scenarios with 2,434 / 937 / 431 / 55 faults (train / val / test / dev).
- Every count by split, topology, fault type, severity class, observability channel, maintenance overlap and concurrency is in `docs/dataset_card.md`, generated from `results/phase3_dataset_summary.json`.
- Phase 2 rerun after the plumbing changes: only the three hash fields in `results/phase2_summary.json` changed. Every diagnostic number is identical.
- Tests (`tests/test_dataset.py`):
  - no test or dev topology, and no test or dev family, in train or val; train is generated graphs only.
  - seeds are fresh and disjoint.
  - a real small build (one train graph, Nextgen) is scanned: the observable tree has only the expected files, columns, event keys and meta keys, no ground-truth words, and no labels or manifests.
  - the status-blind variant has no status events.
  - the freeze holds, and the freeze guard refuses changes.
  - the built dataset's train / val manifests contain no test or dev topology.
- Not done in Phase 3 (parked): Alert nodes from threshold rules, and the sensitivity run with pre-tuning ranges. Both belong to Phase 4.

### 2026-10-30: Phase 3 fresh-clone check (full, as chosen by Zareen): passed

- Cloned `origin/main` at `4dab96c`. No `.env` and no `data/`.
- New throwaway env `nettwin-freshcheck` from `environment.yml`.
- Phase 0, 1, 2 and 3 scripts all exited 0; the Phase 3 build took 21.8 minutes on 6 workers.
- `git status` was clean after regeneration: every committed result, figure, `docs/dataset_card.md` and `results/phase3_dataset_summary.json` are byte-identical.
- All 19,543 files under `data/` (KG CSV, Phase 2 demo, the full v1 dataset in both variants) are byte-identical to the local build.
- `pytest`: 96 passed.
- Clone (about 7.6 GB) and env deleted afterwards.

### 2026-10-30: Phase 3 audit fixes, by ID (`docs/audit_phase3.md`)

No frozen file changed (`test_simulator_is_frozen` still passes against freeze `dcdea810...`). Only `src/nettwin/dataset.py`, `scripts/phase3_build_dataset.py`, tests and docs changed.

- **P1 (major):**
  - Keyed scenario ids alone would not have fixed this. Before changing anything I checked the leak on validation only: reading just the topology from `meta.json`, running the public sampler for that topology's few public candidate seeds, and matching the observed maintenance log recovered the exact faults for 5 of 5 val scenarios, without using the id at all. The maintenance schedule fingerprints the seed, and the seed blocks were public.
  - I reported this to Zareen, who chose secret seeds and accepted that this means regenerating telemetry.
  - Fix:
    - Each scenario seed is now HMAC-SHA256(key, split | topology | slot), 60 bits.
    - Scenario ids are a separate keyed permutation within each split (`tr00000`..), so an id reveals neither topology, order nor seed.
    - The key is a 32-byte local secret in `.nettwin_dataset_key`, gitignored and never printed. The builder refuses to run without it.
    - The id to seed and slot mapping is only in `truth/` manifests.
    - `config.dataset.seed_base` is now unused; it was left in the frozen config rather than edited.
  - The dataset was rebuilt in full (20.2 minutes, 7.5 GB, same frozen simulator, new random draws).
  - Test `test_no_key_cannot_recover_faults`: on a real keyed build, an attacker with only the repo enumerates every seed the repo makes public for that topology (the old seed block, the id as a number, all reserved ranges) and runs the sampler. The true seed is never among them, and no candidate reproduces the observed maintenance log. With the key, the slot gives the seed back.
  - README section "Reproducing the datasets (needs the key)".
- **P2 (minor):**
  - The card's severity table now uses observable faults as the denominator, as T2 does, with unobservable counts in their own column.
  - A new table shows the T2 mix by split and by source (generated vs real).
  - Rebuilt val, observable only: routing_misconfig is 16.7% subtle / 49.2% obvious overall, a **miss**, matching the earlier T2 result. Generated graphs alone hit (20.2% / 47.1%). The real validation networks miss badly (3.6% / 57.1%).
- **P3 (minor):** the card now states that the variants differ in two ways: status events, and how down links are reported.
- **P4 (minor):**
  - The card reports fault starts per 3-hour bin of sim time, computed by the script, and the reason for the night excess (maintenance-linked placement).
  - It also states that models must not get absolute time of day in a way that encodes the maintenance-timing pattern, nor time since scenario start. Diurnal-normalised traffic features are fine. This is a parked item for Phases 4 to 6.
- **P5 (minor):** the dataset manifests give unobservable faults the impact class `unobservable`. The original symptom-size class is kept as `symptom_class`. This is done in `dataset.py`; the simulator's own output is unchanged.
- **P6 (minor):**
  - Labels are now written per variant with a new `telemetry_present` column: some link metric was observed in that cell, in that variant.
  - It differs between variants, because status_blind blanks down links. `label` itself is identical in both.
  - RQ1 scoring must state whether cells without telemetry are scored.
- **P7 (minor):** the card's limitations note the file-size signal (audit AUC 0.62) and that file metadata must never be a model input.
- The test count is now 99: 3 new dataset tests replace and extend the earlier ones.

## Parked ideas

Ideas that came up early, with the phase they belong to. Do not act on them before that phase.

- Done in Phase 2: ECMP cap 4 checked, loads within 3.3% of cap 16, kept.
- Phase 3 or 4: Alert nodes are still empty. Generate alerts from observed telemetry with simple threshold rules (needed for `get_active_alerts` and B3). Alert CSV columns will mix `value` and `value:float` when some alerts have no value; fix the export then.
- Done after the audit: generated graphs replace topohub's 100 shipped Gabriel graphs (400 graphs, every size 15 to 60). The shipped graphs are no longer in any split; `gabriel/20/1` is only a test fixture.
- Phase 3: build a status-blind dataset variant (now supported: `spec["status_blind"] = True`) as an ablation of shortcut 1.
- Phase 3: freeze `config.yaml` and record its sha256 here before building datasets (audit M8). Call `faults.check_fresh_seed` for every scenario seed (audit m12). Train on `generated_split(cfg)["train"]` only.
- Phase 4: score concurrent faults with set-based metrics, and report isolated vs maintenance-linked vs concurrent faults separately (audit M2). Exclude unobservable faults from headline metrics and report them separately (audit M5). Report RQ1 PR-AUC for root and symptomatic labels separately (audit M7).
- Phase 4: sensitivity run: regenerate validation with the pre-tuning fault ranges and report B1 to B3 on both (audit M8).
- Phase 4 and 5: report AARNet in a separate 'seen during development' column; never pool it with test.
- Phase 4, 5 and 6: report every result for link_down and node_down twice, with status events (default telemetry) and with the status-blind variant (audit re-verify N2). Never present the default-telemetry numbers for these types as evidence of hard detection.
- Phase 3: the dataset builder must use neutral scenario ids (`simulator.check_neutral_id` enforces this) and keep any description in the ground-truth manifest (audit re-verify N7).
- Phase 3: report fault results per type and per impact class; never average link_down / node_down / flapping with the hard types.
- Phases 4 to 6 (audit P4, P7): never give a model absolute time of day, day of week from scenario start, or time since scenario start in a way that lets it learn when faults or maintenance happen (fault starts lean toward night; faults never start on day 0). Diurnal traffic modelling, for example comparing a value with the same time on earlier days, is fine. Never use file metadata (sizes, timestamps, paths) as features. Per-class tables must exclude `impact_class == "unobservable"` faults (audit P5), and RQ1 must state how cells with `telemetry_present == False` are scored (audit P6).
- Phase 4: B3 can be built directly on `shared_elements` plus `paths_through` for the healthy-path penalty.
- Phase 7 / 10: a coastline basemap for the AARNet map (Natural Earth, or plotly geo in Streamlit) would look better; needs a data download, so ask first.
- Optional, first to cut: test-import the CSV into a real Neo4j instance.
