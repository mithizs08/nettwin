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

### 2026-10-30: Phase 3 audit-fix fresh-clone check (with the key): passed

- Cloned `origin/main` at `2752660`. No `.env`, no `data/`, no key.
- New throwaway env `nettwin-freshcheck` from `environment.yml`.
- Without the key, `scripts/phase3_build_dataset.py` refused to run (exit 1, "the dataset key is local only").
- With the local key copied in, the Phase 0, 1, 2 and 3 scripts all exited 0.
- `git status` was clean after regeneration.
- All 19,543 files under `data/` are byte-identical to the local build, so reproduction with the key is exact.
- `pytest`: 99 passed.
- Clone (including its copy of the key) and env deleted afterwards.

### 2026-10-30: Phase 4 pre-registration (written before any validation run)

Committed before `scripts/phase4_baselines.py` is run on validation. Nothing in this entry was chosen after seeing a validation number.

**Alerts (parked item, `src/nettwin/alerts.py`).**
- Built only from `observable/` files (`nettwin.observed.load_observed` refuses a `truth/` path) plus the operator's static inventory (topology, services, SLAs, normal routing).
- Rules: link status down and PoP unreachable (from the lagged events), util >= 90%, discards >= 0.05%, one-hop latency >= 2x its trailing one-day EWMA and >= +1 ms, one-hop probe missing with counters present, no link telemetry at all (info), service probe loss >= 99% (unreachable), 15-minute mean probe loss >= the service's SLA loss, latency >= the SLA latency.
- Every metric rule must hold for 2 consecutive steps and is raised on the second, so an alert active at t uses data up to t only.
- One change made on train before this entry: SLA loss was first judged per 5-minute sample. On 20 train scenarios one lost probe (1 of 600 = 0.17%) breached the 0.1% gold SLA, so it moved to a 15-minute mean (an SLA reporting interval). SLA loss alerts stay noisy (about 165 per scenario-week on train) because background probe-loss bursts are a deliberate distractor (audit C2). That is kept as realistic NOC noise.
- Every alert gets a float `value`, so the KG CSV export no longer mixes `value` and `value:float` (Phase 1 parked note). PoP alerts attach to the PoP's links in the KG, since the frozen schema has no Alert -ON-> PoP.

**Harness (`src/nettwin/evaluation.py`), shared by Phases 4 to 6.**
- Detection (RQ1): link cells at steps >= 288 (day 0 has no faults by construction and is left out for every method).
  - Cells with no telemetry are scored like any other cell, the same way for every method (audit P6). A present-only PR-AUC is also stored.
  - Two tasks, never pooled: root (label 2 vs 0, label 1 left out) and symptomatic (label 1 vs 0, label 2 left out).
  - A positive belongs to a fault if it is on one of the fault's root links inside its window (root task), or inside the window plus the 3-step status lag (symptomatic task). Positives of faults outside a reported group are dropped, not counted as negatives. Symptomatic cells owned by no fault are dropped and counted.
  - PR-AUC is average precision on scores binned per method. Bins are fixed from train, and the binned AP matched sklearn's exact AP within 0.0005 on train.
- Localisation (RQ2): per fault, the method gets the oracle window [t_start, t_end) and ranks all links and PoPs.
  - Rank is pessimistic: ties count against the true root.
  - Set-based for concurrent faults: the roots of other faults with overlapping windows are removed from the ranking first.
  - Top-1 and top-3.
- False alarms: 1-hour windows from step 288.
  - A window is a fault window if it meets any fault (observable or not) plus the 3-step lag. Otherwise it is maintenance-only if it meets a change window plus a 12-step tail. Otherwise it is fault-free.
  - Alarm = any link cell >= the method's threshold. The threshold is fixed on train so that 5% of train fault-free windows alarm.
  - Reported: the alarm rate on val fault-free and maintenance-only windows (also by source), and the fault detection rate at that threshold (a cell labelled 1 or 2 inside the fault window alarms).
- Groups, always separate:
  - each fault type
  - hard (link_degradation, demand_surge, routing_misconfig) and easy (link_down, node_down, link_flapping), never pooled with each other
  - severity class within hard / easy
  - unobservable faults on their own and excluded from every other group
  - fault placement (isolated, maintenance-linked, concurrent)
  - generated vs real validation networks
  - scenarios with or without concurrent faults
  - with_status vs status_blind (separate runs)
- CIs: 95% percentile bootstrap over scenarios, 1,000 resamples, seed 42.

**Calibration on train only.** From the train manifest, `default_rng(3003)` picks 100 scenarios to fit B2 (unsupervised, 200,000 cells, no labels) and 200 others to fix the bins and the thresholds. This is done per variant. Test and dev are never read; the script refuses those split names, and a test checks it only reads train and val.

**Baselines (`src/nettwin/baselines.py`).**
- B1: EWMA z-scores per link on util (two-sided), one-hop latency and discards (upward).
  - Std floors at measurement noise; each z must hold for 2 steps. Missing values give no score.
  - Span and persistence were picked on 30 train scenarios: 12 h and 2 steps gave root AP 0.018, against 0.003 to 0.008 for 1 h / 3 h spans or no persistence. That was the only B1 tuning.
- B2: Isolation Forest (100 trees) on 10 per-link features: util relative to its trailing day, util minus the same step yesterday, util minus its recent EWMA, recent util std, latency relative to its trailing day, latency change vs yesterday, discards, recent max discards, and two missing-data flags. Not tuned.
- B3: tomography over alerts. "Paths" are each service's normal routing paths (ECMP weighted), plus every link and every PoP as a one-element path. score(e) = (alerted paths with e / all alerted paths) x (alerted paths with e / all paths with e).
  - Per step for detection; over the window for localisation.
  - Info alerts (no telemetry) are not used. Maintenance is not used to suppress alerts. Normal routing is the t = 0 routing.
- For B1 and B2, a PoP's score is the mean of its links' scores. For localisation, an element's score is the 0.9 quantile of its per-step scores over the window.
- Banned inputs respected: `Observed` has no clock, no feature uses the step index, time of day or file metadata, and the one-day lag is a diurnal comparison. Tests check that the scores are causal (cutting the scenario changes nothing earlier) and that no feature name refers to time or files.

**What I have already seen (train only).** A dry run on 40 train scenarios (with_status, 30 calibration scenarios). Root PR-AUC, hard / easy:
- B1: 0.003 / 0.038
- B2: 0.014 / 0.053
- B3: 0.019 / 0.155

Top-3, hard / easy:
- B1: 0.18 / 0.58
- B2: 0.26 / 0.55
- B3: 0.21 / 0.88

Maintenance-only alarm rate: 0.24 / 0.15 / 0.14 (B1 / B2 / B3).

**Expectations for validation (each will be reported as met or not met).**
- E1. Hard types are hard: root PR-AUC on hard types < 0.10 for every baseline in both variants, and below that method's easy-type PR-AUC.
- E2. With status events, B3 has the best root PR-AUC and top-3 on easy types (B3 > B2 and B3 > B1).
- E3. Status-blind hurts the alert-based method most: B3's top-3 on link_down drops by at least 0.15 from with_status to status_blind. B2 (missing-data flags) drops less than B3 on link_down.
- E4. No baseline localises hard faults well: top-1 <= 0.35 for every method on hard types, both variants.
- E5. node_down top-1 is low for B1 and B2 (< 0.30, PoP score is a mean of link scores) and higher for B3.
- E6. Unobservable faults are near chance for every method: top-3 <= 0.15.
- E7. Fault-free alarm rate on val generated graphs within 3% to 8% for every method (calibrated at 5% on train). Maintenance-only alarm rate at least twice the fault-free rate for every method (drains and planned outages look like faults to all three).
- E8. Severity matters: within hard types, top-3 for subtle is below obvious for every method, and the detection rate at threshold for subtle hard faults is <= 0.30.
- E9. Real validation networks are not easier: hard-type top-3 on real <= generated + 0.05 for every method.
- E10. Concurrent scenarios are harder for B3: its easy-type top-3 is lower in scenarios with concurrent faults than without.
- E11. For B1 and B2 on hard types, symptomatic PR-AUC >= root PR-AUC (symptoms spread over many links).
- E12. Sensitivity run (pre-tuning fault ranges, same slots, same train calibration): every method's hard-type top-3 and root PR-AUC are at least as high as on v1, and top-3 rises by >= 0.05 for at least two of the three methods.

### 2026-10-30: Phase 4 results (validation only)

Run after the pre-registration commit `22653c2`. Outputs:
- `results/phase4_baselines_val.json`
- `results/phase4_sensitivity_pretune.json`
- `results/phase4_pretune_dataset.json`
- `results/phase4_tables.md` (every table, with 95% scenario-bootstrap CIs)
- `results/phase4_expectations.json` (E1 to E12, checked by `scripts/phase4_tables.py`; replaced after the audit by `results/phase4_claims.json`)

Val is 468 scenarios. With status events, it has 432 observable hard faults, 446 observable easy faults and 43 unobservable faults. No test or dev data was read. Calibration used train only: 100 scenarios fit B2, and 200 fixed the bins and thresholds. Binned AP matched exact AP on train to within 0.0001.

**Headline (with_status, observable faults).**

| | B1 | B2 | B3 |
|---|---|---|---|
| Root PR-AUC, hard | 0.006 | 0.050 | 0.032 |
| Root PR-AUC, easy | 0.037 | 0.069 | 0.175 |
| Top-1 / top-3, hard | 0.12 / 0.28 | 0.22 / 0.38 | 0.13 / 0.19 |
| Top-1 / top-3, easy | 0.12 / 0.45 | 0.20 / 0.56 | 0.74 / 0.87 |
| False alarms, fault-free / maintenance-only hours | 4.3% / 26.9% | 3.9% / 18.6% | 5.1% / 19.8% |

**Pre-registered expectations: 7 of 12 met.**
- E1 **not met**. Hard-type root PR-AUC is below 0.10 everywhere, as expected. But in status_blind, B1 and B2 score easy types below hard types: easy 0.000 for B1, 0.012 for B2. They barely see blanked down links.
- E2 met: B3 is best on easy types with status events (root PR-AUC 0.175, top-3 0.87).
- E3 **not met**. B3's link_down top-3 did drop by 0.31 without status events, as expected. But B2 dropped more (0.57, from 0.86 to 0.30), not less. Its missing-data flags did not make blanked links stand out from ordinary gaps and probe timeouts. *Corrected after audit Q3: that explanation depends on the 0.9-quantile window score. With a window mean, B2 keeps status-blind link_down top-3 at 0.59 (audit), so the flags do mark blanked links; the noise on other links hides them under q90.*
- E4 met: hard top-1 is at most 0.22.
- E5 met: node_down top-1 is 0.00 for B1 and 0.01 for B2, against 0.95 for B3.
- E6 **not met**. Unobservable faults are not all near chance:
  - B2 puts 18% of unobservable hard faults (6 of 34) in its top 3.
  - In status_blind, B2 puts 40% of unobservable easy faults in its top 3, and B3 13% (n = 15).
  - Finding: the observability flags (audit N8) count a channel as visible only where a value was observed. A link blanked in status_blind therefore counts as "unobservable", yet the blanking itself is visible as missing data. The flag definition undercounts what a detector can see.
  - *Corrected after audit Q6: this explanation fits the status-blind easy hits (B2 ranks blanked roots 1st or 2nd, against control ranks 32 to 87). It does not fit the hard hits, which also occur with status events and sit near B2's own rate on fault-free control windows (6 of 33 against 3 of 33, audit). "Near chance" should have been judged against such a control rate, not a uniform 3 / n.*
- E7 met: fault-free alarm rate on generated val graphs is 4.5% to 5.0%. Maintenance-only rates are 2.8x to 6.3x the fault-free rates across both variants.
- E8 met: subtle hard top-3 is 0.03 to 0.27, against obvious 0.38 to 0.58. Subtle detection rate is at most 0.25.
- E9 **not met**, for B1 and, in status_blind, for B3 (0.235 on real against 0.182 + 0.05; corrected after audit Q5): B1 hard top-3 is 0.35 on real networks against 0.26 on generated graphs. Both differences are inside the bootstrap noise (audit Q12). Real validation networks are smaller (fewer elements to rank), and their misconfigs and surges are more often obvious (dataset card).
- E10 met: B3 easy top-3 is 0.78 in concurrent scenarios, 0.88 without.
- E11 **not met**, for B2 only: symptomatic PR-AUC (0.030) is below root PR-AUC (0.050). B1 met it (0.012 vs 0.006).
- E12 met. With the pre-tuning fault ranges, hard top-3 rises by 0.06 to 0.08 for B1, B2 and B3, and hard root PR-AUC rises too, in both variants. The difficulty tuning moved results in the expected direction. *Corrected after audit Q8: the builds pair only 878 of 921 faults (a misconfig's draws shift every later draw in 29 scenarios); on matched faults E12 still holds, and the apparent routing_misconfig fall (0.10 to 0.09) is noise (+0.009 on matched faults).*

**Other findings (surprising or worth carrying forward).**
- Isolation Forest (B2) is the best baseline on hard types (top-3 0.38). Tomography (B3) is best on easy types and the worst on hard ones: demand_surge top-3 is 0.05 and routing_misconfig 0.10. Surges and misconfigs rarely raise alerts at the root, and the root of a surge is a PoP with no alerts of its own. PLAN.md's "expect B3 to be strong" holds only for link_down, node_down and flapping. *Corrected after audit Q3 and Q7: pooled over hard types the B2 lead over B1 holds only under the q90 window score (with a window max for B1 the gap is +0.007, CI -0.030 to +0.040). Per type: B1 is at least as good as B2 on demand_surge top-3, B2 ties B3 on link_down top-3 (B3 is ahead on link_down top-1 and PR-AUC only), B3 has the worst node_down root PR-AUC, and "B3 worst on hard types" holds for top-1 / top-3, not root PR-AUC (B1 is lowest).*
- B1 and B2 almost never localise flapping (top-3 0.05 / 0.06), while B3 reaches 0.74. ~~The 0.9-quantile window score dilutes intermittent symptoms~~ (not supported, audit Q3: max and mean give 0.05 to 0.07 too); link telemetry carries little flapping signal, and flapping shows mostly in status events and end-to-end loss.
- Maintenance-only hours have the highest false-alarm *rate* for every method: 19% to 27% of them alarm with status events (11% to 23% status-blind), against 4% to 5% of fault-free hours. *Corrected after audit Q4: they are not the main source by count. Fault-free hours are 17 times more numerous, so maintenance hours supply only 18% to 27% of all false alarms.* Phase 6's change-log tool should still be measured on these hours.
- Without status events, B1 cannot see down links at all: easy root PR-AUC 0.000, and node_down top-3 0.00. That is by design (it ignores missing data). B3 keeps node_down top-3 at 0.93 through service reachability, but link_down falls to 0.55.
- Cell-level root PR-AUC is low for every method (at most 0.19 for any type). The positives are dominated by long gray-failure and misconfig windows with weak per-cell symptoms (66,732 of 95,462 with_status root cells).
- On real validation networks, B1 and B2 alarm less often in fault-free hours (2.4% and 1.7%, against 4.8% and 4.5% on generated graphs). B3 alarms more often (5.9%).
- Alerts: about 135 SLA-loss alerts per scenario-week, and only 24% start inside an observable fault window. Status-down and discard alerts are the most specific (72% and 75%).
- No symptomatic cell was left without an owning fault.

**Sensitivity run (parked M8 item, done).** Validation was rebuilt with the pre-tuning fault ranges in `data/datasets/v1_pretune`, with the same key, slots and ids (*corrected after audit Q8: ids, slots, seeds and maintenance logs match, but only 878 of 921 faults; 29 scenarios with a misconfig have a different fault list*). It took 4.8 minutes and 1.8 GB. No frozen file was edited; the ranges were overridden in memory, plus a runtime patch to restore uniform surge sampling. The fault mix is harsher: demand_surge is 8.7% subtle against 29.0% in v1, and routing_misconfig 60.2% obvious against 49.2%. Baselines were re-scored with the v1 train calibration; the comparison is in `results/phase4_tables.md`.

### 2026-10-30: Phase 4 fresh-clone check (with the key): passed after one fix

- Cloned `origin/main` at `25ace04`. No `.env`, no `data/`, no key. New throwaway env `nettwin-freshcheck` from `environment.yml`.
- Without the key, `scripts/phase4_build_pretune.py` refused to run (exit 1).
- With the key copied in, these all exited 0:
  - Phase 0, 1, 2 and 3 scripts
  - `phase4_baselines.py`
  - `phase4_build_pretune.py`
  - the sensitivity run
  - `phase4_tables.py`
- All 24,225 files under `data/kg`, `data/scenarios`, `data/datasets/v1` and `data/datasets/v1_pretune` are byte-identical to the local build. `pytest`: 108 passed.
- Every committed Phase 4 result reproduced byte for byte except one: `results/phase4_pretune_dataset.json`, which included the build's run time. Fix (`53e6f8c`): the timing is now printed, not stored. Rebuilding locally gave the same file apart from that field.
- A clean clone at `53e6f8c` then rebuilt the pre-tuning data with the key and left `git status` clean. To save time, that re-check reused the `nettwin` env with the clone's own `src` on the path.
- Clones (including their copies of the key) and the throwaway env were deleted.

### 2026-10-30: Phase 4 audit fixes, by ID (`docs/audit_phase4.md`)

No frozen simulator file changed (`test_simulator_is_frozen` passes against freeze `dcdea810...`). Telemetry is byte-identical; only ground-truth manifests gained a channel. "Before" is commit `d6ec794`; "after" is the rerun on validation. All numbers come from `results/phase4_baselines_val.json`, `results/phase4_sensitivity_pretune.json`, `results/phase4_claims.json` and `results/phase4_tables.md`, or from the saved `d6ec794` versions of them. Values are with status events, q90 window, unless a variant or aggregation is named.

- **Telemetry-absence channel (requested with the fixes, follows audit Q6 and E6).**
  - A fifth observability channel, `obs_telemetry_absence` (`dataset.absence_channel`): a root link is blanked (present with status, fully missing in status_blind) for at least 2 consecutive steps inside [t_start, t_end + 3), outside every change window on that link or its end PoPs.
  - Always false in with_status. `observable` is recomputed as any channel, and the impact class follows.
  - The dataset was rebuilt with the same key: 24,225 telemetry and label files unchanged, manifests, card and `results/phase3_dataset_summary.json` updated.
  - Status-blind unobservable faults: train 149 to 141, val 49 to 47, test 27 to 27, dev 3 to 3.
  - The channel is set on 312 status-blind val faults; most were already observable through other channels.
- **Q1 (major): B3 as pre-registered, B3+ kept as a disclosed variant.**
  - B3's own link path is now the link alone. B3+ is the version first run, whose link paths also hold the two end PoPs. Both are reported.
  - node_down top-1, B3 / B3+: 0.610 / 0.947 with status events, 0.701 / 0.367 status-blind. Easy top-1: 0.633 / 0.743 and 0.440 / 0.415.
  - Top-3 moves by at most 0.03, and cell scores for links are identical (link PR-AUC equal).
  - Claim C9 "B3+ ranks node_down roots first more often" is supported with status events (+0.34, CI +0.28 to +0.40) and contradicted status-blind (-0.33, CI -0.39 to -0.28). Counting link alerts for the end routers helps when reachability alerts exist and hurts when they do not.
- **Q2 (major): expected ranks under ties, and a PoP aggregation chosen on train.**
  - Top-k credit is now the expected value under random tie-breaking. The pessimistic rank is still stored.
  - The PoP score for B1 and B2 is chosen on 200 train scenarios per variant from mean, max and second-largest link score, by mean top-3 over observable train faults.
    - B1: second_max (train top-3 0.225, against 0.222 mean and 0.204 max). The margin over mean is 0.003, and the rule is applied as written.
    - B2: mean (0.349, against 0.324 and 0.294).
  - node_down top-1: B1 0.000 to 0.000, B2 0.007 to 0.094. PoP roots top-1, expected / pessimistic: B2 0.094 / 0.007 (easy), 0.069 / 0.000 (hard).
  - Side effect of B1's second_max: demand_surge top-3 falls from 0.283 to 0.136, and B1 hard top-3 from 0.280 to 0.233.
  - New tables report link roots and PoP roots separately.
  - Test `test_a_pop_or_a_spur_link_can_rank_first`: a PoP whose links are all anomalous gets non-zero top-1 credit (the old rule gave 0), and a spur link outranks its leaf PoP under second_max.
- **Q3 (major) and Q12 (minor): window aggregations and paired-CI verdicts.**
  - Every localisation metric is reported under q90 (pre-registered), max and mean.
  - Each headline claim and each expectation is checked with paired scenario-bootstrap differences: supported if the 95% CI is above 0, contradicted if below, inconclusive if it crosses 0 (`evaluation.verdict`).
  - Hard top-3 under q90 / max / mean: B1 0.233 / 0.344 / 0.236, B2 0.383 / 0.326 / 0.404.
  - Status-blind link_down top-3 for B2: 0.325 / 0.119 / 0.603.
- **Q4 to Q8 (minor): wording.** The Phase 4 results entry above and the parked rule were corrected in place, with each change marked.
  - Q4: maintenance-only hours have the highest false-alarm rate, not the most false alarms. They supply 25.0% / 20.2% / 18.2% of false alarms (B1 / B2 / B3), now in the tables.
  - Q5: E9 also missed for B3 status-blind.
  - Q6: the E6 explanation fits the status-blind easy hits only.
  - Q7: per-type exceptions.
  - Q8: the pre-tuning comparison now uses only faults matched in both builds (same scenario, type, root, start, end; observable in both): 836 with status events, 832 status-blind, 405 of them hard. The script docstring was corrected.
- **Q9 (minor): causal alerts.**
  - An alert's `value` is now the reading on its raise step, not the episode peak. `t_end` is the clear step, known only from then on.
  - `alerts.alerts_at(alerts, t)` gives the list as known at t, with `t_end` hidden for alerts still open. `add_to_kg` documents that it must be given that view.
  - Test `test_alert_fields_never_use_future_steps`: cutting the data at 5 points changes no field of any earlier alert except open ones.
  - Alert counts are unchanged (with status events, identical alert statistics). No metric used `value`.
- **Q10 (minor): calibration on each method's own validation scores.**
  - Bins come from a 2% random sample of all of a method's validation cell scores (no labels). The threshold is set so 5% of validation fault-free windows alarm. B2 is still fitted on train.
  - The validation fault-free alarm rate is therefore 5.0% by construction (before, at train thresholds: 4.3% / 3.9% / 5.1%), and E7's fault-free part is close to trivially met.
  - Maintenance-only alarm rates with status events: B1 26.9% to 28.4%, B2 18.6% to 21.6%, B3 19.8% to 18.9%.
  - Binned AP is within 0.0001 of exact AP on the first 40 val scenarios for every method and variant.
  - For Phase 5: a supervised model is trained on train, calibrated on validation the same way, and applied unchanged to test.
- **Q11 (minor): symptoms of concurrent faults.**
  - A symptomatic cell eligible for several faults by time goes to the faults whose KG neighbourhood contains the link: normal paths through the root, plus links within 2 hops.
  - 6,318 cells were reassigned and 576 stayed ambiguous (kept with every eligible fault), per variant.
  - "Detected at threshold" now counts only the fault's own root cells and its own symptoms.
  - Hard symptomatic PR-AUC moved by at most 0.001 (B1 0.0123 to 0.0121, B2 0.0305 to 0.0301, B3 0.0305 to 0.0295). B3 easy detection at threshold fell from 0.758 to 0.686, mostly through the threshold change (Q10).
- Not changed: the oracle window (the same for every method, flagged by the audit for RQ4), and the bootstrap over scenarios rather than topologies (audit Q12; RQ3 will add per-network numbers).

**Claim verdicts after the fixes (q90 / max / mean).**

| Claim | Verdict |
|---|---|
| C1 B2 has the best hard-type top-3 | supported / **inconclusive** / supported (B2 - B1 under max: -0.018, CI -0.066 to +0.027) |
| C2 B3 has the best easy-type top-3 (with status) | supported in all three |
| C3 B3 has the best easy-type root PR-AUC (with status) | supported |
| C4 B2 has the best hard-type root PR-AUC | **inconclusive** (status-blind B2 - B3 +0.011, CI -0.005 to +0.026) |
| C5 B3 localises hard faults worse than B1 and B2 | **inconclusive** / supported / **inconclusive** (B1 - B3 +0.038, CI -0.010 to +0.084) |
| C6 B3 localises flapping better than B1 and B2 | supported in all three |
| C7 maintenance-only hours alarm more often than fault-free hours | supported for every method and variant |
| C8 link_down top-3 drops without status events | supported in all three |
| C9 B3+ beats B3 on node_down top-1 | supported with status, **contradicted** status-blind |
| C10 real vs generated hard top-3 differ | **inconclusive** for every method (point +0.02 to +0.07) |

**Expectations re-checked with CIs (q90; registered verdict in brackets).**
- E1 not met [not met].
- E2 met [met].
- E3 not met [not met]; inconclusive under mean.
- E4 met [met].
- E5 met [met]; now with expected ties, so no longer met by construction.
- **E6 inconclusive [not met].**
- E7 met [met].
- **E8 inconclusive [met]**: B1 subtle detection 0.25, CI crosses 0.30.
- **E9 inconclusive [not met].**
- **E10 inconclusive [met].**
- E11 not met [not met].
- **E12 inconclusive [met]**: every matched top-3 and PR-AUC rise is supported, but the rise exceeds 0.05 with a CI above 0 only for B3.

The registered verdicts stand as the pre-registered result. The CI verdicts are what the evidence supports.

**Conclusions that changed.**
- B3's node_down top-1 of 0.95 belonged to B3+. Pre-registered B3 reaches 0.61 with status events, and B3+ is worse without status events.
- B1 and B2 are no longer locked out of ranking a PoP first: B2 node_down top-1 is 0.09. B1 is still 0.00.
- "B2 is the best baseline on hard types" holds under q90 and mean, not under max (B1 ties).
- "B3 is the worst on hard types" holds against B2 only. "B2 has the best hard root PR-AUC" is not shown status-blind.
- Maintenance hours have the highest false-alarm rate, not the most false alarms.
- Real and generated validation networks show no detectable difference for hard faults.
- The flapping explanation was withdrawn: link telemetry carries little flapping signal under any aggregation.
- Five expectation verdicts are inside the noise.

### 2026-10-30: Phase 4 audit-fix fresh-clone check (with the key): passed

- Cloned `origin/main` at `3e855f9`. No `.env`, no `data/`, no key. New throwaway env `nettwin-freshcheck` from `environment.yml`.
- Without the key, `scripts/phase4_build_pretune.py` refused to run (exit 1).
- With the key copied in, these all exited 0:
  - Phase 0, 1, 2 and 3 scripts
  - `phase4_build_pretune.py`
  - `phase4_baselines.py`
  - `phase4_tables.py`
- `git status` was clean after regeneration: every committed result (including the claims file, the tables and the regenerated dataset card) is byte-identical.
- All 24,225 files under `data/kg`, `data/scenarios`, `data/datasets/v1` and `data/datasets/v1_pretune` are byte-identical to the local build. `pytest`: 113 passed.
- Clone (including its copy of the key) and env deleted afterwards.

### 2026-10-30: Phase 5 pre-registration (written before any training run that is evaluated)

Committed before `scripts/phase5_gnn.py` trains or evaluates anything on validation. What I ran before this entry, all on train scenarios only, none on validation:
- a timing smoke test: 60 train scenarios, 1 epoch per architecture, no evaluation;
- a crash test of the whole script with `--smoke`: 1 epoch, train scenarios standing in for validation, written to a scratch folder and not used for any choice. Its K override did not reach the worker processes, so it used K = 12; that only affected the crash test.

**Model (`src/nettwin/gnn/`), fixed before training.**
- Graph: the line graph (nodes are links; two links are adjacent when they share a PoP; edge attribute = log degree of that PoP).
- Inputs per link per step: the 10 B2 window features over the last W = 12 steps, standardised with train statistics. These carry the same banned-feature rules as Phase 4 (no clock, no time since the start, no file metadata, causal).
- Static per-link inputs: log length and mean log end-PoP degree; the "static path features" are log normal-path count and log service count.
- The model sees link telemetry only: no status events and no service probes. That is a known handicap against B3, which uses alerts from both.
- Architecture: causal temporal encoder (two 1D convs, kernel 3, last step plus mean), then 2 message-passing layers with residuals and LayerNorm: GraphSAGE, GAT (4 heads, edge attribute) or GINE (edge attribute).
- Detection head: per link per step, trained on label >= 1.
- Localisation head: one logit per link and per PoP (a PoP gets the mean of its links' embeddings plus its log degree), softmax over the elements of the graph. The target is the uniform distribution over the roots of the observable faults active at that step.
- Loss: BCE + cross-entropy.
- Ablations at the selected configuration:
  - MLP: same features and heads, per-link layers instead of message passing. PoP pooling stays, because it defines the PoP elements.
  - No static path features.
  - Shuffled edges: line-graph edges relabelled by a fixed random permutation per topology, at training and at evaluation.
- One model per run is trained on both variants mixed. It is evaluated on each variant separately.
- Localisation uses the same oracle incident window as every other method, with the element probabilities aggregated over it by q90, max and mean.

**Tuning budget, fixed now. Nothing outside it will be trained or evaluated.**
- Training data: train split only. K = 12 steps per train scenario and variant (half inside observable fault windows, half anywhere after day 0): 29,568 samples. AdamW, lr 2e-3, weight decay 1e-4, batch 32, 6 epochs, gradient clipping 5. These are not tuned.
- Phase A: 6 configurations, {SAGE, GAT, GINE} x hidden {32, 64}, seed 0.
- Selection is on validation only, on a fixed subset of 60 val scenarios (`default_rng(3008)`, both variants). Selection score = mean of hard-type and easy-type expected top-3 (q90) over observable faults.
  - Each architecture keeps its better hidden size (ties go to the smaller one).
  - The best architecture is the one with the highest score.
- Phase B: seeds 1 and 2 for each architecture at its chosen hidden size (6 runs), and the 3 ablations at the best configuration with seeds 0, 1, 2 (9 runs).
- Total: 21 training runs. CPU only, 4 threads, one run at a time in the main process. A first attempt at 2 parallel runs stalled the 8 GB machine in swap, so the samples are now held once. Measured: about 70 s per epoch, so about 7 minutes per run and 2.5 hours in all. Measured times are reported.
- Full-scenario prediction encodes each link's sequence once instead of re-encoding every 12-step window. A test checks it equals the window-by-window model for all four layer types.
- Final evaluation: the 9 architecture models and the 9 ablation models are run with B1, B2, B3 and B3+ in one pass over all 468 val scenarios, so all bootstrap draws are paired.
- If a run crashes or a bug is found, it is fixed and the run repeated, and this is logged. No configuration is added.

**Scoring.** The shared Phase 4 harness, unchanged, with every breakdown.
- Bins for the GNN are a fixed label-free logit grid on its probabilities. Its alarm threshold comes from its own validation fault-free windows (5%).
- Every PR-AUC is shown next to its base rate.
- Per-network numbers for the 5 real validation networks.
- Seed variance (per-seed values, mean and sd).
- Verdicts against B1, B2, B3 and B3+ use the seed-mean GNN, whose bootstrap arrays share the baselines' draws.

**Expectations (GNN = the selected configuration, seed mean; checked under q90, max and mean where localisation is involved).** "met" means every component is supported, "not met" means one is contradicted, otherwise "inconclusive".
- X1. GNN beats B2 on hard-type top-3, both variants.
- X2. GNN beats B2 on hard-type root PR-AUC, both variants.
- X3. Easy types: with status events, GNN top-3 >= B3 - 0.05; status-blind, GNN top-3 > B3.
- X4. B3 beats the GNN on link_flapping top-3 with status events (the GNN sees no status events or service probes).
- X5. Synthetic-to-real transfer holds: real minus generated top-3 > -0.10 for hard and easy types, both variants.
- X6. The graph helps: GNN beats the MLP ablation on hard top-3 and easy top-1, both variants.
- X7. Shuffled edges hurt: GNN beats the shuffled-edges ablation on hard and easy top-3, both variants.
- X8. GNN has the best symptomatic PR-AUC against all four baselines, hard and easy, both variants.
- X9. At 5% fault-free alarms, the GNN alarms less than B2 in maintenance-only hours, both variants.
- X10. No architecture is clearly better: every pairwise hard top-3 difference between SAGE, GAT and GINE is inconclusive.
- Not an expectation, reported descriptively: the effect of dropping static path features, and the seed sd.

### 2026-10-31: Phase 5 results (validation only)

Run after the pre-registration commit `495d470`, exactly within the budget: 21 training runs, no extra configuration, no rerun. Outputs:
- `results/phase5_gnn_val.json` (every model and baseline, every breakdown, seed summaries, paired verdicts)
- `results/phase5_expectations.json` (X1 to X10)
- `results/phase5_tables.md`
- `results/phase5_timings.json` (wall-clock; the only file that does not reproduce byte for byte)

No test or dev data was read. Models are in `data/models/phase5/` (gitignored, rebuilt by the script).

**Training and selection.**
- 29,568 training samples from train, built in 211 s.
- Runs took 426 to 741 s each on 4 CPU threads, 2.4 hours for all 21. Evaluating 18 GNN models and 4 baselines on both variants took 75 minutes; the whole script took 4.4 hours.
- Parameter counts: GAT-h32 12,099; SAGE-h32 13,891; GINE-h32 14,083; the MLP ablation 11,843.
- Selection subset scores (60 val scenarios, seed 0) for h32 / h64: SAGE 0.765 / 0.744, GAT 0.807 / 0.757, GINE 0.783 / 0.770. Every architecture keeps h32, and **GAT-h32 is the selected configuration**.
- Seed sd of hard top-3 is 0.006 to 0.016 for every configuration. Seed sd of hard root PR-AUC is larger (0.007 to 0.034).

**Headline against the baselines (q90 window, GAT-h32 mean of 3 seeds).**

| | B1 | B2 | B3 | B3+ | GAT-h32 |
|---|---|---|---|---|---|
| Hard top-1 / top-3, with status | 0.13 / 0.23 | 0.26 / 0.38 | 0.12 / 0.20 | 0.14 / 0.20 | **0.49 / 0.63** |
| Easy top-1 / top-3, with status | 0.18 / 0.41 | 0.25 / 0.57 | 0.63 / 0.87 | 0.74 / 0.88 | 0.75 / 0.88 |
| Easy top-1 / top-3, status-blind | 0.01 / 0.03 | 0.08 / 0.26 | 0.44 / 0.70 | 0.42 / 0.72 | **0.76 / 0.88** |
| Hard root PR-AUC, with status (base rate 0.0023) | 0.006 | 0.050 | 0.032 | 0.032 | **0.076** |
| Easy root PR-AUC, with status (base rate 0.0004) | 0.037 | 0.069 | 0.175 | 0.175 | 0.218 |
| Maintenance-only alarm rate at 5% fault-free alarms, with status | 28% | 22% | 19% | 19% | **48%** |

Paired verdicts, GAT-h32 minus each baseline:
- Hard types: every top-1, top-3 (all three window aggregations), root PR-AUC and symptomatic PR-AUC difference is supported against B1, B2, B3 and B3+, in both variants. Hard top-3 against B2 is +0.25 (CI +0.20 to +0.29) with status.
- Easy types with status events: the differences against B3 and B3+ are inconclusive for top-3 (+0.01, CI -0.02 to +0.05). Top-1 beats B3 but not B3+.
- Easy types status-blind: every difference is supported.

**Pre-registered expectations.**

| ID | Expectation | Verdict (q90 / max / mean) |
|---|---|---|
| X1 | GNN beats B2 on hard top-3, both variants | met / met / met |
| X2 | GNN beats B2 on hard root PR-AUC | met |
| X3 | Easy: GNN >= B3 - 0.05 with status, > B3 status-blind | met / met / met |
| X4 | B3 beats the GNN on flapping top-3 with status | met / met / met (B3 - GNN +0.14, CI +0.04 to +0.23) |
| X5 | Transfer: real minus generated top-3 > -0.10, hard and easy | **inconclusive**. Hard types are better on real networks (0.67 against 0.62). Easy types are lower on real networks (0.83 against 0.89), and the CI of the margin crosses 0 |
| X6 | The graph helps: GNN beats the MLP ablation (no message passing; PoP pooling and graph-derived static inputs kept) on hard top-3 and easy top-1 | **not met**. Hard top-3 +0.014 (CI -0.011 to +0.040), inconclusive. Easy top-1 with status -0.023 (CI -0.045 to -0.002): the MLP is better |
| X7 | Shuffled edges hurt | **inconclusive**. Hard +0.022 (CI -0.001 to +0.044), easy +0.009 |
| X8 | GNN has the best symptomatic PR-AUC | met (both variants, hard and easy, against all four) |
| X9 | GNN alarms less than B2 in maintenance hours | **not met** (contradicted). 48% against 22% with status, 47% against 14% status-blind |
| X10 | No architecture is clearly better | met (all pairwise differences inconclusive) |

**What this means, and what was surprising.**
- **The large gain over the baselines comes from supervised training, not from message passing.**
  - The MLP ablation (same features and heads, no message passing) reaches hard top-3 0.615 against 0.629, with no detectable difference. Shuffled edges cost at most 0.02, and dropping static path features costs about 0.02.
  - Corrected after the audit (R4): the MLP ablation means "no message passing", not "no graph". It keeps the link-to-PoP mean pooling for the PoP head and the graph-derived static inputs (end-PoP degree, normal-path and service counts). It also inherits hidden size 32, chosen for the GAT on validation, with no selection of its own; if anything that favours the GNN, so "the MLP is as good" is conservative. No result here supports a graph-free claim.
  - For RQ1 and RQ2 the honest reading is that "learning which element is the root" is what helps. On this data the graph structure adds little beyond PoP pooling (link-to-PoP incidence, which every variant keeps).
  - This must be stated in the write-up rather than credited to the GNN.
- **The GNN localises demand surges far better than anything else**: top-3 0.67 against at most 0.25 (B2, status-blind). Surge roots are PoPs, and the trained PoP head learns the "all incident links rise together" pattern that no baseline encodes.
- **The GNN is no better than B3 on easy faults when status events exist.** It never sees status events or service probes. It is worse on flapping (0.59 against 0.73 top-3), and worse than B3+ on node_down top-1 (0.87 against 0.95).
- **Without status events the GNN keeps its easy-type accuracy** (top-3 0.88). B3 drops to 0.70 and B2 to 0.26.
  - Corrected after the audit (R3): this is the GNN recognising the simulator's down-link rendering in the link telemetry (util exactly 0 with latency missing when status events exist; every metric missing status-blind), not hard detection. The GNN never sees status events, so the with-status link_down / node_down numbers come from the same rendering.
  - The audit's counterfactuals on validation: removing the signature from the true root drops its top-3 to 0.04 to 0.09; painting it on a healthy link or PoP makes the GNN rank that element first 85% to 98% of the time.
  - So the link_down / node_down numbers (top-3 0.99 to 1.00) mean "recognises the down-link rendering", in both variants. They are never evidence of hard detection.
- **The GNN's detection head raises far more false alarms in maintenance hours**: 48%, against 19% to 28% for the baselines, at the same 5% fault-free rate. Maintenance-only hours supply 35% of its false alarms.
  - Drains and planned outages shift traffic exactly like faults. Since those cells are label 0, the model has no input that tells them apart (it does not see the change log).
  - The maintenance log is the obvious missing input. It is a Phase 6 tool, and was deliberately not given to the GNN.
- **Per-cell root PR-AUC improves but stays low** (hard 0.076, base rate 0.0023). On link_degradation it is below B2 (-0.015, CI -0.026 to -0.005). Localisation on an oracle window is far easier than per-cell detection.
- **Unobservable faults**: GNN top-3 is 0.11 (hard, with status), below B2's 0.18. There is no sign that it localises from a prior on which elements tend to be roots.
- **Real validation networks**: GNN hard top-3 per network is 0.42 (Quest) to 0.84 (Aconet), at or above every baseline on each network.
- Binned against exact AP for the GNN on the first 40 val scenarios: 0.146 / 0.1462 with status, 0.1444 / 0.1446 status-blind.

**Not done in Phase 5.**
- GNNExplainer / gradient attribution (PLAN section 6), parked for Phase 6, where the agent's `explain_gnn` tool needs it.
- No test-set run, as instructed. RQ3 on test is for after this review.

### 2026-10-31: Phase 5 fresh-clone check (with the key): passed

- Cloned `origin/main` at `3b8846a`. No `.env`, no `data/`, no key. New throwaway env from `environment.yml`.
- Without the key, `scripts/phase4_build_pretune.py` refused to run (exit 1).
- With the key copied in, these all exited 0:
  - Phase 0, 1, 2 and 3 scripts
  - `phase4_build_pretune.py`, `phase4_baselines.py`, `phase4_tables.py`
  - `phase5_gnn.py` (all 21 training runs plus evaluation, about 4.3 h), `phase5_tables.py`
- `git status` after regeneration showed two changed files:
  - `results/phase5_timings.json`: wall-clock, expected.
  - `results/phase5_tables.md`: it prints the train seconds per run and the wall-clock line from the timings file.
- Every other committed result is byte-identical, including `results/phase5_gnn_val.json` and `results/phase5_expectations.json`, so all metrics, verdicts and seed values reproduce exactly. The tables script reads only those two files plus the timings file, and rerunning it locally reproduces the committed tables byte for byte, so the table difference comes from the timings alone. The clone was deleted before I could diff the table, so this is inferred, not observed line by line.
- All 24,225 files under `data/kg`, `data/scenarios`, `data/datasets/v1` and `data/datasets/v1_pretune` are byte-identical to the local build. `pytest`: 122 passed.
- Clone (including its copy of the key) and env deleted afterwards.

### 2026-10-31: Phase 5 audit fixes R1 to R5, and the maintenance-blanking check (validation only)

The independent audit is `docs/audit_phase5.md`: 1 major and 4 minor findings, none critical, no leakage. Fixes by ID. Nothing was retrained. `scripts/phase5_freeze.py` loads the saved checkpoints and refits B2 on train exactly as before. It then rescores all 468 validation scenarios once (34 min).

**R1 (major): no frozen test path.** Fixed.
- `scripts/phase5_freeze.py` saves the validation calibration under `data/models/phase5/frozen/`:
  - `calibration.json`: bin edges and alarm threshold for B1, B2, B3, B3+, GAT-h32 seeds 0 to 2 and MLP seeds 0 to 2, in both variants;
  - `baselines_<variant>.pkl`: the train-fitted baselines, with their PoP aggregation.
- It refuses to save unless every recomputed threshold equals the one in `results/phase5_gnn_val.json`. All 20 (10 methods, 2 variants) are equal, and the spot-checked top-3 and top-1 values match to 4 decimals.
- `results/phase5_frozen.json` records the sha256 of the 6 checkpoints and the 3 frozen files.
- `scripts/phase5_test.py` is new. It:
  - only loads these files and refuses to run if any hash differs;
  - refuses to run if the test pre-registration is missing from this log;
  - refuses to overwrite its output, so test is scored once;
  - reads ground truth only after every method has scored a scenario.
- The tests check, from the syntax tree, that the test script never calls `calibrate`, `threshold_for`, `make_edges`, `fit`, `train` or an optimiser step, and never imports the scripts that do. They also check that it refuses on a hash mismatch or a missing file.
- The audit's hash prefixes were mislabelled. `d81615b9…` is `gat-h32-s1-shuffled_edges.pt`, and `aaaab0be…` is `gat-h32-s2-mlp.pt`. The files are unchanged since training (02:55), and they reproduce the committed numbers exactly.
- The stale pre-Q10 `data/models/phase4/calib_*.pkl` were deleted. Nothing referenced them.

**R2 (minor): selection subset inside the scored set.** Fixed.
- `results/phase5_heldout_val.json` gives the headline on the 408 scenarios outside the 60-scenario selection subset, next to all 468. The held-out version is now the validation headline.
- With status events (held-out, then all 468):
  - GNN hard top-3: 0.606 against 0.629. Hard top-1: 0.473 against 0.489.
  - Easy top-3: 0.887 against 0.880.
  - Hard root PR-AUC: 0.074 against 0.076.
- Status-blind: hard top-3 0.604 against 0.627, easy top-3 0.885 against 0.878.
- No verdict changes. Held-out hard top-3 against B2 is +0.229 (CI +0.179 to +0.280) with status and +0.215 (CI +0.165 to +0.265) status-blind, both supported. Every hard-type difference against B1, B2, B3 and B3+ stays supported.
- MLP comparison:
  - Hard top-3 GNN minus MLP: +0.004 (CI -0.024 to +0.033), still inconclusive.
  - Hard root PR-AUC: GNN better with status (+0.012, CI +0.005 to +0.018) but not status-blind.
  - Easy root PR-AUC: GNN better in both variants.

**R3 (minor): link_down / node_down.** Wording corrected in place in the Phase 5 results entry and the README. The 0.99 to 1.00 top-3 means "recognises the simulator's down-link rendering", in both variants. It is never evidence of hard detection.

**R4 (minor): MLP wording.** The X6 row and the "supervised training, not message passing" paragraph now say "no message passing", not "no graph". The MLP keeps PoP pooling and graph-derived static inputs, and it inherits the GAT's hidden size without its own selection.

**R5 (minor): split guard.**
- `nettwin.splits.guard` reads the split from the data path (the folder after `observable/` or `truth/`, after resolving `..`). It raises `PermissionError` outside the allowed set.
- Every data read in `phase4_baselines.py`, `phase5_gnn.py` and `phase5_freeze.py` goes through it with {train, val}, and every read in `phase5_test.py` with {test, dev}.
- The argv string matches are removed. A test checks that a test or dev path raises in the Phase 4 loader and the Phase 5 evaluator, and that a val path raises in the test script.

**Hypothesis: the GNN's maintenance-hour alarms come from the blanking signature.** Validation, all 468 scenarios, frozen thresholds (`results/phase5_maintenance_blanking_val.json`).
- A maintenance-only window counts as blanked if some link shows the down-link signature on 2 or more of its steps. The signature is util 0 with latency missing (with status events), or every metric missing (status-blind).
- Blanked windows are 681 of 3,276 maintenance-only windows with status events, and 807 of 3,276 status-blind.
- **Partly supported.**
  - GNN seed mean, blanked against not blanked: 0.898 against 0.375 with status events, and 0.765 against 0.373 status-blind. The difference is supported for every seed in both variants.
  - In blanked alarm windows the top-scoring link is itself blanked in 72% of the GNN's alarms (with status) and 55% (status-blind).
- **But blanking is not the main source of the excess.**
  - Every baseline also alarms far more in blanked windows (B2 0.526 against 0.135 with status events).
  - Only 39% (with status) and 40% (status-blind) of the GNN's maintenance alarms fall in blanked windows.
  - Against B2, the GNN's excess maintenance alarms (seed mean) split as follows:
    - with status: 254 in blanked windows and 625 in windows without blanking;
    - status-blind: 458 in blanked windows and 609 without.
  - By maintenance action, the GNN alarms in 51% of drain windows against B2's 17%, in both variants. Drains shift traffic without blanking anything.
  - So most of the excess is in maintenance windows without blanking: drains, which shift traffic, and the hour after maintenance ends, which also counts as maintenance-only. The split between those two was not measured. The planned-down rendering is the smaller part.
- Consequence for Phase 6:
  - The change log has to cover drains, not only planned "down" actions.
  - Masking blanked links alone could remove at most 29% of the GNN's excess over B2 with status events (254 of 879 alarms), and 43% status-blind (458 of 1,067).
- Recorded because it was asked for, whatever it showed. No method or threshold was changed because of it.

**Also fixed.** `results/phase5_tables.md` no longer prints the wall-clock training times, so it is byte-reproducible. Those times stay in `results/phase5_timings.json`, which differs by design.

## Parked ideas

Ideas that came up early, with the phase they belong to. Do not act on them before that phase.

- Done in Phase 2: ECMP cap 4 checked, loads within 3.3% of cap 16, kept.
- Done in Phase 4: Alert nodes are still empty. Generate alerts from observed telemetry with simple threshold rules (needed for `get_active_alerts` and B3). Alert CSV columns will mix `value` and `value:float` when some alerts have no value; fix the export then.
- Done after the audit: generated graphs replace topohub's 100 shipped Gabriel graphs (400 graphs, every size 15 to 60). The shipped graphs are no longer in any split; `gabriel/20/1` is only a test fixture.
- Phase 3: build a status-blind dataset variant (now supported: `spec["status_blind"] = True`) as an ablation of shortcut 1.
- Phase 3: freeze `config.yaml` and record its sha256 here before building datasets (audit M8). Call `faults.check_fresh_seed` for every scenario seed (audit m12). Train on `generated_split(cfg)["train"]` only.
- Done in Phase 4 (harness): score concurrent faults with set-based metrics, and report isolated vs maintenance-linked vs concurrent faults separately (audit M2). Exclude unobservable faults from headline metrics and report them separately (audit M5). Report RQ1 PR-AUC for root and symptomatic labels separately (audit M7).
- Done in Phase 4: sensitivity run: regenerate validation with the pre-tuning fault ranges and report B1 to B3 on both (audit M8).
- Phase 4 and 5: report AARNet in a separate 'seen during development' column; never pool it with test.
- Phase 4, 5 and 6: report every result for link_down and node_down twice, with status events (default telemetry) and with the status-blind variant (audit re-verify N2). Never present the default-telemetry numbers for these types as evidence of hard detection.
- Phase 3: the dataset builder must use neutral scenario ids (`simulator.check_neutral_id` enforces this) and keep any description in the ground-truth manifest (audit re-verify N7).
- Phase 3: report fault results per type and per impact class; never average link_down / node_down / flapping with the hard types.
- Phases 4 to 6 (audit P4, P7): never give a model absolute time of day, day of week from scenario start, or time since scenario start in a way that lets it learn when faults or maintenance happen (fault starts lean toward night; faults never start on day 0). Diurnal traffic modelling, for example comparing a value with the same time on earlier days, is fine. Never use file metadata (sizes, timestamps, paths) as features. Per-class tables must exclude `impact_class == "unobservable"` faults (audit P5), and RQ1 must state how cells with `telemetry_present == False` are scored (audit P6).
- Done in Phase 4 (B3 uses service paths plus one-element paths): B3 can be built directly on `shared_elements` plus `paths_through` for the healthy-path penalty.
- Phase 7 / 10: a coastline basemap for the AARNet map (Natural Earth, or plotly geo in Streamlit) would look better; needs a data download, so ask first.
- Optional, first to cut: test-import the CSV into a real Neo4j instance.
- Phases 5 and 6: the observability flags miss missing-data evidence (Phase 4, E6). A link blanked in status_blind is flagged unobservable, yet the blanking is visible. When reporting unobservable faults, say so; do not treat B2's or the GNN's hits on them as impossible.
- Phase 6: maintenance-only hours have the highest false-alarm rate for every baseline (19% to 27% with status events), though they supply only about a fifth to a quarter of all false alarms (audit Q4). The agent's change-log tool should be measured on exactly these windows.
- Phase 5: report the GNN under q90, max and mean window scores, with expected-rank ties, link and PoP roots separately, calibration on validation scores only, and paired-CI claim verdicts (`evaluation.verdict`). Compare against both B3 and B3+.
- Phase 6: agent tools must return alerts through `alerts.alerts_at(alerts, t)` (audit Q9). The agent and the localisers it is compared with must get the same window (audit section 6).
- Phase 6: implement `explain_gnn` (gradient attribution over the GAT-h32 models) and give the agent the change log. The GNN's maintenance-hour false alarms (48%) are the clearest case for it. Most of its excess is in drain and post-maintenance windows, not in blanked ones (Phase 5 audit-fix entry), so the change log must cover drains.
- Write-up: credit the GNN's localisation gain to supervised training, not to message passing (Phase 5, X6 and X7). Any RQ1 claim that "graph-aware beats per-link" must cite the MLP ablation.
