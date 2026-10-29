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

## Parked ideas

Ideas that came up early, with the phase they belong to. Do not act on them before that phase.

- Done in Phase 2: ECMP cap 4 checked, loads within 3.3% of cap 16, kept.
- Phase 3 or 4: Alert nodes are still empty. Generate alerts from observed telemetry with simple threshold rules (needed for `get_active_alerts` and B3). Alert CSV columns will mix `value` and `value:float` when some alerts have no value; fix the export then.
- Phase 3: topohub ships only seeds 0 to 9 per Gabriel size, so n in 15..60 gives 10 sizes x 10 seeds = 100 graphs for train + validation. The six Gabriel graphs used for Phase 2 diagnostics (sizes 20/30/40/50/25/35, seeds 1 to 6) can go in either split; they only tuned the simulator, not any detector.
- Phase 3: consider a status-blind dataset variant (no status events, down links reported as missing) as an ablation of shortcut 1.
- Phase 3: report fault results per type and per impact class; never average link_down / node_down / flapping with the hard types.
- Phase 4: B3 can be built directly on `shared_elements` plus `paths_through` for the healthy-path penalty.
- Phase 7 / 10: a coastline basemap for the AARNet map (Natural Earth, or plotly geo in Streamlit) would look better; needs a data download, so ask first.
- Optional, first to cut: test-import the CSV into a real Neo4j instance.
