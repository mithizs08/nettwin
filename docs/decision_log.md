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

## Parked ideas

Ideas that came up early, with the phase they belong to. Do not act on them before that phase.

- Phase 2: if ECMP cap 4 noticeably skews AARNet link loads, raise it or weight paths. Alert CSV columns will mix `value` and `value:float` when some alerts have no value; fix the export when alerts are populated.
- Phase 4: B3 can be built directly on `shared_elements` plus `paths_through` for the healthy-path penalty.
- Phase 7 / 10: a coastline basemap for the AARNet map (Natural Earth, or plotly geo in Streamlit) would look better; needs a data download, so ask first.
- Optional, first to cut: test-import the CSV into a real Neo4j instance.
