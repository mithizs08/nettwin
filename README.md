# NetTwin

A knowledge-graph digital twin of a telecom backbone, with GNN anomaly localisation and an LLM root-cause agent. Telemetry is synthetic and topologies are public (Internet Topology Zoo / SNDlib via `topohub`). None of this is Telstra's network.

Work in progress. See `PLAN.md` for the plan and `docs/decision_log.md` for decisions.

## Setup

```bash
conda env create -f environment.yml
conda activate nettwin
pip install -e .
python scripts/phase0_check_env.py
pytest
```

## Phase 1: topology and knowledge graph

```bash
python scripts/phase1_build_kg.py
```

Builds the typed knowledge graph (PoP, Link, RoutingPath, Service; Alert, Maintenance and Runbook as schema) for the dev network (AARNet) and the validation_real networks, exports Neo4j-style CSV to `data/kg/`, and writes `results/phase1_kg_summary.json`, `results/figures/aarnet_kg.png` and `results/figures/aarnet_map.png`. Services and SLAs are synthetic.

## Phase 2: telemetry simulator

```bash
python scripts/phase2_simulate.py
```

Simulates synthetic traffic, routing (via the Phase 1 KG), utilisation, latency and loss with six fault types and distractors, deterministic from seed. Writes sanity figures to `results/figures/phase2_*.png`, the pre-registered difficulty targets (validation networks only) to `results/phase2_summary.json`, and demo scenarios to `data/scenarios/phase2_demo/` (observable telemetry and ground truth in separate folders). All telemetry is synthetic.

## Topology splits and test-driven decisions

Splits are in `PLAN.md` section 3 and `config.yaml`. AARNet is the dev / demo network: it was used to develop and test the simulator in Phases 1 and 2, so its numbers are reported separately as "seen during development". The clean test set is Geant2012, germany50 and five Topology Zoo networks picked by a rule that was pre-registered in `docs/decision_log.md` before it was run (`scripts/select_topologies.py`).

Decisions that were made while looking at a test (or now-dev) topology, as PLAN.md section 3 requires us to disclose (audit IDs from `docs/audit_phases_0_2.md`):
- L2: capacity planning from a noisy history week was triggered by AARNet peaks (median 0.78). Also confirmed on a Gabriel graph.
- L3 / L4: the ECMP cap of 4 was checked against AARNet loads and set with AARNet, Geant2012 and germany50 ECMP counts in view.
- L5: the routing metric floor of 1 was motivated by AARNet's 0 km links.
- L6: the rule "0 km links get the largest adjacent capacity" was designed on AARNet. It now also runs on generated graphs with dual-router PoPs.
- L9: per-PoP time zones were built for real geography. Generated graphs now get pseudo-longitudes, so the feature is exercised in training and validation too.

## Known realistic signatures (read before Phases 4 to 6)

A down interface on a reachable router reads exactly 0 traffic and 0 discards, and its one-hop probe fails. So in the default (non-blind) telemetry, "util exactly 0 with latency missing" identifies a down link almost perfectly: 99.7% of such cells are down links in the audit's re-check (`docs/audit_phases_0_2_reverify.md`, N2). This is what real SNMP shows, so it stays, but it means link_down and node_down are easy to detect from link telemetry alone, even without status events. Telemetry also changes at the fault's start step, so the status-event lag does not delay onset for a telemetry-based detector. Results for link_down and node_down must therefore be reported twice: with status events and the default telemetry, and with the status-blind variant (`spec["status_blind"] = True`, where down links are fully missing and no status or reachability events exist).

## Phase 3: datasets

```bash
python scripts/phase3_build_dataset.py --estimate     # disk and time estimate first (about 8 GB, 3.6 CPU-hours)
python scripts/phase3_build_dataset.py --workers 6    # full build into data/datasets/v1 (gitignored)
```

Builds train (generated graphs), val (generated validation graphs plus validation_real), test (the 7 test networks) and dev (AARNet) scenarios, each in a `with_status` and a `status_blind` variant, with observable data and ground truth in separate trees. The simulator is frozen first (`results/simulator_freeze.json`), and the builder refuses to run if it changed without a logged reason. Counts are in `docs/dataset_card.md` and `results/phase3_dataset_summary.json`.
