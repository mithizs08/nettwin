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

Builds the typed knowledge graph (PoP, Link, RoutingPath, Service; Alert, Maintenance and Runbook as schema) for the three test topologies, exports Neo4j-style CSV to `data/kg/`, and writes `results/phase1_kg_summary.json`, `results/figures/aarnet_kg.png` and `results/figures/aarnet_map.png`. Services and SLAs are synthetic.
