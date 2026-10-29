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
