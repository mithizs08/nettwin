"""Phase 0: confirm the environment and test topologies load. Writes results/phase0_env_check.json."""
import json
from pathlib import Path

import lightgbm  # noqa: F401  imported before torch to avoid an OpenMP clash on macOS (see decision log)
import networkx as nx
import topohub
import torch
import torch_geometric
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    out = {"torch": torch.__version__, "torch_geometric": torch_geometric.__version__, "topologies": {}}
    t = cfg["topologies"]
    for name in t["dev"] + t["validation_real"] + t["test"]:  # loading only: sizes and connectivity
        g = nx.node_link_graph(topohub.get(name, use_names=cfg["topologies"]["use_names"]), edges="edges")
        out["topologies"][name] = {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "connected": nx.is_connected(g),
        }
    results = ROOT / cfg["paths"]["results"]
    results.mkdir(exist_ok=True)
    (results / "phase0_env_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
