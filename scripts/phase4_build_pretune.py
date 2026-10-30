"""Phase 4 sensitivity data (parked audit M8 item): the validation split rebuilt with the fault ranges as they were
BEFORE the two pre-registered difficulty adjustments (decision log, "Pre-registered targets, results").

No frozen file is edited. The frozen config is loaded and only its `faults` ranges are replaced in memory with the
pre-tuning values; the demand-surge factor, sampled log-uniform after adjustment 1, is sampled uniform again through a
runtime patch of `faults._loguniform` for that one range. Same secret key and slots as v1, so every scenario keeps its
seed, topology and id, and the two builds pair up scenario by scenario.

    python scripts/phase4_build_pretune.py --workers 6      # writes data/datasets/v1_pretune (val only)
"""
import argparse
import copy
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import yaml

from nettwin import dataset, faults

ROOT = Path(__file__).resolve().parents[1]
OUT = "data/datasets/v1_pretune"
PRE_TUNING = {   # values from git ad950f7:config.yaml and faults.py, and the adjustment notes in the decision log
    "link_degradation": {"loss_pct": [0.01, 2.0]},
    "link_flapping": {"max_down_frac": [0.01, 0.5], "flaps_per_step": [0.5, 6], "convergence_s": [0.05, 1.0]},
    "demand_surge": {"factor": [1.3, 4.0]},
    "routing_misconfig": {"minimal_prob": 0.5},
}
UNIFORM_SURGE = [1.3, 4.0]


def pretune_cfg(cfg):
    c = copy.deepcopy(cfg)
    for ft, ranges in PRE_TUNING.items():
        c["faults"][ft].update(ranges)
    return c


def _patch():
    """Surge factor uniform on [1.3, 4] as before adjustment 1. Both samplers draw exactly one uniform, so the random
    streams stay aligned with v1."""
    orig = faults._loguniform

    def sample(rng, lo_hi):
        if list(lo_hi) == UNIFORM_SURGE:
            return faults._uniform(rng, lo_hi)
        return orig(rng, lo_hi)

    faults._loguniform = sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    dataset.check_frozen(ROOT)
    cfg = pretune_cfg(yaml.safe_load((ROOT / "config.yaml").read_text()))
    key = dataset.load_key(ROOT)
    root = ROOT / OUT
    tasks = [t for t in dataset.plan(cfg, key) if t["split"] == "val"]
    t0 = time.time()
    manifests = {}
    with Pool(args.workers, initializer=_patch) as pool:
        for task, out in pool.imap(dataset._build_topology, [(t, cfg, root) for t in tasks]):
            for variant, entries in out.items():
                manifests.setdefault(variant, []).extend(entries)
    summary = {"pre_tuning_ranges": PRE_TUNING, "surge_factor_sampling": "uniform"}   # no timings: the file must reproduce
    for variant, entries in manifests.items():
        entries.sort(key=lambda e: e["scenario_id"])
        d = root / "truth" / "val" / variant
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(entries, indent=1) + "\n")
        mix = {}
        for ft in ("link_degradation", "demand_surge", "routing_misconfig", "link_flapping"):
            fs = [f for e in entries for f in e["faults"] if f["type"] == ft]
            obs = [f for f in fs if f["impact"]["observable"]]
            c = Counter(f["impact"]["impact_class"] for f in obs)
            mix[ft] = {"n": len(fs), "observable": len(obs), "subtle": round(c["subtle"] / len(obs), 4),
                       "obvious": round(c["obvious"] / len(obs), 4), "unobservable": len(fs) - len(obs)}
        summary[variant] = {"scenarios": len(entries), "faults": sum(len(e["faults"]) for e in entries), "observable_mix": mix}
    (ROOT / "results" / "phase4_pretune_dataset.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"built in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
