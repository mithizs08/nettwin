"""Phase 11 network-clustered bootstrap (after the audit, docs/audit_phases_11_12.md, m4). Not pre-registered; no API call.

The pre-registered CIs resample questions, but the 252 test questions come from 7 networks, and questions on one
network share its graph. Here whole networks are resampled with replacement (1,000 resamples, seed 42), each bringing
all of its questions, paired across conditions. With 7 networks these CIs are rough; a group with one network has
no between-network variation to resample, so its CI is not estimable and is left empty. Reads only the saved test
log and question file. Writes `results/phase11/<run>/clustered.json`.

    python scripts/phase11_clustered.py --run test
"""
import argparse
import json
from pathlib import Path

import numpy as np

from nettwin.evaluation import ci, verdict

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("Q0", "Q1", "Q2")
PAIRS = (("Q2", "Q1"), ("Q2", "Q0"), ("Q1", "Q0"))


def cluster_weights(n, n_boot, seed):
    """(1 + n_boot, n) network weights; row 0 is the full sample."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(float)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="test")
    args = ap.parse_args()
    run_dir = ROOT / "results" / "phase11" / args.run
    split = "test" if args.run == "test" else "val"
    qs = {q["qid"]: q for q in map(json.loads, (ROOT / f"results/phase11/questions_{split}.jsonl").read_text().splitlines())}
    recs = {(r["qid"], r["condition"]): r for r in map(json.loads, (run_dir / "log.jsonl").read_text().splitlines())}
    qids = sorted(q for q in qs if all((q, c) in recs for c in CONDITIONS))
    groups = {"all": qids} | {f"size={s}": [q for q in qids if qs[q]["size"] == s] for s in ("small", "medium", "large")}
    out = {"run": args.run, "pre_registered": False, "resampled_unit": "network", "n_boot": 1000, "seed": 42, "groups": {}}
    for g, ids in groups.items():
        nets = sorted({qs[q]["network"] for q in ids})
        W = cluster_weights(len(nets), 1000, 42)
        res = {"n_questions": len(ids), "n_networks": len(nets), "networks": nets, "f1": {}, "comparisons": {}}
        boots = {}
        for c in CONDITIONS:
            s = np.array([sum(recs[(q, c)]["f1"] for q in ids if qs[q]["network"] == n) for n in nets])
            k = np.array([sum(qs[q]["network"] == n for q in ids) for n in nets], float)
            boots[c] = (W @ s) / (W @ k)
            res["f1"][c] = {"value": round(float(boots[c][0]), 4), "ci": ci(boots[c][1:]) if len(nets) > 1 else [None, None]}
        for a, b in PAIRS:
            v = verdict(boots[a] - boots[b])
            if len(nets) < 2:
                v = {"verdict": "not estimable (one network)", "point": v["point"], "ci": [None, None]}
            res["comparisons"][f"f1|{a} vs {b}"] = v
        out["groups"][g] = res
    (run_dir / "clustered.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({g: {"networks": r["n_networks"], **{k: (v["point"], v["ci"], v["verdict"]) for k, v in r["comparisons"].items()}}
                      for g, r in out["groups"].items()}, indent=1))


if __name__ == "__main__":
    main()
