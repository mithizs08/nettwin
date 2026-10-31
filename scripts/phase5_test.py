"""Phase 5 test scoring (audit Phase 5, R1). Load only: no training, no fitting, no calibration.

It loads the saved checkpoints, the pickled baselines (B2 fitted on train) and the frozen validation calibration (bins
and thresholds) listed in results/phase5_frozen.json, and refuses to run if any sha256 differs, if the test
pre-registration is not in docs/decision_log.md, or if results/phase5_test.json already exists (the test set is scored
once). Every method's scores come from observable telemetry only; ground truth is read after scoring, for the harness.
Reads the test and dev splits only. AARNet (dev, seen during development) is reported separately and never pooled.

    python scripts/phase5_test.py --workers 3
"""
import argparse
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from nettwin.baselines import index_for
from nettwin.evaluation import (EASY, HARD, WINDOW_AGGS, aggregate, ci, labels_matrix, related_links, scenario_record,
                                verdict)
from nettwin.gnn.method import gnn_method, sha256
from nettwin.observed import freeze_cfg, inventory, load_observed
from nettwin.splits import guard

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SPLITS = frozenset({"test", "dev"})
SPLITS = ("test", "dev")
OUT = ROOT / "results" / "phase5_test.json"
FROZEN = ROOT / "results" / "phase5_frozen.json"
PREREG = "### 2026-10-31: Phase 5 test pre-registration"
BASELINES = ("B1", "B2", "B3", "B3+")
CTX = {}


def G(path):
    return guard(path, ALLOWED_SPLITS)


def verify(frozen, root=ROOT):
    """Every file the scoring loads must match its recorded sha256."""
    bad = [rel for rel, h in frozen["files"].items() if not (root / rel).exists() or sha256(root / rel) != h]
    if bad:
        sys.exit(f"refusing to score: sha256 mismatch or missing file: {bad}")


def _init(ctx):
    CTX.update(ctx)
    torch.set_num_threads(1)
    CTX["gnn"] = {k: gnn_method(k, (ROOT / "data" / "models" / "phase5" / f"{k}.pt").read_bytes(), ctx["cfg_s"]) for k in ctx["models"]}
    CTX["manifests"] = {}


def _score_one(args):
    split, variant, sid = args
    R = CTX["root"]
    obs = load_observed(G(R / "observable" / split / variant / sid))
    kg = inventory(obs.topology, CTX["cfg_s"])
    ix = index_for(obs, kg)
    methods = {**{k: CTX["baselines"][variant][k] for k in BASELINES}, **CTX["gnn"]}
    states = {name: m.score(obs, kg, ix) for name, m in methods.items()}
    # scoring is complete; ground truth is read only now, for the harness
    key = (split, variant)
    if key not in CTX["manifests"]:
        CTX["manifests"][key] = {e["scenario_id"]: e for e in json.loads(G(R / "truth" / split / variant / "manifest.json").read_text())}
    entry = CTX["manifests"][key][sid]
    lab, pres = labels_matrix(pd.read_parquet(G(R / "truth" / split / variant / entry["files"]["labels"])), ix.links)
    related = [related_links(kg, ix, f["root_cause"]) for f in entry["faults"]]
    cal = CTX["calibration"][variant]
    recs = {name: scenario_record(m, states[name], ix, entry, lab, pres, np.asarray(cal[name]["edges"]), related)
            for name, m in methods.items()}
    return sid, recs, entry


def scenario_ids(root, split, variant):
    return sorted(p.name for p in G(root / "observable" / split / variant).iterdir() if p.is_dir())


def network_groups(networks):
    g = {}
    for t in networks:
        name = t.split("/")[-1]
        for fam, types in (("hard", HARD), ("easy", EASY)):
            g[f"{fam}|network={name}"] = (lambda f, types=types: f["observable"] and f["type"] in types, lambda s, t=t: s["topology"] == t)
    return g


def comparisons(B, best, mlp):
    """The verdicts pre-registered in the decision log (Phase 5 test pre-registration)."""
    out = {}
    groups = ("hard", "easy") + tuple(f"type={t}" for t in EASY + HARD)
    for g in groups:
        for b in BASELINES:
            for agg in WINDOW_AGGS:
                for k in ("top1", "top3"):
                    out[f"{g}|{k}@{agg}|GNN - {b}"] = verdict(B[best][(g, f"{k}@{agg}")] - B[b][(g, f"{k}@{agg}")])
            for m in ("pr_auc_root", "pr_auc_symptomatic"):
                out[f"{g}|{m}|GNN - {b}"] = verdict(B[best][(g, m)] - B[b][(g, m)])
    for b in BASELINES:
        out[f"all|far_maintenance_only|{b} - GNN"] = verdict(B[b][("all", "far_maintenance_only")] - B[best][("all", "far_maintenance_only")])
    for agg in WINDOW_AGGS:
        out[f"hard|top3@{agg}|GNN - MLP"] = verdict(B[best][("hard", f"top3@{agg}")] - B[mlp][("hard", f"top3@{agg}")])
        out[f"easy|top1@{agg}|GNN - MLP"] = verdict(B[best][("easy", f"top1@{agg}")] - B[mlp][("easy", f"top1@{agg}")])
    return out


def replications(C):
    """X1, X2, X3, X4, X6, X8 and X9 as worded on validation, from the test comparisons (both variants, q90 / max / mean)."""
    def ok(keys, predicted="supported"):
        vs = [C[k]["verdict"] for k in keys]
        opposite = "contradicted" if predicted == "supported" else "supported"
        return "met" if all(v == predicted for v in vs) else "not met" if opposite in vs else "inconclusive"
    out = {}
    for agg in WINDOW_AGGS:
        out.setdefault("X1", {})[agg] = ok([f"{v}|hard|top3@{agg}|GNN - B2" for v in C["_variants"]])
        out.setdefault("X4", {})[agg] = ok([f"with_status|type=link_flapping|top3@{agg}|GNN - B3"], predicted="contradicted")
        out.setdefault("X6", {})[agg] = ok([f"{v}|hard|top3@{agg}|GNN - MLP" for v in C["_variants"]]
                                           + [f"{v}|easy|top1@{agg}|GNN - MLP" for v in C["_variants"]])
        out.setdefault("X3", {})[agg] = {"with_status: GNN - B3 + 0.05": C[f"with_status|easy|top3@{agg}|GNN - B3 + 0.05"]["verdict"],
                                         "status_blind: GNN - B3": C[f"status_blind|easy|top3@{agg}|GNN - B3"]["verdict"]}
        out["X3"][agg]["overall"] = "met" if all(v == "supported" for v in out["X3"][agg].values()) else \
            "not met" if "contradicted" in out["X3"][agg].values() else "inconclusive"
    out["X2"] = {"any": ok([f"{v}|hard|pr_auc_root|GNN - B2" for v in C["_variants"]])}
    out["X8"] = {"any": ok([f"{v}|{fam}|pr_auc_symptomatic|GNN - {b}" for v in C["_variants"] for fam in ("hard", "easy") for b in BASELINES])}
    out["X9"] = {"any": ok([f"{v}|all|far_maintenance_only|B2 - GNN" for v in C["_variants"]])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    if OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} exists: the test set is scored once (delete it only to reproduce the run)")
    if PREREG not in (ROOT / "docs" / "decision_log.md").read_text():
        sys.exit("refusing to score: the test pre-registration is not in docs/decision_log.md")
    frozen = json.loads(FROZEN.read_text())
    verify(frozen)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg_s = freeze_cfg(cfg)
    root = ROOT / cfg["dataset"]["root"]
    seed = cfg["seeds"]["global"]
    mdir = ROOT / "data" / "models" / "phase5" / "frozen"
    calibration = json.loads((mdir / "calibration.json").read_text())["variants"]
    baselines = {v: pickle.loads((mdir / f"baselines_{v}.pkl").read_bytes()) for v in frozen["variants"]}
    best, models = frozen["best"], frozen["models"]
    gnn_seeds = [m for m in models if not m.endswith("-mlp")]
    mlp_seeds = [m for m in models if m.endswith("-mlp")]
    networks = {s: cfg["topologies"][s] for s in SPLITS}
    t0 = time.time()
    out = {"note": "Phase 5 test scoring, run once with the frozen validation calibration. Telemetry is synthetic; "
                   "topologies are public. AARNet (dev) was seen during development and is reported separately.",
           "frozen_files": frozen["files"], "best": best, "models": models, "splits": {}}
    ctx = {"root": root, "cfg_s": cfg_s, "baselines": baselines, "models": models, "calibration": calibration}
    with Pool(args.workers, initializer=_init, initargs=(ctx,)) as pool:
        for split in SPLITS:
            S, C = {}, {"_variants": list(frozen["variants"])}
            for variant in frozen["variants"]:
                ids = scenario_ids(root, split, variant)
                res = pool.map(_score_one, [(split, variant, sid) for sid in ids], chunksize=2)
                res.sort(key=lambda r: r[0])
                meta = [{"source": "real", "topology": e["topology"],
                         "concurrent": any(f["placement"] == "concurrent" for f in e["faults"])} for _, _, e in res]
                V, B = {"n_scenarios": len(res), "methods": {}, "thresholds": {}}, {}
                for name in list(BASELINES) + models:
                    cal = calibration[variant][name]
                    V["thresholds"][name] = cal["threshold"]
                    V["methods"][name], B[name] = aggregate([r[1][name] for r in res], meta, len(cal["edges"]) + 1, cal["threshold"],
                                                            n_boot=args.n_boot, seed=seed, extra_groups=network_groups(networks[split]))
                B[best] = {k: np.mean([B[m][k] for m in gnn_seeds], axis=0) for k in B[gnn_seeds[0]]}
                mlp = f"{best}-mlp"
                B[mlp] = {k: np.mean([B[m][k] for m in mlp_seeds], axis=0) for k in B[mlp_seeds[0]]}
                seeds = {}
                for lb, members in ((best, gnn_seeds), (mlp, mlp_seeds)):
                    for g in ("hard", "easy"):
                        for metric in [f"{k}@{a}" for a in WINDOW_AGGS for k in ("top1", "top3")] + ["pr_auc_root", "pr_auc_symptomatic"]:
                            vals = [float(B[m][(g, metric)][0]) for m in members]
                            seeds.setdefault(lb, {})[f"{g}:{metric}"] = {
                                "per_seed": [round(x, 4) for x in vals], "mean": round(float(np.mean(vals)), 4),
                                "sd": round(float(np.std(vals, ddof=1)), 4), "seed_mean_ci": ci(B[lb][(g, metric)][1:])}
                    for which in ("far_fault_free", "far_maintenance_only"):
                        vals = [float(B[m][("all", which)][0]) for m in members]
                        seeds[lb][f"all:{which}"] = {"per_seed": [round(x, 4) for x in vals], "mean": round(float(np.mean(vals)), 4),
                                                     "seed_mean_ci": ci(B[lb][("all", which)][1:])}
                V["seed_summary"] = seeds
                for k, c in comparisons(B, best, mlp).items():
                    C[f"{variant}|{k}"] = c
                for agg in WINDOW_AGGS:
                    if variant == "with_status":
                        C[f"{variant}|easy|top3@{agg}|GNN - B3 + 0.05"] = verdict(B[best][("easy", f"top3@{agg}")] - B["B3"][("easy", f"top3@{agg}")] + 0.05)
                S[variant] = V
                print(f"[{split} {variant}] {len(res)} scenarios ({time.time() - t0:.0f}s)", flush=True)
            out["splits"][split] = {"variants": S, "comparisons": {k: v for k, v in C.items() if k != "_variants"},
                                    "replications": replications(C)}
    enc = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    OUT.write_text(json.dumps(out, indent=1, default=enc) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
