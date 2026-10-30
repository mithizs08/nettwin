"""Phase 4: threshold alerts, baselines B1 to B3 and the shared evaluation harness, on VALIDATION only.

Train is used to fit B2 (unsupervised) and to fix, per method and variant, the score bins and the alarm threshold
(5% of train fault-free windows). Nothing here reads the test or dev splits.

    python scripts/phase4_baselines.py --workers 6
    python scripts/phase4_baselines.py --dataset data/datasets/v1_pretune --calib data/models/phase4 \
        --out results/phase4_sensitivity_pretune.json        # sensitivity run (reuses the v1 train calibration)
"""
import argparse
import json
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from nettwin.alerts import RULES, generate_alerts
from nettwin.baselines import METHODS, WINDOW_QUANTILE, B1EwmaZ, index_for
from nettwin.evaluation import (FAR_TARGET, LAG, WARMUP, aggregate, ap_from_hists, labels_matrix, make_edges,
                                scenario_record, threshold_for, to_bins, window_classes)
from nettwin.observed import freeze_cfg, inventory, load_observed

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("with_status", "status_blind")
N_FIT, N_CAL = 100, 200
CTX = {}


def _init(ctx):
    CTX.update(ctx)


def _load(split, variant, entry):
    R = CTX["root"]
    obs = load_observed(R / "observable" / split / variant / entry["scenario_id"])
    kg = inventory(obs.topology, CTX["cfg_s"])
    ix = index_for(obs, kg)
    lab, pres = labels_matrix(pd.read_parquet(R / "truth" / split / variant / entry["files"]["labels"]), ix.links)
    return obs, kg, ix, lab, pres


def _calibrate_one(args):
    split, variant, entry = args
    obs, kg, ix, lab, pres = _load(split, variant, entry)
    rng = np.random.default_rng(int(entry["scenario_id"][2:]))
    out = {}
    for name, m in CTX["methods"].items():
        link = m.score(obs, kg, ix)[0]
        sc, lb = link[WARMUP:], lab[WARMUP:]
        neg = sc[lb == 0]
        _, _, cls = window_classes(entry, link.shape[0])
        starts = np.arange(WARMUP, link.shape[0], 12)
        mx = np.array([link[s:s + 12].max() for s in starts])
        out[name] = {
            "pos2": sc[lb == 2], "pos1": sc[lb == 1],
            "neg": neg[rng.random(len(neg)) < 0.02],
            "neg_all_root_check": (sc[lb != 1], lb[lb != 1] == 2),
            "free_max": mx[cls == 0],
        }
    return out


def _evaluate_one(args):
    split, variant, entry = args
    obs, kg, ix, lab, pres = _load(split, variant, entry)
    recs = {}
    for name, m in CTX["methods"].items():
        state = m.score(obs, kg, ix)
        recs[name] = scenario_record(m, state, ix, entry, lab, pres, CTX["edges"][name])
    alerts = generate_alerts(obs, kg)
    in_fault = np.zeros(obs.n_steps + 1, bool)
    for f in entry["faults"]:
        if f["impact"]["observable"]:
            in_fault[f["t_start"]:min(f["t_end"] + LAG, obs.n_steps)] = True
    a = alerts[alerts.t_start >= WARMUP]
    stats = {"count": Counter(a.rule), "in_fault": Counter(a.rule[in_fault[a.t_start.to_numpy()]])}
    return entry["scenario_id"], recs, stats


def calibrate(cfg, root, variant, workers):
    entries = json.loads((root / "truth" / "train" / variant / "manifest.json").read_text())
    rng = np.random.default_rng(cfg["seeds"]["training"])
    pick = rng.permutation(len(entries))
    fit_e = [entries[i] for i in sorted(pick[:N_FIT])]
    cal_e = [entries[i] for i in sorted(pick[N_FIT:N_FIT + N_CAL])]
    b2 = METHODS["B2"](seed=cfg["seeds"]["training"])
    b2.fit([load_observed(root / "observable" / "train" / variant / e["scenario_id"]) for e in fit_e])
    methods = {"B1": B1EwmaZ(), "B2": b2, "B3": METHODS["B3"]()}
    ctx = {"root": root, "cfg_s": freeze_cfg(cfg), "methods": methods}
    with Pool(workers, initializer=_init, initargs=(ctx,)) as pool:
        parts = pool.map(_calibrate_one, [("train", variant, e) for e in cal_e], chunksize=4)
    calib = {"methods": methods, "edges": {}, "threshold": {}, "check": {}, "fit_ids": [e["scenario_id"] for e in fit_e],
             "cal_ids": [e["scenario_id"] for e in cal_e]}
    for name in methods:
        pos = np.concatenate([p[name]["pos2"] for p in parts] + [p[name]["pos1"] for p in parts])
        neg = np.concatenate([p[name]["neg"] for p in parts])
        edges = make_edges(pos, neg)
        calib["edges"][name] = edges
        calib["threshold"][name] = threshold_for(np.concatenate([p[name]["free_max"] for p in parts]))
        # harness check: binned AP against exact AP on the train calibration set (root task)
        s = np.concatenate([p[name]["neg_all_root_check"][0] for p in parts])
        y = np.concatenate([p[name]["neg_all_root_check"][1] for p in parts])
        b = to_bins(s, edges)
        nb = len(edges) + 1
        binned = float(ap_from_hists(np.bincount(b[y], minlength=nb), np.bincount(b[~y], minlength=nb)))
        calib["check"][name] = {"exact_ap_root": round(float(average_precision_score(y, s)), 4), "binned_ap_root": round(binned, 4),
                                "n_bins": nb}
    return calib


def scen_meta(cfg, entry):
    real = set(cfg["topologies"]["validation_real"])
    return {"source": "real" if entry["topology"] in real else "generated",
            "concurrent": any(f["placement"] == "concurrent" for f in entry["faults"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dataset", default=None, help="dataset root (default: config dataset.root)")
    ap.add_argument("--calib", default=None, help="reuse a saved train calibration from this folder")
    ap.add_argument("--out", default="results/phase4_baselines_val.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    root = ROOT / (args.dataset or cfg["dataset"]["root"])
    calib_dir = ROOT / "data" / "models" / "phase4"
    out = {"split": "val", "dataset": str(root.relative_to(ROOT)), "far_target_train": FAR_TARGET,
           "window_quantile": WINDOW_QUANTILE, "alert_rules": {k: list(v) for k, v in RULES.items()}, "variants": {}}
    for variant in VARIANTS:
        t0 = time.time()
        if args.calib:
            calib = pickle.loads((ROOT / args.calib / f"calib_{variant}.pkl").read_bytes())
        else:
            calib = calibrate(cfg, ROOT / cfg["dataset"]["root"], variant, args.workers)
            calib_dir.mkdir(parents=True, exist_ok=True)
            (calib_dir / f"calib_{variant}.pkl").write_bytes(pickle.dumps(calib))
        print(f"[{variant}] calibration ready ({time.time() - t0:.0f}s): thresholds {calib['threshold']}", flush=True)
        entries = json.loads((root / "truth" / "val" / variant / "manifest.json").read_text())
        ctx = {"root": root, "cfg_s": freeze_cfg(cfg), "methods": calib["methods"], "edges": calib["edges"]}
        with Pool(args.workers, initializer=_init, initargs=(ctx,)) as pool:
            res = pool.map(_evaluate_one, [("val", variant, e) for e in entries], chunksize=4)
        res.sort(key=lambda r: r[0])
        by_id = {e["scenario_id"]: e for e in entries}
        meta = [scen_meta(cfg, by_id[sid]) for sid, _, _ in res]
        v = {"calibration": {"train_fit_scenarios": len(calib["fit_ids"]), "train_cal_scenarios": len(calib["cal_ids"]),
                             "threshold": calib["threshold"], "binned_vs_exact_ap": calib["check"]},
             "methods": {}}
        for name in calib["methods"]:
            recs = [r[name] for _, r, _ in res]
            nb = len(calib["edges"][name]) + 1
            v["methods"][name] = aggregate(recs, meta, nb, calib["threshold"][name], n_boot=args.n_boot, seed=cfg["seeds"]["global"])
        cnt, inf = Counter(), Counter()
        for _, _, st in res:
            cnt.update(st["count"])
            inf.update(st["in_fault"])
        v["alerts"] = {"per_scenario": {k: round(cnt[k] / len(res), 2) for k in RULES},
                       "share_starting_in_observable_fault_window": {k: (round(inf[k] / cnt[k], 4) if cnt[k] else None) for k in RULES}}
        out["variants"][variant] = v
        print(f"[{variant}] {len(res)} val scenarios evaluated ({time.time() - t0:.0f}s)", flush=True)
    path = ROOT / args.out
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    if any("test" in a or "/dev" in a for a in sys.argv[1:]):
        sys.exit("Phase 4 runs on validation only")
    main()
