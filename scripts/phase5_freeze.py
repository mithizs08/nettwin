"""Phase 5 audit fixes R1 and R2, and the maintenance-blanking check, on VALIDATION only (docs/audit_phase5.md).

Run after `scripts/phase5_gnn.py`. It trains nothing new. It loads the saved checkpoints of the models that will be
scored on test (the selected configuration's 3 seeds and its MLP ablation's 3 seeds), refits B2 on train exactly as
Phase 4 does, and rescores all of validation once to:
- R1: freeze each method's validation calibration (bin edges, alarm threshold) and the fitted baselines under
  data/models/phase5/frozen/, check that it equals what `results/phase5_gnn_val.json` used, and record the sha256 of
  every checkpoint and frozen file in results/phase5_frozen.json (`scripts/phase5_test.py` refuses to run on a mismatch);
- R2: report the headline on the 408 validation scenarios outside the 60-scenario selection subset, next to all 468;
- test whether the GNN's maintenance-hour alarms come from the down-link (blanking) signature.

    python scripts/phase5_freeze.py --workers 3
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase4_baselines as P4  # noqa: E402
import phase5_gnn as P5  # noqa: E402

from nettwin.baselines import index_for  # noqa: E402
from nettwin.evaluation import (MAINT_TAIL, aggregate, boot_weights, ci, labels_matrix,  # noqa: E402
                                related_links, scenario_record, threshold_for, verdict, window_classes)
from nettwin.gnn.method import gnn_method, sha256  # noqa: E402
from nettwin.observed import freeze_cfg, inventory, load_observed  # noqa: E402

G = P4.G
ROOT = Path(__file__).resolve().parents[1]
VARIANTS = P5.VARIANTS
MDIR = ROOT / "data" / "models" / "phase5"
FDIR = MDIR / "frozen"
SIG_STEPS = 2          # a link counts as blanked in a window if it shows the down-link signature on >= 2 steps
CTX = {}


def frozen_models(best):
    """The runs scored on test: the selected configuration and its MLP ablation, every seed."""
    return [f"{best}-s{s}" for s in P5.SEEDS] + [f"{best}-s{s}-mlp" for s in P5.SEEDS]


def signature(obs, blind):
    """(T, L) the down-link rendering in observable telemetry (audit Phase 5, section 3): util exactly 0 with latency
    missing when status events exist; every metric missing in the status-blind variant."""
    if blind:
        return np.isnan(obs.util) & np.isnan(obs.latency) & np.isnan(obs.discards)
    return (obs.util == 0) & np.isnan(obs.latency)


def _init(ctx):
    CTX.update(ctx)
    CTX["gnn"] = {k: gnn_method(k, b, ctx["cfg_s"]) for k, b in ctx["states"].items()}


def _evaluate_one(args):
    variant, entry = args
    R = CTX["root"]
    obs = load_observed(G(R / "observable" / "val" / variant / entry["scenario_id"]))
    kg = inventory(obs.topology, CTX["cfg_s"])
    ix = index_for(obs, kg)
    lab, pres = labels_matrix(pd.read_parquet(G(R / "truth" / "val" / variant / entry["files"]["labels"])), ix.links)
    related = [related_links(kg, ix, f["root_cause"]) for f in entry["faults"]]
    starts, ends, cls = window_classes(entry, obs.n_steps)
    sig = signature(obs, variant == "status_blind")
    blanked = np.stack([sig[s:e].sum(0) >= SIG_STEPS for s, e in zip(starts, ends)])          # (windows, L)
    actions = [sorted({m["action"] for m in entry["maintenance"] if s < m["t_end"] + MAINT_TAIL and m["t_start"] < e})
               for s, e in zip(starts, ends)]
    recs, top_blank = {}, {}
    methods = {**{k: CTX["baselines"][variant][k] for k in P4.METHOD_NAMES}, **CTX["gnn"]}
    for name, m in methods.items():
        state = m.score(obs, kg, ix)
        edges = CTX["edges"][name] if name in P4.METHOD_NAMES else P5.GNN_EDGES
        recs[name] = scenario_record(m, state, ix, entry, lab, pres, edges, related)
        arg = np.array([np.unravel_index(np.argmax(state[0][s:e]), state[0][s:e].shape)[1] for s, e in zip(starts, ends)])
        top_blank[name] = blanked[np.arange(len(starts)), arg]
    win = {"cls": cls, "any_blanked": blanked.any(1), "n_blanked": blanked.sum(1), "actions": actions, "top_blanked": top_blank}
    return entry["scenario_id"], recs, win


# ------------------------------------------------------------------ R2: held-out headline

HEAD = [(g, m) for g in ("hard", "easy") for m in ("top1@q90", "top3@q90", "top1@max", "top3@max", "top1@mean", "top3@mean",
                                                    "pr_auc_root", "pr_auc_symptomatic")] + [("all", "far_maintenance_only")]


def headline(recs, meta, models, edges, thr, n_boot, seed, best):
    """GNN seed means and baselines on one set of scenarios, with paired verdicts and X1 / X2 recomputed."""
    B, out = {}, {"n_scenarios": len(meta), "methods": {}}
    for name in list(P4.METHOD_NAMES) + models:
        nb = len(edges[name]) + 1
        summ, B[name] = aggregate([r[name] for r in recs], meta, nb, thr[name], n_boot=n_boot, seed=seed)
        out["methods"][name] = {f"{g}:{m}": float(B[name][(g, m)][0]) for g, m in HEAD}
        out["methods"][name]["n_faults_hard"] = summ["groups"]["hard"]["n_faults"]
        out["methods"][name]["n_faults_easy"] = summ["groups"]["easy"]["n_faults"]
    for lb, members in ((best, [m for m in models if not m.endswith("-mlp")]), (f"{best}-mlp", [m for m in models if m.endswith("-mlp")])):
        B[lb] = P5.mean_boots([B[m] for m in members])
        out["methods"][f"{lb} (seed mean)"] = {f"{g}:{m}": float(B[lb][(g, m)][0]) for g, m in HEAD}
        for g, m in HEAD:
            out["methods"][f"{lb} (seed mean)"][f"{g}:{m}:ci"] = ci(B[lb][(g, m)][1:])
    comps = {}
    for g, m in HEAD:
        for b in P4.METHOD_NAMES:
            if m.startswith("far"):   # alarm rates: lower is better
                comps[f"{g}|{m}|{b} - GNN"] = verdict(B[b][(g, m)] - B[best][(g, m)])
            else:
                comps[f"{g}|{m}|GNN - {b}"] = verdict(B[best][(g, m)] - B[b][(g, m)])
        if g != "all":
            comps[f"{g}|{m}|GNN - MLP"] = verdict(B[best][(g, m)] - B[f"{best}-mlp"][(g, m)])
    out["comparisons"] = comps
    return out, B


# ------------------------------------------------------------------ maintenance-blanking hypothesis

def blanking_check(recs, wins, models, thr, n_boot, seed):
    """Alarm rate (window max >= the method's frozen threshold) in maintenance-only and fault-free windows, split by
    whether some link shows the down-link signature in the window. Rates are pooled over windows; CIs resample
    scenarios with the shared bootstrap weights."""
    S = len(recs)
    W = boot_weights(S, n_boot, seed)
    out = {}
    for name in list(P4.METHOD_NAMES) + models:
        cells = {}
        for cls_name, cls in (("maintenance_only", 1), ("fault_free", 0)):
            for part, sel in (("blanked", True), ("not_blanked", False)):
                n = np.array([np.sum((w["cls"] == cls) & (w["any_blanked"] == sel)) for w in wins], float)
                a = np.array([np.sum((w["cls"] == cls) & (w["any_blanked"] == sel) & (r[name]["windows"]["max"] >= thr[name]))
                              for r, w in zip(recs, wins)], float)
                tb = np.array([np.sum((w["cls"] == cls) & (w["any_blanked"] == sel) & (r[name]["windows"]["max"] >= thr[name])
                                      & w["top_blanked"][name]) for r, w in zip(recs, wins)], float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = (W @ a) / (W @ n)
                cells[(cls_name, part)] = v
                out.setdefault(name, {})[f"{cls_name}|{part}"] = {
                    "rate": None if not n.sum() else round(float(v[0]), 4), "ci": ci(v[1:]), "n_windows": int(n.sum()),
                    "n_alarms": int(a.sum()), "alarms_whose_top_link_is_blanked": int(tb.sum())}
            d = cells[(cls_name, "blanked")] - cells[(cls_name, "not_blanked")]
            out[name][f"{cls_name}|blanked - not_blanked"] = verdict(d)
        for act in ("down", "drain", "noop"):
            n = np.array([sum(c == 1 and act in ac for c, ac in zip(w["cls"], w["actions"])) for w in wins], float)
            a = np.array([sum(c == 1 and act in ac and mx >= thr[name] for c, ac, mx in zip(w["cls"], w["actions"], r[name]["windows"]["max"]))
                          for r, w in zip(recs, wins)], float)
            out[name][f"maintenance_only|action={act}"] = {"rate": round(float(a.sum() / n.sum()), 4) if n.sum() else None,
                                                            "n_windows": int(n.sum()), "n_alarms": int(a.sum())}
        m = out[name]
        tot = m["maintenance_only|blanked"]["n_alarms"] + m["maintenance_only|not_blanked"]["n_alarms"]
        m["share_of_maintenance_alarms_in_blanked_windows"] = round(m["maintenance_only|blanked"]["n_alarms"] / tot, 4) if tot else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke-out", default=None, help="crash test: 6 val scenarios per variant, outputs to this scratch folder")
    args = ap.parse_args()
    global FDIR
    out_dir = ROOT / "results"
    if args.smoke_out:
        out_dir = Path(args.smoke_out)
        FDIR = out_dir / "frozen"
        args.n_boot = 20
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg_s = freeze_cfg(cfg)
    root = ROOT / cfg["dataset"]["root"]
    seed = cfg["seeds"]["global"]
    t0 = time.time()
    val_res = json.loads((ROOT / "results" / "phase5_gnn_val.json").read_text())
    best = val_res["best"]
    models = frozen_models(best)
    states = {k: (MDIR / f"{k}.pt").read_bytes() for k in models}

    methods_b, selection_b = P4.fit_and_select(cfg, root, args.workers, cfg_s)
    assert selection_b["chosen"] == val_res["baseline_pop_aggregation"], "baseline PoP aggregation differs from Phase 5"
    FDIR.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        (FDIR / f"baselines_{v}.pkl").write_bytes(pickle.dumps(methods_b[v], protocol=4))

    val = {v: json.loads(G(root / "truth" / "val" / v / "manifest.json").read_text()) for v in VARIANTS}
    pick = sorted(np.random.default_rng(cfg["seeds"]["training"] + 5).permutation(len(val["with_status"]))[:P5.N_SELECT])
    select_ids = {val["with_status"][i]["scenario_id"] for i in pick}
    assert select_ids == {val["status_blind"][i]["scenario_id"] for i in pick}
    if args.smoke_out:
        val = {v: [e for e in val[v] if e["scenario_id"] in select_ids][:3] + [e for e in val[v] if e["scenario_id"] not in select_ids][:3]
               for v in VARIANTS}

    calib = {"note": "validation calibration, frozen for test: bins (edges) and alarm thresholds per method and variant",
             "split": "val", "baseline_pop_aggregation": selection_b["chosen"], "models": models, "variants": {}}
    heldout, blanking, repro = {}, {}, {}
    for v in VARIANTS:
        ctxb = {"root": root, "cfg_s": cfg_s, "methods": methods_b, "split": "val"}
        edges, thr = P4.calibrate(root, v, val[v], args.workers, ctxb)
        ctx = {"root": root, "cfg_s": cfg_s, "baselines": methods_b, "edges": edges, "states": states, "threads": 1}
        with Pool(args.workers, initializer=_init, initargs=(ctx,)) as pool:
            res = pool.map(_evaluate_one, [(v, e) for e in val[v]], chunksize=2)
        res.sort(key=lambda r: r[0])
        by_id = {e["scenario_id"]: e for e in val[v]}
        for k in models:
            edges[k] = P5.GNN_EDGES
            thr[k] = threshold_for(np.concatenate([r[1][k]["windows"]["max"][r[1][k]["windows"]["cls"] == 0] for r in res]))
        # R1: the frozen calibration must be the one the committed validation results used
        repro[v] = {k: {"threshold": thr[k], "committed": val_res["variants"][v]["thresholds"][k],
                        "equal": thr[k] == val_res["variants"][v]["thresholds"][k]} for k in list(P4.METHOD_NAMES) + models}
        calib["variants"][v] = {k: {"threshold": thr[k], "edges": [float(x) for x in edges[k]]} for k in list(P4.METHOD_NAMES) + models}
        # R2
        meta = [P5.scen_meta(cfg, by_id[sid]) for sid, _, _ in res]
        recs = [r[1] for r in res]
        keep = [i for i, (sid, _, _) in enumerate(res) if sid not in select_ids]
        heldout[v] = {}
        heldout[v]["all"], _ = headline(recs, meta, models, edges, thr, args.n_boot, seed, best)
        heldout[v]["heldout"], _ = headline([recs[i] for i in keep], [meta[i] for i in keep], models, edges, thr, args.n_boot, seed, best)
        for k in models:   # the all-468 numbers must equal the committed ones
            for g, m in (("hard", "top3@q90"), ("easy", "top1@q90")):
                c = val_res["variants"][v]["methods"][k]["groups"][g]["localisation"][m.split("@")[1]][m.split("@")[0]]["value"]
                repro[v][k][f"{g}:{m}"] = {"now": round(heldout[v]["all"]["methods"][k][f"{g}:{m}"], 4), "committed": c}
        # maintenance blanking
        blanking[v] = blanking_check(recs, [r[2] for r in res], models, thr, args.n_boot, seed)
        print(f"[{v}] done ({time.time() - t0:.0f}s)", flush=True)

    ok = all(x["equal"] and all(y["now"] == y["committed"] for kk, y in x.items() if ":" in kk)
             for v in VARIANTS for x in repro[v].values())
    if not ok and not args.smoke_out:
        print(json.dumps(repro, indent=1))
        sys.exit("the rescored validation calibration differs from results/phase5_gnn_val.json; nothing frozen")
    (FDIR / "calibration.json").write_text(json.dumps(calib, indent=1) + "\n")
    files = {f"data/models/phase5/{k}.pt": sha256(MDIR / f"{k}.pt") for k in models}
    files.update({f"data/models/phase5/frozen/{p.name}": sha256(p) for p in sorted(FDIR.iterdir())})
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = {"note": "sha256 of every file scripts/phase5_test.py loads; it refuses to run if any differs",
              "best": best, "models": models, "baselines": list(P4.METHOD_NAMES), "variants": list(VARIANTS),
              "selection_subset_size": len(select_ids), "files": files, "validation_reproduced": repro}
    enc = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    (out_dir / "phase5_frozen.json").write_text(json.dumps(frozen, indent=1, default=enc) + "\n")
    (out_dir / "phase5_heldout_val.json").write_text(json.dumps(
        {"note": "R2: validation headline on all 468 scenarios and on the 408 outside the 60-scenario selection subset "
                 "(the held-out version is the headline). Paired scenario bootstrap within each set.",
         "best": best, "selection_subset": sorted(select_ids), "variants": heldout}, indent=1, default=enc) + "\n")
    (out_dir / "phase5_maintenance_blanking_val.json").write_text(json.dumps(
        {"note": "Alarm rates at each method's frozen 5% fault-free threshold, in maintenance-only and fault-free 1-hour "
                 f"windows, split by whether some link shows the down-link signature on >= {SIG_STEPS} steps of the window "
                 "(with_status: util 0 and latency missing; status_blind: every metric missing). Validation, all 468 scenarios.",
         "hypothesis": "the GNN's maintenance-only alarm rate is higher in windows with a blanked link than without",
         "variants": blanking}, indent=1, default=enc) + "\n")
    print(f"froze {len(files)} files ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
