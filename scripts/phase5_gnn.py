"""Phase 5: line-graph GNN (GraphSAGE, GAT, GINE) with a temporal encoder, ablations, and evaluation with the shared
harness on VALIDATION only (budget and expectations pre-registered in docs/decision_log.md before any training).

    python scripts/phase5_gnn.py --workers 6      # train (train split), select (val subset), evaluate (val)
    python scripts/phase5_tables.py

Nothing here reads the test or dev splits.
"""
import argparse
import io
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase4_baselines as P4  # noqa: E402

from nettwin.baselines import Method, b2_features, index_for, window_score  # noqa: E402
from nettwin.evaluation import (EASY, HARD, WARMUP, WINDOW_AGGS, aggregate, ap_from_hists, labels_matrix,  # noqa: E402
                                related_links, scenario_record, threshold_for, tie_credit, to_bins, verdict)
from nettwin.gnn.data import graph_for, training_samples  # noqa: E402
from nettwin.gnn.models import NetTwinGNN, predict  # noqa: E402
from nettwin.gnn.train import shuffle_perm, train  # noqa: E402
from nettwin.observed import freeze_cfg, inventory, load_observed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("with_status", "status_blind")
# ---- pre-registered budget (decision log, Phase 5 pre-registration) ----
K_PER_SCENARIO = 12          # training steps per train scenario and variant
EPOCHS = 6
BATCH = 32
LR = 2e-3
TRAIN_THREADS = 4
CONVS = ("sage", "gat", "gine")
HIDDEN = (32, 64)
SEEDS = (0, 1, 2)
N_SELECT = 60                # fixed validation subset for selection (both variants)
ABLATIONS = ("mlp", "no_path_static", "shuffled_edges")
GNN_EDGES = 1 / (1 + np.exp(-np.linspace(-16, 16, 4001)))   # fixed, label-free bins for probabilities
CTX = {}


def _init(ctx):
    CTX.update(ctx)
    torch.set_num_threads(ctx.get("threads", 1))


# ------------------------------------------------------------------ data

def _samples_one(args):
    variant, entry = args
    R = CTX["root"]
    obs = load_observed(R / "observable" / "train" / variant / entry["scenario_id"])
    kg = inventory(obs.topology, CTX["cfg_s"])
    ix = index_for(obs, kg)
    lab, _ = labels_matrix(pd.read_parquet(R / "truth" / "train" / variant / entry["files"]["labels"]), ix.links)
    rng = np.random.default_rng([int(entry["scenario_id"][2:]), VARIANTS.index(variant), 5])
    x, d, r = training_samples(obs, entry, lab, CTX["cfg_s"], K_PER_SCENARIO, rng)
    return [(obs.topology, x[i], d[i], r[i]) for i in range(len(x))]


def _select_data_one(args):
    variant, entry = args
    obs = load_observed(CTX["root"] / "observable" / CTX.get("split", "val") / variant / entry["scenario_id"])
    return {"topology": obs.topology, "feat": b2_features(obs).astype(np.float16), "entry": entry, "variant": variant}


# ------------------------------------------------------------------ models

def run_key(conv, hidden, seed, ablation=None):
    return f"{conv}-h{hidden}-s{seed}" + (f"-{ablation}" if ablation else "")


def cfg_label(spec):
    conv, hidden, _, ablation = spec
    return f"{conv}-h{hidden}" + (f"-{ablation}" if ablation else "")


def _train_one(spec):
    conv, hidden, seed, ablation = spec
    torch.set_num_threads(TRAIN_THREADS)
    model, info = train(CTX["samples"], CTX["graphs"], "mlp" if ablation == "mlp" else conv, hidden, seed, EPOCHS, BATCH, LR,
                        path_static=ablation != "no_path_static", shuffle=ablation == "shuffled_edges", log=lambda *a: None)
    sel = None
    if ablation is None and seed == 0:
        sel = selection_score(model, shuffle_seed=None)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return run_key(conv, hidden, seed, ablation), {"spec": spec, "state": buf.getvalue(), "info": info, "selection": sel}


def selection_score(model, shuffle_seed):
    """Mean expected top-3 (q90 window) of observable faults on the fixed validation subset, hard and easy averaged."""
    credit = {"hard": [], "easy": []}
    for d in CTX["select"]:
        g = graph_for(d["topology"], CTX["cfg_s"])
        _, elem = predict(model, d["feat"].astype(np.float32), g)
        e = d["entry"]
        for f in e["faults"]:
            if not f["impact"]["observable"]:
                continue
            true = g.elements.index(f["root_cause"])
            drop = {g.elements.index(o["root_cause"]) for o in e["faults"]
                    if o["root_cause"] != f["root_cause"] and o["t_start"] < f["t_end"] and f["t_start"] < o["t_end"]}
            keep = np.ones(len(g.elements), bool)
            keep[list(drop | {true})] = False
            c = tie_credit(window_score(elem[f["t_start"]:f["t_end"]], "q90"), true, keep)
            credit["hard" if f["type"] in HARD else "easy"].append(c["top3"])
    out = {k: round(float(np.mean(v)), 4) for k, v in credit.items()}
    out["score"] = round((out["hard"] + out["easy"]) / 2, 4)
    return out


def load_model(rec):
    conv, hidden, seed, ablation = rec["spec"]
    m = NetTwinGNN(conv="mlp" if ablation == "mlp" else conv, hidden=hidden, path_static=ablation != "no_path_static")
    m.load_state_dict(torch.load(io.BytesIO(rec["state"])))
    m.eval()
    return m


class GNNMethod(Method):
    """Wraps a trained GNN for the shared harness: detection = per-link probability, localisation = element
    probabilities aggregated over the incident window like every other method."""
    def __init__(self, name, model, cfg_s, shuffle_seed=None):
        self.name, self.model, self.cfg_s, self.shuffle_seed = name, model, cfg_s, shuffle_seed

    def score(self, obs, kg, ix):
        g = graph_for(obs.topology, self.cfg_s)
        perm = shuffle_perm(obs.topology, g.n_links, self.shuffle_seed) if self.shuffle_seed is not None else None
        det, elem = predict(self.model, b2_features(obs).astype(np.float32), g, perm)
        return det, np.zeros((det.shape[0], g.n_pops), np.float32), elem

    def localise(self, state, t0, t1, agg="q90"):
        return window_score(state[2][t0:t1], agg)


# ------------------------------------------------------------------ evaluation

def _eval_init(ctx):
    _init(ctx)
    CTX["gnn"] = {}
    for key, rec in ctx["models"].items():
        conv, hidden, seed, ablation = rec["spec"]
        CTX["gnn"][key] = GNNMethod(key, load_model(rec), ctx["cfg_s"], seed if ablation == "shuffled_edges" else None)


def _evaluate_one(args):
    variant, entry, check = args
    R, split = CTX["root"], CTX.get("split", "val")
    obs = load_observed(R / "observable" / split / variant / entry["scenario_id"])
    kg = inventory(obs.topology, CTX["cfg_s"])
    ix = index_for(obs, kg)
    lab, pres = labels_matrix(pd.read_parquet(R / "truth" / split / variant / entry["files"]["labels"]), ix.links)
    related = [related_links(kg, ix, f["root_cause"]) for f in entry["faults"]]
    recs, exact = {}, {}
    methods = {**{k: CTX["baselines"][variant][k] for k in P4.METHOD_NAMES}, **CTX["gnn"]}
    for name, m in methods.items():
        state = m.score(obs, kg, ix)
        edges = CTX["edges"][variant][name] if name in P4.METHOD_NAMES else GNN_EDGES
        recs[name] = scenario_record(m, state, ix, entry, lab, pres, edges, related)
        if check and name not in P4.METHOD_NAMES:
            sc, lb = state[0][WARMUP:], lab[WARMUP:]
            exact[name] = (sc[lb != 1], lb[lb != 1] == 2)
    return entry["scenario_id"], recs, exact


def scen_meta(cfg, entry):
    m = P4.scen_meta(cfg, entry)
    m["topology"] = entry["topology"]
    return m


def network_groups(cfg):
    g = {}
    for t in cfg["topologies"]["validation_real"]:
        name = t.split("/")[-1]
        for fam, types in (("hard", HARD), ("easy", EASY)):
            g[f"{fam}|network={name}"] = (lambda f, types=types: f["observable"] and f["type"] in types, lambda s, t=t: s["topology"] == t)
    return g


def mean_boots(boot_list):
    keys = boot_list[0].keys()
    return {k: np.mean([b[k] for b in boot_list], axis=0) for k in keys}


# ------------------------------------------------------------------ pre-registered expectations X1 to X11

def expectations(B, best, abl, archs):
    """B[(variant, method)][(group, metric)] -> bootstrap array; GNN methods are seed means."""
    def get(m, grp, metric, var):
        return B[(var, m)][(grp, metric)]

    def comp(label, diff):
        return {"component": label, **verdict(diff)}

    def overall(parts):
        vs = [p["verdict"] for p in parts]
        return "met" if all(v == "supported" for v in vs) else "not met" if "contradicted" in vs else "inconclusive"

    E = {}

    def expect(eid, text, fn, aggs=WINDOW_AGGS, predicted="supported"):
        per = {}
        for agg in aggs:
            parts = fn(agg)
            if predicted == "inconclusive":
                vs = [p["verdict"] for p in parts]
                v = "met" if all(x == "inconclusive" for x in vs) else "not met"
            else:
                v = overall(parts)
            per[agg] = {"verdict": v, "components": parts}
        E[eid] = {"expectation": text, "by_window_aggregation": per}

    T3, T1 = "top3@{}", "top1@{}"
    G = best
    expect("X1", "GNN beats B2 on hard-type top-3, both variants",
           lambda a: [comp(f"{v}: GNN - B2", get(G, "hard", T3.format(a), v) - get("B2", "hard", T3.format(a), v)) for v in VARIANTS])
    expect("X2", "GNN beats B2 on hard-type root PR-AUC, both variants",
           lambda a: [comp(f"{v}: GNN - B2", get(G, "hard", "pr_auc_root", v) - get("B2", "hard", "pr_auc_root", v)) for v in VARIANTS], aggs=("any",))
    expect("X3", "Easy types: with status events GNN top-3 >= B3 - 0.05; status-blind GNN top-3 > B3",
           lambda a: [comp("with_status: GNN - B3 + 0.05", get(G, "easy", T3.format(a), "with_status") - get("B3", "easy", T3.format(a), "with_status") + 0.05),
                      comp("status_blind: GNN - B3", get(G, "easy", T3.format(a), "status_blind") - get("B3", "easy", T3.format(a), "status_blind"))])
    expect("X4", "B3 beats the GNN on link_flapping top-3 with status events",
           lambda a: [comp("B3 - GNN", get("B3", "type=link_flapping", T3.format(a), "with_status") - get(G, "type=link_flapping", T3.format(a), "with_status"))])
    expect("X5", "Synthetic-to-real transfer: real minus generated top-3 > -0.10 for hard and easy, both variants",
           lambda a: [comp(f"{v}: {fam} real - generated + 0.10", get(G, f"{fam}|source=real", T3.format(a), v) - get(G, f"{fam}|source=generated", T3.format(a), v) + 0.10)
                      for v in VARIANTS for fam in ("hard", "easy")])
    expect("X6", "The graph helps: GNN beats the MLP ablation on hard top-3 and easy top-1, both variants",
           lambda a: [comp(f"{v}: GNN - MLP hard top-3", get(G, "hard", T3.format(a), v) - get(abl["mlp"], "hard", T3.format(a), v)) for v in VARIANTS]
           + [comp(f"{v}: GNN - MLP easy top-1", get(G, "easy", T1.format(a), v) - get(abl["mlp"], "easy", T1.format(a), v)) for v in VARIANTS])
    expect("X7", "Shuffled edges hurt: GNN beats the shuffled-edges ablation on hard and easy top-3, both variants",
           lambda a: [comp(f"{v}: GNN - shuffled {fam}", get(G, fam, T3.format(a), v) - get(abl["shuffled_edges"], fam, T3.format(a), v))
                      for v in VARIANTS for fam in ("hard", "easy")])
    expect("X8", "GNN has the best symptomatic PR-AUC, hard and easy, both variants",
           lambda a: [comp(f"{v}: GNN - {m} {fam}", get(G, fam, "pr_auc_symptomatic", v) - get(m, fam, "pr_auc_symptomatic", v))
                      for v in VARIANTS for fam in ("hard", "easy") for m in P4.METHOD_NAMES], aggs=("any",))
    expect("X9", "GNN alarms less than B2 in maintenance-only hours (at 5% fault-free alarms), both variants",
           lambda a: [comp(f"{v}: B2 - GNN", get("B2", "all", "far_maintenance_only", v) - get(G, "all", "far_maintenance_only", v)) for v in VARIANTS], aggs=("any",))
    expect("X10", "No architecture is clearly better: SAGE, GAT and GINE pairwise hard top-3 differences are all inconclusive",
           lambda a: [comp(f"{v}: {x} - {y}", get(archs[x], "hard", T3.format(a), v) - get(archs[y], "hard", T3.format(a), v))
                      for v in VARIANTS for i, x in enumerate(CONVS) for y in CONVS[i + 1:]], predicted="inconclusive")
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3, help="evaluation workers (each holds every model; 8 GB machines: 3)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="crash test on TRAIN scenarios only, tiny settings, writes to the scratch folder given by --smoke-out")
    ap.add_argument("--smoke-out", default=None)
    args = ap.parse_args()
    global K_PER_SCENARIO, EPOCHS, N_SELECT
    split = "val"
    if args.smoke:
        K_PER_SCENARIO, EPOCHS, N_SELECT, args.n_boot, split = 2, 1, 4, 20, "train"
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg_s = freeze_cfg(cfg)
    root = ROOT / cfg["dataset"]["root"]
    t0 = time.time()
    timings = {}

    # training samples (train only)
    jobs = [(v, e) for v in VARIANTS for e in json.loads((root / "truth" / "train" / v / "manifest.json").read_text())]
    with Pool(args.workers, initializer=_init, initargs=({"root": root, "cfg_s": cfg_s},)) as pool:
        samples = [s for part in pool.map(_samples_one, jobs, chunksize=8) for s in part]
    graphs = {t: graph_for(t, cfg_s) for t in sorted({s[0] for s in samples})}
    timings["build_samples_s"] = round(time.time() - t0, 1)
    print(f"{len(samples)} training samples ({timings['build_samples_s']}s)", flush=True)

    # fixed validation subset for selection
    val = {v: json.loads((root / "truth" / "val" / v / "manifest.json").read_text()) for v in VARIANTS} if not args.smoke else \
        {v: json.loads((root / "truth" / "train" / v / "manifest.json").read_text())[-8:] for v in VARIANTS}
    pick = sorted(np.random.default_rng(cfg["seeds"]["training"] + 5).permutation(len(val["with_status"]))[:N_SELECT])
    with Pool(args.workers, initializer=_init, initargs=({"root": root, "cfg_s": cfg_s, "split": split},)) as pool:
        select = pool.map(_select_data_one, [(v, val[v][i]) for v in VARIANTS for i in pick])
    for d in select:
        graph_for(d["topology"], cfg_s)

    # phase A: 6 configs, seed 0; phase B: seeds 1 and 2 of each architecture at its chosen hidden size, and the ablations
    # runs one at a time in this process, so the samples are held once (8 GB machine)
    _init({"root": root, "cfg_s": cfg_s, "samples": samples, "graphs": graphs, "select": select, "threads": TRAIN_THREADS})
    runs = {}
    for spec in [(c, h, 0, None) for c in CONVS for h in HIDDEN]:
        key, rec = _train_one(spec)
        runs[key] = rec
        print(f"trained {key}: {rec['info']['train_seconds']}s, selection {rec['selection']}", flush=True)
    chosen = {c: max(HIDDEN, key=lambda h: (runs[run_key(c, h, 0)]["selection"]["score"], -h)) for c in CONVS}
    best_conv = max(CONVS, key=lambda c: (runs[run_key(c, chosen[c], 0)]["selection"]["score"], -CONVS.index(c)))
    best_h = chosen[best_conv]
    print(f"chosen hidden {chosen}; best {best_conv} h{best_h}", flush=True)
    specs = [(c, chosen[c], s, None) for c in CONVS for s in SEEDS[1:]] + [(best_conv, best_h, s, a) for a in ABLATIONS for s in SEEDS]
    for spec in specs:
        key, rec = _train_one(spec)
        runs[key] = rec
        print(f"trained {key}: {rec['info']['train_seconds']}s", flush=True)
    timings["train_runs_s"] = {k: r["info"]["train_seconds"] for k, r in runs.items()}
    del samples, select
    CTX.clear()
    mdir = ROOT / "data" / "models" / ("phase5_smoke" if args.smoke else "phase5")
    mdir.mkdir(parents=True, exist_ok=True)
    for k, r in runs.items():
        (mdir / f"{k}.pt").write_bytes(r["state"])

    # evaluation: the 9 architecture models and 9 ablation models, plus the baselines, in one pass (paired bootstrap)
    final = {k: r for k, r in runs.items() if r["spec"][2] in SEEDS and (r["spec"][3] is not None or r["spec"][1] == chosen[r["spec"][0]])}
    methods_b, selection_b = P4.fit_and_select(cfg, root, args.workers, cfg_s)
    B, out = {}, {"split": "val", "budget": {"k_per_scenario": K_PER_SCENARIO, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "convs": CONVS,
                                              "hidden": HIDDEN, "seeds": SEEDS, "n_select": N_SELECT, "ablations": ABLATIONS},
                  "selection": {k: r["selection"] for k, r in runs.items() if r["selection"]}, "chosen_hidden": chosen,
                  "best": f"{best_conv}-h{best_h}", "baseline_pop_aggregation": selection_b["chosen"],
                  "n_params": {k: r["info"]["n_params"] for k, r in final.items()},
                  "train_history": {k: r["info"]["history"] for k, r in runs.items()}, "variants": {}}
    groups = network_groups(cfg)
    label = {k: cfg_label(r["spec"]) for k, r in final.items()}
    labels = sorted(set(label.values()))
    te = time.time()
    for variant in VARIANTS:
        entries = val[variant]
        ctxb = {"root": root, "cfg_s": cfg_s, "methods": methods_b, "split": split}
        edges, thr = P4.calibrate(root, variant, entries, args.workers, ctxb)
        ctxe = {"root": root, "cfg_s": cfg_s, "baselines": methods_b, "edges": {variant: edges}, "models": final, "threads": 1, "split": split}
        with Pool(args.workers, initializer=_eval_init, initargs=(ctxe,)) as pool:
            res = pool.map(_evaluate_one, [(variant, e, i < P4.N_EXACT_CHECK) for i, e in enumerate(entries)], chunksize=2)
        res.sort(key=lambda r: r[0])
        by_id = {e["scenario_id"]: e for e in entries}
        meta = [scen_meta(cfg, by_id[sid]) for sid, _, _ in res]
        V = {"methods": {}, "thresholds": {}, "binned_vs_exact_ap": {}}
        for name in list(P4.METHOD_NAMES) + sorted(final):
            recs = [r[1][name] for r in res]
            if name in P4.METHOD_NAMES:
                e, th = edges[name], thr[name]
            else:
                e = GNN_EDGES
                th = threshold_for(np.concatenate([r["windows"]["max"][r["windows"]["cls"] == 0] for r in recs]))
            V["thresholds"][name] = th
            V["methods"][name], B[(variant, name)] = aggregate(recs, meta, len(e) + 1, th, n_boot=args.n_boot,
                                                                seed=cfg["seeds"]["global"], extra_groups=groups)
            if name not in P4.METHOD_NAMES:
                s = np.concatenate([r[2][name][0] for r in res if r[2]])
                y = np.concatenate([r[2][name][1] for r in res if r[2]])
                b, nb = to_bins(s, e), len(e) + 1
                V["binned_vs_exact_ap"][name] = {"exact_ap_root": round(float(average_precision_score(y, s)), 4),
                                                 "binned_ap_root": round(float(ap_from_hists(np.bincount(b[y], minlength=nb), np.bincount(b[~y], minlength=nb))), 4)}
        # seed means for every GNN configuration (the bootstrap draws are shared, so seed means stay paired)
        for lb in labels:
            B[(variant, f"{lb}-mean")] = mean_boots([B[(variant, k)] for k in final if label[k] == lb])
        out["variants"][variant] = V
        print(f"[{variant}] evaluated ({time.time() - te:.0f}s)", flush=True)
    timings["evaluate_s"] = round(time.time() - te, 1)

    # seed summaries and verdicts
    summary = {}
    for v in VARIANTS:
        for lb in labels:
            members = sorted(k for k in final if label[k] == lb)
            for grp, metric in (("hard", "top3@q90"), ("easy", "top3@q90"), ("hard", "top1@q90"), ("easy", "top1@q90"),
                                ("hard", "pr_auc_root"), ("easy", "pr_auc_root"), ("hard", "pr_auc_symptomatic"), ("easy", "pr_auc_symptomatic")):
                vals = [float(B[(v, k)][(grp, metric)][0]) for k in members]
                summary.setdefault(v, {}).setdefault(lb, {})[f"{grp}:{metric}"] = {
                    "per_seed": [round(x, 4) for x in vals], "mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=1)), 4),
                    "seed_mean_ci": [round(float(x), 4) for x in np.percentile(B[(v, f"{lb}-mean")][(grp, metric)][1:], [2.5, 97.5])]}
    out["seed_summary"] = summary
    best = f"{best_conv}-h{best_h}-mean"
    abl = {a: f"{best_conv}-h{best_h}-{a}-mean" for a in ABLATIONS}
    archs = {c: f"{c}-h{chosen[c]}-mean" for c in CONVS}
    comps = {}
    for v in VARIANTS:
        for g in ("hard", "easy") + tuple(f"type={t}" for t in EASY + HARD):
            for agg in WINDOW_AGGS:
                for m in P4.METHOD_NAMES:
                    for metric in (f"top1@{agg}", f"top3@{agg}"):
                        comps[f"{v}|{g}|{metric}|GNN - {m}"] = verdict(B[(v, best)][(g, metric)] - B[(v, m)][(g, metric)])
            for m in P4.METHOD_NAMES:
                for metric in ("pr_auc_root", "pr_auc_symptomatic"):
                    comps[f"{v}|{g}|{metric}|GNN - {m}"] = verdict(B[(v, best)][(g, metric)] - B[(v, m)][(g, metric)])
    E = expectations(B, best, abl, archs)
    out["comparisons_vs_baselines"] = comps
    enc = lambda o: o.item() if hasattr(o, "item") else (list(o) if isinstance(o, tuple) else str(o))  # noqa: E731
    rdir = Path(args.smoke_out) if args.smoke else ROOT / "results"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "phase5_gnn_val.json").write_text(json.dumps(out, indent=1, default=enc) + "\n")
    (rdir / "phase5_expectations.json").write_text(json.dumps(E, indent=1, default=enc) + "\n")
    timings["total_s"] = round(time.time() - t0, 1)
    (rdir / "phase5_timings.json").write_text(json.dumps(timings, indent=1) + "\n")
    print(f"wrote results ({timings['total_s']}s)")


if __name__ == "__main__":
    if any("test" in a or "/dev" in a for a in sys.argv[1:]):
        sys.exit("Phase 5 runs on validation only")
    main()
