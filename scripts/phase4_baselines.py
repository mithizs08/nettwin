"""Phase 4: threshold alerts, baselines B1 to B3 (and the disclosed variant B3+) and the shared evaluation harness,
on VALIDATION only. Revised after docs/audit_phase4.md (fixes logged by ID in docs/decision_log.md).

- Train: fit B2 (unsupervised, 100 scenarios) and choose the PoP aggregation for B1 and B2 (200 other scenarios).
- Validation: each method's bins and alarm threshold come from its own validation scores (audit Q10); then every
  metric, on v1 and on the pre-tuning sensitivity copy (same calibration, paired by scenario), plus the claim checks
  with paired bootstrap CIs (audit Q3, Q8, Q12).
Nothing here reads the test or dev splits.

    python scripts/phase4_baselines.py --workers 6
    python scripts/phase4_tables.py
"""
import argparse
import json
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
from nettwin.baselines import METHODS, POP_AGGS, WINDOW_QUANTILE, index_for, pop_scores
from nettwin.evaluation import (EASY, FAR_TARGET, HARD, LAG, WARMUP, WINDOW_AGGS, aggregate, ap_from_hists, fault_key,
                                fault_table, labels_matrix, make_edges, related_links, scenario_record,
                                threshold_for, tie_credit, to_bins, verdict, window_classes)
from nettwin.observed import freeze_cfg, inventory, load_observed

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("with_status", "status_blind")
METHOD_NAMES = ("B1", "B2", "B3", "B3+")
N_FIT, N_SEL = 100, 200
SAMPLE_FRAC = 0.02
N_EXACT_CHECK = 40
CTX = {}


def _init(ctx):
    CTX.update(ctx)


def _load(root, split, variant, entry):
    obs = load_observed(root / "observable" / split / variant / entry["scenario_id"])
    kg = inventory(obs.topology, CTX["cfg_s"])
    ix = index_for(obs, kg)
    lab, pres = labels_matrix(pd.read_parquet(root / "truth" / split / variant / entry["files"]["labels"]), ix.links)
    return obs, kg, ix, lab, pres


def _select_one(args):
    """Train: localisation credit (q90 window) of each observable fault for B1 / B2 under every PoP aggregation."""
    variant, entry = args
    obs, kg, ix, lab, pres = _load(CTX["root"], "train", variant, entry)
    out = []
    for name in ("B1", "B2"):
        m = CTX["methods"][variant][name]
        link = m.score(obs, kg, ix)[0]
        for agg in POP_AGGS:
            state = (link, pop_scores(link, ix, agg))
            for f in fault_table(entry):
                if not f["observable"]:
                    continue
                true = ix.elements.index(f["root"])
                drop = {ix.elements.index(g["root_cause"]) for g in entry["faults"]
                        if g["root_cause"] != f["root"] and g["t_start"] < f["t_end"] and f["t_start"] < g["t_end"]}
                keep = np.ones(len(ix.elements), bool)
                keep[list(drop | {true})] = False
                c = tie_credit(m.localise(state, f["t_start"], f["t_end"], "q90"), true, keep)
                out.append((name, agg, f["type"] in HARD, c["top1"], c["top3"]))
    return out


def _calibrate_one(args):
    """Validation pass 1 (no labels used): a random sample of each method's cell scores and its fault-free window maxima."""
    variant, entry = args
    obs, kg, ix, _, _ = _load(CTX["root"], "val", variant, entry)
    rng = np.random.default_rng(int(entry["scenario_id"][2:]))
    out = {}
    for name in METHOD_NAMES:
        link = CTX["methods"][variant][name].score(obs, kg, ix)[0][WARMUP:]
        _, _, cls = window_classes(entry, obs.n_steps)
        mx = np.array([link[s:s + 12].max() for s in range(0, link.shape[0], 12)])
        out[name] = {"sample": link.ravel()[rng.random(link.size) < SAMPLE_FRAC], "free_max": mx[cls == 0]}
    return out


def _evaluate_one(args):
    root, variant, entry, check = args
    obs, kg, ix, lab, pres = _load(root, "val", variant, entry)
    related = [related_links(kg, ix, f["root_cause"]) for f in entry["faults"]]
    recs, exact = {}, {}
    for name in METHOD_NAMES:
        m = CTX["methods"][variant][name]
        state = m.score(obs, kg, ix)
        recs[name] = scenario_record(m, state, ix, entry, lab, pres, CTX["edges"][variant][name], related)
        if check:
            sc, lb = state[0][WARMUP:], lab[WARMUP:]
            exact[name] = (sc[lb != 1], lb[lb != 1] == 2)
    alerts = generate_alerts(obs, kg)
    in_fault = np.zeros(obs.n_steps + 1, bool)
    for f in entry["faults"]:
        if f["impact"]["observable"]:
            in_fault[f["t_start"]:min(f["t_end"] + LAG, obs.n_steps)] = True
    a = alerts[alerts.t_start >= WARMUP]
    stats = {"count": Counter(a.rule), "in_fault": Counter(a.rule[in_fault[a.t_start.to_numpy()]])}
    return entry["scenario_id"], recs, stats, exact


def scen_meta(cfg, entry):
    real = set(cfg["topologies"]["validation_real"])
    return {"source": "real" if entry["topology"] in real else "generated",
            "concurrent": any(f["placement"] == "concurrent" for f in entry["faults"])}


def fit_and_select(cfg, root, workers, cfg_s):
    """Train only: B2 per variant, then the PoP aggregation for B1 and B2 (best mean top-3 over both variants, q90)."""
    methods, sel_entries = {}, {}
    for variant in VARIANTS:
        entries = json.loads((root / "truth" / "train" / variant / "manifest.json").read_text())
        pick = np.random.default_rng(cfg["seeds"]["training"]).permutation(len(entries))
        fit_e = [entries[i] for i in sorted(pick[:N_FIT])]
        sel_entries[variant] = [entries[i] for i in sorted(pick[N_FIT:N_FIT + N_SEL])]
        b2 = METHODS["B2"](seed=cfg["seeds"]["training"])
        b2.fit([load_observed(root / "observable" / "train" / variant / e["scenario_id"]) for e in fit_e])
        methods[variant] = {"B1": METHODS["B1"](), "B2": b2, "B3": METHODS["B3"](), "B3+": METHODS["B3+"]()}
    ctx = {"root": root, "cfg_s": cfg_s, "methods": methods}
    with Pool(workers, initializer=_init, initargs=(ctx,)) as pool:
        rows = [r for part in pool.map(_select_one, [(v, e) for v in VARIANTS for e in sel_entries[v]], chunksize=4) for r in part]
    df = pd.DataFrame(rows, columns=["method", "agg", "hard", "top1", "top3"])
    table = {m: {a: {"top1": round(float(d.top1.mean()), 4), "top3": round(float(d.top3.mean()), 4),
                     "top3_hard": round(float(d[d.hard].top3.mean()), 4), "top3_easy": round(float(d[~d.hard].top3.mean()), 4)}
                 for a, d in g.groupby("agg")} for m, g in df.groupby("method")}
    choice = {m: max(POP_AGGS, key=lambda a: (table[m][a]["top3"], table[m][a]["top1"], -POP_AGGS.index(a))) for m in table}
    for variant in VARIANTS:
        for m, a in choice.items():
            methods[variant][m].pop_agg = a
    selection = {"train_scenarios_per_variant": N_SEL, "fault_evaluations_both_variants": int(len(df) / (2 * len(POP_AGGS))), "window": "q90",
                 "rule": "highest mean top-3 over observable train faults, both variants (then top-1)", "table": table, "chosen": choice}
    return methods, selection


def calibrate(root, variant, entries, workers, ctx):
    """Bins and thresholds from each method's own validation scores (audit Q10)."""
    with Pool(workers, initializer=_init, initargs=(ctx,)) as pool:
        parts = pool.map(_calibrate_one, [(variant, e) for e in entries], chunksize=4)
    edges, thr = {}, {}
    for name in METHOD_NAMES:
        edges[name] = make_edges(np.concatenate([p[name]["sample"] for p in parts]))
        thr[name] = threshold_for(np.concatenate([p[name]["free_max"] for p in parts]))
    return edges, thr


def matched_groups(v1_entries, pre_entries):
    """Audit Q8: faults present in both builds (same scenario, type, root, start, end) and observable in both."""
    keys = {}
    for entries in (v1_entries, pre_entries):
        ks = set()
        for e in entries:
            for f in fault_table(e):
                if f["observable"]:
                    ks.add(fault_key(f))
        keys[id(entries)] = ks
    both = keys[id(v1_entries)] & keys[id(pre_entries)]
    return {f"{fam}|matched": (lambda f, types=types: f["observable"] and f["type"] in types and fault_key(f) in both, None)
            for fam, types in (("hard", HARD), ("easy", EASY))}, len(both)


def evaluate(root, variant, entries, workers, ctx, cfg, extra, n_boot):
    with Pool(workers, initializer=_init, initargs=(ctx,)) as pool:
        res = pool.map(_evaluate_one, [(root, variant, e, i < N_EXACT_CHECK) for i, e in enumerate(entries)], chunksize=4)
    res.sort(key=lambda r: r[0])
    by_id = {e["scenario_id"]: e for e in entries}
    meta = [scen_meta(cfg, by_id[sid]) for sid, _, _, _ in res]
    out, boots, check = {"methods": {}}, {}, {}
    for name in METHOD_NAMES:
        edges = ctx["edges"][variant][name]
        out["methods"][name], boots[name] = aggregate([r[1][name] for r in res], meta, len(edges) + 1, ctx["thr"][variant][name],
                                                      n_boot=n_boot, seed=cfg["seeds"]["global"], extra_groups=extra)
        s = np.concatenate([r[3][name][0] for r in res if r[3]])
        y = np.concatenate([r[3][name][1] for r in res if r[3]])
        b, nb = to_bins(s, edges), len(edges) + 1
        check[name] = {"exact_ap_root": round(float(average_precision_score(y, s)), 4),
                       "binned_ap_root": round(float(ap_from_hists(np.bincount(b[y], minlength=nb), np.bincount(b[~y], minlength=nb))), 4),
                       "scenarios": N_EXACT_CHECK, "n_bins": nb}
    cnt, inf = Counter(), Counter()
    for _, _, st, _ in res:
        cnt.update(st["count"])
        inf.update(st["in_fault"])
    out["alerts"] = {"per_scenario": {k: round(cnt[k] / len(res), 2) for k in RULES},
                     "share_starting_in_observable_fault_window": {k: (round(inf[k] / cnt[k], 4) if cnt[k] else None) for k in RULES}}
    out["binned_vs_exact_ap"] = check
    return out, boots


# ------------------------------------------------------------------ claims and expectations (audit Q3, Q12)

def claims_and_expectations(B):
    """B[(dataset, variant, method)][(group, metric)] -> bootstrap array (full-sample value first). Localisation
    metrics are checked under every window aggregation."""
    def get(m, grp, metric, var="with_status", ds="v1"):
        return B[(ds, var, m)][(grp, metric)]

    def comp(label, diff):
        return {"component": label, **verdict(diff)}

    def overall(parts):
        vs = [p["verdict"] for p in parts]
        return "met" if all(v == "supported" for v in vs) else "not met" if "contradicted" in vs else "inconclusive"

    claims = []

    def claim(cid, text, fn, aggs=WINDOW_AGGS):
        per = {}
        for agg in aggs:
            parts = fn(agg)
            per[agg] = {"overall": {"met": "supported", "not met": "contradicted"}.get(overall(parts), "inconclusive"), "components": parts}
        claims.append({"id": cid, "claim": text, "by_window_aggregation": per})

    T3, T1 = "top3@{}", "top1@{}"
    claim("C1", "B2 has the best hard-type top-3", lambda a: [comp(f"{v}: B2 - {m}", get("B2", "hard", T3.format(a), v) - get(m, "hard", T3.format(a), v))
                                                         for v in VARIANTS for m in ("B1", "B3", "B3+")])
    claim("C2", "B3 has the best easy-type top-3 (with status events)",
          lambda a: [comp(f"B3 - {m}", get("B3", "easy", T3.format(a)) - get(m, "easy", T3.format(a))) for m in ("B1", "B2")])
    claim("C3", "B3 has the best easy-type root PR-AUC (with status events)",
          lambda a: [comp(f"B3 - {m}", get("B3", "easy", "pr_auc_root") - get(m, "easy", "pr_auc_root")) for m in ("B1", "B2")], aggs=("any",))
    claim("C4", "B2 has the best hard-type root PR-AUC",
          lambda a: [comp(f"{v}: B2 - {m}", get("B2", "hard", "pr_auc_root", v) - get(m, "hard", "pr_auc_root", v)) for v in VARIANTS for m in ("B1", "B3")], aggs=("any",))
    claim("C5", "B3 localises hard faults worse than B1 and B2 (top-3)",
          lambda a: [comp(f"{v}: {m} - B3", get(m, "hard", T3.format(a), v) - get("B3", "hard", T3.format(a), v)) for v in VARIANTS for m in ("B1", "B2")])
    claim("C6", "B3 localises flapping better than B1 and B2 (top-3)",
          lambda a: [comp(f"{v}: B3 - {m}", get("B3", "type=link_flapping", T3.format(a), v) - get(m, "type=link_flapping", T3.format(a), v))
                     for v in VARIANTS for m in ("B1", "B2")])
    claim("C7", "Maintenance-only hours alarm more often than fault-free hours (every method)",
          lambda a: [comp(f"{v}: {m}", get(m, "all", "far_maintenance_only", v) - get(m, "all", "far_fault_free", v)) for v in VARIANTS for m in METHOD_NAMES], aggs=("any",))
    claim("C8", "Without status events link_down top-3 drops (every method)",
          lambda a: [comp(m, get(m, "type=link_down", T3.format(a)) - get(m, "type=link_down", T3.format(a), "status_blind")) for m in METHOD_NAMES])
    claim("C9", "B3+ ranks node_down roots first more often than B3 (top-1)",
          lambda a: [comp(v, get("B3+", "type=node_down", T1.format(a), v) - get("B3", "type=node_down", T1.format(a), v)) for v in VARIANTS], aggs=("q90",))
    claim("C10", "Hard faults on real validation networks differ from generated graphs (top-3, real - generated)",
          lambda a: [comp(f"{v}: {m}", get(m, "hard|source=real", T3.format(a), v) - get(m, "hard|source=generated", T3.format(a), v))
                     for v in VARIANTS for m in METHOD_NAMES])

    E = {}

    def expect(eid, fn, aggs=WINDOW_AGGS):
        per = {}
        for agg in aggs:
            parts = fn(agg)
            per[agg] = {"verdict": overall(parts), "components": parts}
        E[eid] = per

    M3 = ("B1", "B2", "B3")
    expect("E1", lambda a: [comp(f"{v}: {m} 0.10 - hard", 0.10 - get(m, "hard", "pr_auc_root", v)) for v in VARIANTS for m in M3]
           + [comp(f"{v}: {m} easy - hard", get(m, "easy", "pr_auc_root", v) - get(m, "hard", "pr_auc_root", v)) for v in VARIANTS for m in M3], aggs=("any",))
    expect("E2", lambda a: [comp(f"B3 - {m} easy root PR-AUC", get("B3", "easy", "pr_auc_root") - get(m, "easy", "pr_auc_root")) for m in ("B1", "B2")]
           + [comp(f"B3 - {m} easy top-3", get("B3", "easy", T3.format(a)) - get(m, "easy", T3.format(a))) for m in ("B1", "B2")])

    def drop(m, a):
        return get(m, "type=link_down", T3.format(a)) - get(m, "type=link_down", T3.format(a), "status_blind")
    expect("E3", lambda a: [comp("B3 drop - 0.15", drop("B3", a) - 0.15), comp("B3 drop - B2 drop", drop("B3", a) - drop("B2", a))])
    expect("E4", lambda a: [comp(f"{v}: {m} 0.35 - hard top-1", 0.35 - get(m, "hard", T1.format(a), v)) for v in VARIANTS for m in M3])
    expect("E5", lambda a: [comp(f"{v}: {m} 0.30 - node_down top-1", 0.30 - get(m, "type=node_down", T1.format(a), v)) for v in VARIANTS for m in ("B1", "B2")]
           + [comp(f"{v}: B3 - {m} node_down top-1", get("B3", "type=node_down", T1.format(a), v) - get(m, "type=node_down", T1.format(a), v))
              for v in VARIANTS for m in ("B1", "B2")])

    def e6(a):
        parts = []
        for v in VARIANTS:
            for m in M3:
                for fam in ("hard", "easy"):
                    x = get(m, f"unobservable|{fam}", T3.format(a), v)
                    if np.isfinite(x[0]):
                        parts.append(comp(f"{v}: {m} {fam} 0.15 - top-3", 0.15 - x))
        return parts
    expect("E6", e6)
    expect("E7", lambda a: [comp(f"{v}: {m} generated fault-free - 0.03", get(m, "source=generated", "far_fault_free", v) - 0.03) for v in VARIANTS for m in M3]
           + [comp(f"{v}: {m} 0.08 - generated fault-free", 0.08 - get(m, "source=generated", "far_fault_free", v)) for v in VARIANTS for m in M3]
           + [comp(f"{v}: {m} maintenance - 2 x fault-free", get(m, "all", "far_maintenance_only", v) - 2 * get(m, "all", "far_fault_free", v))
              for v in VARIANTS for m in M3], aggs=("any",))
    expect("E8", lambda a: [comp(f"{v}: {m} obvious - subtle top-3", get(m, "hard|class=obvious", T3.format(a), v) - get(m, "hard|class=subtle", T3.format(a), v))
                            for v in VARIANTS for m in M3]
           + [comp(f"{v}: {m} 0.30 - subtle detected", 0.30 - get(m, "hard|class=subtle", "detected", v)) for v in VARIANTS for m in M3])
    expect("E9", lambda a: [comp(f"{v}: {m} generated + 0.05 - real", get(m, "hard|source=generated", T3.format(a), v) + 0.05 - get(m, "hard|source=real", T3.format(a), v))
                            for v in VARIANTS for m in M3])
    expect("E10", lambda a: [comp(f"{v}: B3 no concurrent - concurrent", get("B3", "easy|scenario=no_concurrent", T3.format(a), v) - get("B3", "easy|scenario=concurrent", T3.format(a), v))
                             for v in VARIANTS])
    expect("E11", lambda a: [comp(f"{v}: {m} symptomatic - root", get(m, "hard", "pr_auc_symptomatic", v) - get(m, "hard", "pr_auc_root", v))
                             for v in VARIANTS for m in ("B1", "B2")], aggs=("any",))

    def e12(a):
        parts = []
        for v in VARIANTS:
            for m in M3:
                d3 = get(m, "hard|matched", T3.format(a), v, "v1_pretune") - get(m, "hard|matched", T3.format(a), v)
                dp = get(m, "hard|matched", "pr_auc_root", v, "v1_pretune") - get(m, "hard|matched", "pr_auc_root", v)
                parts += [comp(f"{v}: {m} top-3 rise", d3), comp(f"{v}: {m} root PR-AUC rise", dp)]
            rises = [verdict(get(m, "hard|matched", T3.format(a), v, "v1_pretune") - get(m, "hard|matched", T3.format(a), v) - 0.05)["verdict"] for m in M3]
            parts.append({"component": f"{v}: top-3 rise >= 0.05 for 2 of 3 methods",
                          "verdict": "supported" if rises.count("supported") >= 2 else "contradicted" if rises.count("contradicted") >= 2 else "inconclusive",
                          "point": None, "ci": [None, None], "per_method": dict(zip(M3, rises))})
        return parts
    if any(k[0] == "v1_pretune" for k in B):
        expect("E12", e12)
    return claims, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg_s = freeze_cfg(cfg)
    v1 = ROOT / cfg["dataset"]["root"]
    pre = ROOT / "data" / "datasets" / "v1_pretune"
    t0 = time.time()
    methods, selection = fit_and_select(cfg, v1, args.workers, cfg_s)
    print(f"train fit and PoP aggregation ({time.time() - t0:.0f}s): {selection['chosen']}", flush=True)
    settings = {"split": "val", "far_target_on_val_fault_free": FAR_TARGET, "window_aggregations": list(WINDOW_AGGS),
                "q90_quantile": WINDOW_QUANTILE, "pop_aggregation": selection, "alert_rules": {k: list(v) for k, v in RULES.items()},
                "methods": list(METHOD_NAMES)}
    results = {"v1": {"dataset": str(v1.relative_to(ROOT)), **settings, "variants": {}}}
    if pre.exists():
        results["v1_pretune"] = {"dataset": str(pre.relative_to(ROOT)), **settings, "calibration": "v1 validation (same bins and thresholds)", "variants": {}}
    B = {}
    for variant in VARIANTS:
        v1_entries = json.loads((v1 / "truth" / "val" / variant / "manifest.json").read_text())
        ctx = {"root": v1, "cfg_s": cfg_s, "methods": methods}
        edges, thr = calibrate(v1, variant, v1_entries, args.workers, ctx)
        ctx.update(edges={variant: edges}, thr={variant: thr})
        extra, n_matched = {}, None
        if pre.exists():
            pre_entries = json.loads((pre / "truth" / "val" / variant / "manifest.json").read_text())
            extra, n_matched = matched_groups(v1_entries, pre_entries)
        for ds, root, entries in (("v1", v1, v1_entries),) + ((("v1_pretune", pre, pre_entries),) if pre.exists() else ()):
            out, boots = evaluate(root, variant, entries, args.workers, ctx, cfg, extra, args.n_boot)
            out["calibration"] = {"thresholds": thr, "n_bins": {m: len(edges[m]) + 1 for m in METHOD_NAMES}}
            if n_matched is not None:
                out["matched_observable_faults_with_other_build"] = n_matched
            results[ds]["variants"][variant] = out
            for m in METHOD_NAMES:
                B[(ds, variant, m)] = boots[m]
            print(f"[{ds} {variant}] evaluated ({time.time() - t0:.0f}s)", flush=True)
    claims, E = claims_and_expectations(B)
    enc = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    (ROOT / "results" / "phase4_baselines_val.json").write_text(json.dumps(results["v1"], indent=1, default=enc) + "\n")
    if "v1_pretune" in results:
        (ROOT / "results" / "phase4_sensitivity_pretune.json").write_text(json.dumps(results["v1_pretune"], indent=1, default=enc) + "\n")
    (ROOT / "results" / "phase4_claims.json").write_text(json.dumps({"claims": claims, "expectations": E}, indent=1, default=enc) + "\n")
    print(f"wrote results ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if any("test" in a or "/dev" in a for a in sys.argv[1:]):
        sys.exit("Phase 4 runs on validation only")
    main()
