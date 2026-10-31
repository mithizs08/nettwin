"""Phase 5 tables from the results JSON files only.

    python scripts/phase5_tables.py      # writes results/phase5_tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ("B1", "B2", "B3", "B3+")
VARIANTS = ("with_status", "status_blind")
TYPES = ("link_down", "node_down", "link_flapping", "link_degradation", "demand_surge", "routing_misconfig")
AGGS = ("q90", "max", "mean")


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def fmt(v, ci=None, nd=3):
    if v is None:
        return "n/a"
    s = f"{v:.{nd}f}"
    if ci and ci[0] is not None:
        s += f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
    return s


class R:
    def __init__(self, res):
        self.res = res
        self.models = sorted(k for k in res["variants"]["with_status"]["methods"] if k not in BASE)
        self.label = {k: k.split("-s")[0] + ("-" + k.split("-", 3)[3] if k.count("-") >= 3 else "") for k in self.models}

    def members(self, lb):
        return [k for k in self.models if self.label[k] == lb]

    def g(self, var, m, grp):
        return self.res["variants"][var]["methods"][m]["groups"][grp]

    def val(self, var, m, grp, key, agg="q90"):
        """Value for a method (baseline) or a configuration label (seed mean of its runs)."""
        def one(name):
            x = self.g(var, name, grp)
            if key.startswith("top"):
                return x["localisation"][agg][key]["value"]
            if key == "detected":
                return x["detected_at_threshold"]["value"]
            return x[key]["value"]
        if m in BASE or m in self.models:
            return one(m)
        vals = [one(k) for k in self.members(m)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    def ci(self, var, m, grp, key, agg="q90"):
        if m not in BASE:
            return None
        x = self.g(var, m, grp)
        return x["localisation"][agg][key]["ci"] if key.startswith("top") else x[key]["ci"]

    def far(self, var, m, which):
        def one(name):
            return self.res["variants"][var]["methods"][name]["false_alarm"][which]["rate"]
        return one(m) if m in BASE or m in self.models else float(np.mean([one(k) for k in self.members(m)]))


HEAD = (("hard", "top1@q90"), ("hard", "top3@q90"), ("easy", "top1@q90"), ("easy", "top3@q90"), ("hard", "pr_auc_root"),
        ("easy", "pr_auc_root"), ("all", "far_maintenance_only"))
HEAD_COLS = ["hard top-1", "hard top-3", "easy top-1", "easy top-3", "hard root PR-AUC", "easy root PR-AUC", "maint. alarms"]


def mean_or_none(xs):
    xs = list(xs)
    return None if any(x is None for x in xs) else float(np.mean(xs))


def summary_value(summ, grp, metric):
    """One headline value from a harness summary (`aggregate` output)."""
    if metric.startswith("far_"):
        return summ["false_alarm"][metric[4:]]["rate"]
    g = summ["groups"][grp]
    if "@" in metric:
        k, agg = metric.split("@")
        return g["localisation"][agg][k]["value"]
    return g[metric]["value"]


def audit_sections(best):
    """Audit Phase 5 fixes: held-out validation headline (R2), the maintenance-blanking check, and test next to validation."""
    md = []
    hp = ROOT / "results" / "phase5_heldout_val.json"
    if hp.exists():
        H = json.loads(hp.read_text())
        rows_m = list(BASE) + [f"{best} (seed mean)", f"{best}-mlp (seed mean)"]
        md += ["## Validation headline without the selection subset (audit R2)", "",
               f"The headline is the {H['variants']['with_status']['heldout']['n_scenarios']} validation scenarios outside the "
               f"{len(H['selection_subset'])}-scenario selection subset; all 468 shown for reference. q90 window; "
               "maintenance alarms at each method's 5% fault-free threshold.", ""]
        for var in VARIANTS:
            rows = []
            for m in rows_m:
                for part, lab in (("heldout", "held-out"), ("all", "all 468")):
                    x = H["variants"][var][part]["methods"][m]
                    rows.append([m, lab] + [fmt(x[f"{g}:{k}"]) for g, k in HEAD])
            md += [f"### {var}", "", table(["method", "scenarios"] + HEAD_COLS, rows), ""]
            vrows = []
            for g, k in HEAD:
                cells = []
                for b in BASE:
                    key = f"{g}|{k}|{b} - GNN" if k.startswith("far") else f"{g}|{k}|GNN - {b}"
                    c = H["variants"][var]["heldout"]["comparisons"][key]
                    cells.append(f"{c['verdict']} ({fmt(c['point'], c['ci'])})")
                mk = f"{g}|{k}|GNN - MLP"
                c = H["variants"][var]["heldout"]["comparisons"].get(mk)
                cells.append(f"{c['verdict']} ({fmt(c['point'], c['ci'])})" if c else "")
                vrows.append([f"{g} {k}"] + cells)
            md += ["Held-out paired verdicts (GNN seed mean against each method; for alarms, the baseline minus the GNN, "
                   "so supported means the GNN alarms less).", "",
                   table(["metric"] + [f"vs {b}" for b in BASE] + ["vs MLP"], vrows), ""]
    bp = ROOT / "results" / "phase5_maintenance_blanking_val.json"
    if bp.exists():
        Bk = json.loads(bp.read_text())
        md += ["## Maintenance-hour alarms and the down-link signature (validation)", "", Bk["note"], ""]
        for var in VARIANTS:
            rows = []
            for m, x in Bk["variants"][var].items():
                def cell(key):
                    c = x[key]
                    return f"{fmt(c['rate'])} ({c['n_alarms']}/{c['n_windows']})"
                d = x["maintenance_only|blanked - not_blanked"]
                tb = x["maintenance_only|blanked"]
                rows.append([m, cell("maintenance_only|blanked"), cell("maintenance_only|not_blanked"),
                             f"{d['verdict']} ({fmt(d['point'], d['ci'])})", fmt(x["share_of_maintenance_alarms_in_blanked_windows"]),
                             f"{tb['alarms_whose_top_link_is_blanked']}/{tb['n_alarms']}",
                             cell("fault_free|blanked"), cell("fault_free|not_blanked")]
                            + [fmt(x[f"maintenance_only|action={a}"]["rate"]) for a in ("down", "drain", "noop")])
            md += [f"### {var}", "", table(["method", "maint. with blanked link", "maint. without", "difference",
                                            "share of maint. alarms in blanked windows", "top link blanked",
                                            "fault-free with blanked link", "fault-free without", "maint. down", "maint. drain", "maint. noop"], rows), ""]
    tp = ROOT / "results" / "phase5_test.json"
    if tp.exists():
        T = json.loads(tp.read_text())
        H = json.loads(hp.read_text())
        gnn = [m for m in T["models"] if not m.endswith("-mlp")]
        mlp = [m for m in T["models"] if m.endswith("-mlp")]
        md += ["## Test results (run once, frozen validation calibration)", "",
               "Test: 7 real networks, 30 scenarios each. Dev: AARNet, seen during development, reported separately. "
               "GNN and MLP are seed means of 3 seeds. Validation columns are the held-out headline (R2).", ""]
        for split, sp in T["splits"].items():
            for var in VARIANTS:
                V = sp["variants"][var]
                rows = []
                for m, members in [(b, [b]) for b in BASE] + [(best, gnn), (f"{best}-mlp", mlp)]:
                    vals = [mean_or_none(([summary_value(V["methods"][k], g, mt) for k in members])) for g, mt in HEAD]
                    hv = H["variants"][var]["heldout"]["methods"][m if m in BASE else f"{m} (seed mean)"]
                    rows.append([m] + [f"{fmt(a)} (val {fmt(hv[f'{g}:{mt}'])})" for a, (g, mt) in zip(vals, HEAD)])
                br = V["methods"]["B1"]["groups"]
                md += [f"### {split}, {var} ({V['n_scenarios']} scenarios)", "",
                       f"Base rates: root hard {br['hard']['pr_auc_root']['base_rate']}, root easy {br['easy']['pr_auc_root']['base_rate']}. "
                       f"Fault-free alarm rate at the frozen thresholds: "
                       + ", ".join(f"{m} {fmt(V['methods'][m]['false_alarm']['fault_free']['rate'])}" for m in list(BASE) + T["models"]) + ".", "",
                       table(["method"] + HEAD_COLS, rows), ""]
                S = V["seed_summary"]
                md += [table(["model", "metric", "per seed", "mean", "sd", "seed-mean CI"],
                             [[lb, k, ", ".join(f"{x:.3f}" for x in S[lb][k]["per_seed"]), S[lb][k]["mean"], S[lb][k].get("sd", ""),
                               fmt(S[lb][k]["mean"], S[lb][k]["seed_mean_ci"])]
                              for lb in S for k in ("hard:top3@q90", "easy:top1@q90", "hard:pr_auc_root", "all:far_maintenance_only")]), ""]
                md += ["Per type, top-3 (q90).", "", table(["type"] + list(BASE) + [best],
                       [[t] + [fmt(summary_value(V["methods"][b], f"type={t}", "top3@q90")) for b in BASE]
                        + [fmt(mean_or_none(([summary_value(V["methods"][k], f"type={t}", "top3@q90") for k in gnn])))] for t in TYPES]), ""]
                nets = sorted({k.split("network=")[1] for k in V["methods"]["B1"]["groups"] if "network=" in k})
                md += ["Per network, top-3 (q90): hard / easy.", "", table(["network"] + list(BASE) + [best],
                       [[n] + [f"{fmt(summary_value(V['methods'][b], f'hard|network={n}', 'top3@q90'))} / {fmt(summary_value(V['methods'][b], f'easy|network={n}', 'top3@q90'))}" for b in BASE]
                        + [f"{fmt(mean_or_none(([summary_value(V['methods'][k], f'hard|network={n}', 'top3@q90') for k in gnn])))} / "
                           f"{fmt(mean_or_none(([summary_value(V['methods'][k], f'easy|network={n}', 'top3@q90') for k in gnn])))}"] for n in nets]), ""]
            C = sp["comparisons"]
            keys = [k for k in C if k.split("|")[1] in ("hard", "easy", "all") and ("q90" in k or "pr_auc" in k or "far" in k or "MLP" in k)]
            md += [f"### {split}: paired verdicts", "", table(["comparison", "verdict", "point [CI]"],
                   [[k, C[k]["verdict"], fmt(C[k]["point"], C[k]["ci"])] for k in keys]), ""]
            md += [f"### {split}: validation expectations replicated", "", table(["id", "q90", "max", "mean", "no window"],
                   [[x] + [(r[a]["overall"] if isinstance(r.get(a), dict) else r.get(a, "")) for a in AGGS + ("any",)]
                    for x, r in sorted(sp["replications"].items())]), ""]
    return md


def main():
    res = json.loads((ROOT / "results" / "phase5_gnn_val.json").read_text())
    E = json.loads((ROOT / "results" / "phase5_expectations.json").read_text())
    tp = ROOT / "results" / "phase5_timings.json"
    timings = json.loads(tp.read_text()) if tp.exists() else None
    r = R(res)
    best = res["best"]
    archs = [f"{c}-h{res['chosen_hidden'][c]}" for c in ("sage", "gat", "gine")]
    abls = [f"{best}-{a}" for a in ("mlp", "no_path_static", "shuffled_edges")]
    rows_methods = list(BASE) + archs + abls
    md = ["# Phase 5 results: GNN on validation", "",
          "Generated by `scripts/phase5_tables.py` from `results/phase5_gnn_val.json` and `results/phase5_expectations.json`. "
          "Validation only; no test data was used. Telemetry is synthetic. GNN rows are the mean of 3 seeds unless stated; "
          "baseline rows carry 95% scenario-bootstrap CIs. Localisation uses the oracle incident window for every method.", ""]

    md += ["## Selection (validation subset of 60 scenarios, both variants, seed 0)", ""]
    md += [table(["configuration", "hard top-3", "easy top-3", "score"],
                 [[k.replace("-s0", ""), v["hard"], v["easy"], v["score"]] for k, v in sorted(res["selection"].items())]), ""]
    md += [f"Chosen hidden sizes: {res['chosen_hidden']}. Best configuration: **{best}**. Baseline PoP aggregation (from Phase 4, train): "
           f"{res['baseline_pop_aggregation']}.", ""]

    for var in VARIANTS:
        md += [f"## {var}", ""]
        rows = []
        for m in rows_methods:
            rb = lambda grp, key: fmt(r.val(var, m, grp, key), r.ci(var, m, grp, key))  # noqa: E731
            rows.append([("**" + m + "**") if m == best else m, rb("hard", "top1"), rb("hard", "top3"), rb("easy", "top1"), rb("easy", "top3"),
                         rb("hard", "pr_auc_root"), rb("easy", "pr_auc_root"), rb("hard", "pr_auc_symptomatic"), rb("easy", "pr_auc_symptomatic"),
                         fmt(r.far(var, m, "maintenance_only"))])
        br = r.g(var, "B1", "hard")
        be = r.g(var, "B1", "easy")
        md += ["Headline (q90 window). PR-AUC base rates: root hard "
               f"{br['pr_auc_root']['base_rate']}, root easy {be['pr_auc_root']['base_rate']}, symptomatic hard "
               f"{br['pr_auc_symptomatic']['base_rate']}, symptomatic easy {be['pr_auc_symptomatic']['base_rate']}. "
               "Maintenance-only alarm rate at each method's 5% fault-free threshold.", ""]
        md += [table(["method", "hard top-1", "hard top-3", "easy top-1", "easy top-3", "hard root PR-AUC", "easy root PR-AUC",
                      "hard sympt. PR-AUC", "easy sympt. PR-AUC", "maint. alarms"], rows), ""]
        S = res["seed_summary"][var]
        md += ["Seed variance (3 seeds): per-seed values, mean, sd, and the bootstrap CI of the seed mean.", ""]
        srows = []
        for lb in archs + abls:
            for key in ("hard:top3@q90", "easy:top3@q90", "hard:pr_auc_root"):
                x = S[lb][key]
                srows.append([lb, key, ", ".join(f"{v:.3f}" for v in x["per_seed"]), f"{x['mean']:.3f}", f"{x['sd']:.3f}", fmt(None if x is None else x["mean"], x["seed_mean_ci"])])
        md += [table(["configuration", "metric", "per seed", "mean", "sd", "seed mean [CI]"], srows), ""]
        md += ["Per type, top-3 under q90 / max / mean (B3 and B3+ use the window union, so they do not change).", ""]
        md += [table(["type"] + list(BASE) + [best],
                     [[t] + [" / ".join(fmt(r.val(var, m, f"type={t}", "top3", a)) for a in AGGS) for m in list(BASE) + [best]] for t in TYPES]), ""]
        md += ["Per type, top-1 and root PR-AUC (q90), with the base rate.", ""]
        md += [table(["type", "base rate"] + [f"{m} top-1" for m in list(BASE) + [best]] + [f"{m} PR-AUC" for m in list(BASE) + [best]],
                     [[t, r.g(var, "B1", f"type={t}")["pr_auc_root"]["base_rate"]] + [fmt(r.val(var, m, f"type={t}", "top1")) for m in list(BASE) + [best]]
                      + [fmt(r.val(var, m, f"type={t}", "pr_auc_root")) for m in list(BASE) + [best]] for t in TYPES]), ""]
        nets = sorted({k.split("network=")[1] for k in res["variants"][var]["methods"]["B1"]["groups"] if "network=" in k})
        md += ["Per real validation network, top-3 (q90): hard / easy.", ""]
        md += [table(["network", "hard faults", "easy faults"] + list(BASE) + [best],
                     [[n, r.g(var, "B1", f"hard|network={n}")["n_faults"], r.g(var, "B1", f"easy|network={n}")["n_faults"]]
                      + [f"{fmt(r.val(var, m, f'hard|network={n}', 'top3'))} / {fmt(r.val(var, m, f'easy|network={n}', 'top3'))}" for m in list(BASE) + [best]]
                      for n in nets]), ""]
        groups = []
        for fam in ("hard", "easy"):
            groups += [f"{fam}|class={c}" for c in ("subtle", "moderate", "obvious")] + [f"{fam}|source=generated", f"{fam}|source=real",
                                                                                       f"{fam}|root=link", f"{fam}|root=pop", f"{fam}|scenario=concurrent",
                                                                                       f"{fam}|placement=maintenance_linked", f"unobservable|{fam}"]
        md += ["Breakdowns, top-3 (q90).", ""]
        md += [table(["group", "faults"] + list(BASE) + [best],
                     [[grp, r.g(var, "B1", grp)["n_faults"]] + [fmt(r.val(var, m, grp, "top3")) for m in list(BASE) + [best]] for grp in groups]), ""]
        comps = res["comparisons_vs_baselines"]
        vrows = []
        for grp in ("hard", "easy"):
            for metric in ("top1@q90", "top3@q90", "top1@max", "top3@max", "top1@mean", "top3@mean", "pr_auc_root", "pr_auc_symptomatic"):
                cells = []
                for m in BASE:
                    c = comps[f"{var}|{grp}|{metric}|GNN - {m}"]
                    cells.append(f"{c['verdict']} ({fmt(c['point'], c['ci'])})")
                vrows.append([grp, metric] + cells)
        md += [f"Paired verdicts, {best} (seed mean) minus each baseline: supported = GNN better with the 95% CI above 0.", ""]
        md += [table(["group", "metric"] + [f"GNN - {m}" for m in BASE], vrows), ""]
    md += ["## Pre-registered expectations X1 to X10", ""]
    md += [table(["id", "expectation", "q90", "max", "mean", "no window"],
                 [[k, x["expectation"]] + [x["by_window_aggregation"][a]["verdict"] if a in x["by_window_aggregation"] else "" for a in AGGS + ("any",)]
                  for k, x in E.items()]), ""]
    md += ["## Training", "", f"Budget: {json.dumps(res['budget'])}.", ""]
    rows = [[k, res["n_params"].get(k, ""), res["train_history"][k][-1]["loss"], res["train_history"][k][-1]["det_loss"], res["train_history"][k][-1]["loc_loss"]]
            for k in sorted(res["train_history"])]
    md += [table(["run", "parameters", "final loss", "detection loss", "localisation loss"], rows), ""]
    md += ["Wall-clock times (training per run, evaluation, total) are in `results/phase5_timings.json`; they are kept out of "
           "this file so that it reproduces byte for byte.", ""]
    if timings:
        print(f"wall-clock: samples {timings['build_samples_s']} s, evaluation {timings['evaluate_s']} s, total {timings['total_s']} s")
    md += audit_sections(best)
    (ROOT / "results" / "phase5_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-40:]) if "--tail" in sys.argv else "wrote results/phase5_tables.md")


if __name__ == "__main__":
    main()
