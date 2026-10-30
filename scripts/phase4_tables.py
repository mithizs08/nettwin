"""Phase 4 tables, claim verdicts and expectation verdicts, from the results JSON files only.

    python scripts/phase4_tables.py      # writes results/phase4_tables.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("B1", "B2", "B3", "B3+")
VARIANTS = ("with_status", "status_blind")
TYPES = ("link_down", "node_down", "link_flapping", "link_degradation", "demand_surge", "routing_misconfig")
AGGS = ("q90", "max", "mean")


def fmt(m):
    if m is None or m.get("value") is None:
        return "n/a"
    lo, hi = m["ci"]
    return f"{m['value']:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{m['value']:.3f}"


def rate(m):
    return fmt({"value": m["rate"], "ci": m["ci"]})


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def grp(res, var, m, g):
    return res["variants"][var]["methods"][m]["groups"][g]


def loc(res, var, m, g, k, agg="q90"):
    return grp(res, var, m, g)["localisation"][agg][k]


def tables(res):
    md = ["## v1 validation", ""]
    sel = res["pop_aggregation"]
    md += ["PoP aggregation for B1 and B2, chosen on train (audit Q2): mean top-1 / top-3 of observable train faults "
           f"({sel['train_scenarios_per_variant']} scenarios per variant, both variants, q90 window, expected ties).", ""]
    md += [table(["method"] + [f"{a} top-1 / top-3" for a in sel["table"]["B1"]] + ["chosen"],
                 [[m] + [f"{d['top1']:.3f} / {d['top3']:.3f}" for d in sel["table"][m].values()] + [sel["chosen"][m]] for m in sel["table"]]), ""]
    for var in VARIANTS:
        V = res["variants"][var]
        md += [f"### {var}", "",
               f"Calibration on each method's own validation scores (audit Q10): thresholds {json.dumps({k: round(x, 4) for k, x in V['calibration']['thresholds'].items()})}. "
               f"Binned against exact AP (root, first {V['binned_vs_exact_ap']['B1']['scenarios']} scenarios): "
               + "; ".join(f"{m} {c['binned_ap_root']} / {c['exact_ap_root']}" for m, c in V["binned_vs_exact_ap"].items()) + ".", ""]
        groups = ["type=" + t for t in TYPES] + ["easy", "hard"]
        n = lambda g: grp(res, var, "B1", g)["n_faults"]  # noqa: E731
        md += ["Detection, root PR-AUC (label 2 vs 0) [95% bootstrap CI]. Observable faults only.", ""]
        md += [table(["group", "faults", "positive cells"] + list(METHODS),
                     [[g, n(g), grp(res, var, "B1", g)["pr_auc_root"]["n_pos"]] + [fmt(grp(res, var, m, g)["pr_auc_root"]) for m in METHODS] for g in groups]), ""]
        md += ["Detection, symptomatic PR-AUC (label 1 vs 0), symptoms attributed by time and KG neighbourhood (audit Q11).", ""]
        md += [table(["group", "positive cells"] + list(METHODS),
                     [[g, grp(res, var, "B1", g)["pr_auc_symptomatic"]["n_pos"]] + [fmt(grp(res, var, m, g)["pr_auc_symptomatic"]) for m in METHODS] for g in groups]), ""]
        md += ["Localisation (oracle window, q90 window score, expected ties), top-1 and top-3.", ""]
        md += [table(["group", "faults"] + [f"{m} top-1" for m in METHODS] + [f"{m} top-3" for m in METHODS],
                     [[g, n(g)] + [fmt(loc(res, var, m, g, "top1")) for m in METHODS] + [fmt(loc(res, var, m, g, "top3")) for m in METHODS] for g in groups]), ""]
        md += ["Window aggregation sensitivity (audit Q3): top-3 under q90 / max / mean. B3 and B3+ use the window union of alerts, so they do not change.", ""]
        md += [table(["group"] + list(METHODS),
                     [[g] + [" / ".join(f"{loc(res, var, m, g, 'top3', a)['value']:.3f}" for a in AGGS) for m in METHODS] for g in groups]), ""]
        md += ["Root kind (audit Q2): top-1 for link roots and PoP roots separately (q90).", ""]
        rk = [f"{fam}|root={k}" for fam in ("easy", "hard") for k in ("link", "pop")]
        md += [table(["group", "faults"] + [f"{m} top-1" for m in METHODS],
                     [[g, n(g)] + [fmt(loc(res, var, m, g, "top1")) for m in METHODS] for g in rk if n(g)]), ""]
        br = []
        for fam in ("hard", "easy"):
            br += [f"{fam}|class={c}" for c in ("subtle", "moderate", "obvious")]
            br += [f"{fam}|placement={p}" for p in ("isolated", "maintenance_linked", "concurrent")]
            br += [f"{fam}|source=generated", f"{fam}|source=real", f"{fam}|scenario=no_concurrent", f"{fam}|scenario=concurrent", f"unobservable|{fam}"]
        md += ["Breakdowns: top-3 (q90), root PR-AUC and detection rate at the validation threshold.", ""]
        md += [table(["group", "faults"] + [f"{m} top-3" for m in METHODS] + [f"{m} root PR-AUC" for m in METHODS] + [f"{m} detected" for m in METHODS],
                     [[g, n(g)] + [fmt(loc(res, var, m, g, "top3")) for m in METHODS] + [fmt(grp(res, var, m, g)["pr_auc_root"]) for m in METHODS]
                      + [fmt(grp(res, var, m, g)["detected_at_threshold"]) for m in METHODS] for g in br]), ""]
        md += ["Per type and source, top-3 (q90).", ""]
        md += [table(["type", "generated faults", "real faults"] + [f"{m} generated" for m in METHODS] + [f"{m} real" for m in METHODS],
                     [[t, n(f"type={t}|source=generated"), n(f"type={t}|source=real")]
                      + [fmt(loc(res, var, m, f"type={t}|source=generated", "top3")) for m in METHODS]
                      + [fmt(loc(res, var, m, f"type={t}|source=real", "top3")) for m in METHODS] for t in TYPES]), ""]
        rows = []
        for m in METHODS:
            M = V["methods"][m]
            fa, bs = M["false_alarm"], M["false_alarm_by_source"]
            rows.append([m, rate(fa["fault_free"]), rate(fa["maintenance_only"]), f"{fa['maintenance_share_of_false_alarms']:.3f}",
                         rate(bs["generated"]["fault_free"]), rate(bs["real"]["fault_free"]), rate(bs["real"]["maintenance_only"])])
        fa0 = V["methods"]["B1"]["false_alarm"]
        md += [f"False alarms per 1-hour window at each method's validation threshold (5% of all validation fault-free windows by construction). "
               f"Windows: {fa0['fault_free']['n_windows']} fault-free, {fa0['maintenance_only']['n_windows']} maintenance-only. "
               "\"Maintenance share\" is the share of all false alarms that fall in maintenance-only hours (audit Q4).", ""]
        md += [table(["method", "fault-free", "maintenance-only", "maintenance share", "fault-free, generated", "fault-free, real", "maintenance-only, real"], rows), ""]
        sc = V["methods"]["B1"]["symptomatic_cells"]
        md += [f"Symptomatic cells: {sc['sym_reassigned']} moved to the related fault among concurrent ones, {sc['sym_ambiguous']} ambiguous "
               f"(kept with every eligible fault), {sc['sym_unowned']} with no fault.", ""]
        md += ["Alerts per scenario-week from step 288, and the share starting inside an observable fault window (+3 steps).", ""]
        md += [table(["rule", "per scenario", "share in fault window"],
                     [[k, V["alerts"]["per_scenario"][k], V["alerts"]["share_starting_in_observable_fault_window"][k]] for k in V["alerts"]["per_scenario"]]), ""]
    return md


def sens_tables(res, sens):
    md = ["## Sensitivity: v1 against pre-tuning fault ranges (validation, same calibration)", "",
          "Matched faults only (audit Q8): same scenario, type, root, start and end in both builds, observable in both. "
          "Top-3 (q90) and root PR-AUC, v1 / pre-tuning.", ""]
    for var in VARIANTS:
        rows = []
        for g in ("hard|matched", "easy|matched"):
            rows.append([g, grp(res, var, "B1", g)["n_faults"]]
                        + [f"{loc(res, var, m, g, 'top3')['value']:.3f} / {loc(sens, var, m, g, 'top3')['value']:.3f}" for m in METHODS]
                        + [f"{grp(res, var, m, g)['pr_auc_root']['value']:.3f} / {grp(sens, var, m, g)['pr_auc_root']['value']:.3f}" for m in METHODS])
        md += [f"### {var}", "", table(["group", "faults"] + [f"{m} top-3" for m in METHODS] + [f"{m} root PR-AUC" for m in METHODS], rows), ""]
        rows = [[g, grp(res, var, "B1", g)["n_faults"], grp(sens, var, "B1", g)["n_faults"]]
                + [f"{loc(res, var, m, g, 'top3')['value']:.3f} / {loc(sens, var, m, g, 'top3')['value']:.3f}" for m in METHODS]
                for g in ["hard", "easy"] + ["type=" + t for t in TYPES]]
        md += ["All faults of each build (not paired), top-3 v1 / pre-tuning:", "",
               table(["group", "faults v1", "faults pre-tuning"] + [f"{m} top-3" for m in METHODS], rows), ""]
    return md


def verdict_tables(cl):
    md = ["## Claims (audit Q3, Q12)", "",
          "Each claim is checked with paired scenario-bootstrap differences. A component is supported if its 95% CI is above 0, "
          "contradicted if below, inconclusive if it crosses 0; a claim holds only if every component is supported. "
          "Localisation claims are checked under each window aggregation; full components are in `results/phase4_claims.json`.", ""]
    rows = []
    for c in cl["claims"]:
        per = c["by_window_aggregation"]
        rows.append([c["id"], c["claim"]] + [per[a]["overall"] if a in per else "" for a in AGGS + ("any",)])
    md += [table(["id", "claim", "q90", "max", "mean", "no window"], rows), ""]
    md += ["## Pre-registered expectations, re-checked with CIs under the revised harness", "",
           "met = every component supported; not met = some component contradicted; inconclusive otherwise.", ""]
    rows = [[k] + [per[a]["verdict"] if a in per else "" for a in AGGS + ("any",)] for k, per in cl["expectations"].items()]
    md += [table(["id", "q90", "max", "mean", "no window"], rows), ""]
    return md


def main():
    res = json.loads((ROOT / "results" / "phase4_baselines_val.json").read_text())
    sp = ROOT / "results" / "phase4_sensitivity_pretune.json"
    sens = json.loads(sp.read_text()) if sp.exists() else None
    cl = json.loads((ROOT / "results" / "phase4_claims.json").read_text())
    md = ["# Phase 4 results (validation only)", "",
          "Generated by `scripts/phase4_tables.py` from `results/phase4_baselines_val.json`, `results/phase4_sensitivity_pretune.json` and "
          "`results/phase4_claims.json`. Revised after `docs/audit_phase4.md`. Telemetry is synthetic; no number here is from the test split.", ""]
    md += verdict_tables(cl)
    md += tables(res)
    if sens is not None:
        md += sens_tables(res, sens)
    (ROOT / "results" / "phase4_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(verdict_tables(cl)))


if __name__ == "__main__":
    main()
