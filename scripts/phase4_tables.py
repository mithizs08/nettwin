"""Phase 4 tables and the pre-registered expectation checks E1 to E12, from the results JSON files only.

    python scripts/phase4_tables.py      # writes results/phase4_tables.md and results/phase4_expectations.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("B1", "B2", "B3")
VARIANTS = ("with_status", "status_blind")
TYPES = ("link_down", "node_down", "link_flapping", "link_degradation", "demand_surge", "routing_misconfig")


def fmt(m):
    if m is None or m.get("value") is None:
        return "n/a"
    lo, hi = m["ci"]
    return f"{m['value']:.3f} [{lo:.3f}, {hi:.3f}]" if lo is not None else f"{m['value']:.3f}"


def g(res, variant, method, group):
    return res["variants"][variant]["methods"][method]["groups"][group]


def v(res, variant, method, group, key):
    return g(res, variant, method, group)[key]["value"]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def tables(res, title):
    md = [f"## {title}", ""]
    for var in VARIANTS:
        V = res["variants"][var]
        md += [f"### {var}", "", f"Train calibration: thresholds {json.dumps({k: round(x, 4) for k, x in V['calibration']['threshold'].items()})}; "
               f"binned vs exact train AP {json.dumps(V['calibration']['binned_vs_exact_ap'])}.", ""]
        groups = ["type=" + t for t in TYPES] + ["easy", "hard"]
        n = lambda grp: g(res, var, "B1", grp)["n_faults"]  # noqa: E731
        md += ["Detection, root PR-AUC (label 2 vs 0) [95% bootstrap CI]. Observable faults only.", ""]
        md += [table(["group", "faults", "positive cells"] + list(METHODS),
                     [[grp, n(grp), g(res, var, "B1", grp)["pr_auc_root"]["n_pos"]] + [fmt(g(res, var, m, grp)["pr_auc_root"]) for m in METHODS] for grp in groups]), ""]
        md += ["Detection, symptomatic PR-AUC (label 1 vs 0).", ""]
        md += [table(["group", "positive cells"] + list(METHODS),
                     [[grp, g(res, var, "B1", grp)["pr_auc_symptomatic"]["n_pos"]] + [fmt(g(res, var, m, grp)["pr_auc_symptomatic"]) for m in METHODS] for grp in groups]), ""]
        md += ["Localisation (oracle window), top-1 and top-3.", ""]
        md += [table(["group", "faults"] + [f"{m} top-1" for m in METHODS] + [f"{m} top-3" for m in METHODS],
                     [[grp, n(grp)] + [fmt(g(res, var, m, grp)["top1"]) for m in METHODS] + [fmt(g(res, var, m, grp)["top3"]) for m in METHODS] for grp in groups]), ""]
        br = []
        for fam in ("hard", "easy"):
            br += [f"{fam}|class={c}" for c in ("subtle", "moderate", "obvious")]
            br += [f"{fam}|placement={p}" for p in ("isolated", "maintenance_linked", "concurrent")]
            br += [f"{fam}|source=generated", f"{fam}|source=real", f"{fam}|scenario=no_concurrent", f"{fam}|scenario=concurrent", f"unobservable|{fam}"]
        md += ["Breakdowns: top-3, root PR-AUC and detection rate at the train threshold.", ""]
        md += [table(["group", "faults"] + [f"{m} top-3" for m in METHODS] + [f"{m} root PR-AUC" for m in METHODS] + [f"{m} detected" for m in METHODS],
                     [[grp, n(grp)] + [fmt(g(res, var, m, grp)["top3"]) for m in METHODS] + [fmt(g(res, var, m, grp)["pr_auc_root"]) for m in METHODS]
                      + [fmt(g(res, var, m, grp)["detected_at_threshold"]) for m in METHODS] for grp in br]), ""]
        md += ["Per type and source, top-3.", ""]
        md += [table(["type", "generated faults", "real faults"] + [f"{m} generated" for m in METHODS] + [f"{m} real" for m in METHODS],
                     [[t, n(f"type={t}|source=generated"), n(f"type={t}|source=real")]
                      + [fmt(g(res, var, m, f"type={t}|source=generated")["top3"]) for m in METHODS]
                      + [fmt(g(res, var, m, f"type={t}|source=real")["top3"]) for m in METHODS] for t in TYPES]), ""]
        rows = []
        for m in METHODS:
            M = V["methods"][m]
            fa, bs = M["false_alarm"], M["false_alarm_by_source"]
            rows.append([m, fmt(fa["fault_free"] | {"value": fa["fault_free"]["rate"]}), fmt(fa["maintenance_only"] | {"value": fa["maintenance_only"]["rate"]}),
                         fmt(bs["generated"]["fault_free"] | {"value": bs["generated"]["fault_free"]["rate"]}),
                         fmt(bs["real"]["fault_free"] | {"value": bs["real"]["fault_free"]["rate"]}),
                         fmt(bs["real"]["maintenance_only"] | {"value": bs["real"]["maintenance_only"]["rate"]})])
        fa0 = V["methods"]["B1"]["false_alarm"]
        md += [f"False alarms per 1-hour window at the train threshold (train fault-free rate 5%). Windows: "
               f"{fa0['fault_free']['n_windows']} fault-free, {fa0['maintenance_only']['n_windows']} maintenance-only.", ""]
        md += [table(["method", "fault-free", "maintenance-only", "fault-free, generated", "fault-free, real", "maintenance-only, real"], rows), ""]
        if "alerts" in V:
            md += ["Alerts per scenario-week from step 288, and the share starting inside an observable fault window (+3 steps).", ""]
            md += [table(["rule", "per scenario", "share in fault window"],
                         [[k, V["alerts"]["per_scenario"][k], V["alerts"]["share_starting_in_observable_fault_window"][k]] for k in V["alerts"]["per_scenario"]]), ""]
    return md


def expectations(res, sens):
    E = {}

    def rec(k, ok, detail):
        E[k] = {"met": bool(ok), "detail": detail}

    d = {f"{var}/{m}": (v(res, var, m, "hard", "pr_auc_root"), v(res, var, m, "easy", "pr_auc_root")) for var in VARIANTS for m in METHODS}
    rec("E1", all(h < 0.10 and h < e for h, e in d.values()), {k: {"hard": h, "easy": e} for k, (h, e) in d.items()})
    ws = {m: (v(res, "with_status", m, "easy", "pr_auc_root"), v(res, "with_status", m, "easy", "top3")) for m in METHODS}
    rec("E2", all(ws["B3"][i] > ws[m][i] for m in ("B1", "B2") for i in (0, 1)), {m: {"pr_auc_root": a, "top3": b} for m, (a, b) in ws.items()})
    drop = {m: v(res, "with_status", m, "type=link_down", "top3") - v(res, "status_blind", m, "type=link_down", "top3") for m in METHODS}
    rec("E3", drop["B3"] >= 0.15 and drop["B2"] < drop["B3"], {"link_down_top3_drop": drop})
    t1 = {f"{var}/{m}": v(res, var, m, "hard", "top1") for var in VARIANTS for m in METHODS}
    rec("E4", all(x <= 0.35 for x in t1.values()), t1)
    nd = {f"{var}/{m}": v(res, var, m, "type=node_down", "top1") for var in VARIANTS for m in METHODS}
    rec("E5", all(nd[f"{var}/B1"] < 0.30 and nd[f"{var}/B2"] < 0.30 and nd[f"{var}/B3"] > max(nd[f"{var}/B1"], nd[f"{var}/B2"]) for var in VARIANTS), nd)
    un = {f"{var}/{m}/{fam}": v(res, var, m, f"unobservable|{fam}", "top3") for var in VARIANTS for m in METHODS for fam in ("hard", "easy")}
    rec("E6", all(x is None or x <= 0.15 for x in un.values()), un)
    fa = {}
    ok = True
    for var in VARIANTS:
        for m in METHODS:
            M = res["variants"][var]["methods"][m]
            ff_gen = M["false_alarm_by_source"]["generated"]["fault_free"]["rate"]
            ff, mo = M["false_alarm"]["fault_free"]["rate"], M["false_alarm"]["maintenance_only"]["rate"]
            fa[f"{var}/{m}"] = {"fault_free_generated": ff_gen, "fault_free": ff, "maintenance_only": mo}
            ok &= 0.03 <= ff_gen <= 0.08 and mo >= 2 * ff
    rec("E7", ok, fa)
    sv = {f"{var}/{m}": {"subtle_top3": v(res, var, m, "hard|class=subtle", "top3"), "obvious_top3": v(res, var, m, "hard|class=obvious", "top3"),
                          "subtle_detected": v(res, var, m, "hard|class=subtle", "detected_at_threshold")} for var in VARIANTS for m in METHODS}
    rec("E8", all(x["subtle_top3"] < x["obvious_top3"] and x["subtle_detected"] <= 0.30 for x in sv.values()), sv)
    rg = {f"{var}/{m}": {"real": v(res, var, m, "hard|source=real", "top3"), "generated": v(res, var, m, "hard|source=generated", "top3")} for var in VARIANTS for m in METHODS}
    rec("E9", all(x["real"] <= x["generated"] + 0.05 for x in rg.values()), rg)
    cc = {var: {"concurrent": v(res, var, "B3", "easy|scenario=concurrent", "top3"), "no_concurrent": v(res, var, "B3", "easy|scenario=no_concurrent", "top3")} for var in VARIANTS}
    rec("E10", all(x["concurrent"] < x["no_concurrent"] for x in cc.values()), cc)
    sy = {f"{var}/{m}": {"symptomatic": v(res, var, m, "hard", "pr_auc_symptomatic"), "root": v(res, var, m, "hard", "pr_auc_root")} for var in VARIANTS for m in ("B1", "B2")}
    rec("E11", all(x["symptomatic"] >= x["root"] for x in sy.values()), sy)
    if sens is not None:
        cmp, ok = {}, True
        for var in VARIANTS:
            rises = 0
            for m in METHODS:
                a3, b3 = v(res, var, m, "hard", "top3"), v(sens, var, m, "hard", "top3")
                ap, bp = v(res, var, m, "hard", "pr_auc_root"), v(sens, var, m, "hard", "pr_auc_root")
                cmp[f"{var}/{m}"] = {"top3_v1": a3, "top3_pretune": b3, "pr_auc_root_v1": ap, "pr_auc_root_pretune": bp}
                ok &= b3 >= a3 and bp >= ap
                rises += b3 - a3 >= 0.05
            ok &= rises >= 2
        rec("E12", ok, cmp)
    return E


def sens_tables(res, sens):
    md = ["## Sensitivity: v1 against pre-tuning fault ranges (validation, same slots, same train calibration)", ""]
    for var in VARIANTS:
        rows = []
        for grp in ["hard", "easy"] + ["type=" + t for t in TYPES]:
            rows.append([grp, g(res, var, "B1", grp)["n_faults"], g(sens, var, "B1", grp)["n_faults"]]
                        + [f"{v(res, var, m, grp, 'top3'):.3f} / {v(sens, var, m, grp, 'top3'):.3f}" for m in METHODS]
                        + [f"{v(res, var, m, grp, 'pr_auc_root'):.3f} / {v(sens, var, m, grp, 'pr_auc_root'):.3f}" for m in METHODS])
        md += [f"### {var} (v1 / pre-tuning)", "",
               table(["group", "faults v1", "faults pre-tuning"] + [f"{m} top-3" for m in METHODS] + [f"{m} root PR-AUC" for m in METHODS], rows), ""]
        full = [[grp] + [fmt(g(sens, var, m, grp)["top3"]) for m in METHODS] for grp in ("hard", "easy")]
        md += ["Pre-tuning top-3 with CIs:", "", table(["group"] + list(METHODS), full), ""]
    return md


def main():
    res = json.loads((ROOT / "results" / "phase4_baselines_val.json").read_text())
    sp = ROOT / "results" / "phase4_sensitivity_pretune.json"
    sens = json.loads(sp.read_text()) if sp.exists() else None
    md = ["# Phase 4 results (validation only)", "",
          "Generated by `scripts/phase4_tables.py` from `results/phase4_baselines_val.json` and `results/phase4_sensitivity_pretune.json`. "
          "Telemetry is synthetic; no number here is from the test split.", ""]
    md += tables(res, "v1 validation")
    if sens is not None:
        md += sens_tables(res, sens)
    E = expectations(res, sens)
    md += ["## Pre-registered expectations", "", table(["id", "met"], [[k, "yes" if x["met"] else "**no**"] for k, x in E.items()]), ""]
    (ROOT / "results" / "phase4_tables.md").write_text("\n".join(md) + "\n")
    (ROOT / "results" / "phase4_expectations.json").write_text(json.dumps(E, indent=1) + "\n")
    print("\n".join(md[-(len(E) + 4):]))


if __name__ == "__main__":
    main()
