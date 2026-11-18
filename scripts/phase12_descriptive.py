"""Phase 12 descriptive analysis (after the audit, docs/audit_phases_11_12.md). Not pre-registered; no API call.

Reads only the saved test run (replay, decisions, per-fault records, timeline) and the truth manifest, and recomputes
the frozen GNN's element scores (deterministic, causal) for the GNN rankings at its pages. Nothing in the frozen run
or its summary is changed. Writes `results/phase12/<run>/descriptive.json` with:

1. Matched-fault root cause (audit M1): top-1 and top-3 at the first detecting page, on the faults that both policies
   of a pair detect, so both are scored on the same faults. The pre-registered "at escalation" metric averages each
   policy over its own detected faults, which differ.
2. Detection with the correct root (audit m1): a fault counts when some page in its window [start, end + 1 hour)
   names its root in the page's top 3 (the shared tie-aware credit, best page in the window). The pre-registered
   detection rate credits any page in the window, whatever element it names.
3. Run record (audit M2 and m8): recovered-wake flags in each decisions file, the position of the first recovered
   decision, and the agent log's cost of the first decisions.

    python scripts/phase12_descriptive.py --run test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase12_replay as P  # noqa: E402
from nettwin import replay as R  # noqa: E402
from nettwin.agent.eval import localisation_credit  # noqa: E402
from nettwin.evaluation import LAG, boot_weights, ci, verdict  # noqa: E402
from nettwin.splits import guard  # noqa: E402

PAIRS = (("Agent", "GNN"), ("Agent", "GNN+CL"))


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines()]


def ratio_boot(W, num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ num) / (W @ den)


def matched_faults(per_fault, sids, W):
    """Top-1 and top-3 at the first detecting page, on the faults both policies of a pair detect."""
    by = {m: {(f["scenario_id"], f["fault_id"]): f for f in per_fault[m]} for m in per_fault}
    idx = {s: i for i, s in enumerate(sids)}
    out = {}
    for a, b in PAIRS:
        both = [k for k in by[a] if by[a][k]["detected"] and by[b][k]["detected"]]
        only_b = [k for k in by[b] if by[b][k]["detected"] and not by[a][k]["detected"]]
        only_a = [k for k in by[a] if by[a][k]["detected"] and not by[b][k]["detected"]]
        n = np.zeros(len(sids))
        for k in both:
            n[idx[k[0]]] += 1
        res = {"n_both_detect": len(both), f"n_only_{a}": len(only_a), f"n_only_{b}": len(only_b)}
        for c in ("top1", "top3"):
            boots = {}
            for m in (a, b):
                s = np.zeros(len(sids))
                for k in both:
                    s[idx[k[0]]] += by[m][k][c]
                boots[m] = ratio_boot(W, s, n)
                res[f"{c}|{m}"] = {"value": round(float(boots[m][0]), 4), "ci": ci(boots[m][1:])}
            res[f"{c}|{a} minus {b}"] = verdict(boots[a] - boots[b])
        for m, ks in ((a, only_a), (b, only_b)):
            if ks:
                res[f"top3|{m}|faults only {m} detects"] = round(float(np.mean([by[m][k]["top3"] for k in ks])), 4)
        out[f"{a} vs {b}"] = res
    return out


def correct_root_detection(run_dir, split, cfg, sids, W):
    """Share of faults with some page in their window whose top 3 names the root (best tie-aware credit)."""
    root = ROOT / cfg["dataset"]["root"]
    truth = {e["scenario_id"]: e for e in json.loads(guard(root / "truth" / split / P.VARIANT / "manifest.json", {split}).read_text())}
    reps = {r["scenario_id"]: r for r in read_jsonl(run_dir / "replay.jsonl")}
    timeline = {t["scenario_id"]: t for t in json.loads((run_dir / "timeline.json").read_text())}
    P.rebuild_elem(run_dir, split, cfg)
    num = {m: np.zeros(len(sids)) for m in P.METHODS}
    den = np.zeros(len(sids))
    for i, sid in enumerate(sids):
        e, rp, tl = truth[sid], reps[sid], timeline[sid]
        elements, elem = rp["elements"], P.GNN_ELEM[sid]
        faults = R.faults_of(e)
        den[i] = len(faults)
        for m in P.METHODS:
            pages = [dict(p, kind="ids") if m == "Agent" else dict(p, kind="scores", ranking=elem[p["t"]]) for p in tl["pages"][m]]
            for f in faults:
                best = 0.0
                for p in pages:
                    if not f["t_start"] <= p["t"] < f["t_end"] + R.DETECT_TAIL:
                        continue
                    sc = R.ranking_vector(p, elements)
                    for g in e["faults"]:
                        r = g["root_cause"]
                        if g is not f and g["t_start"] <= p["t"] < g["t_end"] + LAG and r in elements and r != f["root_cause"]:
                            sc[elements.index(r)] = -np.inf
                    best = max(best, localisation_credit(sc, elements, f["root_cause"])["top3"])
                num[m][i] += best
    boots = {m: ratio_boot(W, num[m], den) for m in P.METHODS}
    out = {"definition": "share of observable faults with some page in [start, end + 12 steps) whose top 3 names the root "
                         "(tie-aware credit, best page counts); other faults' roots active at the page are dropped first",
           "n_faults": int(den.sum())}
    for m in P.METHODS:
        out[m] = {"value": round(float(boots[m][0]), 4), "ci": ci(boots[m][1:]), "numerator": round(float(num[m].sum()), 4)}
    for a, b in PAIRS + (("GNN+CL", "GNN"),):
        out[f"{a} minus {b}"] = verdict(boots[a] - boots[b])
    return out


def run_record(run_dir):
    decs = read_jsonl(run_dir / "decisions.jsonl")
    log = read_jsonl(run_dir / "agent_log.jsonl")
    rec = [i for i, r in enumerate(decs) if r.get("recovered_from_log")]
    first = [r for r in log if r["round"] == 1]
    return {"decisions": len(decs), "agent_log_wakes": len(log), "recovered_from_log_flags": len(rec),
            "decisions_written_before_first_recovered": rec[0] if rec else None,
            "first_decisions": len(first), "first_decisions_cost_usd": round(sum(r["cost_usd"] for r in first), 4),
            "rechecks": len(log) - len(first), "rechecks_cost_usd": round(sum(r["cost_usd"] for r in log if r["round"] == 2), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="test")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    run_dir = P.OUT / args.run
    split = "test" if args.run == "test" else "val"
    sids = [json.loads(l)["scenario_id"] for l in (run_dir / "replay.jsonl").read_text().splitlines()]
    W = boot_weights(len(sids), 1000, 42)   # the same scenario resamples as the frozen summary
    per_fault = json.loads((run_dir / "per_fault.json").read_text())
    out = {"run": args.run, "pre_registered": False, "n_scenarios": len(sids), "n_boot": 1000, "seed": 42,
           "matched_faults": matched_faults(per_fault, sids, W),
           "correct_root_detection": correct_root_detection(run_dir, split, cfg, sids, W),
           "run_record": run_record(run_dir)}
    (run_dir / "descriptive.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
