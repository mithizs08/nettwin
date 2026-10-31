"""Phase 6: LLM root-cause agent (conditions A0, A1, A2) on incident windows, scored with the shared harness's
localisation credit next to B1, B2, B3, B3+ and the frozen GNN on the same windows.

    python scripts/phase6_agent.py --split val --limit 10 --model dev --name smoke     # smoke test
    python scripts/phase6_agent.py --estimate results/phase6/val_smoke                  # cost estimate from a run
    python scripts/phase6_agent.py --split val --model eval --name full                 # full validation eval

Single process, low memory. Development and debugging on validation only. `--split test` refuses to run unless the
Phase 6 test pre-registration is in docs/decision_log.md, the frozen GNN and baseline hashes match
results/phase5_frozen.json, and the output folder does not exist yet (test is run once).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import yaml

from nettwin.agent.eval import sample_incidents, score_agent, score_reference, summarise
from nettwin.agent.loop import run_incident
from nettwin.agent.rag import RunbookIndex
from nettwin.agent.tools import Incident
from nettwin.baselines import index_for
from nettwin.gnn.method import gnn_method, sha256
from nettwin.observed import freeze_cfg, inventory, load_observed
from nettwin.splits import guard

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("with_status", "status_blind")
CONDITIONS = ("A0", "A1", "A2")
BASELINES = ("B1", "B2", "B3", "B3+")
PREREG = "### 2026-10-31: Phase 6 pre-registration"
SAMPLE_SEED = 6006
N_PER_TYPE, N_PER_DISTRACTOR = 20, 15
PAIRS = [("A1", "A0"), ("A2", "A1"), ("A2", "A0"), ("A2", "GNN"), ("A2", "B3"), ("A2", "B3+"), ("A2", "B2"), ("A1", "B3"), ("A1", "GNN")]


def load_env():
    """ANTHROPIC_API_KEY from .env (never printed)."""
    import os
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def frozen_assets(cfg_s):
    frozen = json.loads((ROOT / "results" / "phase5_frozen.json").read_text())
    for rel, h in frozen["files"].items():
        if sha256(ROOT / rel) != h:
            sys.exit(f"refusing: sha256 mismatch for {rel}")
    mdir = ROOT / "data" / "models" / "phase5"
    cal = json.loads((mdir / "frozen" / "calibration.json").read_text())["variants"]
    baselines = {v: pickle.loads((mdir / "frozen" / f"baselines_{v}.pkl").read_bytes()) for v in VARIANTS}
    seeds = [m for m in frozen["models"] if not m.endswith("-mlp")]
    gnns = {k: gnn_method(k, (mdir / f"{k}.pt").read_bytes(), cfg_s) for k in seeds}
    return cal, baselines, gnns, seeds


def estimate(run_dir, acfg, plan):
    """Cost per incident and condition from a run's log, priced for each configured model on the same token counts."""
    rows = [json.loads(l) for l in (run_dir / "agent_log.jsonl").read_text().splitlines()]
    df = pd.DataFrame([{"condition": r["condition"], **r["usage"], "cost": r["cost_usd"], "seconds": r["seconds"],
                        "calls": r["n_tool_calls"]} for r in rows])
    per = df.groupby("condition").mean(numeric_only=True)
    out = {"from_run": str(run_dir.relative_to(ROOT)), "runs": len(df), "mean_per_incident": per.round(4).to_dict(orient="index"),
           "estimates_usd": {}}
    for model, p in acfg["pricing_usd_per_mtok"].items():
        c = ((per.input_tokens * p["input"] + per.cache_creation_input_tokens * p["cache_write_5m"]
              + per.cache_read_input_tokens * p["cache_read"] + per.output_tokens * p["output"]) / 1e6)
        out["estimates_usd"][model] = {"per_incident_by_condition": c.round(5).to_dict(),
                                       **{name: round(float(c.sum() * n), 2) for name, n in plan.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--model", default="dev", choices=["dev", "eval"])
    ap.add_argument("--name", default="run")
    ap.add_argument("--limit", type=int, default=None, help="first N incidents only (smoke test)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--estimate", default=None, help="print the cost estimate from an existing run folder and exit")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    acfg = cfg["agent"]
    if args.estimate:
        n_inc = len(EASY_HARD) * N_PER_TYPE + 2 * N_PER_DISTRACTOR
        plan = {f"full_{s}_{len(VARIANTS)}_variants ({n_inc} incidents x {len(VARIANTS)} variants)": n_inc * len(VARIANTS) for s in ("val", "test")}
        print(json.dumps(estimate(ROOT / args.estimate, acfg, plan), indent=1))
        return
    split = args.split
    out_dir = ROOT / "results" / "phase6" / f"{split}_{args.name}"
    if split == "test":
        if PREREG not in (ROOT / "docs" / "decision_log.md").read_text():
            sys.exit("refusing: the Phase 6 test pre-registration is not in docs/decision_log.md")
        if out_dir.exists():
            sys.exit(f"refusing: {out_dir.relative_to(ROOT)} exists (the test eval is run once)")
    G = lambda p: guard(p, {split})  # noqa: E731
    load_env()
    model = acfg["dev_model"] if args.model == "dev" else acfg["eval_model"]
    cfg_s = freeze_cfg(cfg)
    root = ROOT / cfg["dataset"]["root"]
    cal, baselines, gnns, seeds = frozen_assets(cfg_s)
    rb = RunbookIndex()
    variants = args.variants.split(",")
    conditions = args.conditions.split(",")
    entries = json.loads(G(root / "truth" / split / "with_status" / "manifest.json").read_text())
    incidents = sample_incidents(entries, N_PER_TYPE, N_PER_DISTRACTOR, SAMPLE_SEED)
    if args.limit:
        rng = np.random.default_rng(SAMPLE_SEED + 1)
        incidents = [incidents[i] for i in sorted(rng.choice(len(incidents), args.limit, replace=False))]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "incidents.json").write_text(json.dumps(incidents, indent=1) + "\n")
    rec_path = out_dir / "records.jsonl"
    done = set()
    if rec_path.exists():
        for l in rec_path.read_text().splitlines():
            r = json.loads(l)
            done.add((r["incident_id"], r["variant"], r["method"]))
    client = anthropic.Anthropic(max_retries=4)
    t0 = time.time()
    with open(out_dir / "agent_log.jsonl", "a") as log, open(rec_path, "a") as rec:
        for inc in incidents:
            for v in variants:
                obs = load_observed(G(root / "observable" / split / v / inc["scenario_id"]))
                kg = inventory(obs.topology, cfg_s)
                ix = index_for(obs, kg)
                elements = ix.elements

                def write(method, score, extra=None):
                    rec.write(json.dumps({"incident_id": inc["incident_id"], "variant": v, "method": method, **score, **(extra or {})}) + "\n")
                    rec.flush()

                for name in BASELINES:
                    if (inc["incident_id"], v, name) not in done:
                        m = baselines[v][name]
                        write(name, score_reference(inc, m, m.score(obs, kg, ix), cal[v][name]["threshold"], elements))
                if (inc["incident_id"], v, "GNN") not in done:
                    per = [score_reference(inc, gnns[k], gnns[k].score(obs, kg, ix), cal[v][k]["threshold"], elements) for k in seeds]
                    write("GNN", {k: float(np.mean([p[k] for p in per])) for k in per[0]}, {"per_seed": per})
                for cond in conditions:
                    if (inc["incident_id"], v, cond) in done:
                        continue
                    incident = Incident(obs, kg, inc["t0"], inc["t1"], cfg_s, gnn=gnns[seeds[0]].model, runbooks=rb)
                    incident.incident_id = f"{inc['incident_id']}|{v}"
                    res = run_incident(client, incident, cond, model, acfg, status_polling=(v == "with_status"), log=log)
                    write(cond, score_agent(inc, res["diagnosis"], elements),
                          {"cost_usd": res["cost_usd"], "seconds": res["seconds"], "n_tool_calls": res["n_tool_calls"], "stop": res["stop"],
                           "ungrounded_items": res["groundedness"]["n_ungrounded_items"], "evidence_items": res["groundedness"]["n_items"],
                           "fault_type": (res["diagnosis"] or {}).get("fault_type"), "root_cause": (res["diagnosis"] or {}).get("root_cause")})
                    print(f"{inc['incident_id']} {v} {cond}: {res['stop']}, {res['n_tool_calls']} calls, ${res['cost_usd']:.4f}, {res['seconds']}s", flush=True)
    recs = [json.loads(l) for l in rec_path.read_text().splitlines()]
    summary = {"split": split, "model": model, "n_incidents": len(incidents), "seconds": round(time.time() - t0, 1), "variants": {}}
    for v in variants:
        by = {}
        for m in list(BASELINES) + ["GNN"] + conditions:
            d = {r["incident_id"]: r for r in recs if r["variant"] == v and r["method"] == m}
            if len(d) == len(incidents):
                by[m] = [d[i["incident_id"]] for i in incidents]
        table, comps = summarise(incidents, by, pairs=[p for p in PAIRS if p[0] in by and p[1] in by])
        agent = [r for r in recs if r["variant"] == v and r["method"] in conditions]
        cost = {c: {"mean_usd": round(float(np.mean([r["cost_usd"] for r in agent if r["method"] == c])), 5),
                    "mean_seconds": round(float(np.mean([r["seconds"] for r in agent if r["method"] == c])), 2),
                    "ungrounded_item_rate": round(sum(r["ungrounded_items"] for r in agent if r["method"] == c)
                                                  / max(1, sum(r["evidence_items"] for r in agent if r["method"] == c)), 4),
                    "no_submission": sum(r["stop"] != "submitted" for r in agent if r["method"] == c)}
                for c in conditions if any(r["method"] == c for r in agent)}
        summary["variants"][v] = {"metrics": table, "comparisons": comps, "agent_cost_latency_groundedness": cost}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"wrote {out_dir.relative_to(ROOT)}/summary.json")


EASY_HARD = ("link_down", "node_down", "link_flapping", "link_degradation", "demand_surge", "routing_misconfig")

if __name__ == "__main__":
    main()
