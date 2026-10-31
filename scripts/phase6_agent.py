"""Phase 6: LLM root-cause agent (conditions A0, A1, A2) on incident windows, scored with the shared harness's
localisation credit next to B1, B2, B3, B3+ and the frozen GNN on the same windows.

    python scripts/phase6_agent.py --split val --limit 10 --model dev --name smoke     # smoke test
    python scripts/phase6_agent.py --estimate results/phase6/val_smoke                  # cost estimate from a run
    python scripts/phase6_agent.py --split val --model eval --name full                 # full validation eval
    python scripts/phase6_agent.py --split test --model eval                            # test, once per model

Development and debugging on validation only. The agent runs of one incident (both variants, every condition) go in
parallel threads (`--threads`); references are scored first in the main thread. `--split test` refuses to run unless
the Phase 6 test pre-registration is in docs/decision_log.md, the frozen GNN and baseline hashes match
results/phase5_frozen.json, and the agent hashes match results/phase6_frozen.json; it also refuses `--limit`, the dev
model, a subset of the pre-registered conditions or variants, and a second test run for the same model under any name
(`nettwin.agent.freeze.test_run_refusal`). Every run records the agent `code_sha256`.
"""
import argparse
import json
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import yaml

from nettwin.agent import load_agent_config
from nettwin.agent.eval import overlaps_logged_change, sample_incidents, score_agent, score_reference, summarise
from nettwin.agent.freeze import agent_hashes, test_out_dir, test_run_refusal
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
# GNN: mean of the 3 frozen seeds. GNN-s0: seed 0 alone, the checkpoint A2's tool uses (Phase 6 audit S7). GNN+CL: the
# GNN with a change-log rule, no fault claimed when the change log shows a drain or planned outage overlapping the window
# (Phase 6 audit S2); its localisation is the GNN's.
REFERENCES = BASELINES + ("GNN", "GNN-s0", "GNN+CL")
PREREG = "### 2026-10-31: Phase 6 pre-registration"
SAMPLE_SEED = 6006
N_PER_TYPE, N_PER_DISTRACTOR = 20, 15
N_CHANGE_OVERLAP = 40           # fault incidents overlapping a logged drain or planned outage, main sample included
# API spend not in any kept log: the smoke attempts deleted before the final smoke run (2026-10-31), an upper bound.
UNLOGGED_SPEND_USD = 0.30
PAIRS = [("A1", "A0"), ("A2", "A1"), ("A2", "A0"), ("A2", "GNN"), ("A2", "B3"), ("A2", "B3+"), ("A2", "B2"), ("A1", "B3"), ("A1", "GNN"),
         ("A2", "GNN+CL"), ("A1", "GNN+CL"), ("A2", "GNN-s0"), ("GNN+CL", "GNN")]


def load_env():
    """ANTHROPIC_API_KEY from .env (never printed)."""
    import os
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def logged_spend():
    """Whole-project Phase 6 API spend so far: every kept agent log plus the unlogged allowance."""
    total = UNLOGGED_SPEND_USD
    for f in (ROOT / "results" / "phase6").glob("*/agent_log.jsonl"):
        total += sum(json.loads(l)["cost_usd"] for l in f.read_text().splitlines())
    return total


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


class LockedLog:
    """One file shared by threads: each write is one whole line under a lock."""
    def __init__(self, f):
        self.f, self.lock = f, threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)
            self.f.flush()

    def flush(self):
        pass


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
    ap.add_argument("--model", default="dev", choices=["dev", "eval", "secondary"])
    ap.add_argument("--name", default="run", help="output folder suffix on validation (test uses test_<model>)")
    ap.add_argument("--limit", type=int, default=None, help="first N incidents only (smoke test; not on test)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--conditions", default=None, help="default: A0,A1,A2 (eval and dev models) or the secondary conditions")
    ap.add_argument("--threads", type=int, default=6, help="parallel agent runs within one incident")
    ap.add_argument("--resume", action="store_true", help="test only: continue the unfinished test run of this model")
    ap.add_argument("--estimate", default=None, help="print the cost estimate from an existing run folder and exit")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    acfg = load_agent_config(ROOT)
    if args.estimate:
        n_inc = len(EASY_HARD) * N_PER_TYPE + 2 * N_PER_DISTRACTOR
        plan = {f"full_{s}_{len(VARIANTS)}_variants ({n_inc} incidents x {len(VARIANTS)} variants)": n_inc * len(VARIANTS) for s in ("val", "test")}
        print(json.dumps(estimate(ROOT / args.estimate, acfg, plan), indent=1))
        return
    split = args.split
    model = acfg[{"dev": "dev_model", "eval": "eval_model", "secondary": "secondary_eval_model"}[args.model]]
    required = list(acfg["secondary_conditions"]) if args.model == "secondary" else list(CONDITIONS)
    conditions = args.conditions.split(",") if args.conditions else required
    variants = args.variants.split(",")
    if args.model == "secondary" and not set(conditions) <= set(acfg["secondary_conditions"]):
        sys.exit(f"the secondary model runs {acfg['secondary_conditions']} only")
    hashes = agent_hashes(ROOT, acfg)
    frozen_path = ROOT / "results" / "phase6_frozen.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    if split == "test":
        if PREREG not in (ROOT / "docs" / "decision_log.md").read_text():
            sys.exit("refusing: the Phase 6 test pre-registration is not in docs/decision_log.md")
        why = test_run_refusal(ROOT, frozen, hashes, model, args.model, args.limit, conditions, variants, required, args.resume)
        if why:
            sys.exit(f"refusing: {why}")
        out_dir = test_out_dir(ROOT, model)
    else:
        out_dir = ROOT / "results" / "phase6" / f"{split}_{args.name}"
    frozen_match = frozen is not None and frozen["code_sha256"] == hashes["code_sha256"]
    print(f"agent code_sha256 {hashes['code_sha256'][:12]} ({'matches' if frozen_match else 'does NOT match'} results/phase6_frozen.json)", flush=True)
    G = lambda p: guard(p, {split})  # noqa: E731
    budget = acfg["project_budget_usd"]
    spent = logged_spend()
    print(f"project API spend so far: ${spent:.2f} of ${budget}", flush=True)
    load_env()
    cfg_s = freeze_cfg(cfg)
    root = ROOT / cfg["dataset"]["root"]
    cal, baselines, gnns, seeds = frozen_assets(cfg_s)
    rb = RunbookIndex()
    entries = json.loads(G(root / "truth" / split / "with_status" / "manifest.json").read_text())
    incidents = sample_incidents(entries, N_PER_TYPE, N_PER_DISTRACTOR, SAMPLE_SEED, N_CHANGE_OVERLAP)
    if args.limit:
        rng = np.random.default_rng(SAMPLE_SEED + 1)
        incidents = [incidents[i] for i in sorted(rng.choice(len(incidents), args.limit, replace=False))]
    out_dir.mkdir(parents=True, exist_ok=True)
    if split == "test" and not (out_dir / "LOCK.json").exists():
        (out_dir / "LOCK.json").write_text(json.dumps({"model": model, "code_sha256": hashes["code_sha256"], "argv": sys.argv[1:],
                                                       "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1) + "\n")
    (out_dir / "incidents.json").write_text(json.dumps(incidents, indent=1) + "\n")
    rec_path = out_dir / "records.jsonl"
    done = set()
    if rec_path.exists():
        for l in rec_path.read_text().splitlines():
            r = json.loads(l)
            done.add((r["incident_id"], r["variant"], r["method"]))
    client = anthropic.Anthropic(max_retries=6)
    t0 = time.time()
    tag = {"code_sha256": hashes["code_sha256"]}
    with open(out_dir / "agent_log.jsonl", "a") as logf, open(rec_path, "a") as recf, ThreadPoolExecutor(args.threads) as pool:
        log, rec = LockedLog(logf), LockedLog(recf)

        def write(inc, v, method, score, extra=None):
            rec.write(json.dumps({"incident_id": inc["incident_id"], "variant": v, "method": method, **score, **(extra or {}), **tag}) + "\n")

        def agent_run(inc, v, obs, kg, elements, cond):
            incident = Incident(obs, kg, inc["t0"], inc["t1"], cfg_s, gnn=gnns[seeds[0]].model, runbooks=rb)
            incident.incident_id = f"{inc['incident_id']}|{v}"
            res = run_incident(client, incident, cond, model, acfg, status_polling=(v == "with_status"), log=log, log_extra=tag)
            g = res["groundedness"]
            write(inc, v, cond, score_agent(inc, res["diagnosis"], elements),
                  {"cost_usd": res["cost_usd"], "seconds": res["seconds"], "n_tool_calls": res["n_tool_calls"],
                   "n_refused_calls": res["n_refused_calls"], "stop": res["stop"], "forced_submit": res["forced_submit"],
                   "ungrounded_items": g["n_ungrounded_items"], "ungrounded_items_strict": g["n_ungrounded_items_strict"],
                   "evidence_items": g["n_items"],
                   "fault_type": (res["diagnosis"] or {}).get("fault_type"), "root_cause": (res["diagnosis"] or {}).get("root_cause")})
            print(f"{inc['incident_id']} {v} {cond}: {res['stop']}, {res['n_tool_calls']} calls, ${res['cost_usd']:.4f}, {res['seconds']}s", flush=True)
            return res["cost_usd"]

        for inc in incidents:
            jobs = []
            for v in variants:
                obs = load_observed(G(root / "observable" / split / v / inc["scenario_id"]))
                kg = inventory(obs.topology, cfg_s)
                ix = index_for(obs, kg)
                elements = ix.elements
                for name in BASELINES:
                    if (inc["incident_id"], v, name) not in done:
                        m = baselines[v][name]
                        write(inc, v, name, score_reference(inc, m, m.score(obs, kg, ix), cal[v][name]["threshold"], elements))
                if not {(inc["incident_id"], v, m) for m in ("GNN", "GNN-s0", "GNN+CL")} <= done:
                    per = [score_reference(inc, gnns[k], gnns[k].score(obs, kg, ix), cal[v][k]["threshold"], elements) for k in seeds]
                    mean = lambda rows: {k: float(np.mean([p[k] for p in rows])) for k in rows[0]}  # noqa: E731
                    changed = overlaps_logged_change([e for e in obs.events if e.get("type") == "maintenance"], inc["t0"], inc["t1"])
                    with_rule = [p | {"claims_fault": p["claims_fault"] and not changed} for p in per]
                    for name, score, extra in (("GNN", mean(per), {"per_seed": per}), ("GNN-s0", per[0], {"seed": seeds[0]}),
                                               ("GNN+CL", mean(with_rule), {"change_log_overlap": changed})):
                        if (inc["incident_id"], v, name) not in done:
                            write(inc, v, name, score, extra)
                for cond in conditions:
                    if (inc["incident_id"], v, cond) not in done:
                        jobs.append((v, obs, kg, elements, cond))
            if jobs and spent >= budget - 1.0:
                sys.exit(f"stopping: project API spend ${spent:.2f} is within $1 of the ${budget} budget")
            for f in [pool.submit(agent_run, inc, *j) for j in jobs]:
                spent += f.result()
    recs = [json.loads(l) for l in rec_path.read_text().splitlines()]
    summary = {"split": split, "model": model, "n_incidents": len(incidents), "seconds": round(time.time() - t0, 1),
               "agent_hashes": hashes, "matches_phase6_frozen": frozen_match, "variants": {}}
    for v in variants:
        by = {}
        for m in list(REFERENCES) + conditions:
            d = {r["incident_id"]: r for r in recs if r["variant"] == v and r["method"] == m}
            if len(d) == len(incidents):
                by[m] = [d[i["incident_id"]] for i in incidents]
        table, comps = summarise(incidents, by, pairs=[p for p in PAIRS if p[0] in by and p[1] in by])
        agent = [r for r in recs if r["variant"] == v and r["method"] in conditions]

        def rate(c, key):
            rs = [r for r in agent if r["method"] == c]
            return round(sum(r[key] for r in rs) / max(1, sum(r["evidence_items"] for r in rs)), 4)

        cost = {c: {"mean_usd": round(float(np.mean([r["cost_usd"] for r in agent if r["method"] == c])), 5),
                    "mean_seconds": round(float(np.mean([r["seconds"] for r in agent if r["method"] == c])), 2),
                    "ungrounded_item_rate": rate(c, "ungrounded_items"),
                    "ungrounded_item_rate_strict": rate(c, "ungrounded_items_strict"),
                    "forced_submissions": sum(r["forced_submit"] for r in agent if r["method"] == c),
                    "refused_calls": sum(r["n_refused_calls"] for r in agent if r["method"] == c),
                    "max_tool_calls": max(r["n_tool_calls"] for r in agent if r["method"] == c),
                    "no_submission": sum(r["stop"] != "submitted" for r in agent if r["method"] == c)}
                for c in conditions if any(r["method"] == c for r in agent)}
        summary["variants"][v] = {"metrics": table, "comparisons": comps, "agent_cost_latency_groundedness": cost}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(f"wrote {out_dir.relative_to(ROOT)}/summary.json")


EASY_HARD = ("link_down", "node_down", "link_flapping", "link_degradation", "demand_surge", "routing_misconfig")

if __name__ == "__main__":
    main()
