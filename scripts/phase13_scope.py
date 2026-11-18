"""Phase 13: a change-scope tool for the agent, scored as a pre-registered follow-up on a fresh incident sample.

    python scripts/phase13_scope.py --tool-dev                       # validation: the tool alone, no API (development)
    python scripts/phase13_scope.py --split val --limit 10 --name smoke
    python scripts/phase13_scope.py --estimate results/phase13/val_smoke
    python scripts/phase13_scope.py --split val --name dev           # the validation run of record
    python scripts/phase13_scope.py --freeze                         # writes results/phase13_frozen.json
    python scripts/phase13_scope.py --split test                     # test, once, through the frozen path

A2 is the frozen Phase 6 condition (`nettwin.agent.loop.run_incident`, hash-checked against results/phase6_frozen.json);
A3 is A2 plus `check_change_scope` (`nettwin.scope_agent`). Both run on Haiku 5.5 with status events (with_status).
The references (GNN mean of 3 seeds, GNN-s0, GNN + change-log rule, and GNN + change-log rule + scope check) get the
same windows. The incidents were never drawn in Phase 6 (`nettwin.scope_agent.sample_phase13`).

Spend guard: an agent run starts only if the Phase 13 logged spend plus RESERVE_USD for every run in flight (this one
included) stays within PHASE_BUDGET_USD, and the project stays $1 under its budget.

`--split test` refuses unless the Phase 13 pre-registration is in docs/decision_log.md, results/phase13_frozen.json
exists and every file in it matches, the Phase 6 agent code still matches results/phase6_frozen.json, the Phase 5 frozen
assets match, and results/phase13/test has no summary (an unfinished run continues with --resume); --limit is refused.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase13"
FROZEN = ROOT / "results" / "phase13_frozen.json"
PREREG = "### 2026-11-18: Phase 13 pre-registration"
VARIANT = "with_status"
SAMPLE_SEED = 1313
SIZES = {"val": {"n_change_overlap": 30, "n_other_per_type": 3, "n_distractor": 10},
         "test": {"n_change_overlap": 40, "n_other_per_type": 5, "n_distractor": 15}}
PHASE_BUDGET_USD = 2.00
RESERVE_USD = 0.08           # set aside per agent run in flight: an upper bound on one Haiku run under the Phase 6 budgets
UNLOGGED_SPEND_USD = 0.30    # Phase 6 allowance, as in scripts/phase6_agent.py
CONDITIONS = ("A2", "A3")
REFERENCES = ("GNN", "GNN-s0", "GNN+CL", "GNN+CL+scope")
PAIRS = (("A3", "A2"), ("A3", "GNN"), ("A3", "GNN+CL"), ("A2", "GNN"), ("A2", "GNN+CL"), ("GNN+CL+scope", "GNN+CL"),
         ("GNN+CL+scope", "GNN"), ("A3", "GNN+CL+scope"))
FROZEN_FILES = ("src/nettwin/change_scope.py", "src/nettwin/scope_agent.py", "scripts/phase13_scope.py", "src/nettwin/kg.py",
                "src/nettwin/alerts.py", "src/nettwin/evaluation.py", "src/nettwin/observed.py", "agent.yaml")
P6_INCIDENTS = {"val": "results/phase6/val_haiku_v2/incidents.json", "test": "results/phase6/test_claude-haiku-5-5/incidents.json"}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_env():
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _sum_cost(files):
    return sum(json.loads(l)["cost_usd"] for f in files for l in f.read_text().splitlines() if l.strip())


def phase_spend():
    return _sum_cost(OUT.glob("*/agent_log.jsonl"))


def project_spend():
    res = ROOT / "results"
    return (UNLOGGED_SPEND_USD + _sum_cost(res.glob("phase6/*/agent_log.jsonl")) + _sum_cost(res.glob("phase7/*agent_log.jsonl"))
            + _sum_cost(res.glob("phase11/*/log.jsonl")) + _sum_cost(res.glob("phase12/*/agent_log.jsonl")) + phase_spend())


class LockedLog:
    def __init__(self, f):
        self.f, self.lock = f, threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)
            self.f.flush()

    def flush(self):
        pass


def frozen_gnns(cfg_s):
    from nettwin.gnn.method import gnn_method
    frozen = json.loads((ROOT / "results" / "phase5_frozen.json").read_text())
    for rel, h in frozen["files"].items():
        if sha256(ROOT / rel) != h:
            sys.exit(f"refusing: sha256 mismatch for {rel}")
    mdir = ROOT / "data" / "models" / "phase5"
    cal = json.loads((mdir / "frozen" / "calibration.json").read_text())["variants"][VARIANT]
    seeds = [m for m in frozen["models"] if not m.endswith("-mlp")]
    return cal, {k: gnn_method(k, (mdir / f"{k}.pt").read_bytes(), cfg_s) for k in seeds}, seeds


def phase6_unchanged(acfg):
    from nettwin.agent.freeze import agent_hashes
    return agent_hashes(ROOT, acfg)["code_sha256"] == json.loads((ROOT / "results" / "phase6_frozen.json").read_text())["code_sha256"]


def incidents_for(split, cfg):
    from nettwin.scope_agent import sample_phase13
    from nettwin.splits import guard
    root = ROOT / cfg["dataset"]["root"]
    entries = json.loads(guard(root / "truth" / split / VARIANT / "manifest.json", {split}).read_text())
    p6 = json.loads((ROOT / P6_INCIDENTS[split]).read_text())
    return sample_phase13(entries, p6, SAMPLE_SEED, **SIZES[split])


# ---------------------------------------------------------------- tool-only development (validation, no API)

def tool_dev(cfg):
    """The tool alone on every eligible validation window of the first 250 validation scenarios: share of
    change-overlapping fault windows and of change-only (distractor) windows where it lists an unexplained symptom, for
    the chosen settings and the grid it was chosen from. Validation only."""
    import itertools
    from nettwin import change_scope as CS
    from nettwin.agent.eval import _overlaps, overlaps_logged_change
    from nettwin.agent.tools import Incident
    from nettwin.evaluation import LAG, MAINT_TAIL, WARMUP
    from nettwin.observed import freeze_cfg, inventory, load_observed
    from nettwin.splits import guard
    cfg_s, root = freeze_cfg(cfg), ROOT / cfg["dataset"]["root"]
    entries = json.loads(guard(root / "truth" / "val" / VARIANT / "manifest.json", {"val"}).read_text())[:250]
    incs = []
    for e in entries:
        items = [("change_overlap_fault", f["type"], f["t_start"], f["t_end"]) for f in e["faults"]
                 if f["impact"]["observable"] and f["t_start"] >= WARMUP and overlaps_logged_change(e["maintenance"], f["t_start"], f["t_end"])]
        items += [("change_only", m["action"], m["t_start"], m["t_end"]) for m in e["maintenance"]
                  if m["action"] in ("drain", "down") and m["t_start"] >= WARMUP
                  and not any(_overlaps(m["t_start"], m["t_end"] + MAINT_TAIL, g["t_start"], g["t_end"] + LAG) for g in e["faults"])]
        if items:
            obs = load_observed(guard(root / "observable" / "val" / VARIANT / e["scenario_id"], {"val"}))
            kg = inventory(obs.topology, cfg_s)
            incs += [(it, Incident(obs, kg, it[2], it[3], cfg_s)) for it in items]
    chosen = (CS.SERVICE_LOSS_EXCESS, CS.UTIL_SHIFT, CS.UTIL_SHIFT_REL)
    grid = []
    for exc, us, ur in itertools.product((1, 2), (0.05, 0.08, 0.10), (0.15, 0.30)):
        CS.SERVICE_LOSS_EXCESS, CS.UTIL_SHIFT, CS.UTIL_SHIFT_REL = exc, us, ur
        flags = {}
        for it, inc in incs:
            flags.setdefault(it[0], []).append(CS.change_scope(inc)["n_unexplained"] > 0)
            if it[0] == "change_overlap_fault":
                flags.setdefault(f"type={it[1]}", []).append(flags[it[0]][-1])
        grid.append({"service_loss_excess": exc, "util_shift": us, "util_shift_rel": ur, "chosen": (exc, us, ur) == chosen,
                     **{k: {"n": len(v), "share_flagged": round(float(np.mean(v)), 3)} for k, v in sorted(flags.items())}})
        print(grid[-1]["service_loss_excess"], us, ur, {k: v["share_flagged"] for k, v in grid[-1].items() if isinstance(v, dict)}, flush=True)
    CS.SERVICE_LOSS_EXCESS, CS.UTIL_SHIFT, CS.UTIL_SHIFT_REL = chosen
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "val_tool_dev.json").write_text(json.dumps({"split": "val", "scenarios": len(entries), "windows": len(incs),
                                                       "chosen": dict(zip(("service_loss_excess", "util_shift", "util_shift_rel"), chosen)),
                                                       "grid": grid}, indent=1) + "\n")
    print("wrote results/phase13/val_tool_dev.json")


# ---------------------------------------------------------------- run

def run(args, cfg, acfg):
    import anthropic
    from nettwin.agent.eval import overlaps_logged_change, score_agent, score_reference
    from nettwin.agent.loop import run_incident
    from nettwin.agent.rag import RunbookIndex
    from nettwin.agent.tools import Incident
    from nettwin.baselines import index_for
    from nettwin.observed import freeze_cfg, inventory, load_observed
    from nettwin.scope_agent import ScopeIncident, run_incident_a3
    from nettwin.splits import guard
    split = args.split
    if not phase6_unchanged(acfg):
        sys.exit("refusing: the Phase 6 agent code no longer matches results/phase6_frozen.json (A2 must be the frozen A2)")
    if split == "test":
        frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else None
        problems = [m for m, bad in (
            ("the Phase 13 pre-registration is not in docs/decision_log.md", PREREG not in (ROOT / "docs" / "decision_log.md").read_text()),
            ("results/phase13_frozen.json is missing", frozen is None),
            ("a frozen file changed", frozen is not None and any(sha256(ROOT / k) != v for k, v in frozen["files"].items())),
            ("results/phase13/test already exists (test is run once; --resume continues only an unfinished run)",
             (OUT / "test").exists() and not (args.resume and not (OUT / "test" / "summary.json").exists())),
            ("--resume needs an unfinished test run", args.resume and not (OUT / "test").exists()),
            ("--limit is not allowed on test", bool(args.limit)),
        ) if bad]
        if problems:
            sys.exit("refusing the test run: " + "; ".join(problems))
        out_dir = OUT / "test"
    else:
        out_dir = OUT / f"val_{args.name}"
    model = acfg["eval_model"]
    cfg_s = freeze_cfg(cfg)
    root = ROOT / cfg["dataset"]["root"]
    G = lambda p: guard(p, {split})  # noqa: E731
    cal, gnns, seeds = frozen_gnns(cfg_s)
    rb = RunbookIndex()
    incidents = incidents_for(split, cfg)
    if args.limit:
        rng = np.random.default_rng(SAMPLE_SEED + 1)
        incidents = [incidents[i] for i in sorted(rng.choice(len(incidents), args.limit, replace=False))]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "incidents.json").write_text(json.dumps(incidents, indent=1) + "\n")
    rec_path = out_dir / "records.jsonl"
    done = {(r["incident_id"], r["method"]) for r in map(json.loads, rec_path.read_text().splitlines())} if rec_path.exists() else set()
    spent = {"phase": phase_spend(), "project": project_spend(), "inflight": 0.0}
    budget = acfg["project_budget_usd"]
    print(f"{len(incidents)} incidents; Phase 13 spend so far ${spent['phase']:.3f} of ${PHASE_BUDGET_USD:.2f}; "
          f"project ${spent['project']:.2f} of ${budget}", flush=True)
    load_env()
    client = anthropic.Anthropic(max_retries=6)
    glock, stopped = threading.Lock(), threading.Event()
    hashes = {k: sha256(ROOT / k) for k in FROZEN_FILES}
    tag = {"phase13_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]}
    t_run = time.time()
    with open(out_dir / "agent_log.jsonl", "a") as logf, open(rec_path, "a") as recf, ThreadPoolExecutor(args.threads) as pool:
        log, rec = LockedLog(logf), LockedLog(recf)

        def write(inc, method, score, extra=None):
            rec.write(json.dumps({"incident_id": inc["incident_id"], "method": method, **score, **(extra or {}), **tag}) + "\n")

        def agent(inc, obs, kg, elements, cond):
            with glock:
                if (spent["phase"] + spent["inflight"] + RESERVE_USD > PHASE_BUDGET_USD
                        or spent["project"] + spent["inflight"] + RESERVE_USD > budget - 1.0):
                    stopped.set()
                    return
                spent["inflight"] += RESERVE_USD
            try:
                cls = ScopeIncident if cond == "A3" else Incident
                x = cls(obs, kg, inc["t0"], inc["t1"], cfg_s, gnn=gnns[seeds[0]].model, runbooks=rb)
                x.incident_id = inc["incident_id"]
                extra = {**tag, "condition": cond}
                res = (run_incident_a3(client, x, model, acfg, status_polling=True, log=log, log_extra=extra) if cond == "A3"
                       else run_incident(client, x, "A2", model, acfg, status_polling=True, log=log, log_extra=extra))
            finally:
                with glock:
                    spent["inflight"] -= RESERVE_USD
            with glock:
                spent["phase"] += res["cost_usd"]
                spent["project"] += res["cost_usd"]
            g = res["groundedness"]
            d = res["diagnosis"] or {}
            write(inc, cond, score_agent(inc, res["diagnosis"], elements),
                  {"cost_usd": res["cost_usd"], "seconds": res["seconds"], "n_tool_calls": res["n_tool_calls"],
                   "n_refused_calls": res["n_refused_calls"], "stop": res["stop"], "forced_submit": res["forced_submit"],
                   "called_scope": any(c["tool"] == "check_change_scope" and not c["refused"] for c in res["calls"]),
                   "called_change_log": any(c["tool"] == "get_change_log" and not c["refused"] for c in res["calls"]),
                   "ungrounded_items": g["n_ungrounded_items"], "ungrounded_items_strict": g["n_ungrounded_items_strict"],
                   "evidence_items": g["n_items"], "fault_type": d.get("fault_type"), "root_cause": d.get("root_cause")})
            print(f"{inc['incident_id']} {cond}: {res['stop']}, {res['n_tool_calls']} calls, ${res['cost_usd']:.4f}", flush=True)

        futures = []
        for inc in incidents:
            obs = load_observed(G(root / "observable" / split / VARIANT / inc["scenario_id"]))
            kg = inventory(obs.topology, cfg_s)
            ix = index_for(obs, kg)
            elements = ix.elements
            scope_inc = ScopeIncident(obs, kg, inc["t0"], inc["t1"], cfg_s)
            if not {(inc["incident_id"], m) for m in REFERENCES + ("SCOPE",)} <= done:
                sc = scope_inc.check_change_scope()
                flagged = sc["n_unexplained"] > 0
                changed = overlaps_logged_change([e for e in obs.events if e.get("type") == "maintenance"], inc["t0"], inc["t1"])
                per = [score_reference(inc, gnns[k], gnns[k].score(obs, kg, ix), cal[k]["threshold"], elements) for k in seeds]
                mean = lambda rows: {k: float(np.mean([p[k] for p in rows])) for k in rows[0]}  # noqa: E731
                rule = [p | {"claims_fault": p["claims_fault"] and not changed} for p in per]
                rule_scope = [p | {"claims_fault": p["claims_fault"] and (not changed or flagged)} for p in per]
                for name, score, extra in (("GNN", mean(per), {}), ("GNN-s0", per[0], {}),
                                           ("GNN+CL", mean(rule), {"change_log_overlap": changed}),
                                           ("GNN+CL+scope", mean(rule_scope), {"change_log_overlap": changed, "scope_flagged": flagged}),
                                           ("SCOPE", {"claims_fault": flagged}, {"n_unexplained": sc["n_unexplained"], "n_active_changes": len(sc["active_changes"])})):
                    if (inc["incident_id"], name) not in done:
                        write(inc, name, score, extra)
            for cond in CONDITIONS:
                if (inc["incident_id"], cond) not in done:
                    futures.append(pool.submit(agent, inc, obs, kg, elements, cond))
        for f in futures:
            f.result()
    if stopped.is_set():
        sys.exit(f"stopped by the spend guard: Phase 13 ${spent['phase']:.3f} of ${PHASE_BUDGET_USD:.2f}; finished runs are kept")
    s = score(out_dir, split)
    s["seconds"] = round(time.time() - t_run, 1)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps({m: {g: {k: v["value"] for k, v in s["metrics"][m][g].items()} for g in ("change_overlap", "distractors")
                          if g in s["metrics"][m]} for m in s["metrics"]}, indent=1))
    print(f"wrote {(out_dir / 'summary.json').relative_to(ROOT)}")


# ---------------------------------------------------------------- scoring

def score(run_dir, split):
    from nettwin.scope_agent import summarise13
    incidents = json.loads((run_dir / "incidents.json").read_text())
    recs = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines()]
    by = {}
    for m in REFERENCES + ("SCOPE",) + CONDITIONS:
        d = {r["incident_id"]: r for r in recs if r["method"] == m}
        if len(d) == len(incidents):
            by[m] = [d[i["incident_id"]] for i in incidents]
    table, comps = summarise13(incidents, by, [p for p in PAIRS if p[0] in by and p[1] in by])
    agents = {}
    for c in CONDITIONS:
        rs = by.get(c, [])
        if not rs:
            continue
        has_change = [r for r, i in zip(rs, incidents) if i["stratum"] != "other_fault"]
        agents[c] = {"runs": len(rs), "cost_usd": round(sum(r["cost_usd"] for r in rs), 4),
                     "mean_cost_usd": round(float(np.mean([r["cost_usd"] for r in rs])), 5),
                     "mean_tool_calls": round(float(np.mean([r["n_tool_calls"] for r in rs])), 2),
                     "forced_submissions": sum(r["forced_submit"] for r in rs), "refused_calls": sum(r["n_refused_calls"] for r in rs),
                     "no_submission": sum(r["stop"] != "submitted" for r in rs),
                     "called_change_log_share": round(float(np.mean([r["called_change_log"] for r in rs])), 3),
                     "called_scope_share_change_windows": round(float(np.mean([r["called_scope"] for r in has_change])), 3) if has_change else None,
                     "ungrounded_item_rate": round(sum(r["ungrounded_items"] for r in rs) / max(1, sum(r["evidence_items"] for r in rs)), 4),
                     "ungrounded_item_rate_strict": round(sum(r["ungrounded_items_strict"] for r in rs) / max(1, sum(r["evidence_items"] for r in rs)), 4)}
    strata = {k: sum(i["stratum"] == k for i in incidents) for k in ("change_overlap", "other_fault", "distractor")}
    out = {"split": split, "variant": VARIANT, "n_incidents": len(incidents), "n_scenarios": len({i["scenario_id"] for i in incidents}),
           "strata": strata, "change_overlap_caused_by_maintenance": sum(i.get("caused_by_maintenance", False) for i in incidents if i["stratum"] == "change_overlap"),
           "metrics": table, "comparisons": comps, "agents": agents, "phase13_spend_usd": round(phase_spend(), 4),
           "project_spend_usd": round(project_spend(), 2)}
    out["expectations"] = expectations(out)
    return out


def expectations(s):
    c, m = s["comparisons"], s["metrics"]
    v = lambda k: c.get(k, {}).get("verdict")  # noqa: E731
    a3 = s["agents"].get("A3", {})
    dist_a3 = m.get("A3", {}).get("distractors", {}).get("claims_fault", {}).get("value")
    e = {
        "H1 A3 > A2 on change-overlap fault recall (supported)": v("change_overlap|claims_fault|A3 vs A2") == "supported",
        "H2 A3 vs A2 on distractor fault claims: inconclusive": v("distractors|claims_fault|A3 vs A2") == "inconclusive",
        "H3 A3 distractor fault claims at most 0.15": dist_a3 is not None and dist_a3 <= 0.15,
        "H4 A3 > A2 on change-overlap top-3 (supported)": v("change_overlap|top3|A3 vs A2") == "supported",
        "H5 A3 vs A2 on other-fault recall and top-3: inconclusive": v("other_fault|claims_fault|A3 vs A2") == "inconclusive"
                                                                     and v("other_fault|top3|A3 vs A2") == "inconclusive",
        "H6 A3 < GNN on change-overlap recall (contradicted)": v("change_overlap|claims_fault|A3 vs GNN") == "contradicted",
        "H7 GNN+CL+scope > GNN+CL on change-overlap recall (supported)": v("change_overlap|claims_fault|GNN+CL+scope vs GNN+CL") == "supported",
        "H8 A3 calls check_change_scope in at least 80% of runs on windows with a logged change": (a3.get("called_scope_share_change_windows") or 0) >= 0.80,
        "H9 Phase 13 spend at most $2": s["phase13_spend_usd"] <= PHASE_BUDGET_USD,
    }
    return {k: bool(x) for k, x in e.items()}


def estimate(run_dir, split, cfg):
    rows = [json.loads(l) for l in (run_dir / "agent_log.jsonl").read_text().splitlines()]
    per = {c: float(np.mean([r["cost_usd"] for r in rows if r.get("condition", r.get("condition")) == c])) for c in CONDITIONS
           if any(r.get("condition") == c for r in rows)}
    n = {sp: len(incidents_for(sp, cfg)) for sp in ("val", "test")}
    return {"from_run": str(run_dir.relative_to(ROOT)), "runs": len(rows), "mean_cost_per_run_usd": {k: round(v, 5) for k, v in per.items()},
            "incidents": n, "estimate_usd": {sp: round(n[sp] * sum(per.values()), 3) for sp in n},
            "phase13_spend_so_far_usd": round(phase_spend(), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"])
    ap.add_argument("--name", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--estimate", default=None)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--score", default=None)
    ap.add_argument("--tool-dev", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from nettwin.agent import load_agent_config
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    acfg = load_agent_config(ROOT)
    if args.tool_dev:
        tool_dev(cfg)
    elif args.estimate:
        out = estimate(ROOT / args.estimate, "val", cfg)
        (OUT / f"estimate_from_{Path(args.estimate).name}.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
    elif args.freeze:
        if (OUT / "test" / "summary.json").exists():
            sys.exit("refusing: test has already been scored")
        FROZEN.write_text(json.dumps({"frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "files": {p: sha256(ROOT / p) for p in FROZEN_FILES}}, indent=1) + "\n")
        print(FROZEN.read_text())
    elif args.score:
        d = ROOT / args.score
        s = score(d, "test" if d.name == "test" else "val")
        print(json.dumps(s["expectations"], indent=1))
    elif args.split:
        run(args, cfg, acfg)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
