"""Phase 12: real-time replay without an oracle incident window.

    python scripts/phase12_replay.py --split val --name smoke --limit 3     # smoke test (first 3 sampled scenarios)
    python scripts/phase12_replay.py --estimate results/phase12/val_smoke    # cost estimate for the full runs
    python scripts/phase12_replay.py --split val --name dev                  # full validation run
    python scripts/phase12_replay.py --freeze                                # hash the code into results/phase12_frozen.json
    python scripts/phase12_replay.py --split test                            # test, once
    python scripts/phase12_replay.py --score results/phase12/test            # rescore a finished run folder

Each sampled scenario (with_status variant) is replayed from the end of the warm-up. The frozen GNN (GAT-h32 seed 0,
frozen validation threshold) alarms; alarms merge into episodes (`nettwin.replay`). Three policies decide which
episodes page an operator:
- GNN: every episode pages when it opens.
- GNN+CL: as GNN, unless a logged drain or planned outage is active (or ended within the hour) when it opens.
- Agent: Haiku 5.5 is woken when the episode opens, sees data up to that step, and escalates, dismisses, or waits and
  is woken once more RECHECK steps later (`nettwin.replay_agent`).
Ground truth is read only after every decision, for scoring. Spend guard: Phase 12 stops before any agent wake that
could take its logged spend plus the wakes in flight past PHASE_BUDGET_USD, or the project within $1 of its budget.
`--split test` refuses unless the Phase 12 pre-registration is in docs/decision_log.md, every hash matches
results/phase12_frozen.json and results/phase5_frozen.json, results/phase12/test does not exist, and --limit is unset.
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

import anthropic
import numpy as np
import yaml

from nettwin import replay as R
from nettwin.agent import load_agent_config
from nettwin.agent.rag import RunbookIndex
from nettwin.alerts import generate_alerts
from nettwin.evaluation import boot_weights, ci, verdict
from nettwin.gnn.method import gnn_method
from nettwin.observed import freeze_cfg, inventory, load_observed
from nettwin.replay_agent import ReplayIncident, run_wake
from nettwin.splits import guard

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase12"
FROZEN = ROOT / "results" / "phase12_frozen.json"
PREREG = "### 2026-11-18: Phase 12 pre-registration"
MODEL = "claude-haiku-5-5"
VARIANT = "with_status"
GNN_RUN = "gat-h32-s0"
SAMPLE_SEED = 12012
PER_NETWORK = {"val": 6, "test": 10}
PHASE_BUDGET_USD = 4.00
RESERVE_USD = 0.02          # set aside per agent wake in flight
UNLOGGED_SPEND_USD = 0.30   # Phase 6 allowance, as in scripts/phase6_agent.py
METHODS = ("GNN", "GNN+CL", "Agent")
PAIRS = (("Agent", "GNN"), ("Agent", "GNN+CL"), ("GNN+CL", "GNN"))
FROZEN_FILES = ("src/nettwin/replay.py", "src/nettwin/replay_agent.py", "scripts/phase12_replay.py",
                "src/nettwin/agent/tools.py", "src/nettwin/agent/loop.py", "src/nettwin/agent/rag.py", "src/nettwin/alerts.py",
                "src/nettwin/baselines.py", "src/nettwin/gnn/models.py", "src/nettwin/gnn/data.py", "src/nettwin/kg.py")


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
            + _sum_cost(res.glob("phase11/*/log.jsonl")) + phase_spend())


def sample(entries, split):
    """PER_NETWORK[split] scenarios per real network (topozoo / sndlib), seeded; generated graphs are not used."""
    rng = np.random.default_rng(SAMPLE_SEED)
    real = [e for e in entries if not e["topology"].startswith("gabaug")]
    out = []
    for topo in sorted({e["topology"] for e in real}):
        pool = [e["scenario_id"] for e in real if e["topology"] == topo]
        out += sorted(rng.choice(pool, PER_NETWORK[split], replace=False).tolist())
    return out


def frozen_gnn(cfg_s):
    frozen = json.loads((ROOT / "results" / "phase5_frozen.json").read_text())
    for rel, h in frozen["files"].items():
        if sha256(ROOT / rel) != h:
            sys.exit(f"refusing: sha256 mismatch for {rel}")
    mdir = ROOT / "data" / "models" / "phase5"
    thr = json.loads((mdir / "frozen" / "calibration.json").read_text())["variants"][VARIANT][GNN_RUN]["threshold"]
    return gnn_method(GNN_RUN, (mdir / f"{GNN_RUN}.pt").read_bytes(), cfg_s).model, thr


class LockedLog:
    def __init__(self, f):
        self.f, self.lock = f, threading.Lock()

    def write(self, s):
        with self.lock:
            self.f.write(s)
            self.f.flush()


# ---------------------------------------------------------------- run

def run(args, cfg):
    split = args.split
    if split == "test" and args.dry:
        sys.exit("refusing: --dry is for validation")
    if split == "test":
        frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else None
        problems = [m for m, bad in (
            ("the Phase 12 pre-registration is not in docs/decision_log.md", PREREG not in (ROOT / "docs" / "decision_log.md").read_text()),
            ("results/phase12_frozen.json is missing", frozen is None),
            ("a frozen file changed", frozen is not None and any(sha256(ROOT / k) != v for k, v in frozen["files"].items())),
            ("results/phase12/test already exists (test is run once)", (OUT / "test").exists()),
            ("--limit is not allowed on test", bool(args.limit)),
        ) if bad]
        if problems:
            sys.exit("refusing the test run: " + "; ".join(problems))
        out_dir = OUT / "test"
    else:
        out_dir = OUT / f"val_{args.name}"
    G = lambda p: guard(p, {split})  # noqa: E731
    cfg_s = freeze_cfg(cfg)
    root = ROOT / cfg["dataset"]["root"]
    model, thr = frozen_gnn(cfg_s)
    acfg = load_agent_config(ROOT)
    prices = acfg["pricing_usd_per_mtok"][MODEL]
    budget = acfg["project_budget_usd"]
    # the manifest is read here only for the list of scenario ids and topologies; faults are read in `score` only
    ids = sample(json.loads(G(root / "truth" / split / VARIANT / "manifest.json").read_text()), split)
    if args.limit:
        ids = ids[::PER_NETWORK[split]][:args.limit]          # the first sampled scenario of each network, in order
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenarios.json").write_text(json.dumps(ids) + "\n")
    rb = RunbookIndex()
    t0 = time.time()

    # 1. replay: GNN stream, episodes and the two reference policies (no API)
    scen = {}
    for sid in ids:
        obs = load_observed(G(root / "observable" / split / VARIANT / sid))
        inv = inventory(obs.topology, cfg_s)
        g, det, elem = R.gnn_stream(obs, model, cfg_s)
        eps = R.episodes(det, elem, g.elements, thr)
        GNN_ELEM[sid] = elem
        scen[sid] = {"obs": obs, "inv": inv, "alerts": generate_alerts(obs, inv), "g": g, "elem": elem, "eps": eps,
                     "pages": {"GNN": R.gnn_pages(eps, elem, g.elements), "GNN+CL": R.gnn_cl_pages(eps, elem, g.elements, obs)}}
        alarm_steps = np.flatnonzero(det.max(1) >= thr)
        with open(out_dir / "replay.jsonl", "a" if sid != ids[0] else "w") as f:
            f.write(json.dumps({"scenario_id": sid, "topology": obs.topology, "n_steps": obs.n_steps, "elements": g.elements,
                                "alarm_steps": alarm_steps.tolist(), "episodes": eps,
                                "pages": {m: [{k: p[k] for k in ("episode_id", "t")} for p in scen[sid]["pages"][m]] for m in ("GNN", "GNN+CL")},
                                "changes": R.logged_changes(obs), "status_events": R.status_events(obs)}) + "\n")
    n_eps = sum(len(s["eps"]) for s in scen.values())
    print(f"replayed {len(ids)} scenarios in {time.time() - t0:.0f}s: {n_eps} alarm episodes", flush=True)
    if args.dry:
        s = score(out_dir, split, cfg, dry=True)
        print(json.dumps({m: {k: s['metrics'][m][k]['value'] for k in s['metrics'][m]} for m in ('GNN', 'GNN+CL')}, indent=1))
        return

    # 2. agent wakes: round 1 at each episode's opening, round 2 re-checks after a wait
    dec_path = out_dir / "decisions.jsonl"
    done = {}
    if dec_path.exists():
        for l in dec_path.read_text().splitlines():
            r = json.loads(l)
            done[(r["scenario_id"], r["episode_id"], r["round"])] = r
    spent = {"phase": phase_spend(), "project": project_spend(), "inflight": 0.0}
    print(f"Phase 12 spend so far ${spent['phase']:.3f} of ${PHASE_BUDGET_USD:.2f}; project ${spent['project']:.2f} of ${budget}", flush=True)
    load_env()
    client = anthropic.Anthropic(max_retries=6)
    glock, stopped = threading.Lock(), threading.Event()

    def wake(sid, ep, rnd, prior):
        s = scen[sid]
        step = ep["open"] + (0 if rnd == 1 else R.RECHECK)
        if step >= s["obs"].n_steps:
            return {"scenario_id": sid, "episode_id": ep["episode_id"], "round": rnd, "t": step, "decision": None, "stop": "scenario_end",
                    "cost_usd": 0.0, "n_tool_calls": 0}
        with glock:
            if (spent["phase"] + spent["inflight"] + RESERVE_USD > PHASE_BUDGET_USD
                    or spent["project"] + spent["inflight"] + RESERVE_USD > budget - 1.0):
                stopped.set()
                return None
            spent["inflight"] += RESERVE_USD
        try:
            inc = ReplayIncident(s["obs"], R.kg_at(s["inv"], s["obs"], step), ep["open"], step + 1, cfg_s, s["alerts"], gnn=model, runbooks=rb)
            res = run_wake(client, inc, ep, MODEL, prices, prior=prior, log=alog,
                           log_extra={"scenario_id": sid, "episode_id": ep["episode_id"], "round": rnd, "t": step})
        finally:
            with glock:
                spent["inflight"] -= RESERVE_USD
        with glock:
            spent["phase"] += res["cost_usd"]
            spent["project"] += res["cost_usd"]
        rec = {"scenario_id": sid, "episode_id": ep["episode_id"], "round": rnd, "t": step, "decision": res["decision"], "stop": res["stop"],
               "cost_usd": res["cost_usd"], "n_tool_calls": res["n_tool_calls"], "seconds": res["seconds"]}
        d = (res["decision"] or {}).get("decision")
        print(f"{sid} {ep['episode_id']} r{rnd}: {d}, {res['n_tool_calls']} calls, ${res['cost_usd']:.4f}", flush=True)
        return rec

    with open(out_dir / "agent_log.jsonl", "a") as af, open(dec_path, "a") as df, ThreadPoolExecutor(args.threads) as pool:
        alog, dlog = LockedLog(af), LockedLog(df)
        for rnd in (1, 2):
            jobs = []
            for sid, s in scen.items():
                for ep in s["eps"]:
                    key = (sid, ep["episode_id"], rnd)
                    if key in done:
                        continue
                    if rnd == 2:
                        first = done.get((sid, ep["episode_id"], 1))
                        if not first or ((first["decision"] or {}).get("decision") != "wait"):
                            continue
                        jobs.append((sid, ep, rnd, (first["decision"] or {}).get("reason", "")))
                    else:
                        jobs.append((sid, ep, rnd, None))
            for rec in pool.map(lambda j: wake(*j), jobs):
                if rec is not None:
                    dlog.write(json.dumps(rec) + "\n")
                    done[(rec["scenario_id"], rec["episode_id"], rec["round"])] = rec
    print(f"agent done in {time.time() - t0:.0f}s; Phase 12 spend ${spent['phase']:.3f}" + (" (STOPPED by the spend guard)" if stopped.is_set() else ""), flush=True)
    if stopped.is_set():
        sys.exit(2)
    s = score(out_dir, split, cfg)
    print(json.dumps({m: {k: s["metrics"][m][k]["value"] for k in s["metrics"][m]} for m in s["metrics"]}, indent=1))


# ---------------------------------------------------------------- scoring (reads ground truth)

def agent_pages(decisions, eps_by_id):
    """Escalations become pages at their decision step, ranked root first, then alternatives."""
    pages = []
    for r in sorted(decisions, key=lambda r: (r["t"], r["episode_id"])):
        d = r["decision"] or {}
        if d.get("decision") == "escalate":
            pages.append({"episode_id": r["episode_id"], "t": r["t"], "kind": "ids",
                          "ranking": [d.get("root_cause")] + list(d.get("alternatives") or [])[:2]})
    return pages


METRICS = {   # name: (numerator, denominator, scale)
    "pages_per_day": ("pages", "days", 1), "false_pages_per_day": ("false_pages", "days", 1),
    "detection_rate": ("detected", "faults", 1), "time_to_detect_min": ("delay_sum", "detected", 5),
    "top1_at_escalation": ("top1_sum", "detected", 1), "top3_at_escalation": ("top3_sum", "detected", 1),
    "top3_all_faults": ("top3_sum", "faults", 1), "missed_during_maintenance": ("change_missed", "change_faults", 1),
    "cost_per_day_usd": ("cost", "days", 1),
}


def score(run_dir, split, cfg, dry=False):
    G = lambda p: guard(p, {split})  # noqa: E731
    root = ROOT / cfg["dataset"]["root"]
    truth = {e["scenario_id"]: e for e in json.loads(G(root / "truth" / split / VARIANT / "manifest.json").read_text())}
    reps = [json.loads(l) for l in (run_dir / "replay.jsonl").read_text().splitlines()]
    decs = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()] if (run_dir / "decisions.jsonl").exists() else []
    rows, per_fault, timeline = {m: [] for m in METHODS}, {m: [] for m in METHODS}, []
    for rp in reps:
        sid, e = rp["scenario_id"], truth[rp["scenario_id"]]
        eps = {x["episode_id"]: x for x in rp["episodes"]}
        d = [r for r in decs if r["scenario_id"] == sid]
        missing = [x for x in eps if not any(r["episode_id"] == x and r["round"] == 1 for r in d)]
        if missing and not dry:
            sys.exit(f"{sid}: no agent decision for {len(missing)} episodes; the run is unfinished")
        # GNN rankings at each page: elem at the page step, recomputed from the logged replay would need the model; the
        # replay stored them in memory only, so they are rebuilt here from the frozen GNN (deterministic, causal)
        elem = GNN_ELEM.get(sid)
        pages = {"GNN": [dict(p, kind="scores", ranking=elem[p["t"]]) for p in rp["pages"]["GNN"]],
                 "GNN+CL": [dict(p, kind="scores", ranking=elem[p["t"]]) for p in rp["pages"]["GNN+CL"]],
                 "Agent": agent_pages(d, eps)}
        cost = sum(r["cost_usd"] for r in d)
        for m in METHODS:
            sc = R.score_scenario(e, pages[m], rp["elements"], rp["n_steps"])
            sc["cost"] = cost if m == "Agent" else 0.0
            rows[m].append(sc)
            per_fault[m] += [dict(f, scenario_id=sid) for f in sc.pop("per_fault")]
        timeline.append({"scenario_id": sid, "topology": rp["topology"], "n_steps": rp["n_steps"], "alarm_steps": rp["alarm_steps"],
                         "episodes": [{k: x[k] for k in ("episode_id", "element", "open", "last", "n_alarms")} for x in rp["episodes"]],
                         "pages": {m: [{"episode_id": p["episode_id"], "t": p["t"]} | ({"ranking": p["ranking"]} if m == "Agent" else {})
                                       for p in pages[m]] for m in METHODS},
                         "decisions": [{k: r[k] for k in ("episode_id", "round", "t")} | {"decision": (r["decision"] or {}).get("decision"),
                                       "root_cause": (r["decision"] or {}).get("root_cause"), "reason": (r["decision"] or {}).get("reason")} for r in d],
                         "faults": [{k: f[k] for k in ("id", "type", "root_cause", "t_start", "t_end")} for f in e["faults"]],
                         "changes": [{k: c[k] for k in ("id", "action", "target", "t_start", "t_end")} for c in e["maintenance"]]})
    S = len(reps)
    W = boot_weights(S, 1000, 42)
    boots, metrics = {}, {}
    for m in METHODS:
        for name, (num, den, k) in METRICS.items():
            a = np.array([r[num] for r in rows[m]], float)
            b = np.array([r[den] for r in rows[m]], float)
            with np.errstate(invalid="ignore", divide="ignore"):
                x = k * (W @ a) / (W @ b)
            boots[(m, name)] = x
            metrics.setdefault(m, {})[name] = {"value": None if not np.isfinite(x[0]) else round(float(x[0]), 4), "ci": ci(x[1:]),
                                               "numerator": float(a.sum()), "denominator": float(b.sum())}
    comps = {f"{name}|{a} vs {b}": verdict(boots[(a, name)] - boots[(b, name)]) for a, b in PAIRS for name in METRICS
             if not (name == "cost_per_day_usd")}
    ad = [r for r in decs if r["round"] == 1]
    agent = {"wakes": len(decs), "episodes": len(ad), "rechecks": sum(r["round"] == 2 for r in decs),
             "first_decisions": {k: sum((r["decision"] or {}).get("decision") == k for r in ad) for k in ("escalate", "dismiss_maintenance", "dismiss_noise", "wait")},
             "no_decision": sum(r["decision"] is None for r in decs), "mean_tool_calls": round(float(np.mean([r["n_tool_calls"] for r in decs])), 2) if decs else None,
             "cost_usd": round(sum(r["cost_usd"] for r in decs), 4), "mean_cost_per_wake": round(float(np.mean([r["cost_usd"] for r in decs])), 6) if decs else None}
    out = {"split": split, "variant": VARIANT, "model": MODEL, "gnn": GNN_RUN, "n_scenarios": S, "scenario_days": float(sum(r["days"] for r in rows["GNN"])),
           "n_faults": int(sum(r["faults"] for r in rows["GNN"])), "n_episodes": sum(len(r["episodes"]) for r in reps),
           "metrics": metrics, "comparisons": comps, "agent": agent, "n_boot": 1000, "seed": 42,
           "settings": {"merge_gap": R.MERGE_GAP, "recheck": R.RECHECK, "detect_tail": R.DETECT_TAIL, "start": R.START}}
    out["expectations"] = expectations(out)
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    (run_dir / "per_fault.json").write_text(json.dumps(per_fault, indent=1) + "\n")
    (run_dir / "timeline.json").write_text(json.dumps(timeline) + "\n")
    return out


GNN_ELEM = {}


def expectations(s):
    """The pre-registered expectations P1 to P8 (decision log, Phase 12 pre-registration)."""
    C, M = s["comparisons"], s["metrics"]
    v = lambda k: C[k]["verdict"]  # noqa: E731
    e = {
        "P1": ("Agent pages per scenario-day below the raw GNN's", v("pages_per_day|Agent vs GNN") == "contradicted"),
        "P2": ("Agent false pages per scenario-day below GNN+CL's", v("false_pages_per_day|Agent vs GNN+CL") == "contradicted"),
        "P3": ("Agent fault detection rate below the raw GNN's", v("detection_rate|Agent vs GNN") == "contradicted"),
        "P4": ("Agent against GNN+CL on fault detection rate: inconclusive", v("detection_rate|Agent vs GNN+CL") == "inconclusive"),
        "P5": ("Agent time to detect later than the raw GNN's, by under 30 minutes",
               v("time_to_detect_min|Agent vs GNN") == "supported" and C["time_to_detect_min|Agent vs GNN"]["point"] < 30),
        "P6": ("Agent against the raw GNN on root-cause top-3 at escalation: inconclusive", v("top3_at_escalation|Agent vs GNN") == "inconclusive"),
        "P7": ("Missed faults during maintenance: Agent below GNN+CL", v("missed_during_maintenance|Agent vs GNN+CL") == "contradicted"),
        "P8": ("Agent cost below $0.05 per scenario-day", (M["Agent"]["cost_per_day_usd"]["value"] or 0) < 0.05),
    }
    return {k: {"expectation": d, "met": bool(ok)} for k, (d, ok) in e.items()}


def rebuild_elem(run_dir, split, cfg):
    """GNN element scores per scenario, recomputed from the frozen model (needed for the GNN rankings at its pages)."""
    G = lambda p: guard(p, {split})  # noqa: E731
    cfg_s = freeze_cfg(cfg)
    model, _ = frozen_gnn(cfg_s)
    root = ROOT / cfg["dataset"]["root"]
    for l in (run_dir / "replay.jsonl").read_text().splitlines():
        sid = json.loads(l)["scenario_id"]
        obs = load_observed(G(root / "observable" / split / VARIANT / sid))
        GNN_ELEM[sid] = R.gnn_stream(obs, model, cfg_s)[2]


# ---------------------------------------------------------------- estimate

def estimate(run_dir, cfg):
    """Full-run cost from a smoke run: mean cost per wake x wakes per alarm episode x episodes per scenario. Episodes per
    scenario on test are scaled from the validation rate per link, by each test network's link count (topology only)."""
    decs = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines()]
    reps = [json.loads(l) for l in (run_dir / "replay.jsonl").read_text().splitlines()]
    per_wake = float(np.mean([d["cost_usd"] for d in decs if d["cost_usd"] > 0]))
    wakes_per_ep = len(decs) / max(1, sum(d["round"] == 1 for d in decs))
    cfg_s = freeze_cfg(cfg)
    links = lambda topo: len(inventory(topo, cfg_s).nodes_of_kind("Link"))  # noqa: E731
    eps_per_link = sum(len(r["episodes"]) for r in reps) / sum(links(r["topology"]) for r in reps)
    out = {"smoke": {"scenarios": len(reps), "episodes": sum(len(r["episodes"]) for r in reps), "wakes": len(decs),
                     "mean_cost_per_wake": round(per_wake, 5), "wakes_per_episode": round(wakes_per_ep, 3),
                     "episodes_per_link_per_scenario": round(eps_per_link, 4), "cost_usd": round(sum(d["cost_usd"] for d in decs), 4)}}
    for split in ("val", "test"):
        nets = cfg["topologies"]["validation_real" if split == "val" else "test"]
        eps = sum(PER_NETWORK[split] * eps_per_link * links(t) for t in nets)
        out[split] = {"scenarios": PER_NETWORK[split] * len(nets), "episodes_est": round(eps), "wakes_est": round(eps * wakes_per_ep),
                      "cost_est_usd": round(eps * wakes_per_ep * per_wake, 2)}
    out["total_est_usd"] = round(out["val"]["cost_est_usd"] + out["test"]["cost_est_usd"], 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"])
    ap.add_argument("--name", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--estimate", default=None)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--score", default=None)
    ap.add_argument("--dry", action="store_true", help="validation only: replay and score the references, no agent and no API")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    if args.estimate:
        print(json.dumps(estimate(ROOT / args.estimate, cfg), indent=1))
    elif args.freeze:
        if (OUT / "test").exists():
            sys.exit("refusing: test has already run")
        FROZEN.write_text(json.dumps({"frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "files": {p: sha256(ROOT / p) for p in FROZEN_FILES}}, indent=1) + "\n")
        print(FROZEN.read_text())
    elif args.score:
        d = ROOT / args.score
        split = "test" if d.name == "test" else "val"
        rebuild_elem(d, split, cfg)
        s = score(d, split, cfg)
        print(json.dumps({m: {k: s["metrics"][m][k]["value"] for k in s["metrics"][m]} for m in s["metrics"]}, indent=1))
    elif args.split:
        run(args, cfg)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
