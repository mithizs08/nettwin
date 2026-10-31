"""Phase 7: agent traces on AARNet (dev) for the demo, and the offline demo bundle.

AARNet was seen during development, so these runs are labelled dev and reported on their own. They never count as
held-out evidence. Incidents come from a fixed rule, not from looking at results: the Phase 6 sampler
(`sample_incidents`, seed 6006) with one incident per fault type and one per distractor action, keeping the six faults and
the drain distractor (7 incidents). Haiku 5.5 A2 and Opus 5.5 A2 run in both variants through the same runner as
Phase 6 (`run_incident`, the frozen tools, runbooks, prompts and GNN checkpoint). B3+ and the GNN references are scored
on the same windows.

The demo bundle (`results/phase7/demo_bundle.json`) holds what the Streamlit app shows: the AARNet map, each
incident's alert timeline, change log, ground truth, utilisation of the links in question, the GNN's top-ranked
elements and both agents' tool traces. The app reads only that file and committed results, so it runs offline with no
data folder and no API key.

    python scripts/phase7_aarnet_traces.py
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase6_agent as p6  # noqa: E402  (the Phase 6 runner's asset loading, spend log and record format)

from nettwin.agent import load_agent_config  # noqa: E402
from nettwin.agent.eval import overlaps_logged_change, sample_incidents, score_agent, score_reference  # noqa: E402
from nettwin.agent.freeze import agent_hashes  # noqa: E402
from nettwin.agent.loop import run_incident  # noqa: E402
from nettwin.agent.rag import RunbookIndex  # noqa: E402
from nettwin.agent.tools import HISTORY, Incident  # noqa: E402
from nettwin.baselines import index_for  # noqa: E402
from nettwin.observed import freeze_cfg, inventory, load_observed  # noqa: E402
from nettwin.splits import guard  # noqa: E402

ROOT = p6.ROOT
SPLIT = "dev"
OUT = ROOT / "results" / "phase7"
MODELS = {"haiku": "eval_model", "opus": "secondary_eval_model"}
CONDITION = "A2"
BUDGET_USD = 3.0           # this script's own cap (Phase 7 instruction); the project cap is checked as in Phase 6
TOP_K = 10


def pick(entries):
    """The fixed rule: the Phase 6 sampler with one per type and per distractor action; keep faults and the drain."""
    inc = sample_incidents(entries, 1, 1, p6.SAMPLE_SEED)
    keep = [i for i in inc if i["kind"] == "fault" or i["type"] == "distractor_drain"]
    for k, i in enumerate(keep):
        i["incident_id"] = f"D{k:02d}"
    return keep


def r3(a):
    return [None if not np.isfinite(x) else round(float(x), 3) for x in a]


def write_summary(incidents, recs, cost, code_sha256):
    """Means over the 7 dev incidents: localisation, fault-type accuracy and fault recall on the 6 faults; the fault
    claim on the 1 distractor (0 or 1). Too few incidents for confidence intervals."""
    kind = {i["incident_id"]: i["kind"] for i in incidents}
    summary = {"split": SPLIT, "note": "dev, seen during development; reported separately, not held-out evidence",
               "n_incidents": len(incidents), "n_faults": sum(k == "fault" for k in kind.values()),
               "n_distractors": sum(k == "distractor" for k in kind.values()), "cost_usd": round(cost, 4), "code_sha256": code_sha256,
               "by_variant": {}}
    for v in p6.VARIANTS:
        out = {}
        for m in ("B3+", "GNN", "GNN+CL", "haiku-A2", "opus-A2"):
            rs = [r for r in recs if r["variant"] == v and r["method"] == m]
            f = [r for r in rs if kind[r["incident_id"]] == "fault"]
            d = [r for r in rs if kind[r["incident_id"]] == "distractor"]
            out[m] = {"faults_top1": round(float(np.mean([r["top1"] for r in f])), 3), "faults_top3": round(float(np.mean([r["top3"] for r in f])), 3),
                      "fault_recall": round(float(np.mean([r["claims_fault"] for r in f])), 3),
                      "distractor_fault_claims": round(float(np.mean([r["claims_fault"] for r in d])), 3)}
            if any("type_correct" in r for r in f):
                out[m]["fault_type_correct"] = round(float(np.mean([r["type_correct"] for r in f])), 3)
        summary["by_variant"][v] = out
    (OUT / "aarnet_dev_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    acfg = load_agent_config(ROOT)
    hashes = agent_hashes(ROOT, acfg)
    frozen = json.loads((ROOT / "results" / "phase6_frozen.json").read_text())
    match = frozen["code_sha256"] == hashes["code_sha256"]
    print(f"agent code_sha256 {hashes['code_sha256'][:12]} ({'matches' if match else 'does NOT match'} results/phase6_frozen.json)", flush=True)
    if not match:
        sys.exit("refusing: the agent code differs from the Phase 6 freeze")
    spent = p6.logged_spend() + sum(json.loads(l)["cost_usd"] for f in OUT.glob("*agent_log.jsonl") for l in f.read_text().splitlines())
    print(f"project API spend so far: ${spent:.2f} of ${acfg['project_budget_usd']}", flush=True)
    p6.load_env()
    cfg_s = freeze_cfg(cfg)
    root = ROOT / cfg["dataset"]["root"]
    G = lambda p: guard(p, {SPLIT})  # noqa: E731
    cal, baselines, gnns, seeds = p6.frozen_assets(cfg_s)
    rb = RunbookIndex()
    incidents = pick(json.loads(G(root / "truth" / SPLIT / "with_status" / "manifest.json").read_text()))
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "aarnet_dev_records.jsonl").exists():
        sys.exit("refusing: results/phase7/aarnet_dev_records.jsonl exists; the traces are run once and never overwritten (audit m3)")
    (OUT / "aarnet_dev_incidents.json").write_text(json.dumps(incidents, indent=1) + "\n")
    print(f"{len(incidents)} incidents: {[(i['incident_id'], i['type']) for i in incidents]}", flush=True)

    tag = {"code_sha256": hashes["code_sha256"], "split": SPLIT}
    client = anthropic.Anthropic(max_retries=6)
    bundle = {"note": "AARNet (public topology, Internet Topology Zoo) with synthetic telemetry; dev split, seen during "
                      "development; not Telstra's network.", "code_sha256": hashes["code_sha256"], "incidents": []}
    kg = inventory("topozoo/Aarnet", cfg_s)
    bundle["topology"] = {"pops": [{"id": p, **{k: kg.G.nodes[p][k] for k in ("name", "x", "y")}} for p in kg.nodes_of_kind("PoP")],
                          "links": [{"id": l, "u": kg.G.nodes[l]["u"], "v": kg.G.nodes[l]["v"], "dist_km": kg.G.nodes[l]["dist_km"]}
                                    for l in kg.nodes_of_kind("Link")]}
    run_cost = 0.0
    with open(OUT / "aarnet_dev_records.jsonl", "w") as recf, ThreadPoolExecutor(8) as pool:
        logs = {m: open(OUT / f"aarnet_dev_{m}_A2_agent_log.jsonl", "w") for m in MODELS}
        locks = {m: p6.LockedLog(f) for m, f in logs.items()}
        rec = p6.LockedLog(recf)

        def write(inc, v, method, score, extra=None):
            rec.write(json.dumps({"incident_id": inc["incident_id"], "variant": v, "method": method, **score, **(extra or {}), **tag}) + "\n")

        def agent_run(inc, v, obs, okg, elements, mkey):
            model = acfg[MODELS[mkey]]
            incident = Incident(obs, okg, inc["t0"], inc["t1"], cfg_s, gnn=gnns[seeds[0]].model, runbooks=rb)
            incident.incident_id = f"{inc['incident_id']}|{v}"
            res = run_incident(client, incident, CONDITION, model, acfg, status_polling=(v == "with_status"), log=locks[mkey], log_extra=tag)
            write(inc, v, f"{mkey}-{CONDITION}", score_agent(inc, res["diagnosis"], elements),
                  {"model": model, "cost_usd": res["cost_usd"], "n_tool_calls": res["n_tool_calls"], "stop": res["stop"],
                   "forced_submit": res["forced_submit"], "fault_type": (res["diagnosis"] or {}).get("fault_type"),
                   "root_cause": (res["diagnosis"] or {}).get("root_cause")})
            print(f"{inc['incident_id']} {v} {mkey} A2: {res['stop']}, {res['n_tool_calls']} calls, ${res['cost_usd']:.4f}", flush=True)
            return res["cost_usd"]

        for inc in incidents:
            entry = {"incident": inc, "variants": {}}
            jobs = []
            for v in p6.VARIANTS:
                obs = load_observed(G(root / "observable" / SPLIT / v / inc["scenario_id"]))
                okg = inventory(obs.topology, cfg_s)
                ix = index_for(obs, okg)
                elements = ix.elements
                m = baselines[v]["B3+"]
                write(inc, v, "B3+", score_reference(inc, m, m.score(obs, okg, ix), cal[v]["B3+"]["threshold"], elements))
                states = {k: gnns[k].score(obs, okg, ix) for k in seeds}
                per = [score_reference(inc, gnns[k], states[k], cal[v][k]["threshold"], elements) for k in seeds]
                changed = overlaps_logged_change([e for e in obs.events if e.get("type") == "maintenance"], inc["t0"], inc["t1"])
                mean = lambda rows: {k: float(np.mean([p[k] for p in rows])) for k in rows[0]}  # noqa: E731
                write(inc, v, "GNN", mean(per), {"per_seed": per})
                write(inc, v, "GNN+CL", mean([p | {"claims_fault": p["claims_fault"] and not changed} for p in per]), {"change_log_overlap": changed})
                # GNN ranking shown in the demo: the reference GNN (mean of the 3 seeds' q90 window scores)
                loc = np.mean([gnns[k].localise(states[k], inc["t0"], inc["t1"], "q90") for k in seeds], 0)
                order = np.argsort(-loc, kind="stable")
                true_rank = int(np.where(np.array(elements)[order] == inc["root"])[0][0]) + 1 if inc["root"] in elements else None
                view = Incident(obs, okg, inc["t0"], inc["t1"], cfg_s)
                n = inc["t1"] - inc["t0"]
                shown = [inc["root"]] if inc["root"].startswith("link:") else []
                shown += [inc.get("change_target")] if inc.get("change_target", "").startswith("link:") else []
                shown += [elements[i] for i in order if elements[i].startswith("link:")][:3]
                shown = list(dict.fromkeys(s for s in shown if s))
                a, b = view.floor, view.now
                util = {l: r3(obs.util[a:b, obs.links.index(l)]) for l in shown if l in obs.links}
                entry["variants"][v] = {
                    "alerts": view.get_active_alerts(-HISTORY, n - 1, max_alerts=400),
                    "changes": view.get_change_log(-HISTORY, n - 1)["changes"],
                    "gnn_top": [{"element": elements[i], "score": round(float(loc[i]), 4)} for i in order[:TOP_K]],
                    "gnn_true_rank": true_rank,
                    "util": {"t_start": -HISTORY, "series": util}}
                for mkey in MODELS:
                    jobs.append((v, obs, okg, elements, mkey))
            bundle["incidents"].append(entry)
            if spent + run_cost >= acfg["project_budget_usd"] - 1.0 or run_cost >= BUDGET_USD - 0.3:
                sys.exit(f"stopping: spend limit (this script ${run_cost:.2f}, project ${spent + run_cost:.2f})")
            for f in [pool.submit(agent_run, inc, *j) for j in jobs]:
                run_cost += f.result()
        for f in logs.values():
            f.close()
    traces = {}
    for mkey in MODELS:
        for l in (OUT / f"aarnet_dev_{mkey}_A2_agent_log.jsonl").read_text().splitlines():
            r = json.loads(l)
            iid, v = r["incident"].split("|")
            traces[(iid, v, mkey)] = {k: r[k] for k in ("model", "diagnosis", "stop", "forced_submit", "n_tool_calls", "n_refused_calls",
                                                         "cost_usd", "seconds", "groundedness")} | {
                "calls": [{"tool": c["tool"], "input": c["input"], "output": c["output"], "error": c["error"], "refused": c["refused"]} for c in r["calls"]]}
    recs = [json.loads(l) for l in (OUT / "aarnet_dev_records.jsonl").read_text().splitlines()]
    for e in bundle["incidents"]:
        iid = e["incident"]["incident_id"]
        for v, x in e["variants"].items():
            x["agents"] = {m: traces.get((iid, v, m)) for m in MODELS}
            x["scores"] = {r["method"]: {k: r.get(k) for k in ("claims_fault", "top1", "top3", "type_correct")} for r in recs
                           if r["incident_id"] == iid and r["variant"] == v}
    bundle["cost_usd"] = round(run_cost, 4)
    (OUT / "demo_bundle.json").write_text(json.dumps(bundle, indent=1, default=str) + "\n")
    write_summary(incidents, recs, run_cost, hashes["code_sha256"])
    print(f"wrote results/phase7/ (this script ${run_cost:.4f})")


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"{time.time() - t:.0f}s")
