"""Phase 13: the Phase 6 agent with one more tool, `check_change_scope`, on a fresh incident sample.

Condition A3 is Phase 6 A2 (same model, prompt, tools, budgets and scoring) plus:
- the tool `check_change_scope` (`nettwin.change_scope`), which asks the inventory KG what each logged drain or planned
  outage in the window can explain and lists the observed symptoms outside it;
- one paragraph in the system prompt asking the agent to call it before deciding that planned work explains the
  symptoms.
A2 runs through the frozen Phase 6 loop (`nettwin.agent.loop.run_incident`), unchanged. A3 runs through `run_incident_a3`
below, a copy of that loop whose only differences are the tool list and the system prompt. Nothing in `nettwin.agent`
is changed, so the Phase 6 freeze (`results/phase6_frozen.json`) still holds and is checked by the runner.

Sampling (`sample_phase13`): incidents never drawn in Phase 6. Any fault or change window that overlaps a Phase 6
incident window of the same split and scenario is excluded. Three strata, in a seeded order:
- `change_overlap`: observable faults after the warm-up, isolated from other faults (LAG), whose window overlaps a
  logged drain or planned outage (the change log as the manifest records it, which is what the operator sees);
- `other_fault`: the same without a change overlap, `n_other_per_type` per fault type;
- `distractor`: a drain or planned outage with no fault within its window plus MAINT_TAIL, `n_distractor` per action.
An incident may share a scenario with another Phase 13 incident (windows never overlap); CIs resample scenarios.
"""
import json
import time

import numpy as np

from nettwin.agent.eval import DISTRACTOR_ACTIONS, _overlaps, overlaps_logged_change
from nettwin.agent.loop import FORCE_MSG, SYSTEM, TOOLS_NOTE, _price, groundedness, incident_prompt
from nettwin.agent.tools import CONDITIONS, Incident, _schema, call_tool, tools_for
from nettwin.change_scope import change_scope
from nettwin.evaluation import EASY, HARD, LAG, MAINT_TAIL, WARMUP, boot_weights, ci, verdict

SCOPE_TOOL = "check_change_scope"
SCOPE_SPEC = ("Check whether the logged drains and planned outages in the window can explain what is observed. Uses the "
              "inventory graph to work out which links each change moves traffic off and onto and which services it "
              "reroutes, then lists the observed symptoms (alerts and traffic shifts) that fall outside that, as known "
              "at the end of the window.", _schema({}, []))
SCOPE_NOTE = ("Planned work can hide a real fault. If the change log shows a drain or planned outage in or near the window, "
              f"call {SCOPE_TOOL} before you decide that planned work explains the symptoms. A symptom outside what the "
              "change can explain needs another cause.")
A3_TOOLS = CONDITIONS["A2"] + (SCOPE_TOOL,)


class ScopeIncident(Incident):
    def check_change_scope(self):
        return change_scope(self)


def tools_a3():
    a2 = tools_for("A2")
    return a2[:-1] + [{"name": SCOPE_TOOL, "description": SCOPE_SPEC[0], "input_schema": SCOPE_SPEC[1]}] + a2[-1:]


def system_a3(max_calls):
    return SYSTEM + "\n\n" + SCOPE_NOTE + "\n\n" + TOOLS_NOTE.format(n=max_calls)


def run_incident_a3(client, inc, model, acfg, status_polling, log=None, log_extra=None):
    """The frozen Phase 6 loop (`run_incident`) with the A3 tool list and system prompt; everything else is the same."""
    max_calls = acfg["max_tool_calls"]
    tools = tools_a3()
    system = system_a3(max_calls)
    messages = [{"role": "user", "content": incident_prompt(inc, "A3", status_polling)}]
    corpus = [messages[0]["content"]]
    usage = {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
    calls, requests, diagnosis, stop, forced, after_force = [], [], None, None, False, 0
    t_start = time.time()
    while True:
        n_exec = sum(not c["refused"] for c in calls)
        over = n_exec >= max_calls or sum(usage.values()) >= acfg["max_tokens_per_incident"]
        if over and not forced:
            forced = True
            messages.append({"role": "user", "content": FORCE_MSG})
        if forced:
            if after_force >= 2:
                stop = "budget"
                break
            after_force += 1
        t0 = time.time()
        resp = client.messages.create(model=model, max_tokens=acfg["max_output_tokens_per_turn"], tools=tools,
                                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                      messages=messages, output_config={"effort": acfg["effort"]}, cache_control={"type": "ephemeral"})
        u = {k: int(getattr(resp.usage, k) or 0) for k in usage}
        for k in usage:
            usage[k] += u[k]
        requests.append({"seconds": round(time.time() - t0, 2), "usage": u, "stop_reason": resp.stop_reason})
        if resp.stop_reason == "refusal":
            stop = "refusal"
            break
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            if len(requests) > acfg["max_tool_calls"] + 3 or forced:
                stop = "no_submission"
                break
            messages.append({"role": "user", "content": FORCE_MSG if forced else "Call submit_diagnosis to finish."})
            continue
        results = []
        for b in uses:
            if b.name == "submit_diagnosis":
                diagnosis = dict(b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": "submitted"})
                continue
            refused = forced or sum(not c["refused"] for c in calls) >= max_calls
            if refused:
                out, err = f"error: tool-call budget ({max_calls}) reached; only submit_diagnosis is accepted now", True
            elif b.name not in A3_TOOLS:
                out, err = f"error: tool {b.name} is not available", True
            else:
                out, err = call_tool(inc, b.name, dict(b.input))
            calls.append({"tool": b.name, "input": dict(b.input), "output": out, "error": err, "refused": refused})
            corpus.append(out)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        if diagnosis is not None:
            stop = "submitted"
            break
        messages.append({"role": "user", "content": results})
    out = {"condition": "A3", "model": model, "diagnosis": diagnosis, "stop": stop, "forced_submit": forced,
           "n_tool_calls": sum(not c["refused"] for c in calls), "n_refused_calls": sum(c["refused"] for c in calls),
           "n_requests": len(requests), "usage": usage,
           "cost_usd": round(_price(usage, acfg["pricing_usd_per_mtok"][model]), 6), "seconds": round(time.time() - t_start, 2),
           "calls": calls, "requests": requests,
           "groundedness": groundedness(diagnosis, "\n".join(corpus), inc.links + inc.pops + inc.services)}
    if log is not None:
        log.write(json.dumps({"incident": getattr(inc, "incident_id", None), **out, **(log_extra or {})}, default=str) + "\n")
        log.flush()
    return out


# ---------------------------------------------------------------- sampling

def sample_phase13(entries, phase6_incidents, seed, n_change_overlap, n_other_per_type, n_distractor):
    """Fresh incidents (see the module docstring). `phase6_incidents` are the Phase 6 incidents of the same split."""
    used = {}
    for i in phase6_incidents:
        used.setdefault(i["scenario_id"], []).append((i["t0"], i["t1"]))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entries))
    change, other = {t: [] for t in EASY + HARD}, {t: [] for t in EASY + HARD}
    distract = {a: [] for a in DISTRACTOR_ACTIONS}
    for i in order:
        e = entries[i]
        faults, prior = e["faults"], used.get(e["scenario_id"], [])
        for f in faults:
            if not f["impact"]["observable"] or f["t_start"] < WARMUP:
                continue
            if any(_overlaps(f["t_start"], f["t_end"], a, b) for a, b in prior):
                continue
            if any(g is not f and _overlaps(f["t_start"], f["t_end"] + LAG, g["t_start"], g["t_end"] + LAG) for g in faults):
                continue
            ov = overlaps_logged_change(e["maintenance"], f["t_start"], f["t_end"])
            inc = {"scenario_id": e["scenario_id"], "kind": "fault", "type": f["type"], "root": f["root_cause"],
                   "t0": f["t_start"], "t1": f["t_end"], "placement": f["placement"], "overlaps_logged_change": ov,
                   "caused_by_maintenance": bool(f.get("caused_by_maintenance")), "stratum": "change_overlap" if ov else "other_fault"}
            (change if ov else other)[f["type"]].append(inc)
        for m in e["maintenance"]:
            if m["action"] not in DISTRACTOR_ACTIONS or m["t_start"] < WARMUP:
                continue
            if any(_overlaps(m["t_start"], m["t_end"] + MAINT_TAIL, g["t_start"], g["t_end"] + LAG) for g in faults):
                continue
            if any(_overlaps(m["t_start"], m["t_end"], a, b) for a, b in prior):
                continue
            distract[m["action"]].append({"scenario_id": e["scenario_id"], "kind": "distractor", "type": f"distractor_{m['action']}",
                                          "root": "none", "change_target": m["target"], "t0": m["t_start"], "t1": m["t_end"],
                                          "stratum": "distractor"})
    out = []
    queues = {t: list(change[t]) for t in EASY + HARD}           # change-overlap faults: one type at a time in turn
    while len(out) < n_change_overlap and any(queues.values()):
        for t in EASY + HARD:
            if queues[t] and len(out) < n_change_overlap:
                out.append(queues[t].pop(0))
    for t in EASY + HARD:
        out += other[t][:n_other_per_type]
    for a in DISTRACTOR_ACTIONS:
        out += distract[a][:n_distractor]
    for k, inc in enumerate(out):
        inc["incident_id"] = f"P{k:03d}"
    return out


# ---------------------------------------------------------------- summaries (scenario-cluster bootstrap)

GROUPS = {"change_overlap": lambda i: i["stratum"] == "change_overlap",
          "change_overlap_caused": lambda i: i["stratum"] == "change_overlap" and i.get("caused_by_maintenance", False),
          "change_overlap_other_element": lambda i: i["stratum"] == "change_overlap" and not i.get("caused_by_maintenance", False),
          "other_fault": lambda i: i["stratum"] == "other_fault",
          "all_faults": lambda i: i["kind"] == "fault",
          "distractors": lambda i: i["kind"] == "distractor"}
METRICS = ("claims_fault", "top1", "top3", "type_correct")


def cluster_weights(incidents, n_boot, seed):
    """Bootstrap weights over scenarios, expanded to incidents: (n_boot + 1, n_incidents), row 0 all ones."""
    scen = sorted({i["scenario_id"] for i in incidents})
    W = boot_weights(len(scen), n_boot, seed)
    col = [scen.index(i["scenario_id"]) for i in incidents]
    return W[:, col]


def summarise13(incidents, by_method, pairs, n_boot=1000, seed=42):
    W = cluster_weights(incidents, n_boot, seed)
    boots, table = {}, {}
    for m, scores in by_method.items():
        for g, fn in GROUPS.items():
            mask = np.array([fn(i) for i in incidents])
            for metric in METRICS:
                v = np.array([np.nan if s.get(metric) is None else float(s[metric]) for s in scores])
                ok = mask & np.isfinite(v)
                if not ok.any():
                    continue
                Wm = W * ok
                with np.errstate(invalid="ignore", divide="ignore"):
                    b = (Wm @ np.nan_to_num(v)) / Wm.sum(1)
                boots[(m, g, metric)] = b
                table.setdefault(m, {}).setdefault(g, {})[metric] = {"value": round(float(b[0]), 4), "ci": ci(b[1:]), "n": int(ok.sum())}
    comps = {}
    for a, b in pairs:
        for (m, g, metric), arr in boots.items():
            if m == a and (b, g, metric) in boots:
                d = arr - boots[(b, g, metric)]
                if metric == "claims_fault" and g == "distractors":
                    d = -d                                   # on distractors, fewer fault claims is better
                comps[f"{g}|{metric}|{a} vs {b}"] = verdict(d)
    return table, comps
