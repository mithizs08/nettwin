"""Agent loop with tool-call and token budgets (Phase 6). Manual tool-use loop on the Messages API, every request and
tool call logged, cost from the token usage and the prices in config.yaml."""
import json
import re
import time

from nettwin.agent.tools import CONDITIONS, FAULT_TYPES, NO_FAULT, call_tool, tools_for

SYSTEM = """You are a network operations engineer diagnosing an incident on an IP backbone.
An incident window has been opened. Decide whether a fault happened in it. If so, name the single root-cause element \
(a link 'link:...' or a router 'pop:...') and the fault type. If the symptoms are fully explained by planned \
maintenance, or nothing is wrong, say so: fault_type 'planned_maintenance' or 'no_fault' with root_cause 'none'.

Fault types: link_down, node_down (a whole router), link_flapping, link_degradation (loss or latency on an up link), \
demand_surge (extra traffic from one router; the root is that router), routing_misconfig (a routing metric changed on \
one link without a logged change; the root is that link).

Times are 5-minute steps relative to the incident window start (step 0); negative steps are earlier. You only see data \
up to the end of the window.

Finish by calling submit_diagnosis exactly once. List up to 2 alternative root elements, most likely first. Every \
evidence item must quote element ids and values exactly as you saw them; do not state values you did not see."""

FORCE_MSG = "Budget reached. Call submit_diagnosis now with your best answer."
A0_NOTE = "You have no tools other than submit_diagnosis; decide from the alert list below."
TOOLS_NOTE = "Use the tools to investigate before submitting. You have at most {n} tool calls."


def incident_prompt(inc, condition, status_polling):
    w = inc.now - inc.t0
    lines = [f"Incident window: steps 0 to {w - 1} ({w * 5} minutes). Interface status polling: "
             f"{'available' if status_polling else 'NOT available (no up/down events; only telemetry)'}."]
    if condition == "A0":
        lines += [A0_NOTE, "Alerts active in the window, as known at its end:",
                  json.dumps(inc.get_active_alerts(), separators=(",", ":"))]
    return "\n".join(lines)


def _price(usage, prices):
    p = prices
    return (usage["input_tokens"] * p["input"] + usage["cache_creation_input_tokens"] * p["cache_write_5m"]
            + usage["cache_read_input_tokens"] * p["cache_read"] + usage["output_tokens"] * p["output"]) / 1e6


def run_incident(client, inc, condition, model, acfg, status_polling, log=None):
    """One incident under one condition. Returns the diagnosis (or None), usage, cost, latency and the full log."""
    max_calls = acfg["max_tool_calls"]
    tools = tools_for(condition)
    system = SYSTEM + "\n\n" + (A0_NOTE if condition == "A0" else TOOLS_NOTE.format(n=max_calls))
    messages = [{"role": "user", "content": incident_prompt(inc, condition, status_polling)}]
    corpus = [messages[0]["content"]]
    usage = {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
    calls, requests, diagnosis, stop, forced, after_force = [], [], None, None, False, 0
    t_start = time.time()
    while True:
        # the tool list never changes within a conversation: changing it invalidates earlier thinking blocks
        over = len(calls) >= max_calls or sum(usage.values()) >= acfg["max_tokens_per_incident"]
        if over and not forced and condition != "A0":
            forced = True
            messages.append({"role": "user", "content": FORCE_MSG})
        if forced:
            if after_force >= 2:
                stop = "budget"
                break
            after_force += 1
        t0 = time.time()
        # two cache breakpoints: the tools + system prefix (identical across incidents of one condition) and, through
        # top-level automatic caching, the conversation so far (reused by the next request of this incident)
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
            if forced:
                out, err = "error: budget reached; only submit_diagnosis is accepted now", True
            elif b.name not in CONDITIONS[condition]:
                out, err = f"error: tool {b.name} is not available", True
            else:
                out, err = call_tool(inc, b.name, dict(b.input))
            calls.append({"tool": b.name, "input": dict(b.input), "output": out, "error": err})
            corpus.append(out)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        if diagnosis is not None:
            stop = "submitted"
            break
        messages.append({"role": "user", "content": results})
    out = {"condition": condition, "model": model, "diagnosis": diagnosis, "stop": stop, "forced_submit": forced,
           "n_tool_calls": len(calls), "n_requests": len(requests), "usage": usage,
           "cost_usd": round(_price(usage, acfg["pricing_usd_per_mtok"][model]), 6), "seconds": round(time.time() - t_start, 2),
           "calls": calls, "requests": requests,
           "groundedness": groundedness(diagnosis, "\n".join(corpus), inc.links + inc.pops + inc.services)}
    if log is not None:
        log.write(json.dumps({"incident": getattr(inc, "incident_id", None), **out}, default=str) + "\n")
        log.flush()
    return out


# ---------------------------------------------------------------- groundedness

ID_RE = re.compile(r"\b(?:link|pop|svc):[^\s,;()\[\]{}\"']+")   # ids without spaces, to catch invented ones
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?")


def _cited_ids(text, known):
    """Inventory ids named in the text (longest first, so names with spaces are found), and other id-like tokens."""
    found, rest = [], text
    for k in sorted(known, key=len, reverse=True):
        if k in rest:
            found.append(k)
            rest = rest.replace(k, " ")
    other = [t.rstrip(".:") for t in ID_RE.findall(rest) if t.split(":", 1)[1].strip(".:")]   # skip placeholders: 'pop:...'
    return found, other, rest


def _numbers(text):
    return [float(x) for x in NUM_RE.findall(text)]


def groundedness(diagnosis, corpus, known=()):
    """Every element id and number cited in the evidence must appear in the tool outputs (or, for A0, the prompt).
    Ids are matched against the incident's inventory (names may contain spaces); an id-like token outside the
    inventory is ungrounded. A number counts as grounded if some number in the corpus equals it after rounding to the
    cited precision."""
    if not diagnosis:
        return {"n_items": 0, "n_ungrounded_items": 0, "ungrounded": []}
    c_ids, c_other, c_rest = _cited_ids(corpus, known)
    in_corpus = set(c_ids) | set(c_other)
    nums = _numbers(c_rest)
    bad = []
    for item in diagnosis.get("evidence", []):
        ids, other, rest = _cited_ids(item, known)
        miss = [i for i in ids + other if i not in in_corpus]
        for raw in NUM_RE.findall(rest):
            nd = len(raw.split(".")[1]) if "." in raw else 0
            v = float(raw)
            if not any(round(c, nd) == v for c in nums):
                miss.append(raw)
        if miss:
            bad.append({"item": item, "not_found": miss})
    return {"n_items": len(diagnosis.get("evidence", [])), "n_ungrounded_items": len(bad), "ungrounded": bad}


def is_fault_claim(diagnosis):
    return bool(diagnosis) and diagnosis.get("fault_type") in FAULT_TYPES


__all__ = ["run_incident", "groundedness", "is_fault_claim", "NO_FAULT"]
