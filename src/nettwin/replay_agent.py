"""The Phase 12 triage agent: woken when a GNN alarm episode opens, it sees only data up to that step and decides to
escalate (page an operator, with a root cause), dismiss (maintenance or noise) or wait and re-check once.

It reuses the frozen Phase 6 tools through a subclass (`ReplayIncident`); nothing in `nettwin.agent` is changed.
Step 0 is the step the episode opened; the agent's data ends at `now - 1` (the decision step). Alerts come through
`alerts_at`, telemetry and GNN scores are cut at `now`, and the KG is the live KG as known at the decision step (only
status events that have arrived). One extra tool, `kg_current_state`, shows what that live KG holds.
"""
import json
import time

from nettwin.agent.loop import _price
from nettwin.agent.tools import CONDITIONS, FAULT_TYPES, HISTORY, TOOL_SPECS, Incident, _schema, call_tool

MAX_TOOL_CALLS = 10
MAX_TOKENS_PER_WAKE = 80_000
MAX_OUTPUT_TOKENS_PER_TURN = 4000
EFFORT = "medium"
DECISIONS = ("escalate", "dismiss_maintenance", "dismiss_noise", "wait")


class ReplayIncident(Incident):
    """Phase 6 tools at a decision step of the replay. Window start t0 = the episode's opening step."""

    def __init__(self, obs, live_kg, t_open, now, cfg_s, alerts, gnn=None, runbooks=None):
        self.obs, self.kg, self.cfg_s = obs, live_kg, cfg_s
        self.t0, self.now = int(t_open), int(now)
        self.floor = self.t0 - HISTORY
        if self.floor < 0:
            raise ValueError(f"episode opens at {self.t0}, before {HISTORY} steps of history")
        self.alerts = alerts
        self.gnn, self.runbooks = gnn, runbooks
        self._gnn_cache = None
        self.links, self.services = obs.links, obs.services
        self.pops = live_kg.nodes_of_kind("PoP")

    def kg_current_state(self):
        """Elements the live KG holds as down at the decision step (from status events only), and recent reroutes."""
        t = self.now - 1
        down = [{"element": e, "down_since": self.rel(max(self.kg.G.nodes[e]["intervals_down"][-1][0], self.floor))}
                for e in sorted(self.kg._down_links | self.kg._down_pops)]
        ev = [e for e in self.kg.events if e["t"] >= self.floor]
        reroutes = [e for e in ev if e["type"] == "reroute"]
        return {"at_step": self.rel(t), "elements_down": down,
                "status_changes": [{"t": self.rel(e["t"]), "type": e["type"], "element": e["element"]} for e in ev if e["type"] != "reroute"][-20:],
                "pop_pairs_rerouted": len(reroutes), "pop_pairs_unreachable": sum(bool(e.get("unreachable")) for e in reroutes),
                "note": "built from interface status and router reachability events as they arrived (polling lag included); "
                        "degradations, surges and metric changes do not appear here"}


TOOLS_USED = CONDITIONS["A2"]
EXTRA_SPECS = {"kg_current_state": ("Elements the live inventory graph marks as down now (from status events as they "
                                    "arrived) and how many router pairs were rerouted.", _schema({}, []))}


def submit_tool(allow_wait):
    opts = list(DECISIONS) if allow_wait else [d for d in DECISIONS if d != "wait"]
    return {"name": "submit_triage", "description": "Submit the triage decision. Call exactly once, at the end.", "strict": True,
            "input_schema": _schema({
                "decision": {"type": "string", "enum": opts},
                "root_cause": {"type": "string", "description": "for escalate: element id of the root cause (link:... or pop:...); otherwise 'none'"},
                "alternatives": {"type": "array", "items": {"type": "string"}, "description": "up to 2 other candidate root elements, most likely first"},
                "fault_type": {"type": "string", "enum": list(FAULT_TYPES) + ["none"]},
                "reason": {"type": "string", "description": "one or two sentences"},
            }, ["decision", "root_cause", "alternatives", "fault_type", "reason"])}


def tool_list(allow_wait):
    return ([{"name": n, "description": TOOL_SPECS[n][0], "input_schema": TOOL_SPECS[n][1]} for n in TOOLS_USED]
            + [{"name": n, "description": d, "input_schema": s} for n, (d, s) in EXTRA_SPECS.items()] + [submit_tool(allow_wait)])


SYSTEM = f"""You are the on-call network operations engineer for an IP backbone, triaging alarms as they happen.
A graph neural network (GNN) watching live telemetry has raised an alarm episode. Decide what to do now:
- escalate: page the operations team; name the single most likely root-cause element (a link 'link:...' or a router \
'pop:...'), up to 2 alternatives and the fault type;
- dismiss_maintenance: the symptoms are explained by planned work in the change log;
- dismiss_noise: nothing is wrong (normal traffic variation, a telemetry gap, a false alarm);
- wait: not enough evidence yet; you will be woken once more {5 * 6} minutes later and must then escalate or dismiss.
Pages cost operator time, so do not page for noise or planned work. Missing a real fault is worse than one extra page.

Fault types: link_down, node_down (a whole router), link_flapping, link_degradation (loss or latency on an up link), \
demand_surge (extra traffic from one router; the root is that router), routing_misconfig (a routing metric changed on \
one link without a logged change; the root is that link).

Times are 5-minute steps relative to the step the episode opened (step 0); negative steps are earlier. You only see \
data up to the current step. You have at most {MAX_TOOL_CALLS} tool calls. Finish by calling submit_triage exactly once."""
FORCE_MSG = "Budget reached. Call submit_triage now with your best decision."


def wake_prompt(inc, ep, prior=None):
    rel = inc.rel
    lines = [f"Alarm episode {ep['episode_id']}: opened at step 0 with the GNN's top root candidate {ep['element']}.",
             f"Links over the alarm threshold at step 0: {', '.join(ep['alarm_links'])}.",
             f"Current step: {rel(inc.now - 1)}. Alarm steps on this element so far: "
             f"{[rel(t) for t in ep['alarm_steps'] if t < inc.now][:20]}."]
    if prior:
        lines.append(f"At step 0 you chose to wait (reason: {prior}). This is the re-check: escalate or dismiss now.")
    return "\n".join(lines)


def run_wake(client, inc, ep, model, prices, prior=None, log=None, log_extra=None):
    """One agent wake. Returns the decision (or None), usage, cost and the tool calls."""
    allow_wait = prior is None
    tools = tool_list(allow_wait)
    messages = [{"role": "user", "content": wake_prompt(inc, ep, prior)}]
    usage = {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
    calls, decision, stop, forced, after = [], None, None, False, 0
    t_start = time.time()
    while stop is None:
        if forced:
            after += 1
            if after > 3:
                stop = "budget"
                break
        resp = client.messages.create(model=model, max_tokens=MAX_OUTPUT_TOKENS_PER_TURN, tools=tools,
                                      system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                                      messages=messages, output_config={"effort": EFFORT}, cache_control={"type": "ephemeral"})
        for k in usage:
            usage[k] += int(getattr(resp.usage, k) or 0)
        if resp.stop_reason == "refusal":
            stop = "refusal"
            break
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            messages.append({"role": "user", "content": "Call submit_triage to finish."})
            forced = True
            continue
        results = []
        for b in uses:
            if b.name == "submit_triage":
                decision = dict(b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": "submitted"})
            elif b.name not in {t["name"] for t in tools}:
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": f"error: unknown tool {b.name!r}"})
            elif forced or len(calls) >= MAX_TOOL_CALLS:
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": "error: no more tool calls; call submit_triage now"})
            else:
                if b.name == "kg_current_state":
                    out, err = json.dumps(inc.kg_current_state(), separators=(",", ":")), False
                else:
                    out, err = call_tool(inc, b.name, dict(b.input))
                calls.append({"tool": b.name, "input": dict(b.input), "error": err, "chars": len(out)})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        if decision is not None:
            stop = "submitted"
            break
        if not forced and (len(calls) >= MAX_TOOL_CALLS or sum(usage.values()) >= MAX_TOKENS_PER_WAKE):
            forced = True
            results.append({"type": "text", "text": FORCE_MSG})
        messages.append({"role": "user", "content": results})
    rec = {"decision": decision, "stop": stop, "n_tool_calls": len(calls), "calls": calls, "usage": usage,
           "cost_usd": round(_price(usage, prices), 6), "seconds": round(time.time() - t_start, 2), "model": model}
    if log is not None:
        log.write(json.dumps({**(log_extra or {}), **rec}) + "\n")
    return rec
