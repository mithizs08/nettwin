"""Agent tools over observable data only (Phase 6, PLAN section 7).

One `Incident` holds what an operator would have when the incident is opened at the end of its window:
- observable telemetry and observed events (no truth files are read here; `load_observed` refuses truth paths);
- the operator inventory KG (`inventory`), alerts generated from observable telemetry and read through `alerts_at`;
- the change log: every planned change (drain, planned outage, no-op work) whose window has started by then;
- the frozen GNN (Phase 5, hash-checked by the caller) and the runbooks.

Time. Every tool works in steps relative to the incident window start (t = 0 is the window start, negative is
earlier; one step is 5 minutes), so no tool exposes time since the scenario start (audit P4, P7). History is visible
from HISTORY steps before the window start: every window starts at least that late, so a range is never clipped at the
scenario start and outputs are the same wherever the window sits in the scenario (Phase 6 audit S3). Nothing at or
after `now` (the window end, the same horizon the other methods use) is returned.
"""
import json
import threading

import numpy as np

from nettwin.alerts import RULES, alerts_at, generate_alerts
from nettwin.baselines import b2_features, window_score
from nettwin.gnn.data import graph_for
from nettwin.gnn.explain import explain
from nettwin.gnn.models import predict

DAY = 288
GNN_LOCK = threading.Lock()       # the shared frozen model: one forward or backward pass at a time when runs are threaded
FAULT_TYPES = ("link_down", "node_down", "link_flapping", "link_degradation", "demand_surge", "routing_misconfig")
NO_FAULT = ("no_fault", "planned_maintenance")
MAX_SERIES_POINTS = 24
HISTORY = DAY                    # visible history before the window start, in steps (= evaluation.WARMUP)
LINK_METRICS = {"util": "util", "latency": "latency", "discards": "discards"}
SERVICE_METRICS = {"latency": "svc_latency", "loss": "svc_loss"}


def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


class Incident:
    def __init__(self, obs, kg, t0, t1, cfg_s, gnn=None, runbooks=None):
        self.obs, self.kg, self.cfg_s = obs, kg, cfg_s
        self.t0, self.now = int(t0), int(t1)                     # window [t0, t1); data up to now - 1
        self.floor = self.t0 - HISTORY                           # earliest visible step
        if self.floor < 0:
            raise ValueError(f"window starts at {self.t0}, before {HISTORY} steps of history")
        self.alerts = generate_alerts(obs, kg)
        self.gnn, self.runbooks = gnn, runbooks
        self._gnn_cache = None
        self.links, self.services = obs.links, obs.services
        self.pops = kg.nodes_of_kind("PoP")

    # ---------------------------------------------------------------- helpers
    def rel(self, t):
        return int(t) - self.t0

    def abs_range(self, t0, t1):
        """Relative [t0, t1] (inclusive) to absolute [a, b) clipped to what is known at `now`."""
        a = max(self.floor, self.t0 + int(t0))
        b = min(self.now, self.t0 + int(t1) + 1)
        if b <= a:
            raise ValueError(f"empty range: [{t0}, {t1}] relative to the window start; data ends at t={self.now - self.t0 - 1}")
        return a, b

    def pop_links(self, pop):
        name = pop[4:]
        return [l for l in self.links if name in (self.kg.G.nodes[l]["u"], self.kg.G.nodes[l]["v"])]

    def element_kind(self, el):
        if el in self.links:
            return "link"
        if el in self.pops:
            return "pop"
        if el in self.services:
            return "service"
        raise ValueError(f"unknown element {el!r}; ids look like 'link:R1--R2', 'pop:R1' or a service id from the alerts")

    # ---------------------------------------------------------------- alerts
    def get_active_alerts(self, t0=None, t1=None, max_alerts=60):
        t0 = 0 if t0 is None else t0
        t1 = self.now - self.t0 - 1 if t1 is None else t1
        a, b = self.abs_range(t0, t1)
        known = alerts_at(self.alerts, self.now - 1)
        end = known.t_end.fillna(np.inf)
        sel = known[(known.t_start < b) & (end > a)]
        sev = {"critical": 0, "major": 1, "minor": 2, "info": 3}
        sel = sel.assign(_s=sel.severity.map(sev)).sort_values(["_s", "t_start", "target"], kind="stable")
        rows = [{"rule": r.rule, "target": r.target, "severity": r.severity, "start": self.rel(max(r.t_start, self.floor)),
                 "end": None if not np.isfinite(r.t_end) else self.rel(r.t_end), "value": _r(r.value)}
                | ({"started_before_visible_history": True} if r.t_start < self.floor else {})
                for r in sel.head(max_alerts).itertuples()]
        return {"window": [t0, t1], "n_alerts": int(len(sel)), "shown": len(rows),
                "counts_by_rule": {k: int(v) for k, v in sel.rule.value_counts().sort_index().items()}, "alerts": rows,
                "rules": {k: v[3] for k, v in RULES.items() if k in set(sel.rule)}}

    # ---------------------------------------------------------------- telemetry
    def _series(self, kind, el, metric):
        if kind == "link":
            if metric not in LINK_METRICS:
                raise ValueError(f"link metrics: {sorted(LINK_METRICS)}")
            return getattr(self.obs, LINK_METRICS[metric])[:, self.links.index(el)]
        if kind == "service":
            if metric not in SERVICE_METRICS:
                raise ValueError(f"service metrics: {sorted(SERVICE_METRICS)}")
            return getattr(self.obs, SERVICE_METRICS[metric])[:, self.services.index(el)]
        raise ValueError("use a link or a service; for a PoP call it on its links (kg_neighbours lists them)")

    def get_metric_series(self, element, metric, t0, t1):
        kind = self.element_kind(element)
        if kind == "pop":
            links = self.pop_links(element)
            return {"element": element, "note": "a PoP has no series of its own; per attached link summaries follow",
                    "links": [self.get_metric_series(l, metric, t0, t1)["summary"] | {"link": l} for l in links]}
        x = self._series(kind, element, metric)
        a, b = self.abs_range(t0, t1)
        seg = x[a:b]
        prev = x[max(self.floor, a - DAY):b - DAY] if b - DAY > self.floor else np.array([])
        n = len(seg)
        k = int(np.ceil(n / MAX_SERIES_POINTS))
        pts = []
        for s in range(0, n, k):
            chunk = seg[s:s + k]
            ok = chunk[np.isfinite(chunk)]
            pts.append({"t": self.rel(a + s), "mean": _r(ok.mean()) if len(ok) else None, "missing": int(len(chunk) - len(ok))})
        ok = seg[np.isfinite(seg)]
        okp = prev[np.isfinite(prev)] if len(prev) else prev
        summ = {"mean": _r(ok.mean()) if len(ok) else None, "min": _r(ok.min()) if len(ok) else None,
                "max": _r(ok.max()) if len(ok) else None, "missing_steps": int(n - len(ok)), "steps": n,
                "zero_steps": int((ok == 0).sum()) if len(ok) else 0,
                "same_window_one_day_earlier_mean": _r(okp.mean()) if len(okp) else None}
        return {"element": element, "metric": metric, "range": [self.rel(a), self.rel(b - 1)], "points_per_bin": k,
                "series": pts, "summary": summ}

    def list_links_by_change(self, metric, t0, t1, top_k=10):
        """Links ranked by the change of a metric in [t0, t1] against the same steps one day earlier (a dashboard view)."""
        if metric not in LINK_METRICS:
            raise ValueError(f"link metrics: {sorted(LINK_METRICS)}")
        a, b = self.abs_range(t0, t1)
        x = getattr(self.obs, LINK_METRICS[metric])
        now = x[a:b]
        out = []
        with np.errstate(invalid="ignore"):
            cur = np.nanmean(now, 0) if a < b else np.full(x.shape[1], np.nan)
            miss = np.isnan(now).mean(0)
            if b - DAY > self.floor:
                base = np.nanmean(x[max(self.floor, a - DAY):b - DAY], 0)
            else:
                base = np.full(x.shape[1], np.nan)
        delta = cur - base
        key = np.where(np.isnan(delta), -np.inf, np.abs(delta))
        for j in np.argsort(-key, kind="stable")[:top_k]:
            out.append({"link": self.links[j], "mean_now": _r(cur[j]), "mean_day_before": _r(base[j]),
                        "change": _r(delta[j]), "missing_fraction": _r(miss[j], 2)})
        fully_missing = [self.links[j] for j in np.flatnonzero(miss == 1.0)]
        return {"metric": metric, "range": [self.rel(a), self.rel(b - 1)], "top_by_absolute_change": out,
                "links_with_no_data_in_range": fully_missing[:20], "n_links_with_no_data": len(fully_missing)}

    # ---------------------------------------------------------------- KG
    def kg_paths_through(self, link, max_paths=15):
        self.element_kind(link)
        paths = self.kg.paths_through(link, t=0)
        return {"element": link, "n_paths": len(paths),
                "paths": [{"path": p, "pops": list(self.kg.G.nodes[p]["pops"])} for p in paths[:max_paths]]}

    def kg_services_on(self, element, max_services=25):
        self.element_kind(element)
        s = self.kg.services_on(element, t=0)
        return {"element": element, "n_services": len(s), "services": s[:max_services]}

    def kg_neighbours(self, element, hops=1):
        kind = self.element_kind(element)
        hops = int(min(max(hops, 1), 2))
        out = {"element": element, "hops": hops, "neighbours": self.kg.neighbours(element, hops)[:40]}
        if kind == "pop":
            out["attached_links"] = self.pop_links(element)
        return out

    def kg_shared_elements(self, alerted, top_k=12):
        """Elements shared by the normal routing paths of the given services (or routing path ids)."""
        paths = []
        for x in alerted:
            if x in self.services:
                paths += self.kg.service_paths(x, t=0)
            elif x in self.kg.G and self.kg.kind(x) == "RoutingPath":
                paths.append(x)
            else:
                raise ValueError(f"{x!r} is not a service or routing path id")
        paths = sorted(set(paths))
        shared = self.kg.shared_elements(paths)
        return {"n_paths": len(paths), "top_shared": [{"element": e, "paths": c} for e, c in shared[:top_k]]}

    # ---------------------------------------------------------------- change log
    def get_change_log(self, t0, t1):
        """Planned changes (drains, planned outages, no-op work) overlapping [t0, t1] that had started by now."""
        a, b = self.abs_range(t0, t1)
        rows = [{"id": e["id"], "action": e["action"], "target": e["target"], "start": self.rel(max(e["t_start"], self.floor)),
                 "end": self.rel(e["t_end"]), "description": e.get("description", "")}
                for e in self.obs.events if e.get("type") == "maintenance"
                and e["t_start"] < self.now and e["t_start"] < b and e["t_end"] > a]
        return {"range": [self.rel(a), self.rel(b - 1)], "changes": sorted(rows, key=lambda r: (r["start"], r["id"])),
                "actions": {"drain": "routing metric raised so traffic leaves the element; nothing is broken",
                            "down": "element taken down on purpose (planned outage)", "noop": "logged work with no effect on traffic"}}

    # ---------------------------------------------------------------- GNN
    def _gnn_scores(self):
        if self._gnn_cache is None:
            g = graph_for(self.obs.topology, self.cfg_s)
            feat = b2_features(self.obs).astype(np.float32)[:self.now]           # causal: nothing at or after now
            with GNN_LOCK:
                det, elem = predict(self.gnn, feat, g)
            self._gnn_cache = (g, feat, det, elem)
        return self._gnn_cache

    def get_gnn_scores(self, t=None, top_k=8, over_window=True):
        g, _, det, elem = self._gnn_scores()
        t = self.now - self.t0 - 1 if t is None else int(t)
        a, b = self.abs_range(0 if over_window else t, t)
        sc = window_score(elem[a:b], "q90") if over_window else elem[b - 1]
        order = np.argsort(-sc, kind="stable")[:top_k]
        dmax = det[a:b].max(0)
        top_links = np.argsort(-dmax, kind="stable")[:5]
        return {"range": [self.rel(a), self.rel(b - 1)], "aggregation": "q90 over steps" if over_window else "single step",
                "root_probability_top": [{"element": g.elements[i], "probability": _r(sc[i], 4)} for i in order],
                "anomaly_probability_top_links": [{"link": g.elements[j], "max_probability": _r(dmax[j], 4)} for j in top_links],
                "note": "probabilities from a graph neural network trained on synthetic incidents; it does not see the change log"}

    def explain_gnn(self, element, t=None):
        g, feat, _, _ = self._gnn_scores()
        if element not in g.elements:
            raise ValueError(f"unknown element {element!r}")
        t = self.now - self.t0 - 1 if t is None else int(t)
        a, b = self.abs_range(t, t)
        with GNN_LOCK:
            out = explain(self.gnn, feat, g, g.elements.index(element), b - 1)
        return {"element": element, "t": self.rel(b - 1), **out}

    # ---------------------------------------------------------------- runbooks
    def search_runbooks(self, query, k=2):
        return {"query": query, "results": [{"runbook": rid, "score": s, "text": text} for rid, s, text in self.runbooks.search(query, k)]}


# ---------------------------------------------------------------- tool schemas per condition

def _schema(props, required):
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


T_RANGE = {"t0": {"type": "integer", "description": f"start step, relative to the incident window start (negative = earlier; history starts at -{HISTORY})"},
           "t1": {"type": "integer", "description": "end step (inclusive), relative to the incident window start"}}

TOOL_SPECS = {
    "get_active_alerts": ("Alerts active at any point in [t0, t1], as known now. Defaults to the incident window.",
                          _schema({"t0": T_RANGE["t0"], "t1": T_RANGE["t1"]}, [])),
    "get_metric_series": ("Telemetry of one link (util, latency, discards) or service (latency, loss) over [t0, t1], "
                          "binned, with a summary and the same window one day earlier. For a PoP, per-link summaries.",
                          _schema({"element": {"type": "string"}, "metric": {"type": "string", "enum": ["util", "latency", "discards", "loss"]}, **T_RANGE},
                                  ["element", "metric", "t0", "t1"])),
    "list_links_by_change": ("Links ranked by how much a metric changed in [t0, t1] against the same steps one day earlier, "
                             "plus links with no data in the range.",
                             _schema({"metric": {"type": "string", "enum": ["util", "latency", "discards"]}, **T_RANGE,
                                      "top_k": {"type": "integer"}}, ["metric", "t0", "t1"])),
    "kg_paths_through": ("Normal routing paths that traverse a link or pass through a PoP.", _schema({"link": {"type": "string"}}, ["link"])),
    "kg_services_on": ("Services whose normal paths use a link or PoP.", _schema({"element": {"type": "string"}}, ["element"])),
    "kg_neighbours": ("Neighbours of a link (links sharing a router) or PoP (adjacent PoPs, plus its attached links).",
                      _schema({"element": {"type": "string"}, "hops": {"type": "integer", "description": "1 or 2"}}, ["element"])),
    "kg_shared_elements": ("Links and PoPs shared by the normal paths of the given services or routing paths, most shared first.",
                           _schema({"alerted": {"type": "array", "items": {"type": "string"}}}, ["alerted"])),
    "get_change_log": ("Planned changes (drains, planned outages, no-op work) overlapping [t0, t1].", _schema(dict(T_RANGE), ["t0", "t1"])),
    "get_gnn_scores": ("Root-cause probabilities from the trained graph model, q90 over [0, t] by default, and its most anomalous links.",
                       _schema({"t": {"type": "integer"}, "top_k": {"type": "integer"}, "over_window": {"type": "boolean"}}, [])),
    "explain_gnn": ("Which link features drive the graph model's probability for an element at step t (gradient x input).",
                    _schema({"element": {"type": "string"}, "t": {"type": "integer"}}, ["element"])),
    "search_runbooks": ("Search the operations runbooks.", _schema({"query": {"type": "string"}}, ["query"])),
}

SUBMIT = {
    "name": "submit_diagnosis",
    "description": "Submit the final diagnosis. Call exactly once, at the end.",
    "strict": True,
    "input_schema": _schema({
        "root_cause": {"type": "string", "description": "element id of the root cause (link:... or pop:...), or 'none' if there is no fault"},
        "fault_type": {"type": "string", "enum": list(FAULT_TYPES + NO_FAULT)},
        "confidence": {"type": "number", "description": "0 to 1"},
        "alternatives": {"type": "array", "items": {"type": "string"}, "description": "up to 2 other candidate root elements, most likely first"},
        "evidence": {"type": "array", "items": {"type": "string"}, "description": "short facts quoting element ids and values exactly as the tools returned them"},
        "recommended_action": {"type": "string"},
    }, ["root_cause", "fault_type", "confidence", "alternatives", "evidence", "recommended_action"]),
}

CONDITIONS = {
    "A0": (),
    "A1": ("get_active_alerts", "get_metric_series", "list_links_by_change", "kg_paths_through", "kg_services_on",
           "kg_neighbours", "kg_shared_elements", "get_change_log"),
}
CONDITIONS["A2"] = CONDITIONS["A1"] + ("get_gnn_scores", "explain_gnn", "search_runbooks")


def tools_for(condition):
    return [{"name": n, "description": TOOL_SPECS[n][0], "input_schema": TOOL_SPECS[n][1]} for n in CONDITIONS[condition]] + [SUBMIT]


def call_tool(inc, name, args):
    """Run one tool; errors come back as text so the agent can correct itself."""
    try:
        fn = getattr(inc, name)
        if name == "kg_paths_through":
            out = fn(args["link"])
        else:
            out = fn(**args)
        return json.dumps(out, separators=(",", ":")), False
    except (ValueError, KeyError, TypeError) as e:
        return f"error: {e}", True
