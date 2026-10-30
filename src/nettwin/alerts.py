"""Threshold alerts from observable telemetry only (Phase 4; parked item from Phase 1).

These are the alerts a NOC would have from an ordinary NMS: interface status, PoP reachability, high utilisation,
interface discards, one-hop latency and probe failures, and per-service SLA breaches on end-to-end probes. Inputs
are an `Observed` scenario (observable tree only) and the operator's static inventory (topology, services, SLAs).
Ground truth is never read here.

Thresholds are fixed operator conventions or the synthetic SLAs in `config.yaml`; none was tuned on any split.
A metric rule must hold for PERSIST consecutive steps. The alert is raised on the step that completes the run, so
an alert active at step t only uses data up to t. Alerts are episodes [t_start, t_end), end exclusive.

Causality of every field (audit Phase 4, Q9): `t_start`, `rule`, `target`, `severity` and `value` are known at the raise
step (`value` is the reading on that step, not the episode peak). `t_end` is the clearing step, known only from
then on; an episode still open when the data ends has t_end = n_steps. Anything that answers "what alerts are there at
time t" (Phase 6 tools) must go through `alerts_at`, which hides the clear time of alerts still open at t.
"""
import numpy as np
import pandas as pd

PERSIST = 2
RULES = {
    # name: (target kind, severity, threshold, description)
    "link_status_down": ("link", "critical", None, "interface reported down by the poller (status event)"),
    "pop_unreachable": ("pop", "critical", None, "router not answering NMS polls (reachability event)"),
    "link_util_high": ("link", "major", 0.90, "utilisation >= 90% of capacity"),
    "link_discards": ("link", "major", 0.05, "interface discards >= 0.05% of packets"),
    "link_latency_high": ("link", "minor", 2.0, "one-hop latency >= 2x its trailing one-day EWMA and >= +1 ms"),
    "link_probe_fail": ("link", "minor", None, "one-hop probe missing while interface counters are present"),
    "link_no_data": ("link", "info", None, "no telemetry at all for the link (collector gap or blanked)"),
    "svc_unreachable": ("service", "critical", 99.0, "end-to-end probe loss >= 99%"),
    "svc_loss_sla": ("service", "major", None, "15-minute mean end-to-end probe loss >= the service's SLA loss"),
    "svc_latency_sla": ("service", "major", None, "end-to-end latency >= the service's SLA latency"),
}
COLUMNS = ["alert_id", "rule", "target", "severity", "t_start", "t_end", "value"]
DAY_STEPS = 288
SLA_STEPS = 3   # SLA loss is judged over a 15-minute reporting interval, not one 5-minute sample


def _episodes(mask, persist=PERSIST):
    """Runs of True of length >= persist in each column of a (T, N) mask, as (col, start, end) with start shifted
    to the step where the run reached `persist` steps (causal)."""
    out = []
    T = mask.shape[0]
    padded = np.vstack([np.zeros((1, mask.shape[1]), bool), mask, np.zeros((1, mask.shape[1]), bool)])
    diff = np.diff(padded.astype(np.int8), axis=0)
    for j in range(mask.shape[1]):
        starts = np.flatnonzero(diff[:, j] == 1)
        ends = np.flatnonzero(diff[:, j] == -1)
        for s, e in zip(starts, ends):
            if e - s >= persist:
                out.append((j, int(s + persist - 1), int(min(e, T))))
    return out


def _trailing_ewma(x, span):
    """EWMA of past observed values only (excludes the current step); NaN where there is no history."""
    return pd.DataFrame(x).shift(1).ewm(span=span, ignore_na=True, min_periods=1).mean().to_numpy()


def _status_episodes(events, etype, down, up, T):
    """Episodes from paired status events: [first `down`, next `up`) per element."""
    out, open_ = [], {}
    for ev in sorted((e for e in events if e["type"] == etype), key=lambda e: (e["t"], e["element"])):
        el = ev["element"]
        if ev["status"] == down and el not in open_:
            open_[el] = ev["t"]
        elif ev["status"] == up and el in open_:
            out.append((el, open_.pop(el), ev["t"]))
    out += [(el, s, T) for el, s in open_.items()]
    return out


def generate_alerts(obs, kg):
    """All alerts for one observed scenario, as a DataFrame with COLUMNS, sorted by (t_start, rule, target)."""
    T = obs.n_steps
    rows = []

    def add(rule, target, s, e, value):
        rows.append((rule, target, RULES[rule][1], s, e, float(value)))

    for el, s, e in _status_episodes(obs.events, "link_status", "down", "up", T):
        add("link_status_down", el, s, e, 1.0)
    for el, s, e in _status_episodes(obs.events, "pop_status", "unreachable", "reachable", T):
        add("pop_unreachable", el if el.startswith("pop:") else f"pop:{el}", s, e, 1.0)

    util, lat, disc = obs.util, obs.latency, obs.discards
    with np.errstate(invalid="ignore"):
        rules = {
            "link_util_high": (util >= RULES["link_util_high"][2], util),
            "link_discards": (disc >= RULES["link_discards"][2], disc),
        }
        base = _trailing_ewma(lat, DAY_STEPS)
        rules["link_latency_high"] = ((lat >= RULES["link_latency_high"][2] * base) & (lat >= base + 1.0), lat / base)
    rules["link_probe_fail"] = (np.isnan(lat) & ~np.isnan(util), np.ones_like(util))
    rules["link_no_data"] = (np.isnan(util) & np.isnan(lat) & np.isnan(disc), np.ones_like(util))
    for rule, (mask, val) in rules.items():
        for j, s, e in _episodes(mask):
            add(rule, obs.links[j], s, e, val[s, j])

    G = kg.G
    sla_loss = np.array([G.nodes[s]["sla_loss_pct"] for s in obs.services])
    sla_lat = np.array([G.nodes[s]["sla_latency_ms"] for s in obs.services])
    sloss, slat = obs.svc_loss, obs.svc_latency
    sla_mean = pd.DataFrame(sloss).rolling(SLA_STEPS, min_periods=1).mean().to_numpy()   # trailing, causal
    with np.errstate(invalid="ignore"):
        unreach = sloss >= RULES["svc_unreachable"][2]
        srules = {
            "svc_unreachable": (unreach, sloss),
            "svc_loss_sla": ((sla_mean >= sla_loss) & ~unreach, sla_mean),
            "svc_latency_sla": (slat >= sla_lat, slat),
        }
    for rule, (mask, val) in srules.items():
        for j, s, e in _episodes(mask):
            add(rule, obs.services[j], s, e, val[s, j])

    df = pd.DataFrame(rows, columns=COLUMNS[1:]).sort_values(["t_start", "rule", "target"], kind="stable").reset_index(drop=True)
    df.insert(0, "alert_id", [f"A{i:05d}" for i in range(len(df))])
    return df


def alerts_at(alerts, t):
    """The alert list as known at step t: alerts raised at or before t; `t_end` is NaN for those still open at t."""
    known = alerts[alerts.t_start <= t].copy()
    known["t_end"] = known["t_end"].where(known["t_end"] <= t).astype(float)
    return known


def active_matrix(alerts, T, targets, severities=("critical", "major", "minor")):
    """(T, len(targets)) bool: some alert of the given severities is active on the target at t."""
    idx = {t: i for i, t in enumerate(targets)}
    out = np.zeros((T, len(targets)), bool)
    for a in alerts.itertuples():
        if a.severity in severities and a.target in idx:
            out[a.t_start:a.t_end, idx[a.target]] = True
    return out


def add_to_kg(kg, alerts):
    """Alert -ON-> Link|Service nodes in the KG. The KG schema has no Alert -ON-> PoP, so a PoP reachability alert is
    attached to every link at that PoP. Pass `alerts_at(alerts, t)` to build the KG as known at t; the full table
    carries clear times that are only known later (t_end is then left out for open alerts)."""
    ids = []
    for a in alerts.itertuples():
        targets = [a.target]
        if a.target.startswith("pop:"):
            targets = [l for l in kg.nodes_of_kind("Link") if a.target in kg._out(l, "CONNECTS")]
        for k, tgt in enumerate(targets):
            aid = a.alert_id if len(targets) == 1 else f"{a.alert_id}.{k}"
            ids.append(kg.add_alert(aid, tgt, a.t_start, metric=a.rule, severity=a.severity, value=a.value))
            if not pd.isna(a.t_end):
                kg.G.nodes[ids[-1]]["t_end"] = int(a.t_end)
    return ids
