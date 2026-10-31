"""Agent evaluation for RQ4 (Phase 6): incident sampling, scoring with the shared harness's localisation credit, the
non-LLM references on the same windows, and paired bootstrap summaries."""
import numpy as np

from nettwin.evaluation import EASY, HARD, LAG, MAINT_TAIL, WARMUP, boot_weights, ci, tie_credit, verdict
from nettwin.agent.loop import is_fault_claim

DISTRACTOR_ACTIONS = ("drain", "down")


def _overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def overlaps_logged_change(changes, t0, t1):
    """A drain or planned outage overlapping the window [t0, t1) (so it has started by the window end). `changes` are
    maintenance records with `action`, `t_start` and `t_end`: the manifest's for sampling, the observed change log for
    the change-log rule."""
    return any(c["action"] in DISTRACTOR_ACTIONS and c["t_start"] < t1 and c["t_end"] > t0 for c in changes)


def sample_incidents(entries, n_per_type, n_per_distractor, seed, n_change_overlap=0):
    """Fault incidents (observable, isolated from other faults by LAG) stratified by type, and distractor windows (a
    drain or a planned outage with no fault within MAINT_TAIL), at most one incident per scenario. The window is the
    fault's [t_start, t_end) or the change's [t_start, t_end); every method gets the same window.

    Then the change-overlap stratum (Phase 6 audit S2): further faults whose window overlaps a logged drain or planned
    outage, from unused scenarios, one fault type at a time in turn, until `n_change_overlap` fault incidents in all
    (main sample included) overlap a change. The main sample does not depend on it."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(entries))
    pools = {t: [] for t in EASY + HARD}
    pools.update({f"distractor_{a}": [] for a in DISTRACTOR_ACTIONS})
    for i in order:
        e = entries[i]
        faults = e["faults"]
        for f in faults:
            if not f["impact"]["observable"] or f["t_start"] < WARMUP:
                continue
            if any(g is not f and _overlaps(f["t_start"], f["t_end"] + LAG, g["t_start"], g["t_end"] + LAG) for g in faults):
                continue
            pools[f["type"]].append({"scenario_id": e["scenario_id"], "kind": "fault", "type": f["type"], "root": f["root_cause"],
                                     "t0": f["t_start"], "t1": f["t_end"], "placement": f["placement"],
                                     "overlaps_maintenance": f.get("overlaps_maintenance", False),
                                     "overlaps_logged_change": overlaps_logged_change(e["maintenance"], f["t_start"], f["t_end"]),
                                     "stratum": "main"})
        for m in e["maintenance"]:
            if m["action"] not in DISTRACTOR_ACTIONS or m["t_start"] < WARMUP:
                continue
            if any(_overlaps(m["t_start"], m["t_end"] + MAINT_TAIL, g["t_start"], g["t_end"] + LAG) for g in faults):
                continue
            pools[f"distractor_{m['action']}"].append({"scenario_id": e["scenario_id"], "kind": "distractor", "type": f"distractor_{m['action']}",
                                                       "root": "none", "change_target": m["target"], "t0": m["t_start"], "t1": m["t_end"],
                                                       "stratum": "main"})
    used, out = set(), []
    for name in list(EASY + HARD) + [f"distractor_{a}" for a in DISTRACTOR_ACTIONS]:
        want = n_per_type if not name.startswith("distractor") else n_per_distractor
        got = 0
        for inc in pools[name]:
            if got == want:
                break
            if inc["scenario_id"] in used:
                continue
            used.add(inc["scenario_id"])
            out.append(inc)
            got += 1
    need = n_change_overlap - sum(i.get("overlaps_logged_change", False) for i in out)
    queues = {t: [i for i in pools[t] if i["overlaps_logged_change"]] for t in EASY + HARD}
    while need > 0 and any(queues.values()):
        for t in EASY + HARD:
            while need > 0 and queues[t]:
                inc = queues[t].pop(0)
                if inc["scenario_id"] not in used:
                    used.add(inc["scenario_id"])
                    out.append(dict(inc, stratum="change_overlap"))
                    need -= 1
                    break
    for k, inc in enumerate(out):
        inc["incident_id"] = f"I{k:03d}"
    return out


def ranking_scores(diagnosis, elements):
    """Agent ranking as a score vector for the harness's expected-rank credit: root 3, alternatives 2 and 1, others 0
    (tied). 'none' or an unknown id ranks nothing."""
    sc = np.zeros(len(elements))
    if not diagnosis:
        return sc
    cands = [diagnosis.get("root_cause")] + list(diagnosis.get("alternatives") or [])[:2]
    seen = []
    for c in cands:
        if isinstance(c, str) and c.startswith("link:") and c not in elements and "--" in c:
            c = "link:" + "--".join(reversed(c[5:].split("--", 1)))      # a link named in reverse (Phase 6 audit S7)
        if c in elements and c not in seen:
            seen.append(c)
    for r, c in enumerate(seen):
        sc[elements.index(c)] = 3 - r
    return sc


def localisation_credit(scores, elements, root):
    """The harness's tie-aware top-1 / top-3 credit (`evaluation.tie_credit`); isolated incidents drop no other root."""
    true = elements.index(root)
    keep = np.ones(len(elements), bool)
    keep[true] = False
    c = tie_credit(np.nan_to_num(np.asarray(scores, float), nan=-np.inf), true, keep)
    return {"top1": float(c["top1"]), "top3": float(c["top3"])}


def score_agent(inc, diagnosis, elements):
    claims = is_fault_claim(diagnosis)
    out = {"claims_fault": claims, "answered": diagnosis is not None}
    if inc["kind"] == "fault":
        out.update(localisation_credit(ranking_scores(diagnosis, elements), elements, inc["root"]))
        out["type_correct"] = bool(diagnosis) and diagnosis.get("fault_type") == inc["type"]
    return out


def score_reference(inc, method, state, threshold, elements):
    """A non-LLM method on the same window: localisation from its q90 window score, and 'claims a fault' if its
    detection score reaches its frozen 5% fault-free threshold anywhere in the window."""
    out = {"claims_fault": bool(np.nanmax(state[0][inc["t0"]:inc["t1"]]) >= threshold)}
    if inc["kind"] == "fault":
        out.update(localisation_credit(method.localise(state, inc["t0"], inc["t1"], "q90"), elements, inc["root"]))
    return out


def _main(i):
    return i.get("stratum", "main") == "main"


# headline groups use the main sample only; change_overlap_faults is every fault incident (main sample and the
# change-overlap stratum) whose window overlaps a logged drain or planned outage
GROUPS = {"faults": lambda i: _main(i) and i["kind"] == "fault", "hard": lambda i: _main(i) and i["type"] in HARD,
          "easy": lambda i: _main(i) and i["type"] in EASY,
          "distractors": lambda i: i["kind"] == "distractor", "distractor_drain": lambda i: i["type"] == "distractor_drain",
          "distractor_down": lambda i: i["type"] == "distractor_down",
          "change_overlap_faults": lambda i: i["kind"] == "fault" and i.get("overlaps_logged_change", False)}
for _t in EASY + HARD:
    GROUPS[f"type={_t}"] = lambda i, t=_t: _main(i) and i["type"] == t


def metric_vectors(incidents, scores):
    """{(group, metric): (values per incident, mask)} for one method; `scores[k]` belongs to incidents[k]."""
    out = {}
    for g, fn in GROUPS.items():
        m = np.array([fn(i) for i in incidents])
        for metric in ("top1", "top3", "type_correct", "claims_fault", "answered"):
            v = np.array([float(s.get(metric, np.nan)) if s.get(metric) is not None else np.nan for s in scores])
            if np.isfinite(v[m]).any():
                out[(g, metric)] = (np.nan_to_num(v), m & np.isfinite(v))
    return out


def summarise(incidents, by_method, n_boot=1000, seed=42, pairs=()):
    """Means with incident-bootstrap CIs per method and group, and paired verdicts for the given (a, b) pairs."""
    W = boot_weights(len(incidents), n_boot, seed)
    vec = {m: metric_vectors(incidents, s) for m, s in by_method.items()}
    boots, table = {}, {}
    for m, V in vec.items():
        for (g, metric), (v, mask) in V.items():
            Wm = W * mask
            with np.errstate(invalid="ignore", divide="ignore"):
                b = (Wm @ v) / Wm.sum(1)
            boots[(m, g, metric)] = b
            table.setdefault(m, {}).setdefault(g, {})[metric] = {"value": round(float(b[0]), 4), "ci": ci(b[1:]), "n": int(mask.sum())}
    comps = {}
    for a, b in pairs:
        for (m, g, metric), arr in boots.items():
            if m == a and (b, g, metric) in boots:
                d = arr - boots[(b, g, metric)]
                if metric == "claims_fault" and g.startswith("distractor"):
                    d = -d                              # on distractors, fewer fault claims is better
                comps[f"{g}|{metric}|{a} vs {b}"] = verdict(d)
    return table, comps
