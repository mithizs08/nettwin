"""Shared evaluation harness (Phase 4, used by every later phase).

Definitions (pre-registered in docs/decision_log.md, Phase 4 entry, then revised after docs/audit_phase4.md; every
revision is logged there by audit ID):

Detection (RQ1), link level. A method gives every (step, link) cell a score. Scored cells are steps
t >= WARMUP (day 0 has no faults by construction, so it is left out for every method), all links, including cells
with no telemetry (`telemetry_present == False`; the same for every method). Two tasks, never pooled:
- root: positives are label-2 cells, negatives label-0 cells; label-1 cells are left out.
- symptomatic: positives are label-1 cells, negatives label-0 cells; label-2 cells are left out.
A root cell belongs to a fault if it is on one of the fault's root links inside [t_start, t_end). A symptomatic cell
is time-eligible for a fault if it lies inside [t_start, t_end + LAG). When several faults are time-eligible, it goes
to those whose KG neighbourhood contains the link (`related_links`: links on the normal routing paths through the root,
plus links within 2 hops of a root link), so concurrent faults do not take each other's symptoms (audit Q11); if none
is related it stays with all of them and is counted as ambiguous. Positives of faults outside the reported group are
dropped, not counted as negatives. PR-AUC = average precision on binned scores (see `make_edges`).

Localisation (RQ2), per fault. The method gets the oracle window [t_start, t_end) and ranks every link and PoP. Ties
are broken at random in expectation (audit Q2): with g elements scored above the root and m others tied with it,
top-k credit = clip((k - g) / (m + 1), 0, 1) and the expected rank is 1 + g + m / 2. The pessimistic rank (ties count
against the root, the pre-registered rule) is kept for comparison. Roots of other faults whose windows overlap are
removed from the ranking first (set-based scoring for concurrent faults). Each method reports every window
aggregation in WINDOW_AGGS (audit Q3).

False alarms. Steps from WARMUP are cut into 1-hour windows. A window is a fault window if it meets any fault's
[t_start, t_end + LAG) (observable or not); otherwise maintenance-only if it meets any change window
[t_start, t_end + MAINT_TAIL); otherwise fault-free. A method alarms in a window if any link cell reaches its threshold.

Calibration (audit Q10). Bins and the alarm threshold are fixed per method from that method's own scores on the
calibration split, never from scores on scenarios the method was fitted on: bins from the quantiles of a random
sample of all scored cells (no labels), the threshold so that FAR_TARGET of the calibration split's fault-free windows
alarm. In Phase 4 the calibration split is validation (baselines are fitted on train), so the validation fault-free
alarm rate is FAR_TARGET by construction and the reported operating point is the detection rate at that rate. A
supervised model (Phase 5) is trained on train and calibrated on validation the same way, then applied unchanged to test.

Headline numbers exclude unobservable faults (impact_class == "unobservable"); they are reported on their own.
Hard types and easy types are never pooled. Confidence intervals: 95% percentile bootstrap over scenarios. The
bootstrap draws depend only on the number of scenarios and the seed, so results for different methods, variants and
datasets with the same scenario list are paired, and `aggregate` returns the bootstrap arrays for paired differences.
"""
import numpy as np
import pandas as pd

WARMUP = 288
LAG = 3
MAINT_TAIL = 12
WIN = 12
FAR_TARGET = 0.05
RELATED_HOPS = 2
WINDOW_AGGS = ("q90", "max", "mean")
HARD = ("link_degradation", "demand_surge", "routing_misconfig")
EASY = ("link_down", "node_down", "link_flapping")
TYPES = EASY + HARD
CLASSES = ("subtle", "moderate", "obvious")
PLACEMENTS = ("isolated", "maintenance_linked", "concurrent")


def labels_matrix(labels_df, links):
    """(T, L) label and telemetry_present arrays in the given link order."""
    t = labels_df["t"].to_numpy()
    j = pd.Categorical(labels_df["link"].astype(str), categories=links).codes
    if (j < 0).any():
        raise ValueError("label links do not match")
    T = int(t.max()) + 1
    lab = np.zeros((T, len(links)), np.int8)
    pres = np.zeros((T, len(links)), bool)
    lab[t, j] = labels_df["label"].to_numpy()
    pres[t, j] = labels_df["telemetry_present"].to_numpy()
    return lab, pres


def root_link_idx(root, ix):
    if root.startswith("link:"):
        return [ix.links.index(root)]
    p = ix.pops.index(root)
    return list(np.flatnonzero(ix.incidence[:, p]))


def related_links(kg, ix, root):
    """Link indices in a fault's KG neighbourhood: links on normal (t = 0) routing paths through the root, plus links
    within RELATED_HOPS of a root link. Uses the static inventory only."""
    rel = set()
    for p in kg.paths_through(root, t=0):
        rel.update(kg.links_of_path(p))
    for j in root_link_idx(root, ix):
        rel.add(ix.links[j])
        rel.update(kg.neighbours(ix.links[j], hops=RELATED_HOPS))
    lidx = {l: i for i, l in enumerate(ix.links)}
    return {lidx[l] for l in rel}


def fault_table(entry):
    out = []
    for f in entry["faults"]:
        im = f["impact"]
        out.append({
            "sid": entry["scenario_id"],
            "id": f["id"],
            "type": f["type"],
            "root": f["root_cause"],
            "root_kind": f["root_cause"].split(":", 1)[0],
            "t_start": f["t_start"],
            "t_end": f["t_end"],
            "impact_class": im["impact_class"],
            "observable": bool(im["observable"]),
            "placement": f["placement"],
            "overlaps_maintenance": bool(f.get("overlaps_maintenance", False)),
        })
    return out


def fault_key(f):
    return (f["sid"], f["type"], f["root"], f["t_start"], f["t_end"])


def make_edges(sample):
    """Bin edges for one method from a random sample of its scores on the calibration split (no labels): dense in the
    upper tail, where positives and the alarm threshold sit."""
    x = np.asarray(sample, float)
    levels = np.unique(np.concatenate([1 - np.logspace(-6, 0, 2500), np.linspace(0, 1, 500)]))
    return np.unique(np.quantile(x, np.clip(levels, 0, 1)))


def to_bins(scores, edges):
    return np.searchsorted(edges, scores, side="right")


def tie_credit(sc, true, keep, ks=(1, 3)):
    """Expected top-k credit and expected rank under random tie-breaking, plus the pessimistic rank."""
    s = sc[true]
    others = sc[keep]
    g = int(np.sum(others > s))
    m = int(np.sum(others == s))
    out = {f"top{k}": float(np.clip((k - g) / (m + 1), 0, 1)) for k in ks}
    out["rank"] = 1 + g + m / 2
    out["rank_pessimistic"] = 1 + g + m
    return out


def scenario_record(method, state, ix, entry, lab, pres, edges, related=None):
    """Everything the harness needs from one scenario for one method, without keeping the full score arrays.
    `related[k]` is the KG neighbourhood (link indices) of fault k (see `related_links`); without it, symptomatic cells
    are attributed by time only."""
    link = state[0]
    T, L = link.shape
    nb = len(edges) + 1
    bins = to_bins(link, edges)
    scored = np.zeros((T, L), bool)
    scored[WARMUP:] = True
    neg = scored & (lab == 0)
    rec = {
        "neg": np.bincount(bins[neg], minlength=nb),
        "neg_present": np.bincount(bins[neg & pres], minlength=nb),
    }
    faults = fault_table(entry)
    pos_bin, pos_fault, pos_task, pos_present, pos_cell = [], [], [], [], []
    sym_t, sym_j = np.nonzero(scored & (lab == 1))
    elig = np.zeros((len(faults), len(sym_t)), bool)
    for k, f in enumerate(faults):
        elig[k] = (sym_t >= f["t_start"]) & (sym_t < f["t_end"] + LAG)
    owner = elig.copy()
    multi = elig.sum(0) > 1
    n_ambiguous = 0
    if related is not None and multi.any():
        rel = np.array([[j in related[k] for j in sym_j] for k in range(len(faults))]) & elig
        resolved = multi & rel.any(0)
        owner[:, resolved] = rel[:, resolved]
        n_ambiguous = int((multi & ~rel.any(0)).sum())
    rec["sym_reassigned"] = int((owner != elig).any(0).sum())
    rec["sym_ambiguous"] = n_ambiguous
    rec["sym_unowned"] = int((~elig.any(0)).sum())
    for k, f in enumerate(faults):
        rl = root_link_idx(f["root"], ix)
        t0, t1 = max(f["t_start"], WARMUP), min(f["t_end"], T)
        hit = -np.inf
        for j in rl:
            tt = np.arange(t0, t1)
            tt = tt[lab[tt, j] == 2]
            pos_bin += list(bins[tt, j]); pos_fault += [k] * len(tt); pos_task += [2] * len(tt); pos_present += list(pres[tt, j])
            pos_cell += list(tt * L + j)
            if len(tt):
                hit = max(hit, float(link[tt, j].max()))
        sel = owner[k]
        st, sj = sym_t[sel], sym_j[sel]
        pos_bin += list(bins[st, sj]); pos_fault += [k] * len(st)
        pos_task += [1] * len(st); pos_present += list(pres[st, sj])
        pos_cell += list(st * L + sj)
        inwin = st < f["t_end"]
        if inwin.any():
            hit = max(hit, float(link[st[inwin], sj[inwin]].max()))
        # detection at threshold: best score on a cell of this fault (its root cells or its own symptoms) in its window
        f["hit_max"] = hit
        true = ix.elements.index(f["root"])
        drop = {ix.elements.index(g["root_cause"]) for g in entry["faults"]
                if g["root_cause"] != f["root"] and g["t_start"] < f["t_end"] and f["t_start"] < g["t_end"]}
        keep = np.ones(len(ix.elements), bool)
        keep[list(drop | {true})] = False
        f["loc"] = {}
        for agg in WINDOW_AGGS:
            sc = np.nan_to_num(np.asarray(method.localise(state, f["t_start"], f["t_end"], agg), float), nan=-np.inf)
            f["loc"][agg] = tie_credit(sc, true, keep)
    rec["pos_bin"] = np.array(pos_bin, np.int32)
    rec["pos_fault"] = np.array(pos_fault, np.int16)
    rec["pos_task"] = np.array(pos_task, np.int8)
    rec["pos_present"] = np.array(pos_present, bool)
    rec["pos_cell"] = np.array(pos_cell, np.int64)
    rec["faults"] = faults
    rec["windows"] = window_maxima(link, entry)
    return rec


def window_classes(entry, T):
    """Class of each 1-hour window from WARMUP: 0 fault-free, 1 maintenance-only, 2 fault."""
    starts = np.arange(WARMUP, T, WIN)
    ends = np.minimum(starts + WIN, T)
    cls = np.zeros(len(starts), np.int8)
    for m in entry["maintenance"]:
        cls[(starts < m["t_end"] + MAINT_TAIL) & (m["t_start"] < ends)] = 1
    for f in entry["faults"]:
        cls[(starts < f["t_end"] + LAG) & (f["t_start"] < ends)] = 2
    return starts, ends, cls


def window_maxima(link, entry):
    T = link.shape[0]
    starts, ends, cls = window_classes(entry, T)
    mx = np.array([link[s:e].max() for s, e in zip(starts, ends)])
    return {"cls": cls, "max": mx}


def threshold_for(window_max_free, target=FAR_TARGET):
    """Smallest score threshold whose alarm rate on fault-free windows is at most `target`."""
    x = np.sort(np.asarray(window_max_free, float))
    k = int(np.ceil((1 - target) * len(x))) - 1
    thr = x[min(max(k, 0), len(x) - 1)]
    return float(np.nextafter(thr, np.inf))


# ------------------------------------------------------------------ aggregation with bootstrap

def ap_from_hists(pos, neg):
    """Average precision from per-bin positive and negative counts (bins ascending in score). Works on the last
    axis, so a (B, nb) stack gives B values."""
    pos = np.asarray(pos, float)[..., ::-1]
    neg = np.asarray(neg, float)[..., ::-1]
    tp = np.cumsum(pos, -1)
    fp = np.cumsum(neg, -1)
    P = tp[..., -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        return np.where(P > 0, (pos * prec).sum(-1) / P, np.nan)


def group_filters(extra=None):
    """name -> (fault filter, scenario filter). Hard and easy types are never pooled."""
    g = {}
    obs = lambda f: f["observable"]  # noqa: E731
    for t in TYPES:
        g[f"type={t}"] = (lambda f, t=t: obs(f) and f["type"] == t, None)
    for fam, types in (("hard", HARD), ("easy", EASY)):
        inf = lambda f, types=types: obs(f) and f["type"] in types  # noqa: E731
        g[fam] = (inf, None)
        for c in CLASSES:
            g[f"{fam}|class={c}"] = (lambda f, inf=inf, c=c: inf(f) and f["impact_class"] == c, None)
        for p in PLACEMENTS:
            g[f"{fam}|placement={p}"] = (lambda f, inf=inf, p=p: inf(f) and f["placement"] == p, None)
        for kind in ("link", "pop"):
            g[f"{fam}|root={kind}"] = (lambda f, inf=inf, kind=kind: inf(f) and f["root_kind"] == kind, None)
        for src in ("generated", "real"):
            g[f"{fam}|source={src}"] = (inf, lambda s, src=src: s["source"] == src)
        g[f"{fam}|scenario=concurrent"] = (inf, lambda s: s["concurrent"])
        g[f"{fam}|scenario=no_concurrent"] = (inf, lambda s: not s["concurrent"])
        g[f"unobservable|{fam}"] = (lambda f, types=types: not f["observable"] and f["type"] in types, None)
    for t in TYPES:
        for src in ("generated", "real"):
            g[f"type={t}|source={src}"] = (g[f"type={t}"][0], lambda s, src=src: s["source"] == src)
    g.update(extra or {})
    return g


def ci(boot):
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return [None, None]
    return [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]


def _val(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def boot_weights(S, n_boot, seed):
    """(1 + n_boot, S) scenario weights; row 0 is the full sample. Depends only on S and the seed (paired across runs)."""
    rng = np.random.default_rng(seed)
    W = rng.multinomial(S, np.full(S, 1 / S), size=n_boot).astype(float)
    return np.vstack([np.ones(S), W])


def aggregate(records, scen_meta, nb, threshold, n_boot=1000, seed=0, extra_groups=None):
    """Metrics with bootstrap CIs per group. Returns (summary dict for JSON, {(group, metric): bootstrap array}),
    where each array has the full-sample value first."""
    S = len(records)
    W = boot_weights(S, n_boot, seed)
    negH = {k: np.vstack([r[k] for r in records]).astype(float) for k in ("neg", "neg_present")}
    out = {"n_scenarios": S, "threshold": threshold, "groups": {}}
    boots = {}

    def ratio(a, n):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (W @ a) / (W @ n)

    def far(cls, mask=None):
        m = np.ones(S) if mask is None else mask.astype(float)
        n = np.array([np.sum(r["windows"]["cls"] == cls) for r in records], float) * m
        a = np.array([np.sum((r["windows"]["cls"] == cls) & (r["windows"]["max"] >= threshold)) for r in records], float) * m
        v = ratio(a, n)
        return {"rate": _val(v[0]), "ci": ci(v[1:]), "n_windows": int(n.sum()), "n_alarms": int(a.sum())}, v

    out["false_alarm"], out["false_alarm_by_source"] = {}, {}
    for name, cls in (("fault_free", 0), ("maintenance_only", 1)):
        out["false_alarm"][name], boots[("all", f"far_{name}")] = far(cls)
    for src in ("generated", "real"):
        mask = np.array([s["source"] == src for s in scen_meta])
        out["false_alarm_by_source"][src] = {}
        for name, cls in (("fault_free", 0), ("maintenance_only", 1)):
            out["false_alarm_by_source"][src][name], boots[(f"source={src}", f"far_{name}")] = far(cls, mask)
    fa = out["false_alarm"]
    tot = fa["fault_free"]["n_alarms"] + fa["maintenance_only"]["n_alarms"]
    out["false_alarm"]["maintenance_share_of_false_alarms"] = _val(fa["maintenance_only"]["n_alarms"] / tot) if tot else None

    loc_keys = [(agg, k) for agg in WINDOW_AGGS for k in ("top1", "top3", "top1_pessimistic", "top3_pessimistic")]
    for name, (ff, sf) in group_filters(extra_groups).items():
        smask = np.array([True if sf is None else sf(s) for s in scen_meta])
        Wg = W * smask
        g = {}
        n_s = np.zeros(S)
        det = np.zeros(S)
        loc = {key: np.zeros(S) for key in loc_keys}
        for i, r in enumerate(records):
            if not smask[i]:
                continue
            for f in r["faults"]:
                if ff(f):
                    n_s[i] += 1
                    det[i] += f["hit_max"] >= threshold
                    for agg, k in loc_keys:
                        lc = f["loc"][agg]
                        loc[(agg, k)][i] += lc[k] if not k.endswith("pessimistic") else lc["rank_pessimistic"] <= int(k[3])
        g["n_faults"] = int(n_s.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            for (agg, k), arr in loc.items():
                v = (Wg @ arr) / (Wg @ n_s)
                g.setdefault("localisation", {}).setdefault(agg, {})[k] = {"value": _val(v[0]), "ci": ci(v[1:])}
                boots[(name, f"{k}@{agg}")] = v
            v = (Wg @ det) / (Wg @ n_s)
        g["detected_at_threshold"] = {"value": _val(v[0]), "ci": ci(v[1:])}
        boots[(name, "detected")] = v
        for task, tname in ((2, "root"), (1, "symptomatic")):
            for present_only, suffix in ((False, ""), (True, "_present_only")):
                posM = np.zeros((S, nb))
                for i, r in enumerate(records):
                    if not smask[i] or not len(r["pos_fault"]):
                        continue
                    ok = np.array([ff(f) for f in r["faults"]], bool)
                    sel = (r["pos_task"] == task) & ok[r["pos_fault"]]
                    if present_only:
                        sel &= r["pos_present"]
                    _, first = np.unique(r["pos_cell"][sel], return_index=True)   # a cell owned by two faults counts once
                    posM[i] = np.bincount(r["pos_bin"][sel][first], minlength=nb)
                nh = negH["neg_present" if present_only else "neg"]
                ap = ap_from_hists(Wg @ posM, Wg @ nh)
                n_pos, n_neg = int(posM.sum()), int(nh[smask].sum())
                g[f"pr_auc_{tname}{suffix}"] = {"value": _val(ap[0]), "ci": ci(ap[1:]), "n_pos": n_pos,
                                                "base_rate": _val(n_pos / (n_pos + n_neg)) if n_pos + n_neg else None}
                boots[(name, f"pr_auc_{tname}{suffix}")] = ap
        out["groups"][name] = g
    out["symptomatic_cells"] = {k: int(sum(r[k] for r in records)) for k in ("sym_unowned", "sym_reassigned", "sym_ambiguous")}
    return out, boots


def verdict(diff, direction=1):
    """Claim 'direction * diff > 0' from a bootstrap array (full-sample value first): supported if the 95% CI lies
    above zero, contradicted if below, inconclusive if it crosses zero (audit Q12)."""
    d = direction * np.asarray(diff, float)
    lo, hi = ci(d[1:])
    if lo is None:
        return {"verdict": "n/a", "point": None, "ci": [None, None]}
    v = "supported" if lo > 0 else "contradicted" if hi < 0 else "inconclusive"
    return {"verdict": v, "point": _val(d[0]), "ci": [lo, hi]}
