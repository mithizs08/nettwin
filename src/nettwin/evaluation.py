"""Shared evaluation harness (Phase 4, used by every later phase).

Definitions (pre-registered in docs/decision_log.md, Phase 4 entry, before any validation run):

Detection (RQ1), link level. A method gives every (step, link) cell a score. Scored cells are steps
t >= WARMUP (day 0 has no faults by construction, so it is left out for every method), all links, including cells
with no telemetry (`telemetry_present == False`; the same for every method). Two tasks, never pooled:
- root: positives are label-2 cells, negatives label-0 cells; label-1 cells are left out.
- symptomatic: positives are label-1 cells, negatives label-0 cells; label-2 cells are left out.
A positive cell belongs to a fault if it is on one of the fault's root links inside [t_start, t_end) (root task), or
inside [t_start, t_end + LAG) (symptomatic task). Positives of faults outside the reported group are dropped, not
counted as negatives. Symptomatic cells that belong to no fault are dropped and counted. Score = average precision
(PR-AUC) on binned scores; bins are fixed per method from train (see `make_edges`), ties share a bin.

Localisation (RQ2), per fault. The method gets the oracle window [t_start, t_end) and ranks every link and PoP.
Rank of the true root is pessimistic (ties count against it). Roots of other faults whose windows overlap are taken
out of the ranking first (set-based scoring for concurrent faults), so finding the other root first is not
penalised. Top-1 / top-3 accuracy.

False alarms. Steps from WARMUP are cut into 1-hour windows. A window is a fault window if it meets any fault's
[t_start, t_end + LAG) (observable or not); otherwise maintenance-only if it meets any change window
[t_start, t_end + MAINT_TAIL); otherwise fault-free. A method alarms in a window if any link cell reaches its
threshold. The threshold is fixed on train so that 5% of train fault-free windows alarm. Reported: alarm rate on
fault-free and on maintenance-only windows, and the fault detection rate at that threshold (a fault counts when a
cell labelled 1 or 2 inside its window alarms).

Headline numbers exclude unobservable faults (impact_class == "unobservable"); they are reported on their own.
Hard types and easy types are never pooled. Confidence intervals: 95% percentile bootstrap over scenarios.
"""
import numpy as np
import pandas as pd

WARMUP = 288
LAG = 3
MAINT_TAIL = 12
WIN = 12
FAR_TARGET = 0.05
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


def fault_table(entry):
    out = []
    for f in entry["faults"]:
        im = f["impact"]
        out.append({
            "type": f["type"],
            "root": f["root_cause"],
            "t_start": f["t_start"],
            "t_end": f["t_end"],
            "impact_class": im["impact_class"],
            "observable": bool(im["observable"]),
            "placement": f["placement"],
            "overlaps_maintenance": bool(f.get("overlaps_maintenance", False)),
        })
    return out


def make_edges(pos_scores, neg_scores):
    """Bin edges for one method, from train scores: dense in the negatives' upper tail and across the positives."""
    neg = np.asarray(neg_scores, float)
    pos = np.asarray(pos_scores, float)
    levels = 1 - np.logspace(-7, 0, 1500)
    e = np.concatenate([np.quantile(neg, np.clip(levels, 0, 1)), np.quantile(pos, np.linspace(0, 1, 500))])
    return np.unique(e)


def to_bins(scores, edges):
    return np.searchsorted(edges, scores, side="right")


def scenario_record(method, state, ix, entry, lab, pres, edges):
    """Everything the harness needs from one scenario for one method, without keeping the full score arrays."""
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
    sym_owned = np.zeros(len(sym_t), bool)
    for k, f in enumerate(faults):
        rl = root_link_idx(f["root"], ix)
        f["root_links"] = rl
        t0, t1 = max(f["t_start"], WARMUP), min(f["t_end"], T)
        for j in rl:
            tt = np.arange(t0, t1)
            tt = tt[lab[tt, j] == 2]
            pos_bin += list(bins[tt, j]); pos_fault += [k] * len(tt); pos_task += [2] * len(tt); pos_present += list(pres[tt, j])
            pos_cell += list(tt * L + j)
        sel = (sym_t >= f["t_start"]) & (sym_t < f["t_end"] + LAG)
        sym_owned |= sel
        pos_bin += list(bins[sym_t[sel], sym_j[sel]]); pos_fault += [k] * int(sel.sum())
        pos_task += [1] * int(sel.sum()); pos_present += list(pres[sym_t[sel], sym_j[sel]])
        pos_cell += list(sym_t[sel] * L + sym_j[sel])
        # localisation with the other overlapping roots taken out
        sc = np.nan_to_num(np.asarray(method.localise(state, f["t_start"], f["t_end"]), float), nan=-np.inf)
        true = ix.elements.index(f["root"])
        drop = {ix.elements.index(g["root_cause"]) for g in entry["faults"]
                if g["root_cause"] != f["root"] and g["t_start"] < f["t_end"] and f["t_start"] < g["t_end"]}
        keep = np.ones(len(sc), bool)
        keep[list(drop | {true})] = False
        f["rank"] = int(1 + np.sum(sc[keep] >= sc[true]))
        # fault detection at threshold: best score on a labelled cell inside the window
        w = lab[f["t_start"]:min(f["t_end"], T)] >= 1
        f["hit_max"] = float(link[f["t_start"]:min(f["t_end"], T)][w].max()) if w.any() else -np.inf
    rec["pos_bin"] = np.array(pos_bin, np.int32)
    rec["pos_fault"] = np.array(pos_fault, np.int16)
    rec["pos_task"] = np.array(pos_task, np.int8)
    rec["pos_present"] = np.array(pos_present, bool)
    rec["pos_cell"] = np.array(pos_cell, np.int64)
    rec["sym_unowned"] = int((~sym_owned).sum())
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


def group_filters():
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
        for src in ("generated", "real"):
            g[f"{fam}|source={src}"] = (inf, lambda s, src=src: s["source"] == src)
        g[f"{fam}|scenario=concurrent"] = (inf, lambda s: s["concurrent"])
        g[f"{fam}|scenario=no_concurrent"] = (inf, lambda s: not s["concurrent"])
        g[f"unobservable|{fam}"] = (lambda f, types=types: not f["observable"] and f["type"] in types, None)
    for t in TYPES:
        for src in ("generated", "real"):
            g[f"type={t}|source={src}"] = (g[f"type={t}"][0], lambda s, src=src: s["source"] == src)
    return g


def _ci(boot):
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    if not len(boot):
        return [None, None]
    return [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]


def _val(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def aggregate(records, scen_meta, nb, threshold, n_boot=1000, seed=0):
    """Metrics with bootstrap CIs per group. `records` and `scen_meta` are lists aligned by scenario."""
    S = len(records)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(S, np.full(S, 1 / S), size=n_boot).astype(float)   # (B, S) resampling counts
    W = np.vstack([np.ones(S), W])                                          # row 0 = the full sample
    negH = {k: np.vstack([r[k] for r in records]).astype(float) for k in ("neg", "neg_present")}
    out = {"n_scenarios": S, "threshold": threshold, "groups": {}}

    # false alarms: per-scenario counts of windows and alarms
    def rate(cls):
        n = np.array([np.sum(r["windows"]["cls"] == cls) for r in records], float)
        a = np.array([np.sum((r["windows"]["cls"] == cls) & (r["windows"]["max"] >= threshold)) for r in records], float)
        v = (W @ a) / (W @ n)
        return {"rate": _val(v[0]), "ci": _ci(v[1:]), "n_windows": int(n.sum())}

    out["false_alarm"] = {"fault_free": rate(0), "maintenance_only": rate(1)}
    out["false_alarm_by_source"] = {}
    for src in ("generated", "real"):
        m = np.array([s["source"] == src for s in scen_meta], float)
        sub = {}
        for name, cls in (("fault_free", 0), ("maintenance_only", 1)):
            n = np.array([np.sum(r["windows"]["cls"] == cls) for r in records], float) * m
            a = np.array([np.sum((r["windows"]["cls"] == cls) & (r["windows"]["max"] >= threshold)) for r in records], float) * m
            v = (W @ a) / (W @ n)
            sub[name] = {"rate": _val(v[0]), "ci": _ci(v[1:]), "n_windows": int(n.sum())}
        out["false_alarm_by_source"][src] = sub

    for name, (ff, sf) in group_filters().items():
        smask = np.array([True if sf is None else sf(s) for s in scen_meta])
        Wg = W * smask
        g = {}
        per_s = np.zeros((4, S))   # n faults, top1, top3, detected at threshold
        for i, r in enumerate(records):
            for f in r["faults"]:
                if smask[i] and ff(f):
                    per_s[:, i] += (1, f["rank"] <= 1, f["rank"] <= 3, f["hit_max"] >= threshold)
        n_f = per_s[0].sum()
        g["n_faults"] = int(n_f)
        for k, key in ((1, "top1"), (2, "top3"), (3, "detected_at_threshold")):
            with np.errstate(invalid="ignore", divide="ignore"):
                v = (Wg @ per_s[k]) / (Wg @ per_s[0])
            g[key] = {"value": _val(v[0]), "ci": _ci(v[1:])}
        for task, tname in ((2, "root"), (1, "symptomatic")):
            for present_only, suffix in ((False, ""), (True, "_present_only")):
                posM = np.zeros((S, nb))
                for i, r in enumerate(records):
                    if not smask[i]:
                        continue
                    ok = np.array([ff(f) for f in r["faults"]], bool)
                    sel = (r["pos_task"] == task) & ok[r["pos_fault"]] if len(r["pos_fault"]) else np.zeros(0, bool)
                    if present_only:
                        sel &= r["pos_present"]
                    _, first = np.unique(r["pos_cell"][sel], return_index=True)   # a cell owned by two faults counts once
                    posM[i] = np.bincount(r["pos_bin"][sel][first], minlength=nb)
                ap = ap_from_hists(Wg @ posM, Wg @ negH["neg_present" if present_only else "neg"])
                g[f"pr_auc_{tname}{suffix}"] = {"value": _val(ap[0]), "ci": _ci(ap[1:]), "n_pos": int(posM.sum())}
        out["groups"][name] = g
    out["symptomatic_cells_without_fault"] = int(sum(r["sym_unowned"] for r in records))
    return out
