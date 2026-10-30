"""Baselines B1 to B3 (Phase 4, PLAN.md section 6).

Every method reads one `Observed` scenario (observable tree only) plus the operator's static inventory, and returns
per-step anomaly scores for every link (T, L) and every PoP (T, P). Localisation ranks links and PoPs together over
a window.

Banned inputs (decision log, parked rule from audit P4 / P7): no absolute time of day, no day of week, no time
since scenario start, no file metadata. `Observed` carries no clock at all (arrays by step position only), and every
score at step t uses data up to t only (tests check both). Comparing a value with the same step one day earlier (a
diurnal lag) is allowed.

- B1: per-link EWMA z-scores on util (two-sided), one-hop latency and discards (upward only). Each z must hold for
  2 consecutive steps (the smaller of the two counts). Score = the largest. Missing values give no score (classic
  per-metric z-score; missingness is ignored). Span and persistence were picked on 30 train scenarios.
- B2: Isolation Forest on per-link windowed features (relative levels, one-day lags, short rolling stats,
  missing-data flags). Fit unsupervised on train cells.
- B3: network tomography heuristic over threshold alerts, as pre-registered. "Paths" are service routing paths, plus
  each link and each PoP as a one-element path. An element's score = (alerted paths containing it / all alerted paths)
  x (alerted paths containing it / all paths containing it): coverage times specificity, so elements also used by many
  healthy paths are penalised.
- B3+: the version first run in Phase 4, a disclosed deviation (audit Q1): a link's own path also contains its two end
  PoPs, so link alerts also count for the routers at each end.

For B1 and B2 a PoP's score is an aggregation of its links' scores (POP_AGGS; the choice is made on train, audit Q2).
Localisation scores an element over a window by one of WINDOW_AGGS (audit Q3); B3 and B3+ use the window union of
alerts for every aggregation, as pre-registered.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from nettwin.alerts import active_matrix, generate_alerts

DAY = 288
WINDOW_QUANTILE = 0.9   # the pre-registered window aggregation ("q90"); "max" and "mean" are reported next to it
POP_AGGS = ("mean", "max", "second_max")


@dataclass
class Index:
    """Links and PoPs of a scenario and how they connect."""
    links: list
    pops: list
    ends: np.ndarray      # (L, 2) PoP index of each link end
    incidence: np.ndarray  # (L, P) 1 where the link touches the PoP

    @property
    def elements(self):
        return self.links + self.pops


def index_for(obs, kg):
    links = kg.nodes_of_kind("Link")
    if links != obs.links:
        raise ValueError("observed links do not match the inventory")
    pops = kg.nodes_of_kind("PoP")
    pidx = {p: i for i, p in enumerate(pops)}
    ends = np.array([(pidx[f"pop:{kg.G.nodes[l]['u']}"], pidx[f"pop:{kg.G.nodes[l]['v']}"]) for l in links])
    inc = np.zeros((len(links), len(pops)))
    inc[np.arange(len(links)), ends[:, 0]] = 1
    inc[np.arange(len(links)), ends[:, 1]] = 1
    return Index(links, pops, ends, inc)


def pop_scores(link_scores, ix, agg="mean"):
    """PoP score from its links' scores (for per-link methods): mean, max, or the second largest (0 for a degree-1
    PoP, so a spur link outranks its leaf PoP unless something else marks the PoP)."""
    if agg == "mean":
        return link_scores @ ix.incidence / ix.incidence.sum(0)
    out = np.zeros((link_scores.shape[0], len(ix.pops)))
    for p in range(len(ix.pops)):
        cols = link_scores[:, ix.incidence[:, p] > 0]
        if agg == "max":
            out[:, p] = cols.max(1)
        elif cols.shape[1] >= 2:
            out[:, p] = np.partition(cols, -2, axis=1)[:, -2]
    return out


def window_score(x, agg):
    if agg == "q90":
        return np.quantile(x, WINDOW_QUANTILE, axis=0)
    if agg == "max":
        return x.max(0)
    if agg == "mean":
        return x.mean(0)
    raise ValueError(agg)


def _past_ewm(x, span):
    """EWMA mean and variance of past values only (excludes the current step)."""
    e = pd.DataFrame(x).shift(1).ewm(span=span, ignore_na=True, min_periods=2)
    return e.mean().to_numpy(), e.var().to_numpy()


def _lag(x, k):
    out = np.full_like(x, np.nan)
    out[k:] = x[:-k]
    return out


class Method:
    name = ""
    pop_agg = "mean"

    def fit(self, scenarios):
        return self

    def score(self, obs, kg, ix):
        """Returns a state tuple starting (link_scores (T, L), pop_scores (T, P)); higher = more anomalous, finite."""
        raise NotImplementedError

    def localise(self, state, t0, t1, agg="q90"):
        """Score per element (links then PoPs) for the window [t0, t1). `state` is what `score` returned."""
        link, pop = state[0], state[1]
        return window_score(np.hstack([link[t0:t1], pop[t0:t1]]), agg)


class B1EwmaZ(Method):
    name = "B1"
    span = 144                             # 12 hours (picked on train from 3 h / 12 h / 1 h)
    persist = 2                            # z must hold for 2 steps, so single probe outliers do not score
    floors = {"util": 0.005, "latency_rel": 0.02, "latency_abs": 0.05, "discards": 0.005}   # measurement noise

    def score(self, obs, kg, ix):
        zs = []
        for x, two_sided, kind in ((obs.util, True, "util"), (obs.latency, False, "latency"), (obs.discards, False, "discards")):
            mu, var = _past_ewm(x, self.span)
            if kind == "latency":
                floor = self.floors["latency_rel"] * np.abs(mu) + self.floors["latency_abs"]
            else:
                floor = self.floors[kind]
            sd = np.maximum(np.sqrt(np.nan_to_num(var)), floor)
            z = (x - mu) / sd
            z = np.nan_to_num(np.abs(z) if two_sided else np.maximum(z, 0), nan=0.0)
            held = z.copy()
            for k in range(1, self.persist):
                held[k:] = np.minimum(held[k:], z[:-k])
                held[:k] = 0
            zs.append(held)
        link = np.max(np.stack(zs), axis=0)
        return link, pop_scores(link, ix, self.pop_agg)


FEATURES = (
    "util_rel_day", "util_minus_yesterday", "util_minus_recent", "util_std_recent",
    "lat_rel_day", "lat_minus_yesterday_rel", "discards", "discards_max_recent",
    "util_missing", "latency_missing",
)


def b2_features(obs):
    """(T, L, F) per-link windowed features, in FEATURES order. No time input."""
    u, lat, d = obs.util, obs.latency, obs.discards
    u_day, _ = _past_ewm(u, DAY)
    u_recent, _ = _past_ewm(u, 12)
    lat_day, _ = _past_ewm(lat, DAY)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = [
            u / u_day,
            u - _lag(u, DAY),
            u - u_recent,
            pd.DataFrame(u).rolling(12, min_periods=2).std().to_numpy() / u_day,
            lat / lat_day,
            (lat - _lag(lat, DAY)) / lat_day,
            d,
            pd.DataFrame(d).rolling(12, min_periods=1).max().to_numpy(),
            np.isnan(u).astype(float),
            np.isnan(lat).astype(float),
        ]
    neutral = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    out = np.stack([np.where(np.isfinite(x), x, n) for x, n in zip(f, neutral)], axis=-1)
    return np.clip(out, -50, 50)


class B2IsolationForest(Method):
    name = "B2"

    def __init__(self, seed=0, n_cells=200_000, n_estimators=100):
        self.seed, self.n_cells, self.n_estimators = seed, n_cells, n_estimators
        self.model = None

    def fit(self, scenarios):
        """Unsupervised: a random sample of cells (after the first day) from the given train scenarios. No labels."""
        rng = np.random.default_rng(self.seed)
        feats = [b2_features(o)[DAY:].reshape(-1, len(FEATURES)) for o in scenarios]
        per = max(1, self.n_cells // len(feats))
        X = np.vstack([f[rng.choice(len(f), size=min(per, len(f)), replace=False)] for f in feats])
        self.model = IsolationForest(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=1).fit(X)
        return self

    def score(self, obs, kg, ix):
        F = b2_features(obs)
        T, L, _ = F.shape
        link = -self.model.score_samples(F.reshape(-1, F.shape[-1])).reshape(T, L)
        return link, pop_scores(link, ix, self.pop_agg)


class B3Tomography(Method):
    name = "B3"
    severities = ("critical", "major", "minor")   # info alerts (no telemetry at all) are not used
    link_path_has_ends = False                     # pre-registered: a link's own path is the link alone

    def _matrices(self, obs, kg, ix):
        S, L, P = len(obs.services), len(ix.links), len(ix.pops)
        E = L + P
        lidx = {l: i for i, l in enumerate(ix.links)}
        pidx = {p: i for i, p in enumerate(ix.pops)}
        M = np.zeros((S + L + P, E))
        for i, s in enumerate(obs.services):
            paths = kg.service_paths(s, t=0)
            for p in paths:
                for l in kg.links_of_path(p):
                    M[i, lidx[l]] += 1 / len(paths)
                for n in kg.G.nodes[p]["pops"]:
                    M[i, L + pidx[f"pop:{n}"]] += 1 / len(paths)
        for j in range(L):
            M[S + j, j] = 1
            if self.link_path_has_ends:
                M[S + j, L + ix.ends[j]] = 1
        for k in range(P):
            M[S + L + k, L + k] = 1
        alerts = generate_alerts(obs, kg)
        T = obs.n_steps
        A = np.hstack([
            active_matrix(alerts, T, obs.services, self.severities),
            active_matrix(alerts, T, ix.links, self.severities),
            active_matrix(alerts, T, ix.pops, self.severities),
        ])
        return A, M

    @staticmethod
    def _tomo(A, M):
        A = A.astype(float)
        a = A @ M                                  # alerted paths through each element (weighted)
        n_alerted = A.sum(-1, keepdims=True)
        total = M.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (a / n_alerted) * (a / total)
        return np.nan_to_num(s, nan=0.0)

    def score(self, obs, kg, ix):
        A, M = self._matrices(obs, kg, ix)
        s = self._tomo(A, M)
        L = len(ix.links)
        return s[:, :L], s[:, L:], (A, M)

    def localise(self, state, t0, t1, agg="q90"):
        A, M = state[2]
        return self._tomo(A[t0:t1].any(0, keepdims=True), M)[0]


class B3Plus(B3Tomography):
    name = "B3+"
    link_path_has_ends = True


METHODS = {"B1": B1EwmaZ, "B2": B2IsolationForest, "B3": B3Tomography, "B3+": B3Plus}
