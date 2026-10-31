"""Capacity forecasting (Phase 8, RQ5): will a link's observed utilisation exceed 0.8 in the next 60 minutes?

Every input is observable and causal: data at or before the prediction time t only, no clock and no step index (the
banned-feature rules of Phases 4 to 6). The one-day lag compares with the same time yesterday. Ground truth is used
only by `window_class`, to stratify the scoring.

Import `lightgbm` before this module (it imports torch through the GNN): torch's bundled libomp crashes multithreaded
LightGBM otherwise (decision log, Phase 0).
"""
import numpy as np
import pandas as pd
import torch

from nettwin.baselines import b2_features
from nettwin.gnn.models import W, same_graph_batch

DAY = 288
HORIZON = 12                 # 60 minutes of 5-minute steps
STRIDE = 6                   # prediction times every 30 minutes
LIMIT = 0.8
LAG_FAULT = 3                # a fault's window counts until t_end + LAG (as the evaluation harness)
CLASSES = ("all", "fault_free", "fault", "maintenance")
LAG_FEATURES = ("u", "u_1", "u_2", "u_3", "u_6", "u_12", "mean12", "max12", "std12", "mean36", "max36", "std36",
                "yest_max", "yest_mean", "u_288", "lat_ratio", "discards", "u_missing", "lat_missing")


def grid(T):
    return np.arange(DAY, T - HORIZON, STRIDE)


def targets(util, times):
    """(n_t, L) breach labels and the mask of pairs with at least one finite future step."""
    fut = np.stack([util[t + 1:t + 1 + HORIZON] for t in times])                 # (n_t, H, L)
    ok = np.isfinite(fut).any(1)
    with np.errstate(invalid="ignore"):
        y = np.where(ok, np.nanmax(np.where(np.isfinite(fut), fut, -np.inf), 1), -np.inf) > LIMIT
    return y, ok


def _ffill(u):
    return pd.DataFrame(u).ffill().to_numpy()


def _yesterday(util, times):
    """Max and mean of utilisation over yesterday's horizon (t - 288, t - 276]."""
    win = np.stack([util[t - DAY + 1:t - DAY + 1 + HORIZON] for t in times])     # (n_t, H, L)
    with np.errstate(invalid="ignore"):
        mx = np.nanmax(np.where(np.isfinite(win), win, -np.inf), 1)
        cnt = np.isfinite(win).sum(1)
        mean = np.where(cnt > 0, np.nansum(np.nan_to_num(win, nan=0.0), 1) / np.maximum(cnt, 1), np.nan)
    return np.where(np.isfinite(mx), mx, np.nan), mean


def persistence(obs, times):
    u = _ffill(obs.util)[times]
    return np.nan_to_num(u, nan=0.0) > LIMIT


def seasonal_naive(obs, times):
    mx, _ = _yesterday(obs.util, times)
    return np.nan_to_num(mx, nan=0.0) > LIMIT


def lag_features(obs, times):
    """(n_t, L, len(LAG_FEATURES)) float32; NaN where unknown (LightGBM handles missing values)."""
    u = obs.util
    uf = _ffill(u)
    df = pd.DataFrame(u)
    roll = {w: (df.rolling(w, min_periods=1).mean().to_numpy(), df.rolling(w, min_periods=1).max().to_numpy(),
                df.rolling(w, min_periods=2).std().to_numpy()) for w in (12, 36)}
    ymax, ymean = _yesterday(u, times)
    lat = obs.latency
    lat_day = pd.DataFrame(lat).ewm(span=DAY, ignore_na=True).mean().shift(1).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        lat_ratio = lat / lat_day
    cols = [uf[times], uf[times - 1], uf[times - 2], uf[times - 3], uf[times - 6], uf[times - 12],
            roll[12][0][times], roll[12][1][times], roll[12][2][times], roll[36][0][times], roll[36][1][times], roll[36][2][times],
            ymax, ymean, u[times - DAY], lat_ratio[times], obs.discards[times],
            np.isnan(u[times]).astype(float), np.isnan(lat[times]).astype(float)]
    return np.stack(cols, -1).astype(np.float32)


def head_scalars(obs, times):
    """The three absolute-level inputs the GNN head gets next to the frozen embedding."""
    uf = np.nan_to_num(_ffill(obs.util), nan=0.0)
    mx12 = pd.DataFrame(obs.util).rolling(12, min_periods=1).max().to_numpy()[times]
    ymax, _ = _yesterday(obs.util, times)
    return np.stack([uf[times], np.nan_to_num(mx12, nan=0.0), np.nan_to_num(ymax, nan=0.0)], -1).astype(np.float32)


@torch.no_grad()
def gnn_embedding(model, obs, graph, times):
    """(n_t, L, hidden) per-link embedding after the frozen model's message-passing layers, at each prediction time.
    The same computation as `NetTwinGNN.forward` up to its heads, with the full-scenario temporal encoding of `predict`."""
    model.eval()
    feat = b2_features(obs).astype(np.float32)
    L = graph.n_links
    seq = np.concatenate([np.repeat(feat[:1], W - 1, 0), feat])
    x = (torch.from_numpy(np.ascontiguousarray(seq.transpose(1, 0, 2))) - model.f_mean) / model.f_std
    enc = model.enc.sequence(x)                                                         # (T, L, h)
    n = len(times)
    b = same_graph_batch(graph, n, graph.edge_index)
    s = (b["static"] - model.s_mean) / model.s_std * model.static_mask
    h = torch.relu(model.mix(torch.cat([enc[times].reshape(n * L, -1), torch.relu(model.static(s))], 1)))
    for conv, norm in zip(model.convs, model.norms):
        if model.conv_type == "mlp":
            m = conv(h)
        elif model.conv_type == "sage":
            m = conv(h, b["edge_index"])
        else:
            m = conv(h, b["edge_index"], b["edge_attr"])
        h = norm(h + torch.relu(m))
    return h.numpy().reshape(n, L, -1)


class Head(torch.nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)


def window_class(entry, times):
    """Per prediction time: 'fault' if a fault window [t_start, t_end + LAG) overlaps (t - H, t + H], else
    'maintenance' if any logged maintenance overlaps it, else 'fault_free'. Ground truth, for stratifying only."""
    lo, hi = times - HORIZON + 1, times + HORIZON + 1          # [lo, hi) = (t - H, t + H]
    def hit(spans):
        out = np.zeros(len(times), bool)
        for a, b in spans:
            out |= (a < hi) & (b > lo)
        return out
    fault = hit([(f["t_start"], f["t_end"] + LAG_FAULT) for f in entry["faults"]])
    maint = hit([(m["t_start"], m["t_end"]) for m in entry["maintenance"]]) & ~fault
    return np.where(fault, "fault", np.where(maint, "maintenance", "fault_free"))


def counts(pred, y, ok, cls):
    """{class: (tp, fp, fn)} over the masked pairs of one scenario; `cls` is per prediction time."""
    out = {}
    for c in CLASSES:
        m = ok & ((cls == c)[:, None] if c != "all" else True)
        p, t = pred[m], y[m]
        out[c] = (int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()))
    return out
