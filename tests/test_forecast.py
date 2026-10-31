"""Phase 8 forecasting inputs: causal, no clock, labels as defined, and the frozen-encoder embedding matches the model."""
import lightgbm  # noqa: F401  (before torch)

import numpy as np
import torch

torch.set_num_threads(1)          # lightgbm is loaded: multithreaded torch would segfault (libomp clash)

from nettwin import forecast as F
from nettwin.gnn.data import graph_for
from nettwin.gnn.models import NetTwinGNN, predict
from nettwin.observed import Observed


def _obs(T=900, L=4, seed=0):
    rng = np.random.default_rng(seed)
    u = np.clip(0.5 + 0.3 * np.sin(np.arange(T) / 40)[:, None] + 0.05 * rng.standard_normal((T, L)), 0, 1)
    u[rng.random((T, L)) < 0.02] = np.nan
    lat = 5 + rng.random((T, L))
    return Observed("s", "topozoo/Aarnet", False, [f"link:{i}" for i in range(L)], [], u, lat, np.zeros((T, L)),
                    np.zeros((T, 0)), np.zeros((T, 0)), [])


def _cut(o, T):
    return Observed(o.scenario_id, o.topology, o.status_blind, o.links, o.services, o.util[:T], o.latency[:T], o.discards[:T],
                    o.svc_latency[:T], o.svc_loss[:T], o.events)


def test_inputs_are_causal():
    o = _obs()
    times = F.grid(o.util.shape[0])[:20]
    t = int(times[-1])
    cut = _cut(o, t + 1)
    for f in (F.lag_features, F.head_scalars, F.persistence, F.seasonal_naive):
        a, b = f(o, times), f(cut, times)
        assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9)), f.__name__


def test_targets_and_window_classes():
    u = np.full((400, 2), 0.5)
    u[310, 0] = 0.9
    times = np.array([298, 300, 310])
    y, ok = F.targets(u, times)
    assert y[:, 0].tolist() == [True, True, False] and not y[:, 1].any() and ok.all()
    entry = {"faults": [{"t_start": 320, "t_end": 330}], "maintenance": [{"t_start": 290, "t_end": 295}]}
    assert F.window_class(entry, np.array([288, 300, 310, 345, 360])).tolist() == ["maintenance", "maintenance", "fault", "fault_free", "fault_free"]


def test_no_time_features():
    assert not any(k in name for name in F.LAG_FEATURES for k in ("time", "hour", "step", "clock", "day_of"))


def test_embedding_matches_model_detection_head():
    import yaml
    from pathlib import Path
    from nettwin.observed import freeze_cfg
    cfg_s = freeze_cfg(yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text()))
    g = graph_for("topozoo/Aarnet", cfg_s)
    torch.manual_seed(0)
    m = NetTwinGNN("gat", 32)
    o = _obs(T=700, L=g.n_links)
    times = F.grid(700)[:5]
    h = F.gnn_embedding(m, o, g, times)
    from nettwin.baselines import b2_features
    det, _ = predict(m, b2_features(o).astype(np.float32), g)
    with torch.no_grad():
        d = torch.sigmoid(m.det(torch.from_numpy(h))).squeeze(-1).numpy()
    assert np.allclose(d, det[times], atol=1e-5)
