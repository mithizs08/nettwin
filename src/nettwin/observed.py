"""Read one scenario from the observable tree only (Phase 4).

Everything detectors, alerts and the agent see goes through `load_observed`. It takes a directory under
`observable/` and never opens anything under `truth/`. Time comes back as a step index only so arrays line up;
detectors must not use it as a feature (decision log, parked rule from audit P4).
"""
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from nettwin.kg import NetTwinKG
from nettwin.topology import load


@dataclass
class Observed:
    scenario_id: str
    topology: str
    status_blind: bool
    links: list            # link ids, column order of the link arrays
    services: list         # service ids, column order of the service arrays
    util: np.ndarray       # (T, L) float, NaN = missing
    latency: np.ndarray    # (T, L) one-hop latency ms
    discards: np.ndarray   # (T, L) interface discard counter, percent
    svc_latency: np.ndarray  # (T, S) end-to-end latency ms
    svc_loss: np.ndarray     # (T, S) end-to-end probe loss, percent
    events: list = field(default_factory=list)

    @property
    def n_steps(self):
        return self.util.shape[0]


def _pivot(df, key, cols):
    names = sorted(df[key].astype(str).unique())
    t = df["t"].to_numpy()
    j = pd.Categorical(df[key].astype(str), categories=names).codes
    T = int(t.max()) + 1
    out = []
    for c in cols:
        a = np.full((T, len(names)), np.nan, dtype=np.float64)
        a[t, j] = df[c].to_numpy(dtype=np.float64)
        out.append(a)
    return names, out


def load_observed(obs_dir):
    d = Path(obs_dir)
    if "truth" in d.parts:
        raise ValueError("detectors read the observable tree only")
    meta = json.loads((d / "meta.json").read_text())
    links, (util, lat, disc) = _pivot(pd.read_parquet(d / "links.parquet"), "link", ["util", "latency_ms", "loss_pct"])
    services, (slat, sloss) = _pivot(pd.read_parquet(d / "services.parquet"), "service", ["latency_ms", "loss_pct"])
    return Observed(
        scenario_id=meta["scenario_id"],
        topology=meta["topology"],
        status_blind=bool(meta["status_blind"]),
        links=links,
        services=services,
        util=util,
        latency=lat,
        discards=disc,
        svc_latency=slat,
        svc_loss=sloss,
        events=json.loads((d / "observed_events.json").read_text()),
    )


@lru_cache(maxsize=None)
def inventory(topology, cfg_items):
    """The operator's static inventory for a topology: the KG at t = 0 (PoPs, links, services with SLAs, normal
    routing). Built from the public topology and the config, no scenario data."""
    cfg = _thaw(cfg_items)
    return NetTwinKG.from_config(load(topology, cfg=cfg), cfg)


def freeze_cfg(cfg):
    """Hashable form of the config for `inventory`'s cache."""
    return json.dumps(cfg, sort_keys=True)


def _thaw(s):
    return json.loads(s)
