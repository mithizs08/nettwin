"""A trained Phase 5 model wrapped for the shared harness, and loading a saved checkpoint by its run name."""
import hashlib
import io
import re

import numpy as np
import torch

from nettwin.baselines import Method, b2_features, window_score
from nettwin.gnn.data import graph_for
from nettwin.gnn.models import NetTwinGNN, predict
from nettwin.gnn.train import shuffle_perm

RUN_RE = re.compile(r"^(sage|gat|gine)-h(\d+)-s(\d+)(?:-(mlp|no_path_static|shuffled_edges))?$")


def parse_run(name):
    """'gat-h32-s1-mlp' -> ('gat', 32, 1, 'mlp')."""
    m = RUN_RE.match(name)
    if not m:
        raise ValueError(f"not a Phase 5 run name: {name}")
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)


def model_from_state(spec, state_bytes):
    conv, hidden, _, ablation = spec
    m = NetTwinGNN(conv="mlp" if ablation == "mlp" else conv, hidden=hidden, path_static=ablation != "no_path_static")
    m.load_state_dict(torch.load(io.BytesIO(state_bytes)))
    m.eval()
    return m


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GNNMethod(Method):
    """Wraps a trained GNN for the shared harness: detection = per-link probability, localisation = element
    probabilities aggregated over the incident window like every other method."""
    def __init__(self, name, model, cfg_s, shuffle_seed=None):
        self.name, self.model, self.cfg_s, self.shuffle_seed = name, model, cfg_s, shuffle_seed

    def score(self, obs, kg, ix):
        g = graph_for(obs.topology, self.cfg_s)
        perm = shuffle_perm(obs.topology, g.n_links, self.shuffle_seed) if self.shuffle_seed is not None else None
        det, elem = predict(self.model, b2_features(obs).astype(np.float32), g, perm)
        return det, np.zeros((det.shape[0], g.n_pops), np.float32), elem

    def localise(self, state, t0, t1, agg="q90"):
        return window_score(state[2][t0:t1], agg)


def gnn_method(name, state_bytes, cfg_s):
    spec = parse_run(name)
    return GNNMethod(name, model_from_state(spec, state_bytes), cfg_s, spec[2] if spec[3] == "shuffled_edges" else None)
