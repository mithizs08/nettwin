"""Phase 5: the GNN pipeline keeps the Phase 4 rules (observable inputs only, causal, no clock) and its fast
full-scenario prediction equals the window-by-window model. Runs on topohub's shipped `gabriel/20/1` (in no split)."""
import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from nettwin.baselines import b2_features
from nettwin.dataset import _truth_entry, blanked_cells
from nettwin.evaluation import labels_matrix
from nettwin.faults import make_fault
from nettwin.gnn.data import W, active_roots, graph_for, training_samples, windows
from nettwin.gnn.models import NetTwinGNN, collate, predict
from nettwin.gnn.train import shuffle_perm, train
from nettwin.observed import freeze_cfg, inventory, load_observed
from nettwin.baselines import index_for
from nettwin.simulator import Simulator, write_scenario
from nettwin.topology import load

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "gabriel/20/1"


@pytest.fixture(scope="module")
def scen(tmp_path_factory):
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text()))
    cfg["simulator"]["days"] = 3
    sim = Simulator(load(FIXTURE), cfg)
    link = max(sim.links, key=lambda l: (len(sim.kg.paths_through(l)), l))
    rng = np.random.default_rng(0)
    faults = [make_fault(sim, "link_down", link, 400, 30, rng, fault_id="F0")]
    for f in faults:
        f.update(placement="isolated", linked_maintenance=None, caused_by_maintenance=False, overlaps_maintenance=False)
    spec = {"scenario_id": "va99998", "topology": sim.name, "seed": 8902, "n_steps": sim.n_steps, "step_minutes": 5,
            "faults": faults, "maintenance": [], "gaps": [], "event_days": []}
    res, blind = sim.simulate(spec, with_blind=True)
    root = tmp_path_factory.mktemp("p5")
    e = write_scenario(res, obs_dir=root / "observable" / "val" / "with_status", truth_dir=root / "truth" / "val" / "with_status")
    entry = _truth_entry(e, res, sim, root, "val", "with_status", {"topology": sim.name, "k": 0}, None)
    obs = load_observed(root / "observable" / "val" / "with_status" / "va99998")
    cs = freeze_cfg(cfg)
    ix = index_for(obs, inventory(FIXTURE, cs))
    lab, _ = labels_matrix(pd.read_parquet(root / "truth" / "val" / "with_status" / entry["files"]["labels"]), ix.links)
    return obs, entry, lab, cs


@pytest.mark.parametrize("conv", ["sage", "gat", "gine", "mlp"])
def test_fast_prediction_equals_window_model(scen, conv):
    obs, _, _, cs = scen
    torch.manual_seed(0)
    model = NetTwinGNN(conv=conv, hidden=16).eval()
    g = graph_for(FIXTURE, cs)
    feat = b2_features(obs).astype(np.float32)
    det, elem = predict(model, feat, g)
    steps = np.array([0, 5, 11, 300, 415, feat.shape[0] - 1])
    xw = windows(feat, steps)
    with torch.no_grad():
        for i, t in enumerate(steps):
            d, _, _ = model(collate([(FIXTURE, xw[i], None, None)], {FIXTURE: g}))
            np.testing.assert_allclose(torch.sigmoid(d).numpy(), det[t], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(elem.sum(1), 1, atol=1e-4)


def test_prediction_is_causal(scen):
    obs, _, _, cs = scen
    torch.manual_seed(1)
    model = NetTwinGNN(conv="sage", hidden=16).eval()
    g = graph_for(FIXTURE, cs)
    feat = b2_features(obs).astype(np.float32)
    full = predict(model, feat, g)
    part = predict(model, feat[:500], g)
    np.testing.assert_allclose(full[0][:500], part[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(full[1][:500], part[1], rtol=1e-5, atol=1e-6)


def test_training_samples_and_targets(scen):
    obs, entry, lab, cs = scen
    x, d, r = training_samples(obs, entry, lab, cs, 8, np.random.default_rng(0))
    assert x.shape[1:] == (len(obs.links), W, 10) and d.shape == (8, len(obs.links))
    f = entry["faults"][0]
    g = graph_for(FIXTURE, cs)
    assert sum(bool(rr) for rr in r) >= 4                              # half the steps inside the fault window
    assert all(rr in ([], [g.elements.index(f["root_cause"])]) for rr in r)
    hidden = copy.deepcopy(entry)
    hidden["faults"][0]["impact"]["observable"] = False
    assert active_roots(hidden, f["t_start"], g.elements) == []        # unobservable faults are never targets


def test_shuffled_edges_is_a_fixed_permutation():
    p = shuffle_perm("gabaug/20/1000001", 30, 0)
    assert sorted(p) == list(range(30)) and (p != np.arange(30)).any()
    assert (p == shuffle_perm("gabaug/20/1000001", 30, 0)).all() and (p != shuffle_perm("gabaug/20/1000001", 30, 1)).any()


def test_training_is_deterministic(scen):
    obs, entry, lab, cs = scen
    x, d, r = training_samples(obs, entry, lab, cs, 8, np.random.default_rng(0))
    samples = [(FIXTURE, x[i], d[i], r[i]) for i in range(len(x))]
    graphs = {FIXTURE: graph_for(FIXTURE, cs)}
    torch.set_num_threads(4)
    a, ia = train(samples, graphs, "gat", 16, 3, 2, batch=4, log=lambda *_: None)
    b, ib = train(samples, graphs, "gat", 16, 3, 2, batch=4, log=lambda *_: None)
    assert ia["history"] == ib["history"]
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def test_phase5_script_reads_train_and_val_only():
    src = (ROOT / "scripts" / "phase5_gnn.py").read_text()
    splits = set(re.findall(r'/ "(\w+)"(?= /)', src))
    assert {"train", "val", "truth", "observable"} <= splits <= {"train", "val", "truth", "observable", "data", "models", "results"}
