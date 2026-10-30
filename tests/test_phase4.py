"""Phase 4: alerts from observable data only, baselines respect the banned-feature rules, and the evaluation harness
computes what the decision log says. Runs on topohub's shipped `gabriel/20/1` (in no split) and toy arrays."""
import copy
import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import average_precision_score

from nettwin.alerts import generate_alerts
from nettwin.baselines import FEATURES, B1EwmaZ, B2IsolationForest, B3Tomography, index_for
from nettwin.dataset import _truth_entry
from nettwin.evaluation import (WARMUP, ap_from_hists, labels_matrix, make_edges, scenario_record, threshold_for,
                                to_bins, window_classes)
from nettwin.faults import make_fault
from nettwin.observed import Observed, freeze_cfg, inventory, load_observed
from nettwin.simulator import Simulator, write_scenario
from nettwin.topology import load

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = "gabriel/20/1"
TIME_WORDS = ("hour", "time", "clock", "day_of", "weekday", "since_start", "file", "size")


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(yaml.safe_load((ROOT / "config.yaml").read_text()))
    c["simulator"]["days"] = 3
    return c


@pytest.fixture(scope="module")
def scen(cfg, tmp_path_factory):
    """One 3-day scenario with a link_down and an overlapping node_down elsewhere, written like the v1 dataset."""
    sim = Simulator(load(FIXTURE), cfg)
    link = max(sim.links, key=lambda l: (len(sim.kg.paths_through(l)), l))
    ends = {sim.kg.G.nodes[link]["u"], sim.kg.G.nodes[link]["v"]}
    pop = next(p for p in sim.pops if p not in ends and all(n not in ends for n in sim.topo.neighbors(p)))
    rng = np.random.default_rng(0)
    faults = [make_fault(sim, "link_down", link, 400, 30, rng, fault_id="F0"),
              make_fault(sim, "node_down", f"pop:{pop}", 410, 10, rng, fault_id="F1")]
    for f in faults:
        f.update(placement="concurrent", linked_maintenance=None, caused_by_maintenance=False, overlaps_maintenance=False)
    spec = {"scenario_id": "va99999", "topology": sim.name, "seed": 8901, "n_steps": sim.n_steps, "step_minutes": 5,
            "faults": faults, "maintenance": [], "gaps": [], "event_days": []}
    res, blind = sim.simulate(spec, with_blind=True)
    root = tmp_path_factory.mktemp("p4")
    out = {}
    for variant, r in (("with_status", res), ("status_blind", blind)):
        e = write_scenario(r, obs_dir=root / "observable" / "val" / variant, truth_dir=root / "truth" / "val" / variant)
        out[variant] = _truth_entry(e, r, sim, root, "val", variant, {"topology": sim.name, "k": 0})
    return root, out, cfg


def _obs(root, variant):
    return load_observed(root / "observable" / "val" / variant / "va99999")


def test_alerts_come_from_observable_data_only(scen, tmp_path):
    root, entries, cfg = scen
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    copy_root = tmp_path / "only_observable"
    shutil.copytree(root / "observable", copy_root / "observable")       # no truth folder at all
    a = generate_alerts(load_observed(copy_root / "observable" / "val" / "with_status" / "va99999"), kg)
    pd.testing.assert_frame_equal(a, generate_alerts(_obs(root, "with_status"), kg))
    with pytest.raises(ValueError, match="observable"):
        load_observed(root / "truth" / "val" / "with_status" / "va99999")
    f0 = entries["with_status"]["faults"][0]
    down = a[(a.rule == "link_status_down") & (a.target == f0["root_cause"])]
    assert len(down) and down.t_start.min() >= f0["t_start"] and down.t_start.min() <= f0["t_start"] + 3
    blind = generate_alerts(_obs(root, "status_blind"), kg)
    assert not blind.rule.isin(["link_status_down", "pop_unreachable"]).any()
    assert (blind.rule == "link_no_data").any()


def test_no_clock_and_no_banned_features():
    fields = {f.name for f in dataclasses.fields(Observed)}
    assert fields == {"scenario_id", "topology", "status_blind", "links", "services", "util", "latency", "discards",
                      "svc_latency", "svc_loss", "events"}
    assert not [f for f in FEATURES if any(w in f for w in TIME_WORDS)]


def test_scores_are_causal(scen):
    """Scores at step t use data up to t only: cutting the scenario at step 500 changes nothing before it."""
    root, _, cfg = scen
    obs = _obs(root, "with_status")
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    ix = index_for(obs, kg)
    cut = dataclasses.replace(obs, util=obs.util[:500], latency=obs.latency[:500], discards=obs.discards[:500],
                              svc_latency=obs.svc_latency[:500], svc_loss=obs.svc_loss[:500],
                              events=[e for e in obs.events if e["t"] < 500])
    b2 = B2IsolationForest(seed=1, n_cells=2000, n_estimators=20).fit([obs])
    for m in (B1EwmaZ(), b2, B3Tomography()):
        full, part = m.score(obs, kg, ix)[0], m.score(cut, kg, ix)[0]
        np.testing.assert_allclose(full[:500], part, err_msg=m.name)


def test_binned_ap_matches_exact():
    rng = np.random.default_rng(0)
    y = rng.random(200_000) < 0.002
    s = rng.normal(size=y.size) + 2.5 * y
    edges = make_edges(s[y], s[~y][::10])
    b = to_bins(s, edges)
    nb = len(edges) + 1
    binned = ap_from_hists(np.bincount(b[y], minlength=nb), np.bincount(b[~y], minlength=nb))
    assert abs(binned - average_precision_score(y, s)) < 0.005
    ties = np.r_[np.ones(10), np.zeros(90)]
    assert np.isclose(ap_from_hists(np.array([0, 10]), np.array([0, 90])), average_precision_score(ties > 0, np.ones(100)))


class Fixed:
    """A stub localiser: fixed score per element."""
    def __init__(self, scores):
        self.scores = scores

    def localise(self, state, t0, t1):
        return self.scores


def test_localisation_is_set_based_and_pessimistic(scen):
    root, entries, cfg = scen
    entry = entries["with_status"]
    obs = _obs(root, "with_status")
    ix = index_for(obs, inventory(FIXTURE, freeze_cfg(cfg)))
    lab, pres = labels_matrix(pd.read_parquet(root / "truth" / "val" / "with_status" / entry["files"]["labels"]), ix.links)
    roots = [ix.elements.index(f["root_cause"]) for f in entry["faults"]]
    link = np.zeros(lab.shape)
    edges = np.array([0.0, 1.0])
    s = np.zeros(len(ix.elements))
    s[roots] = 1.0                                         # both concurrent roots tie at the top
    rec = scenario_record(Fixed(s), (link,), ix, entry, lab, pres, edges)
    assert [f["rank"] for f in rec["faults"]] == [1, 1]    # the other true root is taken out, not counted against
    rec = scenario_record(Fixed(np.zeros(len(ix.elements))), (link,), ix, entry, lab, pres, edges)
    assert all(f["rank"] == len(ix.elements) - 1 for f in rec["faults"])   # all tied: worst rank (one root removed)


def test_record_counts_cells_as_defined(scen):
    root, entries, cfg = scen
    entry = entries["status_blind"]
    obs = _obs(root, "status_blind")
    ix = index_for(obs, inventory(FIXTURE, freeze_cfg(cfg)))
    lab, pres = labels_matrix(pd.read_parquet(root / "truth" / "val" / "status_blind" / entry["files"]["labels"]), ix.links)
    rec = scenario_record(Fixed(np.zeros(len(ix.elements))), (np.zeros(lab.shape),), ix, entry, lab, pres, np.array([0.5]))
    assert rec["neg"].sum() == np.sum(lab[WARMUP:] == 0)               # every label-0 cell after warmup, with or without telemetry
    assert rec["neg_present"].sum() == np.sum((lab[WARMUP:] == 0) & pres[WARMUP:])
    assert np.sum(rec["pos_task"] == 2) == np.sum(lab[WARMUP:] == 2)   # every root cell belongs to a fault
    assert not rec["pos_present"][rec["pos_task"] == 2].all()          # status_blind: down root links have no telemetry


def test_window_classes_and_threshold():
    entry = {"faults": [{"t_start": 300, "t_end": 310}], "maintenance": [{"t_start": 400, "t_end": 410}]}
    starts, ends, cls = window_classes(entry, 600)
    assert starts[0] == WARMUP and cls[0] == 0 and cls[1] == 2         # 288-299 is clear, 300-311 meets the fault
    assert cls[list(starts).index(408)] == 1 and cls[list(starts).index(420)] == 1   # change window + 12-step tail
    assert cls[list(starts).index(432)] == 0
    x = np.arange(1000.0)
    thr = threshold_for(x, 0.05)
    assert np.mean(x >= thr) <= 0.05 and np.mean(x >= thr - 1) > 0.05


def test_b3_localises_a_failed_link(scen):
    root, entries, cfg = scen
    obs = _obs(root, "with_status")
    kg = inventory(FIXTURE, freeze_cfg(cfg))
    ix = index_for(obs, kg)
    m = B3Tomography()
    state = m.score(obs, kg, ix)
    f0, f1 = entries["with_status"]["faults"]
    top = np.argsort(-m.localise(state, f0["t_start"], f0["t_end"]))[:3]
    assert ix.elements.index(f0["root_cause"]) in top or ix.elements.index(f1["root_cause"]) in top


def test_phase4_script_reads_train_and_val_only():
    import re
    src = (ROOT / "scripts" / "phase4_baselines.py").read_text()
    splits = set(re.findall(r'/ "(\w+)" /', src)) | set(re.findall(r'\("(\w+)", variant', src))
    assert {"train", "val"} <= splits <= {"train", "val", "truth", "observable", "data", "models"}
