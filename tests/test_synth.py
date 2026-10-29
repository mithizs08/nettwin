"""Generated topologies and the topology split (audit C1, M4, Q6)."""
import json
import random

import networkx as nx
import pytest
import yaml

from nettwin.synth import generate, generated_split, reference_stats
from nettwin.topology import local_longitude


@pytest.fixture(scope="module")
def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def test_generated_graphs_are_valid_and_deterministic(cfg):
    for n, seed in [(15, 1000000), (37, 1000022), (60, 1000045)]:
        g = generate(n, seed, cfg)
        assert g.number_of_nodes() == n and nx.is_connected(g)
        assert all(local_longitude(g, v) is not None for v in g)
        h = generate(n, seed, cfg)
        assert sorted(g.edges(data=True)) == sorted(h.edges(data=True))


def test_generator_leaves_global_random_alone(cfg):
    random.seed(123)
    expect = random.random()
    random.seed(123)
    generate(30, 1000010, cfg)
    assert random.random() == expect


def test_structure_tracks_reference_pool(cfg):
    names = generated_split(cfg)["train"][::10]
    degs = [2 * g.number_of_edges() / g.number_of_nodes() for g in (generate(int(x.split("/")[1]), int(x.split("/")[2]), cfg) for x in names)]
    ref = sorted(s["avg_degree"] for s in reference_stats(cfg))
    assert ref[len(ref) // 10] - 0.2 < sorted(degs)[len(degs) // 2] < ref[-len(ref) // 10] + 0.2


def test_split_is_disjoint_stratified_and_large_enough(cfg):
    sp = generated_split(cfg)
    assert not set(sp["train"]) & set(sp["val"])
    assert len(sp["val"]) >= 60 and len(sp["train"]) + len(sp["val"]) == cfg["topologies"]["synthetic"]["count"]
    lo, hi = cfg["topologies"]["synthetic"]["n_range"]
    assert {int(x.split("/")[1]) for x in sp["val"]} == set(range(lo, hi + 1))


def test_reference_pool_is_clean(cfg):
    sel = json.loads(open(cfg["topologies"]["reference_pool_file"]).read())
    t = cfg["topologies"]
    forbidden = set(t["dev"]) | set(t["test"]) | set(t["validation_real"]) | set(t["skip"])
    pool = set(sel["reference_pool"])
    assert not pool & forbidden
    fam = sel["family_of"]
    used = {fam[x] for x in t["dev"] + t["test"] + t["validation_real"] if x in fam}
    assert not {fam[x] for x in pool} & used
