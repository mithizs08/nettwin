"""Phase 11 KG question answering: scoring, answer normalisation, neutral ids and gold answers (audit m7)."""
import itertools
import json
import re
from pathlib import Path

import networkx as nx
import pytest
import yaml

from nettwin import kgqa as K
from nettwin.topology import link_id, load

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def q(kind, answer):
    return {"answer_kind": kind, "answer": answer}


# ---------------------------------------------------------------- scoring and normalisation

def test_set_scoring_exact_and_partial():
    gold = q("set", ["svc:S001", "svc:S002"])
    assert K.score(gold, {"items": ["S002", "svc:S001"]}) == (1.0, 1.0)
    em, f1 = K.score(gold, {"items": ["svc:S001", "svc:S003"]})
    assert em == 0.0 and f1 == pytest.approx(0.5)
    assert K.score(gold, {"items": []}) == (0.0, 0.0)


def test_empty_gold_set_scores_only_an_empty_answer():
    gold = q("set", [])
    assert K.score(gold, {"items": []}) == (1.0, 1.0)
    assert K.score(gold, {"items": ["svc:S001"]}) == (0.0, 0.0)


def test_link_ids_are_order_free():
    gold = q("set", [f"link:{link_id('P02', 'P09')}"])
    assert K.score(gold, {"items": ["link:P09--P02"]}) == (1.0, 1.0)
    assert K.score(gold, {"items": ["link:P02--P09"]}) == (1.0, 1.0)


def test_path_exact_in_either_direction_and_f1_over_links():
    gold = q("path", ["pop:P01", "pop:P02", "pop:P03", "pop:P04"])
    assert K.score(gold, {"items": ["P01", "P02", "P03", "P04"]}) == (1.0, 1.0)
    assert K.score(gold, {"items": ["P04", "P03", "P02", "P01"]}) == (1.0, 1.0)
    em, f1 = K.score(gold, {"items": ["P01", "P02", "P05", "P04"]})   # one of three links right
    assert em == 0.0 and f1 == pytest.approx(1 / 3)
    assert K.score(gold, {"items": ["P01", "P04"]}) == (0.0, 0.0)     # the endpoints alone earn nothing


def test_number_scoring_and_number_given_as_an_item():
    gold = q("number", 7)
    assert K.score(gold, {"items": [], "number": 7}) == (1.0, 1.0)
    assert K.score(gold, {"items": [], "number": 7.0}) == (1.0, 1.0)
    assert K.score(gold, {"items": [], "number": 8}) == (0.0, 0.0)
    assert K.score(gold, {"items": ["7"]}) == (1.0, 1.0)
    assert K.score(gold, {"items": ["seven"]}) == (0.0, 0.0)


def test_no_answer_scores_zero():
    for kind, ans in (("set", ["svc:S001"]), ("path", ["pop:P01", "pop:P02"]), ("number", 3)):
        assert K.score(q(kind, ans), None) == (0.0, 0.0)


# ---------------------------------------------------------------- neutral ids and gold answers

@pytest.fixture(scope="module")
def first_val():
    name = CFG["topologies"]["validation_real"][0]
    return name, K.build_network(name, "val-1", CFG, K.QUESTION_SEED)


def test_neutral_ids_are_a_relabelling_of_the_real_topology(first_val):
    name, kg = first_val
    real = load(name)
    g = K.neutral_topology(name, "val-1", K.QUESTION_SEED)
    assert nx.is_isomorphic(real, g)
    assert all(re.fullmatch(r"P\d+", n) for n in g.nodes)
    text = "\n".join(K.triples(kg))
    assert not [n for n in real.nodes if len(n) >= 4 and re.search(r"\b" + re.escape(n) + r"\b", text)]


def test_committed_questions_hold_no_real_pop_name():
    names = {n for s in ("validation_real", "test") for t in CFG["topologies"][s] for n in load(t).nodes if len(n) >= 4}
    for split in ("val", "test"):
        text = (ROOT / f"results/phase11/questions_{split}.jsonl").read_text()
        assert not [n for n in names if re.search(r"\b" + re.escape(n) + r"\b", text)]


def test_gold_answers_match_networkx_on_one_network(first_val):
    """link metric, links at a PoP and the lowest-cost path, recomputed from the graph without the KG query code."""
    _, kg = first_val
    g = K.neutral_topology(CFG["topologies"]["validation_real"][0], "val-1", K.QUESTION_SEED)
    qs = [x for x in map(json.loads, (ROOT / "results/phase11/questions_val.jsonl").read_text().splitlines()) if x["network"] == "val-1"]
    metric = {link_id(u, v): d["metric"] for u, v, d in g.edges(data=True)}
    checked = 0
    for x in qs:
        ids = re.findall(r"(link:P\d+--P\d+|pop:P\d+)", x["question"])
        if x["template"] == "link_metric":
            assert x["answer"] == metric[ids[0][5:]]
        elif x["template"] == "pop_links":
            p = ids[0][4:]
            assert x["answer"] == sorted(f"link:{link_id(p, n)}" for n in g[p])
        elif x["template"] == "shortest_path":
            h = nx.Graph([(u, v, {"w": metric[link_id(u, v)]}) for u, v in g.edges])
            ps = list(itertools.islice(nx.shortest_simple_paths(h, ids[0][4:], ids[1][4:], weight="w"), 2))
            assert [f"pop:{n}" for n in ps[0]] in (x["answer"], x["answer"][::-1])
        else:
            continue
        checked += 1
    assert checked >= 6


def test_tools_return_errors_instead_of_raising(first_val):
    _, kg = first_val
    assert K.call_tool(kg, "no_such_tool", {})[1] is True
    assert K.call_tool(kg, "kg_element_info", {})[1] is True
    out, err = K.call_tool(kg, "kg_element_info", {"element": "P01"})
    assert not err and "P01" in out
