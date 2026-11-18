"""Phase 11: LLM question answering over the knowledge graph.

Questions come from 12 templates (6 types x 2) on the validation and test topologies. Every gold answer is computed by
the existing KG code (`NetTwinKG` and its query API, plus `apply_change` on a copy for what-if questions). PoPs get
neutral ids (P01, P02, ...) from a seeded shuffle and each network a neutral name, so a model cannot answer from
general knowledge of real cities or operators.

Conditions (Haiku 5.5): Q0 closed-book (the question only), Q1 retrieval (BM25 top-k triples from a text serialisation
of the graph), Q2 tool use (wrappers over the KG query API). Scoring is exact: set F1 and exact match per question.
Telemetry is not used here; the networks are public topologies and none of this is Telstra's network.
"""
import copy
import itertools
import json
import math
import re
from collections import Counter

import networkx as nx
import numpy as np

from nettwin.agent.rag import tokens
from nettwin.kg import NetTwinKG, pop_node
from nettwin.topology import link_id, load

TYPES = {"one_hop": ("link_metric", "pop_links"), "multi_hop": ("services_on_link", "links_of_service"),
         "what_if": ("pop_fail_lost", "link_fail_rerouted"), "path": ("shortest_path", "second_shortest_path"),
         "count": ("gold_services_on_pop", "paths_on_link"), "time": ("links_changed", "services_rerouted")}
TEMPLATE_TYPE = {t: k for k, ts in TYPES.items() for t in ts}
PER_TEMPLATE = 3                 # questions per template per network
QUESTION_SEED = 11011
CHANGE_STEPS = (12, 24, 36, 48, 60, 72)
SIZE_BINS = (("small", 0, 20), ("medium", 21, 35), ("large", 36, 10**6))   # by number of PoPs
BM25_K = 50                      # triples retrieved for Q1


def size_bin(n_pops):
    return next(name for name, lo, hi in SIZE_BINS if lo <= n_pops <= hi)


# ---------------------------------------------------------------- networks with neutral ids

def neutral_topology(name, alias, seed):
    """The topology with PoPs renamed P01.. in a seeded random order and a neutral network name."""
    topo = load(name)
    rng = np.random.default_rng([seed, *map(ord, name)])
    order = [sorted(topo.nodes)[i] for i in rng.permutation(topo.number_of_nodes())]
    width = max(2, len(str(len(order))))
    mapping = {n: f"P{i + 1:0{width}d}" for i, n in enumerate(order)}
    g = nx.Graph(name=alias, geo=False, demands={})
    for n in sorted(topo.nodes, key=mapping.get):
        g.add_node(mapping[n], pos=topo.nodes[n]["pos"])
    for u, v, d in topo.edges(data=True):
        a, b = mapping[u], mapping[v]
        g.add_edge(a, b, link_id=link_id(a, b), dist_km=d["dist_km"], metric=d["metric"])
    return g


def change_history(kg, seed):
    """A fixed, seeded schedule on the KG: two busy links go down and come back, and one link's metric is tripled and
    reset. Everything is restored by the last step, so the end state equals step 0."""
    rng = np.random.default_rng(seed)
    links = kg.nodes_of_kind("Link")
    busy = sorted(links, key=lambda l: (-len(kg.paths_through(l, 0)), l))[: max(6, len(links) // 3)]
    l1, l2, l3 = (busy[i] for i in rng.choice(len(busy), 3, replace=False))
    m = kg.G.nodes[l2]["metric"]
    plan = [(CHANGE_STEPS[0], {"links_down": [l1]}), (CHANGE_STEPS[1], {"metrics": {l2: 3 * m}}),
            (CHANGE_STEPS[2], {"links_up": [l1]}), (CHANGE_STEPS[3], {"links_down": [l3]}),
            (CHANGE_STEPS[4], {"metrics": {l2: None}}), (CHANGE_STEPS[5], {"links_up": [l3]})]
    for t, change in plan:
        kg.apply_change(t, reason="scheduled change", **change)
    return kg


def build_network(name, alias, cfg, seed):
    topo = neutral_topology(name, alias, seed)
    kg = NetTwinKG.from_config(topo, cfg)
    return change_history(kg, seed + 1)


# ---------------------------------------------------------------- gold answers (existing KG code)

def connects(kg, link):
    return sorted(kg._out(link, "CONNECTS"))


def links_at(kg, pop):
    return sorted(kg._in(pop, "CONNECTS"))


def service_links(kg, svc, t):
    return sorted({l for p in kg.service_paths(svc, t) for l in kg.links_of_path(p)})


def what_if(kg, element):
    """Apply a failure to a copy of the KG one step after its last change; per service: paths before and after."""
    k = copy.deepcopy(kg)
    t = k.t + 1
    before = {s: set(k.service_paths(s, k.t)) for s in k.nodes_of_kind("Service")}
    k.apply_change(t, **({"pops_down": [element]} if k.kind(element) == "PoP" else {"links_down": [element]}))
    after = {s: set(k.service_paths(s, t)) for s in before}
    lost = sorted(s for s in before if before[s] and not after[s])
    rerouted = sorted(s for s in before if after[s] and after[s] != before[s])
    return k, t, lost, rerouted


def k_shortest(kg, a, b, k=3):
    """The k cheapest simple paths a -> b on the routing graph (link metrics), with costs."""
    h = kg._routing_graph()
    out = []
    for p in itertools.islice(nx.shortest_simple_paths(h, a, b, weight="metric"), k):
        out.append((p, sum(h.edges[u, v]["metric"] for u, v in zip(p, p[1:]))))
    return out


def gen_questions(net, kg, split, rng):
    """PER_TEMPLATE questions per template on one network. Elements are drawn so each answer is unique and non-trivial."""
    pops, links, svcs = kg.nodes_of_kind("PoP"), kg.nodes_of_kind("Link"), kg.nodes_of_kind("Service")
    cls = {s: kg.G.nodes[s]["cls"] for s in svcs}
    pick = lambda xs, n=PER_TEMPLATE: [xs[i] for i in sorted(rng.choice(len(xs), min(n, len(xs)), replace=False))]  # noqa: E731
    qs = []

    def add(template, text, kind, answer, **extra):
        qs.append({"network": net, "split": split, "template": template, "type": TEMPLATE_TYPE[template], "question": text,
                   "answer_kind": kind, "answer": answer, **extra})

    S0 = "at step 0 (the normal state)"
    for l in pick(links):
        add("link_metric", f"What is the routing metric of {l}?", "number", kg.G.nodes[l]["metric"])
    for p in pick(pops):
        add("pop_links", f"Which links attach to {p}?", "set", links_at(kg, p))
    for l in pick([l for l in links if kg.services_on(l, 0)]):
        add("services_on_link", f"Which services use a routing path that traverses {l} {S0}?", "set", kg.services_on(l, 0))
    for s in pick(svcs):
        add("links_of_service", f"Which links are on the routing paths of {s} {S0}?", "set", service_links(kg, s, 0))
    for p in pick(pops):
        add("pop_fail_lost", f"If {p} fails (from the normal state), which services lose every routing path?", "set",
            what_if(kg, p)[2])
    cand = [l for l in links if what_if(kg, l)[3]]
    for l in pick(cand):
        add("link_fail_rerouted", f"If {l} fails (from the normal state), which services get a different set of routing "
            "paths but keep at least one?", "set", what_if(kg, l)[3])
    pairs = [(a, b) for a, b in itertools.combinations(pops, 2)]
    rng.shuffle(pairs)
    sp, ssp = [], []
    for a, b in pairs:
        act = kg.paths_between(a, b, 0)
        if len(act) != 1:
            continue
        seq = kg.G.nodes[act[0]]["pops"]
        if len(seq) < 3:
            continue
        ks = k_shortest(kg, kg.G.nodes[a]["name"], kg.G.nodes[b]["name"])
        names = [pop_node(x) for x in ks[0][0]]
        assert names == [pop_node(x) for x in seq] or names == [pop_node(x) for x in seq][::-1]
        if len(sp) < PER_TEMPLATE:
            sp.append((a, b, [pop_node(x) for x in ks[0][0]]))
        elif len(ssp) < PER_TEMPLATE and len(ks) == 3 and ks[0][1] < ks[1][1] < ks[2][1]:
            ssp.append((a, b, [pop_node(x) for x in ks[1][0]], ks[1][1]))
        if len(sp) == len(ssp) == PER_TEMPLATE:
            break
    for a, b, path in sp:
        add("shortest_path", f"List, in order, the PoPs on the lowest-cost routing path from {a} to {b} {S0} "
            "(cost = sum of link routing metrics).", "path", path)
    for a, b, path, cost in ssp:
        add("second_shortest_path", f"List, in order, the PoPs on the second-lowest-cost loop-free path from {a} to {b} "
            f"{S0} (cost = sum of link routing metrics; the lowest-cost path is unique).", "path", path, cost=cost)
    for p in pick(pops):
        add("gold_services_on_pop", f"How many gold-SLA services use a routing path through {p} (endpoints included) {S0}?",
            "number", sum(cls[s] == "gold" for s in kg.services_on(p, 0)))
    for l in pick(links):
        add("paths_on_link", f"How many routing paths traverse {l} {S0}?", "number", len(kg.paths_through(l, 0)))
    windows = [(a, b) for a, b in itertools.combinations(range(0, 85, 6), 2) if b - a >= 12]
    for t0, t1 in pick(windows):
        ev = kg.changes_between(t0, t1)
        add("links_changed", f"Which links went down, came back up or had their routing metric changed between step {t0} "
            f"and step {t1} (inclusive)?", "set", sorted({e["element"] for e in ev if e["type"] in ("link_down", "link_up", "metric_change")}))
    for t0, t1 in pick(windows):
        ev = kg.changes_between(t0, t1)
        pairs_changed = {tuple(e["pair"]) for e in ev if e["type"] == "reroute"}
        add("services_rerouted", f"Which services had their set of routing paths changed between step {t0} and step {t1} "
            "(inclusive)?", "set", sorted({s for pr in pairs_changed for s in kg._pair_services[pr]}))
    return qs


def make_questions(split_networks, cfg):
    """split_networks: [(split, real name, alias)]. Returns (questions, {alias: kg}, {alias: meta})."""
    questions, kgs, meta = [], {}, {}
    for i, (split, name, alias) in enumerate(split_networks):
        kg = build_network(name, alias, cfg, QUESTION_SEED + 97 * i)
        kgs[alias] = kg
        n = len(kg.nodes_of_kind("PoP"))
        meta[alias] = {"split": split, "n_pops": n, "n_links": len(kg.nodes_of_kind("Link")),
                       "n_services": len(kg.nodes_of_kind("Service")), "size": size_bin(n)}
        rng = np.random.default_rng([QUESTION_SEED, i])
        for q in gen_questions(alias, kg, split, rng):
            q["size"] = meta[alias]["size"]
            q["n_pops"] = n
            questions.append(q)
    for j, q in enumerate(questions):
        q["qid"] = f"{q['split']}-{j:03d}"
    return questions, kgs, meta


# ---------------------------------------------------------------- Q1: text serialisation and BM25

def _iv(ivs):
    return ", ".join(f"steps {s} to {e - 1}" if e is not None else f"from step {s}" for s, e in ivs) or "never"


def triples(kg):
    """One short line per fact. Paths keep their activity intervals; the change log is included."""
    out = []
    for p in kg.nodes_of_kind("PoP"):
        out.append(f"{p} is a PoP")
    for l in kg.nodes_of_kind("Link"):
        a, b = connects(kg, l)
        out += [f"{l} CONNECTS {a}", f"{l} CONNECTS {b}", f"{l} routing_metric {kg.G.nodes[l]['metric']}",
                f"{l} length_km {kg.G.nodes[l]['dist_km']:g}"]
    for s in kg.nodes_of_kind("Service"):
        d = kg.G.nodes[s]
        out += [f"{s} endpoint {pop_node(d['src'])}", f"{s} endpoint {pop_node(d['dst'])}", f"{s} sla_class {d['cls']}"]
        for p in sorted(kg._out(s, "USES")):
            out.append(f"{s} USES {p} (active {_iv(kg.G.nodes[p]['intervals'])})")
    for p in kg.nodes_of_kind("RoutingPath"):
        for i, l in enumerate(kg.links_of_path(p)):
            out.append(f"{p} TRAVERSES {l} (hop {i + 1}; path active {_iv(kg.G.nodes[p]['intervals'])})")
    for e in kg.events:
        if e["type"] == "reroute":
            out.append(f"step {e['t']} reroute {pop_node(e['pair'][0])} {pop_node(e['pair'][1])}: removed "
                       f"{' '.join(e['removed']) or 'none'}; added {' '.join(e['added']) or 'none'}")
        elif e["type"] == "metric_change":
            out.append(f"step {e['t']} metric_change {e['element']} from {e['old']} to {e['new']}")
        else:
            out.append(f"step {e['t']} {e['type']} {e['element']}")
    return out


ID_RE = re.compile(r"\b(?:link:P\d+--P\d+|pop:P\d+|svc:S\d+|path:P\d+(?:>P\d+)+|P\d+|S\d+)\b")


def qa_tokens(text):
    """The project's BM25 tokens plus each element id as one token, so 'link:P03--P17' also matches as a whole
    (link ids normalised to sorted endpoints, bare 'P07' and 'S004' to 'pop:P07' and 'svc:S004')."""
    return tokens(text) + [_norm_el(m).lower() for m in ID_RE.findall(text)]


class TripleIndex:
    """BM25 over the triples (`qa_tokens`: the project's tokenizer plus whole element ids)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.tf = [Counter(qa_tokens(t)) for t in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / len(self.len)
        df = Counter(w for c in self.tf for w in c)
        self.idf = {w: math.log(1 + (len(docs) - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.post = {}
        for i, c in enumerate(self.tf):
            for w in c:
                self.post.setdefault(w, []).append(i)
        self.k1, self.b = k1, b

    def search(self, query, k=BM25_K):
        sc = Counter()
        for w in set(qa_tokens(query)):
            for i in self.post.get(w, ()):
                f = self.tf[i][w]
                sc[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return [self.docs[i] for i, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


# ---------------------------------------------------------------- Q2: tools over the KG query API

def _j(x):
    return json.dumps(x, separators=(",", ":"))


def _norm_el(x):
    """Accept 'P07', 'pop:P07', 'link:P09--P02' (either order), 'S004', 'svc:S004'."""
    x = str(x).strip()
    if re.fullmatch(r"P\d+", x):
        return f"pop:{x}"
    if re.fullmatch(r"S\d+", x):
        return f"svc:{x}"
    m = re.fullmatch(r"link:(P\d+)--(P\d+)", x)
    if m:
        return f"link:{link_id(m.group(1), m.group(2))}"
    return x


def path_info(kg, p):
    """A path's PoPs and links in order and its total routing metric in the normal state (base link metrics)."""
    ls = kg.links_of_path(p)
    return {"path": p, "pops": [pop_node(x) for x in kg.G.nodes[p]["pops"]], "links": ls,
            "total_metric": sum(kg.G.nodes[l]["metric"] for l in ls), "active": kg.G.nodes[p]["intervals"]}


def element_info(kg, e):
    k = kg.kind(e)
    d = kg.G.nodes[e]
    if k == "PoP":
        return {"element": e, "kind": k, "links": links_at(kg, e)}
    if k == "Link":
        return {"element": e, "kind": k, "connects": connects(kg, e), "routing_metric": d["metric"], "length_km": d["dist_km"]}
    if k == "Service":
        return {"element": e, "kind": k, "endpoints": [pop_node(d["src"]), pop_node(d["dst"])], "sla_class": d["cls"],
                "sla_latency_ms": d["sla_latency_ms"], "sla_loss_pct": d["sla_loss_pct"]}
    if k == "RoutingPath":
        return path_info(kg, e) | {"kind": k, "used_by": sorted(kg._in(e, "USES"))}
    raise ValueError(f"{e} is a {k}")


EL = {"type": "string", "description": "element id: pop:P07, link:P02--P09, svc:S004 or path:P02>P05>P09"}
T = {"type": "integer", "description": "time step (default 0, the normal state)"}
TOOLS = [
    {"name": "kg_list_elements", "description": "List every element of one kind. Services come with endpoints and SLA class.",
     "input_schema": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["PoP", "Link", "Service"]}}, "required": ["kind"]}},
    {"name": "kg_element_info", "description": "Attributes of one element: a PoP's links; a link's endpoints, routing metric and "
     "length; a service's endpoints and SLA; a routing path's PoPs, links, total metric, active intervals and services.",
     "input_schema": {"type": "object", "properties": {"element": EL}, "required": ["element"]}},
    {"name": "kg_paths_through", "description": "Routing paths active at step t that traverse a link or pass through a PoP (endpoints included).",
     "input_schema": {"type": "object", "properties": {"element": EL, "t": T}, "required": ["element"]}},
    {"name": "kg_services_on", "description": "Services whose active routing paths at step t use a link, PoP or routing path.",
     "input_schema": {"type": "object", "properties": {"element": EL, "t": T}, "required": ["element"]}},
    {"name": "kg_paths_between", "description": "Active routing paths between two PoPs at step t (several if equal-cost).",
     "input_schema": {"type": "object", "properties": {"pop_a": EL, "pop_b": EL, "t": T}, "required": ["pop_a", "pop_b"]}},
    {"name": "kg_service_paths", "description": "Routing paths a service uses at step t.",
     "input_schema": {"type": "object", "properties": {"service": EL, "t": T}, "required": ["service"]}},
    {"name": "kg_neighbours", "description": "Same-kind physical neighbours within `hops` (PoP: adjacent PoPs; link: links sharing a PoP).",
     "input_schema": {"type": "object", "properties": {"element": EL, "hops": {"type": "integer"}}, "required": ["element"]}},
    {"name": "kg_changes_between", "description": "Change log entries with t0 <= step <= t1: link_down, link_up, metric_change "
     "and reroute (per PoP pair: removed and added paths, unreachable flag).",
     "input_schema": {"type": "object", "properties": {"t0": {"type": "integer"}, "t1": {"type": "integer"}}, "required": ["t0", "t1"]}},
    {"name": "kg_what_if_down", "description": "Fail one link or PoP on a copy of the graph, starting from the normal state, and "
     "return the change log the graph records: per PoP pair, the paths removed and added and whether it became unreachable. "
     "Does not change the real graph.",
     "input_schema": {"type": "object", "properties": {"element": EL}, "required": ["element"]}},
]
SUBMIT = {"name": "submit_answer", "description": "Submit the final answer once. Put element ids in `items` (for a path: the "
          "PoP ids in order from source to destination). For a numeric question put the number in `number` and leave items empty.",
          "input_schema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}},
                                                           "number": {"type": ["number", "null"]}}, "required": ["items"]}}


def call_tool(kg, name, args):
    """Run one tool. Returns (text, is_error). Errors are returned to the model, not raised."""
    try:
        t = int(args.get("t", 0))
        if name == "kg_list_elements":
            kind = args["kind"]
            if kind == "Service":
                return _j([[s, pop_node(kg.G.nodes[s]["src"]), pop_node(kg.G.nodes[s]["dst"]), kg.G.nodes[s]["cls"]]
                           for s in kg.nodes_of_kind("Service")]), False
            return _j(kg.nodes_of_kind(kind)), False
        if name == "kg_element_info":
            return _j(element_info(kg, _norm_el(args["element"]))), False
        if name == "kg_paths_through":
            return _j(kg.paths_through(_norm_el(args["element"]), t)), False
        if name == "kg_services_on":
            return _j(kg.services_on(_norm_el(args["element"]), t)), False
        if name == "kg_paths_between":
            return _j(kg.paths_between(_norm_el(args["pop_a"]), _norm_el(args["pop_b"]), t)), False
        if name == "kg_service_paths":
            return _j(kg.service_paths(_norm_el(args["service"]), t)), False
        if name == "kg_neighbours":
            return _j(kg.neighbours(_norm_el(args["element"]), int(args.get("hops", 1)))), False
        if name == "kg_changes_between":
            return _j(kg.changes_between(int(args["t0"]), int(args["t1"]))), False
        if name == "kg_what_if_down":
            k, t1, _, _ = what_if(kg, _norm_el(args["element"]))
            ev = [{kk: v for kk, v in e.items() if kk != "t"} for e in k.changes_between(t1, t1)]
            return _j(ev), False
        return f"error: unknown tool {name}", True
    except (KeyError, ValueError, TypeError) as e:
        return f"error: {type(e).__name__}: {e}", True


# ---------------------------------------------------------------- scoring

def normalise(answer, kind):
    """The submitted answer as a set (set), a PoP sequence (path) or a number (number)."""
    if answer is None:
        return None
    items = [_norm_el(x) for x in (answer.get("items") or [])]
    if kind == "number":
        n = answer.get("number")
        if n is None and len(items) == 1:
            try:
                n = float(items[0])
            except ValueError:
                n = None
        return None if n is None else float(n)
    if kind == "path":
        return [x if x.startswith("pop:") else _norm_el(x) for x in items]
    return set(items)


def _f1(pred, gold):
    if not pred and not gold:
        return 1.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


def path_links(seq):
    return {link_id(a.removeprefix("pop:"), b.removeprefix("pop:")) for a, b in zip(seq, seq[1:])}


def score(q, answer):
    """(exact match, F1) for one question. Paths: exact if the PoP sequence matches (either direction), F1 over the
    links of the path (so the two endpoints given in the question earn no credit). Numbers: both are 1 if equal."""
    pred = normalise(answer, q["answer_kind"])
    if pred is None:
        return 0.0, 0.0
    if q["answer_kind"] == "number":
        ok = float(abs(pred - q["answer"]) < 1e-9)
        return ok, ok
    if q["answer_kind"] == "path":
        gold = q["answer"]
        return float(pred == gold or pred == gold[::-1]), _f1(path_links(pred), path_links(gold))
    gold = {_norm_el(x) for x in q["answer"]}
    return float(pred == gold), _f1(pred, gold)
