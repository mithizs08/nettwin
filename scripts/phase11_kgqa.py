"""Phase 11: LLM question answering over the knowledge graph (Q0 closed-book, Q1 BM25 retrieval, Q2 KG tools).

    python scripts/phase11_kgqa.py --build-questions            # write results/phase11/questions_{val,test}.jsonl
    python scripts/phase11_kgqa.py --split val --name smoke --limit 12
    python scripts/phase11_kgqa.py --split val --name dev       # development run on validation
    python scripts/phase11_kgqa.py --freeze                     # hash the code and questions into results/phase11_frozen.json
    python scripts/phase11_kgqa.py --split test                 # test, once
    python scripts/phase11_kgqa.py --score results/phase11/test # tables and paired verdicts

All conditions use claude-haiku-5-5. Spend guard: this phase stops before any question once its logged spend plus the
questions in flight could pass PHASE_BUDGET_USD, and the whole project stays within $1 of project_budget_usd.
`--split test` refuses unless the Phase 11 pre-registration is in docs/decision_log.md, the hashes match
results/phase11_frozen.json, the output folder does not exist, and no --limit or condition subset is given.
"""
import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import numpy as np
import yaml

from nettwin.agent import load_agent_config
from nettwin.agent.loop import _price
from nettwin.evaluation import boot_weights, ci, verdict
from nettwin.kgqa import BM25_K, SUBMIT, TOOLS, TYPES, SIZE_BINS, TripleIndex, call_tool, make_questions, score, triples

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase11"
FROZEN = ROOT / "results" / "phase11_frozen.json"
PREREG = "### 2026-11-18: Phase 11 pre-registration"
MODEL = "claude-haiku-5-5"
CONDITIONS = ("Q0", "Q1", "Q2")
PHASE_BUDGET_USD = 3.00
RESERVE_USD = 0.03               # set aside per question in flight (above the largest smoke-test question cost)
MAX_TOOL_CALLS = 20
MAX_TOKENS_PER_QUESTION = 150_000
MAX_OUTPUT_TOKENS_PER_TURN = 4000
EFFORT = "medium"
UNLOGGED_SPEND_USD = 0.30        # Phase 6 allowance, as in scripts/phase6_agent.py

SYSTEM = """You answer questions about the inventory graph of one IP backbone network.
Element ids: routers (PoPs) 'pop:P07'; links 'link:P02--P09' (endpoints sorted); customer services 'svc:S004', each \
between two endpoint PoPs with an SLA class gold, silver or bronze; routing paths 'path:P02>P05>P09' (the PoP sequence).
Each PoP pair is routed on its lowest-cost paths (cost = sum of link routing metrics; up to 4 equal-cost paths are \
kept), and a service uses the routing paths between its two endpoint PoPs. Time is in integer steps. Step 0 is the \
normal state; a change log records link outages, metric changes and the reroutes they cause.
Finish by calling submit_answer exactly once: element ids in `items` (for a path, the PoP ids in order from source to \
destination) or a number in `number`. If the answer is an empty set, submit an empty items list."""
NOTE = {"Q0": "You have no access to this network's graph. Answer from the question alone with your best guess.",
        "Q1": f"Below the question are facts retrieved from a text version of the whole graph (the {BM25_K} best BM25 "
              "matches for the question). They may be incomplete. Answer from them.",
        "Q2": f"Use the tools to query the graph before answering. You have at most {MAX_TOOL_CALLS} tool calls."}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen_files():
    return [ROOT / "src/nettwin/kgqa.py", ROOT / "scripts/phase11_kgqa.py", OUT / "questions_val.jsonl", OUT / "questions_test.jsonl"]


def load_env():
    import os
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def phase_spend():
    return sum(json.loads(l)["cost_usd"] for f in OUT.glob("*/log.jsonl") for l in f.read_text().splitlines())


def project_spend():
    res = ROOT / "results"
    logs = list(res.glob("phase6/*/agent_log.jsonl")) + list(res.glob("phase7/*agent_log.jsonl"))
    return UNLOGGED_SPEND_USD + phase_spend() + sum(json.loads(l)["cost_usd"] for f in logs for l in f.read_text().splitlines())


def networks(cfg):
    t = cfg["topologies"]
    return ([("val", n, f"val-{i + 1}") for i, n in enumerate(t["validation_real"])]
            + [("test", n, f"test-{i + 1}") for i, n in enumerate(t["test"])])


def read_questions(split):
    return [json.loads(l) for l in (OUT / f"questions_{split}.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- one question

def ask(client, q, condition, kg, index, prices, log, lock):
    system = SYSTEM + "\n\n" + NOTE[condition]
    text = q["question"]
    if condition == "Q1":
        text += "\n\nRetrieved facts:\n" + "\n".join(index.search(q["question"]))
    messages = [{"role": "user", "content": text}]
    tools = (TOOLS if condition == "Q2" else []) + [SUBMIT]
    usage = {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
    calls, answer, stop, nudges = [], None, None, 0
    t_start = time.time()
    while stop is None:
        over = len(calls) >= MAX_TOOL_CALLS or sum(usage.values()) >= MAX_TOKENS_PER_QUESTION
        resp = client.messages.create(model=MODEL, max_tokens=MAX_OUTPUT_TOKENS_PER_TURN, tools=tools,
                                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                      messages=messages, output_config={"effort": EFFORT}, cache_control={"type": "ephemeral"})
        for k in usage:
            usage[k] += int(getattr(resp.usage, k) or 0)
        if resp.stop_reason == "refusal":
            stop = "refusal"
            break
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            nudges += 1
            if nudges > 2:
                stop = "no_answer"
                break
            messages.append({"role": "user", "content": "Call submit_answer to finish."})
            continue
        results = []
        for b in uses:
            if b.name == "submit_answer":
                answer = dict(b.input)
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": "submitted"})
            elif over or len(calls) >= MAX_TOOL_CALLS or condition != "Q2":
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                "content": "error: no more tool calls; call submit_answer now"})
            else:
                out, err = call_tool(kg, b.name, dict(b.input))
                calls.append({"tool": b.name, "input": dict(b.input), "error": err, "chars": len(out)})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        if answer is not None:
            stop = "submitted"
            break
        if over and nudges > 2:
            stop = "budget"
            break
        if over:
            nudges += 1
        messages.append({"role": "user", "content": results})
    em, f1 = score(q, answer)
    rec = {"qid": q["qid"], "condition": condition, "model": MODEL, "answer": answer, "em": em, "f1": f1, "stop": stop,
           "n_tool_calls": len(calls), "calls": calls, "usage": usage, "cost_usd": round(_price(usage, prices), 6),
           "seconds": round(time.time() - t_start, 2)}
    with lock:
        log.write(json.dumps(rec) + "\n")
        log.flush()
    return rec


def run(args, cfg):
    split = args.split
    conditions = tuple(args.conditions.split(",")) if args.conditions else CONDITIONS
    if split == "test":
        log_text = (ROOT / "docs" / "decision_log.md").read_text()
        frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else None
        problems = [m for m, bad in (
            ("the Phase 11 pre-registration is not in docs/decision_log.md", PREREG not in log_text),
            ("results/phase11_frozen.json is missing", frozen is None),
            ("a frozen file changed", frozen is not None and any(sha256(ROOT / k) != v for k, v in frozen["files"].items())),
            ("results/phase11/test already exists (test is scored once)", (OUT / "test").exists()),
            ("--limit is not allowed on test", bool(args.limit)),
            ("every condition must run on test", conditions != CONDITIONS),
        ) if bad]
        if problems:
            sys.exit("refusing the test run: " + "; ".join(problems))
        out_dir = OUT / "test"
    else:
        out_dir = OUT / f"val_{args.name}"
    questions = read_questions(split)
    regenerated, kgs, _ = make_questions(networks(cfg), cfg)   # all networks: each seed depends on the network's position
    regenerated = [q for q in regenerated if q["split"] == split]
    if [{k: q[k] for k in ("qid", "question", "answer")} for q in regenerated] != \
            [{k: q[k] for k in ("qid", "question", "answer")} for q in questions]:
        sys.exit(f"questions_{split}.jsonl does not match what the KG code generates")
    if args.limit:
        rng = np.random.default_rng(11)
        questions = [questions[i] for i in sorted(rng.choice(len(questions), args.limit, replace=False))]
    indexes = {a: TripleIndex(triples(k)) for a, k in kgs.items()}
    prices = load_agent_config(ROOT)["pricing_usd_per_mtok"][MODEL]
    budget = load_agent_config(ROOT)["project_budget_usd"]
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    log_path = out_dir / "log.jsonl"
    if log_path.exists():
        done = {(r["qid"], r["condition"]) for r in map(json.loads, log_path.read_text().splitlines())}
    jobs = [(q, c) for q in questions for c in conditions if (q["qid"], c) not in done]
    spent = {"phase": phase_spend(), "project": project_spend(), "inflight": 0.0}
    print(f"{len(jobs)} runs; Phase 11 spend so far ${spent['phase']:.3f} of ${PHASE_BUDGET_USD:.2f}; "
          f"project ${spent['project']:.2f} of ${budget}", flush=True)
    load_env()
    client = anthropic.Anthropic(max_retries=6)
    lock, glock = threading.Lock(), threading.Lock()
    stopped = threading.Event()
    t0 = time.time()

    def job(q, c):
        with glock:
            if (spent["phase"] + spent["inflight"] + RESERVE_USD > PHASE_BUDGET_USD
                    or spent["project"] + spent["inflight"] + RESERVE_USD > budget - 1.0):
                stopped.set()
                return None
            spent["inflight"] += RESERVE_USD
        try:
            r = ask(client, q, c, kgs[q["network"]], indexes[q["network"]], prices, logf, lock)
        finally:
            with glock:
                spent["inflight"] -= RESERVE_USD
        with glock:
            spent["phase"] += r["cost_usd"]
            spent["project"] += r["cost_usd"]
        print(f"{q['qid']} {c} {q['template']}: f1 {r['f1']:.2f}, {r['n_tool_calls']} calls, ${r['cost_usd']:.4f}, {r['stop']}", flush=True)
        return r

    with open(log_path, "a") as logf, ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(lambda j: job(*j), jobs))
    print(f"done in {time.time() - t0:.0f}s; Phase 11 spend ${spent['phase']:.3f}" + (" (STOPPED by the spend guard)" if stopped.is_set() else ""), flush=True)
    if stopped.is_set():
        sys.exit(2)


# ---------------------------------------------------------------- scoring

def summarise(run_dir, split):
    qs = {q["qid"]: q for q in read_questions(split)}
    recs = {}
    for r in map(json.loads, (run_dir / "log.jsonl").read_text().splitlines()):
        recs[(r["qid"], r["condition"])] = r
    conds = [c for c in CONDITIONS if any(k[1] == c for k in recs)]
    qids = sorted(q for q in qs if all((q, c) in recs for c in conds))
    groups = {"all": qids}
    groups |= {f"type={t}": [q for q in qids if qs[q]["type"] == t] for t in TYPES}
    groups |= {f"template={t}": [q for q in qids if qs[q]["template"] == t] for ts in TYPES.values() for t in ts}
    groups |= {f"size={s}": [q for q in qids if qs[q]["size"] == s] for s, _, _ in SIZE_BINS}
    groups = {g: v for g, v in groups.items() if v}
    out = {"split": split, "model": MODEL, "n_questions": len(qids), "conditions": conds, "metrics": {}, "comparisons": {},
           "cost": {}, "n_boot": 1000, "seed": 42}
    for g, ids in groups.items():
        W = boot_weights(len(ids), 1000, 42)
        vals = {}
        for c in conds:
            for m in ("em", "f1"):
                x = np.array([recs[(q, c)][m] for q in ids])
                b = W @ x / len(ids)
                vals[(c, m)] = b
                out["metrics"].setdefault(g, {}).setdefault(c, {})[m] = {"value": round(float(b[0]), 4), "ci": ci(b[1:]), "n": len(ids)}
        for a, bb in (("Q2", "Q1"), ("Q2", "Q0"), ("Q1", "Q0")):
            if a in conds and bb in conds:
                for m in ("em", "f1"):
                    out["comparisons"][f"{g}|{m}|{a} vs {bb}"] = verdict(vals[(a, m)] - vals[(bb, m)])
    for c in conds:
        rs = [recs[(q, c)] for q in qids]
        out["cost"][c] = {"total_usd": round(sum(r["cost_usd"] for r in rs), 4), "mean_usd": round(float(np.mean([r["cost_usd"] for r in rs])), 6),
                          "mean_tool_calls": round(float(np.mean([r["n_tool_calls"] for r in rs])), 2),
                          "max_tool_calls": int(max(r["n_tool_calls"] for r in rs)),
                          "mean_seconds": round(float(np.mean([r["seconds"] for r in rs])), 2),
                          "not_submitted": sum(r["stop"] != "submitted" for r in rs)}
    out["expectations"] = expectations(out) if set(conds) == set(CONDITIONS) else {}
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def expectations(s):
    """The pre-registered expectations E1 to E7 (decision log, Phase 11 pre-registration)."""
    M, C = s["metrics"], s["comparisons"]
    f1 = lambda g, c: M[g][c]["f1"]["value"]  # noqa: E731
    v = lambda k: C[k]["verdict"]  # noqa: E731
    hard = [t for t in ("type=what_if", "type=path", "type=time") if t in M]
    e = {
        "E1": ("Q2 beats Q1 on overall F1", v("all|f1|Q2 vs Q1") == "supported"),
        "E2": ("Q1 beats Q0 on overall F1", v("all|f1|Q1 vs Q0") == "supported"),
        "E3": ("Q0 overall F1 below 0.15", f1("all", "Q0") < 0.15),
        "E4": ("Q1 F1 at least 0.7 on one-hop questions and below 0.4 on what-if, path and time questions",
               f1("type=one_hop", "Q1") >= 0.7 and all(f1(t, "Q1") < 0.4 for t in hard)),
        "E5": ("Q2 F1 at least 0.8 on one-hop, multi-hop and count questions",
               all(f1(t, "Q2") >= 0.8 for t in ("type=one_hop", "type=multi_hop", "type=count"))),
        "E6": ("Q2 beats Q1 on F1 in every graph-size group", all(v(f"size={x}|f1|Q2 vs Q1") == "supported"
                                                                for x, _, _ in SIZE_BINS if f"size={x}" in M)),
        "E7": ("Q2 mean cost below $0.01 per question", s["cost"]["Q2"]["mean_usd"] < 0.01),
    }
    return {k: {"expectation": d, "met": bool(ok)} for k, (d, ok) in e.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-questions", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--split", choices=["val", "test"])
    ap.add_argument("--name", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--score", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    if args.build_questions:
        if FROZEN.exists():
            sys.exit("refusing: the questions are frozen (results/phase11_frozen.json)")
        qs, _, meta = make_questions(networks(cfg), cfg)
        OUT.mkdir(parents=True, exist_ok=True)
        for split in ("val", "test"):
            (OUT / f"questions_{split}.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs if q["split"] == split))
        (OUT / "networks.json").write_text(json.dumps(meta, indent=1) + "\n")
        print({s: sum(q["split"] == s for q in qs) for s in ("val", "test")})
    elif args.freeze:
        if (OUT / "test").exists():
            sys.exit("refusing: test has already run")
        FROZEN.write_text(json.dumps({"frozen": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "files": {str(p.relative_to(ROOT)): sha256(p) for p in frozen_files()}}, indent=1) + "\n")
        print(FROZEN.read_text())
    elif args.score:
        d = Path(args.score)
        s = summarise(d, "test" if d.name == "test" else "val")
        print(json.dumps({c: s["metrics"]["all"][c] for c in s["conditions"]}, indent=1), json.dumps(s["cost"], indent=1))
    elif args.split:
        run(args, cfg)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
