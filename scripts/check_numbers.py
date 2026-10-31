"""Phase 10 number checker: every number in README.md and docs/report.md must match a results file.

    python scripts/check_numbers.py              # check; exit 1 and list the misses if any number is untraced
    python scripts/check_numbers.py --propose    # (re)write docs/number_sources.json with a proposed source per number

Two rules:
- A number written with a CI, "X [lo, hi]", must match a point and CI stored together in one result, inside the
  results its section reports from (SCOPES).
- Every other number must have an entry in `docs/number_sources.json` naming one results file and one JSON path, and
  the value at that path, rounded to the decimals written, must equal it (also as a percentage for numbers written
  with %, ignoring the sign, since differences are written as "+0.044"). `--propose` picks a candidate path for each
  number from the words around it; the picks were then reviewed by hand. A flat pool of every number in `results/`
  would hold nearly every 3-decimal value, so it could not catch a wrong number; one path per number can.

The script also runs a mutation test: it shifts every checked number by one in its last digit and counts how many
shifted numbers would still pass. That is the checker's power, printed so it is not overstated.

Not checked, by rule: inline code and code blocks (commands, paths), section numbers and model versions (Haiku 5.5),
integers 0 to 20 written as plain counts, citations (years in "[Author 2011]" and the whole References section),
and the constants in ALLOW.
"""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCS = ("README.md", "docs/report.md")
MANIFEST = ROOT / "docs" / "number_sources.json"
# Constants that define the setup (from PLAN.md and the pre-registrations), not results; each is listed with its reason.
ALLOW = {
    "80%": "breach threshold, pre-registered (PLAN section 6)",
    "0.8": "breach threshold, pre-registered (PLAN section 6)",
    "60": "breach horizon in minutes, pre-registered",
    "30": "forecast cadence in minutes (every 6 steps), pre-registered",
    "95%": "confidence level of every CI",
    "1000": "bootstrap resamples (shared harness)",
    "100": "project API budget in USD (pre-test design change)",
    "$100": "project API budget in USD (pre-test design change)",
    "32": "length in bytes of the dataset key",
    "600": "file permissions of the dataset key",
    "0.10": "X5 transfer threshold, pre-registered (Phase 5)",
}
# Sources: (files, regex on the dotted JSON path). Only these paths can be named for a section's numbers.
SOURCES = {
    "p5": ["phase5_test.json", "phase5_heldout_val.json"],
    "p6": ["phase6/test_claude-haiku-5-5/summary.json", "phase6/test_claude-opus-5-5/summary.json", "phase6/test_haiku_vs_opus.json"],
    "p7": ["phase7/aarnet_dev_summary.json"],
    "p8": ["phase8_test.json"],
    "data": ["phase1_kg_summary.json", "phase3_dataset_summary.json"],
}
# Each doc heading (prefix match on its text) names the sources its numbers come from. An unmapped heading is an error.
SCOPES = {
    "NetTwin": ["data"], "Read this first": ["data"], "Run the demo": [], "Setup and reproduction": [],
    "Reproducing the datasets": [], "Where things are": [], "License": [], "Headline results": ["p5", "p6", "p8"],
    "Results in detail": ["p5", "p6", "p8"],
    "Descriptive, not pre-registered": ["p6", "p7"], "Why measure": [], "What was built": ["data"], "RQ1 and RQ2": ["p5"],
    "RQ3": ["p5"], "RQ1 ablation": ["p5"], "RQ4": ["p6"], "RQ5": ["p8"], "Descriptive results": ["p6", "p7", "p8"],
    "Negative results": ["p5", "p6", "p8"], "How the work was kept honest": [], "Reproduce and run": [], "References": [],
}
# An allowed constant that is also a plausible result counts as the constant only in this context.
ALLOW_CONTEXT = {"95%": r"\s+(CIs?\b|bootstrap|scenario|confidence)"}
TOKEN = re.compile(r"(?<![\w.\-/:#])([-+−]?\$?)(\d+(?:,\d{3})*(?:\.\d+)?)(%?)(?![\w/:]|\.\d|-\w)")
TRIPLE = re.compile(r"([-+−]?\d+\.\d+)\s*\[([-+−]?\d+\.\d+),\s*([-+−]?\d+\.\d+)\]")


def leaves(obj, path, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaves(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            leaves(v, f"{path}.{i}" if path else str(i), out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append((path, float(obj)))


def triples(obj, out):
    """(point, lo, hi) for every result that stores a point and its CI together."""
    if isinstance(obj, dict):
        ci = obj.get("ci", obj.get("seed_mean_ci"))
        pt = next((obj[k] for k in ("value", "point", "mean", "rate") if isinstance(obj.get(k), (int, float))), None)
        if pt is not None and isinstance(ci, list) and len(ci) == 2 and None not in ci:
            out.append((pt, *ci))
        for v in obj.values():
            triples(v, out)
    elif isinstance(obj, list):
        for v in obj:
            triples(v, out)


def derived():
    """Totals computed from committed result files (formula next to each), addressed as derived:<name>."""
    out = {}
    logs = sorted(RES.glob("phase6/*/agent_log.jsonl")) + sorted((RES / "phase7").glob("*agent_log.jsonl"))
    costs = {p: sum(json.loads(l)["cost_usd"] for l in p.read_text().splitlines()) for p in logs}
    out["haiku test cost"] = costs[RES / "phase6" / "test_claude-haiku-5-5" / "agent_log.jsonl"]
    out["opus test cost"] = costs[RES / "phase6" / "test_claude-opus-5-5" / "agent_log.jsonl"]
    out["project API spend"] = sum(costs.values()) + 0.30          # all agent logs + $0.30 unlogged allowance
    t = json.loads((RES / "phase8_test.json").read_text())["splits"]["test"]["metrics"]["persistence"]
    out["phase8 fault-free share of breaches"] = t["fault_free"]["n_breaches"] / t["all"]["n_breaches"]
    hv = json.loads((RES / "phase6" / "test_haiku_vs_opus.json").read_text())["variants"]
    verdicts = [c["verdict"] for v in hv.values() for c in v["comparisons"].values()]
    out["haiku vs opus comparisons"] = len(verdicts)
    out["haiku vs opus supported"] = verdicts.count("supported")
    out["haiku vs opus inconclusive"] = verdicts.count("inconclusive")
    h = json.loads((RES / "phase6" / "test_claude-haiku-5-5" / "summary.json").read_text())
    m = h["variants"]["with_status"]["metrics"]["A2"]
    out["phase6 stratum faults"] = h["n_incidents"] - m["faults"]["top1"]["n"] - m["distractors"]["claims_fault"]["n"]
    out["phase6 main-sample change-overlap faults"] = m["change_overlap_faults"]["claims_fault"]["n"] - out["phase6 stratum faults"]
    p5 = json.loads((RES / "phase5_test.json").read_text())["splits"]["test"]["variants"]["with_status"]["methods"]
    for g in ("type=link_degradation", "type=routing_misconfig", "type=link_flapping", "hard|network=York", "hard|network=Ibm"):
        # GNN 3-seed mean of hard top-3 (q90) for one group, as in results/phase5_tables.md
        out[f"phase5 GNN seed mean {g} top3"] = sum(p5[f"gat-h32-s{k}"]["groups"][g]["localisation"]["q90"]["top3"]["value"] for k in range(3)) / 3
    for v in ("with_status", "status_blind"):
        cost = {k: json.loads((RES / "phase6" / f"test_claude-{k}-5-5" / "summary.json").read_text())["variants"][v]
                ["agent_cost_latency_groundedness"]["A2"]["mean_usd"] for k in ("haiku", "opus")}
        out[f"opus/haiku A2 cost ratio {v}"] = cost["opus"] / cost["haiku"]
    return out


_cache = {}


def source_values(rel):
    """{path: value} for one results file (or 'derived')."""
    if rel not in _cache:
        if rel == "derived":
            _cache[rel] = derived()
        else:
            out = []
            leaves(json.loads((RES / rel).read_text()), "", out)
            _cache[rel] = dict(out)
    return _cache[rel]


def strip_code(text):
    """Drop code, section numbers ("### 4.1") and model versions ("Haiku 5.5"): names, not results."""
    text = re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    text = re.sub(r"^(#+ )[\d.]+", r"\1", text, flags=re.M)
    text = re.sub(r"\b(Haiku|Opus|Sonnet) \d\.\d\b", r"\1", text)
    return re.sub(r"`[^`\n]*`", "", text)


def roundings(v, dec):
    """A stored value ending in 5 can be written either way: half up (0.0115 as 1.2%) or as Python formats the binary
    float, which is how the results tables print it (0.0865 as 0.086). Both are the same stored value."""
    return {float(Decimal(repr(abs(v))).quantize(Decimal(1).scaleb(-dec), rounding=ROUND_HALF_UP)), round(abs(v), dec)}


def same(num, pct, v):
    if v is None:
        return False
    dec = len(num.split(".")[1]) if "." in num else 0
    x = abs(float(num.replace(",", "")))
    return x in roundings(v, dec) or (pct and x in roundings(v * 100, dec))


def triple_ok(t, trips):
    """The point is matched ignoring sign; the CI bounds are matched with their sign."""
    dec = [len(s.split(".")[1]) for s in t]
    x, lo, hi = (float(s.replace("−", "-")) for s in t)
    return any(abs(round(p, dec[0])) == abs(x) and round(a, dec[1]) == lo and round(b, dec[2]) == hi for p, a, b in trips)


def doc_numbers():
    """Yield (key, doc, line no, line, heading, scope sources, kind, match) for every checked number in the docs."""
    for rel in DOCS:
        heading, seen, header = None, {}, ""
        for i, line in enumerate(strip_code((ROOT / rel).read_text()).splitlines(), 1):
            if line.startswith("#"):
                heading = line.lstrip("# ").strip()
                key = next((k for k in SCOPES if heading.startswith(k)), None)
                if key is None:
                    sys.exit(f"{rel}:{i}: heading '{heading}' has no scope in SCOPES")
                files = [f for s in SCOPES[key] for f in SOURCES[s]]
                continue
            header = line if line.startswith("|") and not header.startswith("|") else header if line.startswith("|") else ""
            spans = []
            for m in TRIPLE.finditer(line):
                spans.append((m.start(), m.end()))
                yield None, rel, i, line, heading, files, "triple", m
            for m in TOKEN.finditer(line):
                sign, num, pct = m.groups()
                if any(a <= m.start(2) < b for a, b in spans):
                    continue
                tok = f"{sign.lstrip('+-−')}{num}{pct}"
                if (tok in ALLOW or num in ALLOW) and re.match(ALLOW_CONTEXT.get(tok, ""), line[m.end():]):
                    continue
                if "." not in num and not pct and int(num.replace(",", "")) <= 20:
                    continue
                if heading.startswith("References") or (re.fullmatch(r"(19|20)\d\d", num) and line[m.end():].startswith("]")):
                    continue
                k = seen.get((heading, num + pct), 0)
                seen[(heading, num + pct)] = k + 1
                yield f"{rel}|{heading}|{num}{pct}|{k}", rel, i, line, heading, files, "single", (m, header)


SYN = [(r"status-blind|status blind|without status", "status_blind"), (r"with status", "with_status"), (r"top-3", "top3"),
       (r"top-1", "top1"), (r"\bgnn\b|gat-h32", "gat-h32 gnn gnn_head"), (r"mlp", "mlp"), (r"haiku", "haiku"), (r"opus", "opus"),
       (r"distractor|maintenance windows|drain", "distractors"), (r"change", "change_overlap_faults"),
       (r"recall", "recall claims_fault"), (r"precision", "precision"), (r"\bf1\b", "f1"), (r"persistence", "persistence"),
       (r"seasonal", "seasonal_naive"), (r"lightgbm", "lightgbm"), (r"encoder|head", "gnn_head"), (r"pr-auc", "pr_auc_root"),
       (r"maintenance", "maintenance_only false_alarm"), (r"\bhard\b", "hard"), (r"\beasy\b", "easy"), (r"fault", "faults"),
       (r"aarnet|dev\b", "dev"), (r"rule|\+ change-log", "gnn+cl"), (r"breach", "n_breaches"), (r"incidents", "n_incidents"),
       (r"scenarios", "n_scenarios"), (r"type accuracy|fault-type", "type_correct"), (r"ungrounded|cited|lenient", "ungrounded_item_rate"),
       (r"strict", "strict"), (r"per incident|cost", "mean_usd")]


def words(text):
    t = text.lower()
    out = set(re.findall(r"[a-z0-9_+=\-]+", t))
    for rx, w in SYN:
        if re.search(rx, t):
            out.update(w.split())
    return out


def propose():
    """Propose a source per number; entries already in the manifest whose value still matches are kept as reviewed."""
    old = {e["key"]: e for e in json.loads(MANIFEST.read_text())} if MANIFEST.exists() else {}
    entries = []
    for key, rel, i, line, heading, files, kind, m in doc_numbers():
        if kind != "single":
            continue
        m, header = m
        num, pct = m.group(2), m.group(3)
        near = words(line[max(0, m.start() - 70):m.start()])
        ctx = words(line) | words(heading) | words(header)
        if line.startswith("|"):
            cells = line.split("|")
            col = line[:m.start()].count("|")
            heads = header.split("|")
            near |= words(cells[1]) | words(heads[col] if col < len(heads) else "")
        cands = [(f, p, v) for f in files + ["derived"] for p, v in source_values(f).items() if same(num, bool(pct), v)]
        # a number written alone is a point, a rate or a count, not one bound of a CI or a per-seed value
        cands = [c for c in cands if not re.search(r"(ci|seed_mean_ci|per_seed)\.\d$|\.sd$|_pessimistic|_optimistic", c[1])] or cands
        near |= {"q90", "localisation", "value", "point", "mean"} | ({"test"} if "dev" not in near else set())

        def score(c):
            toks = set(re.split(r"[.|:@ ]+", (c[0] + "." + c[1]).lower())) | {t for t in re.split(r"[./_\-]", c[0].lower())}
            return (3 * len(toks & near) + len(toks & ctx) - 2 * ("val" in c[0]) - 0.01 * len(c[1]), c[0] == "derived")
        cands.sort(key=score, reverse=True)
        best = cands[0] if cands else (None, None, None)
        e = {"key": key, "source": best[0], "path": best[1], "line": f"{rel}:{i}",
             "context": line.strip()[max(0, m.start() - 70):m.start() + 20], "reviewed": "NEW, not reviewed"}
        o = old.get(key)
        if o and o.get("source") and same(num, bool(pct), source_values(o["source"]).get(o["path"])):
            e.update(source=o["source"], path=o["path"], reviewed=o.get("reviewed"))
        entries.append(e)
    MANIFEST.write_text(json.dumps(entries, indent=1) + "\n")
    for e in entries:
        if e["reviewed"].startswith("NEW"):
            print(f"{e['line']:22s} {e['key'].split('|')[2]:8s} {e['source']}  {e['path']}   <- {e['context']}")


def main():
    if "--propose" in sys.argv:
        return propose()
    manifest = {e["key"]: e for e in json.loads(MANIFEST.read_text())}
    misses, n, used, mut = [], 0, set(), [0, 0]
    for key, rel, i, line, heading, files, kind, m in doc_numbers():
        n += 1
        if kind == "triple":
            trips = []
            for f in files:
                triples(json.loads((RES / f).read_text()), trips)
            if not triple_ok(m.groups(), trips):
                misses.append(f"{rel}:{i}: {m.group(0)} (no result with this point and CI in {SCOPES_OF(heading)})    | {line.strip()[:90]}")
            continue
        m, _ = m
        num, pct = m.group(2), m.group(3)
        e = manifest.get(key)
        used.add(key)
        if e is None or e["source"] is None or e.get("reviewed", "").startswith("NEW"):
            misses.append(f"{rel}:{i}: {m.group(0)} has no source in {MANIFEST.name}    | {line.strip()[:90]}")
            continue
        if e["source"] != "derived" and e["source"] not in files:
            misses.append(f"{rel}:{i}: {m.group(0)} source {e['source']} is outside the section's scope")
            continue
        v = source_values(e["source"]).get(e["path"])
        if not same(num, bool(pct), v):
            misses.append(f"{rel}:{i}: {m.group(0)} != {v} at {e['source']}:{e['path']}")
            continue
        dec = len(num.split(".")[1]) if "." in num else 0
        step = 10 ** -dec
        for d in (-step, step):
            mut[0] += 1
            mut[1] += not same(f"{abs(float(num.replace(',', ''))) + d:.{dec}f}", bool(pct), v)
    stale = sorted(set(manifest) - used)
    print(f"checked {n} numbers in {', '.join(DOCS)}: CI triples against their section's results, other numbers against "
          f"one named path each ({MANIFEST.relative_to(ROOT)})")
    print(f"mutation test: {mut[1]} of {mut[0]} numbers shifted by one in the last digit ({mut[1] / max(mut[0], 1):.0%}) are caught")
    print("allowed constants (not results): " + "; ".join(f"{k} ({v})" for k, v in ALLOW.items()))
    if stale:
        print(f"{len(stale)} stale manifest entries (no longer in the docs): " + "; ".join(stale))
    if misses:
        print(f"{len(misses)} untraced:")
        print("\n".join(misses))
        sys.exit(1)
    print("all numbers trace to a results file")


def SCOPES_OF(heading):
    return next((", ".join(SCOPES[k]) for k in SCOPES if heading.startswith(k)), "?")


if __name__ == "__main__":
    main()
