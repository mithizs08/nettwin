"""Haiku 5.5 against Opus 5.5 on the Phase 6 test runs (A0 and A2), paired over the same incidents.

Not pre-registered as a hypothesis (the pre-test design change reports Opus as a secondary result with its own A2 - A0
verdicts); this is a descriptive comparison with the same bootstrap (1000 resamples, seed 42) and verdict rule. It only
reads the committed test records; nothing is rerun.

    python scripts/phase6_test_compare.py
"""
import json
from pathlib import Path

from nettwin.agent.eval import summarise

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"Haiku": "test_claude-haiku-5-5", "Opus": "test_claude-opus-5-5"}
VARIANTS = ("with_status", "status_blind")
PAIRS = [("Opus-A2", "Haiku-A2"), ("Opus-A0", "Haiku-A0")]
GROUPS = ("faults", "hard", "easy", "distractors", "change_overlap_faults")


def main():
    d = ROOT / "results" / "phase6"
    incidents = json.loads((d / RUNS["Haiku"] / "incidents.json").read_text())
    assert incidents == json.loads((d / RUNS["Opus"] / "incidents.json").read_text()), "the two test runs differ in incidents"
    ids = [i["incident_id"] for i in incidents]
    out = {"note": "descriptive, not pre-registered; paired over the same test incidents", "variants": {}}
    for v in VARIANTS:
        by_method = {}
        for model, run in RUNS.items():
            recs = [json.loads(l) for l in (d / run / "records.jsonl").read_text().splitlines()]
            for c in ("A0", "A2"):
                got = {r["incident_id"]: r for r in recs if r["variant"] == v and r["method"] == c}
                by_method[f"{model}-{c}"] = [got[i] for i in ids]
        table, comps = summarise(incidents, by_method, pairs=PAIRS)
        out["variants"][v] = {"metrics": {m: {g: t[g] for g in GROUPS if g in t} for m, t in table.items()},
                              "comparisons": {k: c for k, c in comps.items() if k.split("|")[0] in GROUPS and "|answered|" not in k}}
    (ROOT / "results" / "phase6" / "test_haiku_vs_opus.json").write_text(json.dumps(out, indent=1) + "\n")
    for v, x in out["variants"].items():
        print(v)
        for k, c in x["comparisons"].items():
            print(f"  {k:45} {c['verdict']:13} {c['point']:+.3f} {c['ci']}")


if __name__ == "__main__":
    main()
