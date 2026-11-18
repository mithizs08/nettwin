"""Phase 13, descriptive and not pre-registered: reads the committed test run only (no API call, no rescoring of the
pre-registered metrics). Splits the agents' fault claims by whether the change-scope tool listed an unexplained
symptom, and gives the tool's flag rate per fault type. Writes results/phase13/test/descriptive.json."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAULT_TYPES = ("link_down", "node_down", "link_flapping", "link_degradation", "demand_surge", "routing_misconfig")


def main(run="test"):
    d = ROOT / "results" / "phase13" / run
    inc = json.loads((d / "incidents.json").read_text())
    rec = collections.defaultdict(dict)
    for l in (d / "records.jsonl").read_text().splitlines():
        x = json.loads(l)
        rec[x["incident_id"]][x["method"]] = x
    out = {"run": f"results/phase13/{run}", "note": "descriptive, not pre-registered; counts over incidents, no CIs"}
    for stratum in ("change_overlap", "distractor"):
        rows = [i for i in inc if i["stratum"] == stratum]
        part = {}
        for flag in (True, False):
            sub = [i for i in rows if rec[i["incident_id"]]["SCOPE"]["claims_fault"] == flag]
            part["tool_flagged" if flag else "tool_flagged_nothing"] = {
                "n": len(sub), **{f"{c}_fault_claims": sum(bool(rec[i["incident_id"]][c]["claims_fault"]) for i in sub) for c in ("A2", "A3")},
                "A3_claim_A2_not": sum(rec[i["incident_id"]]["A3"]["claims_fault"] and not rec[i["incident_id"]]["A2"]["claims_fault"] for i in sub),
                "A2_claim_A3_not": sum(rec[i["incident_id"]]["A2"]["claims_fault"] and not rec[i["incident_id"]]["A3"]["claims_fault"] for i in sub)}
        if stratum == "change_overlap":
            part["by_type"] = {t: {"n": sum(i["type"] == t for i in rows),
                                   "tool_flagged": sum(i["type"] == t and rec[i["incident_id"]]["SCOPE"]["claims_fault"] for i in rows)}
                               for t in FAULT_TYPES}
            part["A3_said_planned_maintenance"] = sum(rec[i["incident_id"]]["A3"]["fault_type"] == "planned_maintenance" for i in rows)
            part["A2_said_planned_maintenance"] = sum(rec[i["incident_id"]]["A2"]["fault_type"] == "planned_maintenance" for i in rows)
        out[stratum] = part
    (d / "descriptive.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
