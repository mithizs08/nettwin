"""Select extra test and validation topologies by the rule pre-registered in docs/decision_log.md (audit C1, M4).

Writes results/topology_selection.json: picks, family of every candidate, and the reference pool.
Prints only names and sizes. Run once; the picks are then copied into config.yaml.
"""
import importlib.resources
import json
import re
from pathlib import Path

import numpy as np
import topohub
import yaml

from nettwin.topology import load

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261029
N_TEST, N_VAL = 5, 5
EXTRA_EXCLUDED_FAMILIES = {"dfn"}  # German research network, near-duplicate of germany50


def stripped(name):
    return re.sub(r"\d+$", "", name).lower()


def first_token(name):
    m = re.match(r"[A-Z]?[a-z0-9]+", name)
    return (m.group(0) if m else name).lower()


def families(names):
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in names:
        for b in names:
            if a < b:
                sa, sb = stripped(a), stripped(b)
                k = 0
                while k < min(len(sa), len(sb)) and sa[k] == sb[k]:
                    k += 1
                if k >= 5 or (len(first_token(a)) >= 2 and first_token(a) == first_token(b)):
                    parent[find(a)] = find(b)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    # family label: shortest stripped name in the group
    return {n: min((stripped(m) for m in g), key=lambda s: (len(s), s)) for g in groups.values() for n in g}


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    data = importlib.resources.files(topohub) / "data"
    zoo = sorted(p.name[:-5] for p in (data / "topozoo").iterdir() if p.name.endswith(".json"))
    sndlib = sorted(p.name[:-5] for p in (data / "sndlib").iterdir() if p.name.endswith(".json"))
    fam = families(zoo)
    dev, test_fixed = ["topozoo/Aarnet"], ["topozoo/Geant2012", "sndlib/germany50"]
    excluded = {fam[n.split("/")[1]] for n in dev + test_fixed if n.startswith("topozoo/")} | EXTRA_EXCLUDED_FAMILIES
    skip = set(cfg["topologies"]["skip"])

    eligible = {}
    for name in zoo:
        key = f"topozoo/{name}"
        if fam[name] in excluded or key in skip:
            continue
        try:
            g = load(key)
        except Exception:  # noqa: BLE001  rule step 4: must load cleanly
            continue
        if 15 <= g.number_of_nodes() <= 60:
            eligible.setdefault(fam[name], []).append((name, g.number_of_nodes(), g.number_of_edges()))
    reps = sorted(max(members) for members in eligible.values())
    order = np.random.default_rng(SEED).permutation(len(reps))
    ranked = [reps[i] for i in order]
    test, val, ref = ranked[:N_TEST], ranked[N_TEST:N_TEST + N_VAL], ranked[N_TEST + N_VAL:]
    picked_families = {fam[n] for n, _, _ in test + val}
    sndlib_skip = sorted(f"sndlib/{s}" for s in sndlib if s.split("-")[0].lower() in picked_families)

    out = {
        "rule": "pre-registered in docs/decision_log.md, 2026-10-29",
        "seed": SEED,
        "excluded_families": sorted(excluded),
        "n_representatives": len(reps),
        "test_added": [{"name": f"topozoo/{n}", "nodes": a, "links": b, "family": fam[n]} for n, a, b in test],
        "validation_real": [{"name": f"topozoo/{n}", "nodes": a, "links": b, "family": fam[n]} for n, a, b in val],
        "reference_pool": [f"topozoo/{n}" for n, _, _ in ref],
        "sndlib_near_duplicates_of_picks": sndlib_skip,
        "family_of": {f"topozoo/{n}": fam[n] for n in zoo},
    }
    (ROOT / "results" / "topology_selection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"representatives: {len(reps)}  reference pool: {len(ref)}")
    for split in ("test_added", "validation_real"):
        print(split, [(d["name"], d["nodes"], d["links"]) for d in out[split]])
    print("sndlib skips from picks:", sndlib_skip)


if __name__ == "__main__":
    main()
