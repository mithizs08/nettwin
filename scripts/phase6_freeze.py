"""Freeze the Phase 6 agent: write the sha256 of its code, runbooks, prompts, tool schemas and agent.yaml to
results/phase6_frozen.json (Phase 6 audit S1). Run on a clean working tree, after committing the agent code.

    python scripts/phase6_freeze.py
"""
import json
import subprocess
import sys
from pathlib import Path

from nettwin.agent import load_agent_config
from nettwin.agent.freeze import agent_hashes

ROOT = Path(__file__).resolve().parents[1]


def main():
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "agent.yaml", "config.yaml"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"refusing: uncommitted changes in src, scripts, agent.yaml or config.yaml:\n{dirty}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    acfg = load_agent_config(ROOT)
    out = {"frozen_at_commit": commit, **agent_hashes(ROOT, acfg)}
    (ROOT / "results" / "phase6_frozen.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote results/phase6_frozen.json ({len(out['files'])} files, code_sha256 {out['code_sha256'][:12]})")


if __name__ == "__main__":
    main()
