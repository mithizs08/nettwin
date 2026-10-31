"""Phase 6 agent freeze and test-run guard (Phase 6 audit S1).

`agent_hashes` hashes everything the agent's behaviour and scoring depend on: the agent package, the runbooks, the
prompts and tool schemas, the shared code the tools and scoring call, the eval script, and agent.yaml (the agent
settings). `scripts/phase6_freeze.py` writes them to `results/phase6_frozen.json`; every run records the combined
`code_sha256`, and a test run refuses on any mismatch.

`test_run_refusal` holds the test rules: one run per model, ever (a fixed folder `results/phase6/test_<model>` with a
LOCK file written when the run starts; resuming an unfinished run is allowed only with identical hashes), no `--limit`,
no dev model, and the full pre-registered conditions and variants.
"""
import hashlib
import json
from pathlib import Path

FROZEN_GLOBS = (
    "agent.yaml",
    "src/nettwin/agent/*.py",
    "src/nettwin/agent/runbooks/*.md",
    "src/nettwin/gnn/explain.py",
    "src/nettwin/gnn/method.py",
    "src/nettwin/gnn/models.py",
    "src/nettwin/gnn/data.py",
    "src/nettwin/alerts.py",
    "src/nettwin/baselines.py",
    "src/nettwin/evaluation.py",
    "src/nettwin/observed.py",
    "src/nettwin/splits.py",
    "scripts/phase6_agent.py",
)
VARIANTS = ("with_status", "status_blind")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def agent_hashes(root, acfg):
    from nettwin.agent import loop, tools
    root = Path(root)
    files = {}
    for g in FROZEN_GLOBS:
        for p in sorted(root.glob(g)):
            files[p.relative_to(root).as_posix()] = _sha(p.read_bytes())
    prompts = json.dumps({"system": loop.SYSTEM, "force": loop.FORCE_MSG, "a0": loop.A0_NOTE, "tools_note": loop.TOOLS_NOTE,
                          "tools": {c: tools.tools_for(c) for c in tools.CONDITIONS}}, sort_keys=True)
    out = {"files": files, "prompt_and_tool_schemas_sha256": _sha(prompts.encode()),
           "agent_config_sha256": _sha(json.dumps(acfg, sort_keys=True).encode())}
    out["code_sha256"] = _sha(json.dumps(out, sort_keys=True).encode())
    return out


def test_out_dir(root, model):
    return Path(root) / "results" / "phase6" / f"test_{model}"


def test_run_refusal(root, frozen, current, model, model_key, limit, conditions, variants, required_conditions, resume):
    """None if a test run may start (or resume); otherwise the reason it is refused."""
    if frozen is None:
        return "results/phase6_frozen.json is missing"
    if frozen["code_sha256"] != current["code_sha256"]:
        diff = sorted(set(frozen["files"].items()) ^ set(current["files"].items()))
        names = sorted({k for k, _ in diff}) + [k for k in ("prompt_and_tool_schemas_sha256", "agent_config_sha256") if frozen[k] != current[k]]
        return f"agent code differs from results/phase6_frozen.json: {names}"
    if limit:
        return "--limit is not allowed on test"
    if model_key == "dev":
        return "the dev model is not allowed on test"
    if sorted(conditions) != sorted(required_conditions):
        return f"this model runs exactly {list(required_conditions)} on test"
    if sorted(variants) != sorted(VARIANTS):
        return f"test runs both variants {list(VARIANTS)}"
    for lock in (Path(root) / "results" / "phase6").glob("*/LOCK.json"):
        info = json.loads(lock.read_text())
        if info["model"] != model:
            continue
        if lock.parent != test_out_dir(root, model):
            return f"a test run for {model} already exists in {lock.parent.name}"
        if not resume:
            return f"the test run for {model} has already started ({lock.parent.name}); it is run once"
        if (lock.parent / "summary.json").exists():
            return f"the test run for {model} is finished; it is run once"
        if info["code_sha256"] != current["code_sha256"]:
            return "resuming needs the same code hash as the run that started"
    out = test_out_dir(root, model)
    if out.exists() and not (out / "LOCK.json").exists():
        return f"{out.name} exists without a LOCK file"
    return None
