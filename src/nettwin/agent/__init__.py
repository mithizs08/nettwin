"""LLM root-cause agent (Phase 6)."""
from pathlib import Path

import yaml

AGENT_CONFIG = "agent.yaml"


def load_agent_config(root):
    """The `agent` settings from agent.yaml (config.yaml is frozen with the simulator; its agent placeholder is unused)."""
    return yaml.safe_load((Path(root) / AGENT_CONFIG).read_text())["agent"]
