"""The Phase 7 demo renders every incident, variant and page from committed results, with no data folder or API key."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_demo_renders_offline(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    for incident in at.selectbox(key="incident").options:
        for v in ("with_status", "status_blind"):
            at.selectbox(key="incident").set_value(incident)
            at.radio(key="variant").set_value(v)
            at.run()
            assert not at.exception, (incident, v)
    for page in ("Test results", "Capacity forecasting", "Limitations"):
        at.radio(key="page").set_value(page).run()
        assert not at.exception, page
        if page == "Capacity forecasting" and at.radio(key="fc_split") is not None:
            at.radio(key="fc_split").set_value("dev").run()
            assert not at.exception, "forecasting dev"
