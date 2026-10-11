"""Headless smoke tests of the Streamlit dashboard with Streamlit's AppTest."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
VIEWS = ["Fleet overview", "Engine detail", "Simulate landing", "Parts plan", "Impact"]


def button(at, text, where=None):
    for b in (where or at).button:
        if text in b.label:
            return b
    raise AssertionError(f"No button containing {text!r}")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def test_every_view_renders_with_banner(app):
    for view in VIEWS:
        app.sidebar.radio(key="view").set_value(view).run()
        assert not app.exception, f"{view}: {app.exception}"
        assert app.header[0].value == view
        assert any("Decision support only" in i.value for i in app.info)


def test_landing_approval_and_fleet_plan(app):
    button(app, "Next landing in the fleet", app.sidebar).click().run()
    assert not app.exception and app.session_state.recent_events

    app.sidebar.radio(key="view").set_value("Parts plan").run()
    button(app, "Approve recommendation").click().run()
    assert not app.exception and len(app.session_state.decisions) == 1

    button(app, "Approve the fleet plan").click().run()
    assert not app.exception and len(app.session_state.decisions) > 1
    inv = app.session_state.inventory
    assert (inv["qty_reserved"] <= inv["qty_on_hand"]).all()
