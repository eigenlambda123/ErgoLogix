import streamlit as st
import importlib
import app


def setup_function():
    # Ensure a fresh session state for each test
    try:
        st.session_state.clear()
    except Exception:
        pass
    app.init_state()


def test_handle_browser_location_result_success(monkeypatch):
    setup_function()
    loc = {'lat': 12.34, 'lon': 56.78}

    called = {'refreshed': False}

    def fake_refresh(force_refresh=True):
        called['refreshed'] = True

    monkeypatch.setattr(app, 'process_environmental_metabolic_metrics', fake_refresh)

    res = app.handle_browser_location_result(loc)

    assert res is True
    assert float(st.session_state.latitude) == 12.34
    assert float(st.session_state.longitude) == 56.78
    assert st.session_state.location_source == 'browser'
    assert called['refreshed'] is True


def test_handle_browser_location_result_fallback(monkeypatch):
    setup_function()
    called = {'fallback': False}

    def fake_fallback(force=True):
        called['fallback'] = True

    monkeypatch.setattr(app, 'refresh_user_location', fake_fallback)

    res = app.handle_browser_location_result(None)

    assert res is False
    assert called['fallback'] is True
