import app


def test_process_message_routes_to_environment_tool(monkeypatch):
    app.st.session_state.clear()
    app.init_state()

    monkeypatch.setattr(app, 'ollama_intent_extractor', lambda *args, **kwargs: None)
    monkeypatch.setattr(app, 'keyword_intent_extractor', lambda text: {'pain_area': 'environment', 'matched_keyword': None})

    called = {}

    def fake_environment_tool(force_refresh=True):
        called['called'] = force_refresh
        app.st.session_state.environment_metrics = {
            'temperature_c': 31.0,
            'humidity_percent': 72.0,
            'thermal_fatigue_multiplier': 1.08,
            'calories_burned': 120.0,
        }
        app.st.session_state.thermal_fatigue_multiplier = 1.08
        return app.st.session_state.environment_metrics

    monkeypatch.setattr(app, 'process_environmental_metabolic_metrics', fake_environment_tool)

    app.process_message('It feels hot and humid in here')

    assert called['called'] is True
    assert app.st.session_state.last_tool == 'process_environmental_metabolic_metrics'
    assert app.st.session_state.pain_area == 'environment'
    assert app.st.session_state.thermal_fatigue_multiplier == 1.08