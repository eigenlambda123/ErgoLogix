import streamlit as st

import app


def setup_function():
    try:
        st.session_state.clear()
    except Exception:
        pass
    app.init_state()


def test_native_tool_calls_execute_in_order(monkeypatch):
    setup_function()

    st.session_state['use_ollama'] = True
    st.session_state['use_native_tool_calls'] = True
    st.session_state['use_orchestrator'] = False

    def fake_native(msg, model=None, host='http://127.0.0.1:11434'):
        return {
            'tools': ['process_posture_neck_metrics', 'process_environmental_metabolic_metrics'],
            'params': {'session_duration_min': 180, 'breaks_taken': 0},
            'assistant_response': 'Running neck and environment checks.',
        }

    monkeypatch.setattr(app, 'ollama_native_tool_orchestrate', fake_native)

    calls = []

    def fake_neck():
        calls.append('neck')
        return {'tool': 'process_posture_neck_metrics', 'area': 'Neck', 'risk_tier': 'Moderate Risk', 'risk_pct': 55.0}

    def fake_env(*args, **kwargs):
        calls.append('env')
        return {'temperature_c': 31.0, 'humidity_percent': 72.0, 'thermal_fatigue_multiplier': 1.12}

    monkeypatch.setattr(app, 'process_posture_neck_metrics', fake_neck)
    monkeypatch.setattr(app, 'process_environmental_metabolic_metrics', fake_env)

    app.process_message('My neck is stiff and room is hot')

    assert calls == ['neck', 'env']
    assert float(st.session_state.get('session_duration_min', 0.0)) == 180.0
    assert float(st.session_state.get('breaks_taken', 1.0)) == 0.0
    msgs = st.session_state.get('messages', [])
    assert any(m.get('from') == 'assistant' and 'Running neck and environment checks.' in m.get('text', '') for m in msgs)


def test_native_no_tool_skips_fallback(monkeypatch):
    setup_function()

    st.session_state['use_ollama'] = True
    st.session_state['use_native_tool_calls'] = True
    st.session_state['use_orchestrator'] = False

    def fake_native(msg, model=None, host='http://127.0.0.1:11434'):
        return {
            'tools': ['no_tool'],
            'params': {},
            'assistant_response': 'Could you share how many hours you have worked?',
        }

    monkeypatch.setattr(app, 'ollama_native_tool_orchestrate', fake_native)

    called = {'keyword': 0}

    def fake_keyword(_text):
        called['keyword'] += 1
        return {'pain_area': 'environment', 'matched_keyword': None}

    monkeypatch.setattr(app, 'keyword_intent_extractor', fake_keyword)

    app.process_message('hello')

    assert called['keyword'] == 0
    assert st.session_state.get('last_tool') == 'no_tool'
    msgs = st.session_state.get('messages', [])
    assert any(m.get('from') == 'assistant' and 'Could you share how many hours' in m.get('text', '') for m in msgs)


def test_native_mode_works_when_use_ollama_toggle_off(monkeypatch):
    setup_function()

    st.session_state['use_ollama'] = False
    st.session_state['use_native_tool_calls'] = True
    st.session_state['use_orchestrator'] = False

    def fake_native(msg, model=None, host='http://127.0.0.1:11434'):
        return {
            'tools': ['no_tool'],
            'params': {},
            'assistant_response': 'I can help—tell me where you feel discomfort.',
        }

    monkeypatch.setattr(app, 'ollama_native_tool_orchestrate', fake_native)

    called = {'multi': 0}

    def fake_multi(_text):
        called['multi'] += 1
        return {'pain_areas': ['lower_back', 'environment'], 'matched_keywords': ['back', 'hot']}

    monkeypatch.setattr(app, 'keyword_intent_extractor_multi', fake_multi)

    app.process_message('hello')

    assert called['multi'] == 0
    assert st.session_state.get('last_tool') == 'no_tool'
    msgs = st.session_state.get('messages', [])
    assert any(m.get('from') == 'assistant' and 'tell me where you feel discomfort' in m.get('text', '') for m in msgs)
