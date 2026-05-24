import types

import streamlit as st

import app


def setup_function():
    try:
        st.session_state.clear()
    except Exception:
        pass
    app.init_state()


def test_keyword_multi_tool_and_runtime_params(monkeypatch):
    setup_function()
    st.session_state['use_ollama'] = False
    st.session_state['use_orchestrator'] = False

    calls = []

    def fake_env(*args, **kwargs):
        calls.append('env')
        return {}

    monkeypatch.setattr(app, 'process_environmental_metabolic_metrics', fake_env)

    app.process_message(
        'My lower back hurts for over 10 hours without a break and it is too hot in here on this couch'
    )

    # Both lumbar and environment should run from one message
    assert st.session_state.get('tool_result', {}).get('tool') == 'process_lumbar_metrics'
    assert 'env' in calls

    # Runtime params should be extracted from text
    assert float(st.session_state.get('session_duration_min', 0.0)) == 600.0
    assert float(st.session_state.get('breaks_taken', 1.0)) == 0.0
    assert st.session_state.get('workspace_mode') == 'sitting'

    queue = st.session_state.get('extracted_params', {}).get('tool_queue', [])
    assert 'process_lumbar_metrics' in queue
    assert 'process_environmental_metabolic_metrics' in queue


def test_orchestrator_tools_list_executes_all(monkeypatch):
    setup_function()
    st.session_state['use_ollama'] = True
    st.session_state['use_orchestrator'] = True

    def fake_llm_orchestrate(msg, model=None):
        return {
            'tools': ['process_posture_neck_metrics', 'process_environmental_metabolic_metrics'],
            'params': {'session_duration_min': 120, 'breaks_taken': 1},
            'assistant_response': 'Running neck and environment checks.',
        }

    monkeypatch.setattr(app, 'orchestrator', types.SimpleNamespace(llm_orchestrate=fake_llm_orchestrate))

    calls = []

    def fake_neck():
        calls.append('neck')
        return {}

    def fake_env(*args, **kwargs):
        calls.append('env')
        return {}

    monkeypatch.setattr(app, 'process_posture_neck_metrics', fake_neck)
    monkeypatch.setattr(app, 'process_environmental_metabolic_metrics', fake_env)

    app.process_message('My neck is stiff and room is hot')

    assert calls == ['neck', 'env']
    assert float(st.session_state.get('session_duration_min', 0.0)) == 120.0
    assert float(st.session_state.get('breaks_taken', 0.0)) == 1.0
    msgs = st.session_state.get('messages', [])
    assert any(m.get('from') == 'assistant' and 'Running neck and environment checks.' in m.get('text', '') for m in msgs)
