import streamlit as st
import importlib
import types

import app


def setup_function():
    # Reset Streamlit session state between tests
    try:
        st.session_state.clear()
    except Exception:
        pass
    app.init_state()


def test_appends_assistant_response(monkeypatch):
    setup_function()

    # Mock orchestrator.llm_orchestrate to return a no-op tool with assistant text and params
    def fake_llm_orchestrate(msg, model=None):
        return {'tool': 'no_tool', 'params': {'foo': 'bar'}, 'assistant_response': 'Assistant: acknowledged'}

    fake_orch = types.SimpleNamespace(llm_orchestrate=fake_llm_orchestrate)
    monkeypatch.setattr(app, 'orchestrator', fake_orch)

    st.session_state['use_ollama'] = True
    st.session_state['use_orchestrator'] = True

    app.process_message('Please analyze my environment')

    # Expect assistant message appended and params stored
    msgs = st.session_state.get('messages', [])
    assert any(m.get('from') == 'assistant' and 'acknowledged' in m.get('text', '') for m in msgs)
    assert st.session_state.get('extracted_params', {}).get('foo') == 'bar'
    assert st.session_state.get('last_tool') == 'no_tool'


def test_triggers_environmental_background(monkeypatch):
    setup_function()

    # Fake future object
    class DummyFuture:
        def __init__(self, res):
            self._res = res

        def done(self):
            return True

        def result(self):
            return self._res

    # Orchestrator instructs to run the environmental tool
    def fake_llm_orchestrate(msg, model=None):
        return {'tool': 'process_environmental_metabolic_metrics', 'params': {}, 'assistant_response': 'Refreshing environment...'}

    fake_orch = types.SimpleNamespace(llm_orchestrate=fake_llm_orchestrate)
    monkeypatch.setattr(app, 'orchestrator', fake_orch)

    # Patch analyze_environment_async used by app to return our dummy future
    def fake_analyze_environment_async(**kwargs):
        return DummyFuture({'thermal_fatigue_multiplier': 1.23})

    monkeypatch.setattr(app, 'analyze_environment_async', fake_analyze_environment_async)

    st.session_state['use_ollama'] = True
    st.session_state['use_orchestrator'] = True

    app.process_message('Refresh my environmental data')

    # The background future should be stored in session state
    fut = st.session_state.get('env_future')
    assert fut is not None
    assert hasattr(fut, 'done') and fut.done()
