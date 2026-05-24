import streamlit as st

import app


def setup_function():
    try:
        st.session_state.clear()
    except Exception:
        pass
    app.init_state()


def test_wrist_handler_updates_state():
    setup_function()
    st.session_state['session_duration_min'] = 180.0
    st.session_state['breaks_taken'] = 0.0
    st.session_state['workspace_mode'] = 'sitting'

    result = app.process_wrist_assessment()

    assert result['tool'] == 'process_wrist_assessment'
    assert isinstance(result['risk_pct'], float)
    assert st.session_state['last_tool'] == 'process_wrist_assessment'
    assert st.session_state['calculated_risk'] == result['risk_pct']
    assert st.session_state['risk_tier'] in {'Low Risk', 'Moderate Risk', 'High Risk'}
    assert 'recommendation' in result


def test_keyword_route_calls_neck_handler(monkeypatch):
    setup_function()
    st.session_state['use_ollama'] = False
    st.session_state['use_orchestrator'] = False

    app.process_message('My neck feels stiff and sore today')

    assert st.session_state['last_tool'] == 'process_posture_neck_metrics'
    assert st.session_state['tool_result'].get('tool') == 'process_posture_neck_metrics'
    assert isinstance(st.session_state['calculated_risk'], float)
