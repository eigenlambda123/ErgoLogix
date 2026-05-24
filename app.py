import streamlit as st
from typing import Optional, Dict
import subprocess
import json
import re


def init_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_tool' not in st.session_state:
        st.session_state.last_tool = 'none'
    if 'pain_area' not in st.session_state:
        st.session_state.pain_area = None
    if 'extracted_params' not in st.session_state:
        st.session_state.extracted_params = {}


def keyword_intent_extractor(text: str) -> Dict[str, Optional[str]]:
    text_l = text.lower()
    pain_keywords = {
        'neck': ['neck', 'cervical', 'nape'],
        'wrist': ['wrist', 'carpal', 'ulnar'],
        'lower_back': ['lower back', 'lumbar', 'back pain', 'lumbar'],
        'shoulder': ['shoulder', 'deltoid'],
        'elbow': ['elbow', 'epicondylitis']
    }
    for area, kws in pain_keywords.items():
        for kw in kws:
            if kw in text_l:
                return {'pain_area': area, 'matched_keyword': kw}
    # fallback: look for posture or environment hints
    if 'hot' in text_l or 'temperature' in text_l or 'humid' in text_l:
        return {'pain_area': 'environment', 'matched_keyword': None}
    return {'pain_area': None, 'matched_keyword': None}


def ollama_intent_extractor(text: str, model: str = 'ergo-intent') -> Optional[Dict[str, Optional[str]]]:
    """Try to extract intent using a local Ollama model via CLI. Returns dict or None on failure.

    This function attempts several common `ollama` CLI invocation forms. If the CLI is missing
    or the command fails, it returns None so the caller can fall back to the keyword extractor.
    """
    prompt = (
        "You are a lightweight intent extractor. Given the user message, return a JSON object"
        " with keys: pain_area (one of 'neck','wrist','lower_back','shoulder','elbow','environment',null)"
        " and matched_keyword (string or null). Only output valid JSON. Message: \"" + text + "\""
    )

    cmds = [
        ['ollama', 'query', model, prompt],
        ['ollama', 'run', model, prompt],
        ['ollama', 'generate', model, prompt],
    ]

    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        except FileNotFoundError:
            # ollama CLI not installed
            return None
        except Exception:
            continue

        if proc.returncode != 0:
            continue

        out = proc.stdout.strip()
        if not out:
            continue

        # Try to find JSON in the output
        m = re.search(r'(\{\s*"pain_area"[\s\S]*\})', out)
        candidate = None
        if m:
            candidate = m.group(1)
        else:
            candidate = out

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and 'pain_area' in parsed:
                return {'pain_area': parsed.get('pain_area'), 'matched_keyword': parsed.get('matched_keyword')}
        except Exception:
            # not valid JSON — skip to next
            continue

    return None


def route_tool_from_intent(intent: Dict[str, Optional[str]]) -> str:
    area = intent.get('pain_area')
    mapping = {
        'neck': 'process_posture_neck_metrics',
        'wrist': 'process_wrist_assessment',
        'lower_back': 'process_lumbar_metrics',
        'shoulder': 'process_shoulder_assessment',
        'elbow': 'process_elbow_assessment',
        'environment': 'process_environmental_metabolic_metrics',
        None: 'fallback_intent_handler'
    }
    return mapping.get(area, 'fallback_intent_handler')


def process_message(msg: str):
    intent = None
    use_ollama = st.session_state.get('use_ollama', True)
    if use_ollama:
        intent = ollama_intent_extractor(msg)
    if intent is None:
        intent = keyword_intent_extractor(msg)
    tool = route_tool_from_intent(intent)
    st.session_state.messages.append({'from': 'user', 'text': msg})
    st.session_state.messages.append({'from': 'system', 'text': f"Routed to: {tool}"})
    st.session_state.last_tool = tool
    st.session_state.pain_area = intent.get('pain_area')
    st.session_state.extracted_params.update(intent)


def main():
    init_state()
    st.set_page_config(page_title='ErgoLogix — Conversational Router', layout='wide')
    st.title('ErgoLogix — Conversational Intent Router')

    # Ollama toggle in sidebar
    if 'use_ollama' not in st.session_state:
        st.session_state.use_ollama = True
    with st.sidebar:
        st.header('Settings')
        st.session_state.use_ollama = st.checkbox('Use Ollama for intent extraction', value=st.session_state.use_ollama)
        st.markdown('If Ollama is unavailable the router will fallback to a keyword extractor.')

    col1, col2 = st.columns([3, 1])

    with col1:
        st.text_area('Describe your discomfort, posture, or environment', height=120, key='user_input')

        def _on_send():
            msg = st.session_state.get('user_input', '').strip()
            if not msg:
                return
            try:
                process_message(msg)
                # Clear the input safely inside the callback
                st.session_state.user_input = ''
                if 'last_error' in st.session_state:
                    del st.session_state['last_error']
            except Exception as e:
                st.session_state.last_error = str(e)

        st.button('Send', on_click=_on_send)
        if st.session_state.get('last_error'):
            st.error(f"Error processing message: {st.session_state.last_error}")

        st.markdown('---')
        st.header('Conversation')
        for m in st.session_state.messages[::-1]:
            if m['from'] == 'user':
                st.write(f"**User:** {m['text']}")
            else:
                st.write(f"*{m['text']}*")

    with col2:
        st.header('Session Summary')
        st.write('**Last tool:**', st.session_state.last_tool)
        st.write('**Pain area:**', st.session_state.pain_area)
        st.write('**Extracted params:**')
        st.json(st.session_state.extracted_params)
        st.markdown('---')
        st.info('This first-pass router uses a keyword fallback. Replace or extend `keyword_intent_extractor` to integrate Ollama or other LLMs.')


if __name__ == '__main__':
    main()
