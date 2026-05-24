import streamlit as st
from typing import Optional, Dict
import subprocess
import json
import re
from typing import Any

try:
    import requests
except Exception:
    requests = None


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
    # Safely embed the user's message using JSON encoding to avoid quoting issues
    user_msg = json.dumps(text)
    prompt = (
        "RESPOND WITH STRICT JSON ONLY. Return a JSON object with keys: \"pain_area\" and \"matched_keyword\". "
        "pain_area must be one of: 'neck','wrist','lower_back','shoulder','elbow','environment', or null. "
        "matched_keyword must be a short string or null.\n\n"
        f"User message: {user_msg}\n\nRespond now with JSON ONLY."
    )

    cmds = [
        ['ollama', 'run', model, prompt],
        ['ollama', 'query', model, prompt],
        ['ollama', 'generate', model, prompt],
    ]

    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        except FileNotFoundError:
            # ollama CLI not installed
            # Try HTTP fallback if available
            http_res = ollama_http_intent_extractor(text, model=model)
            return http_res
        except Exception:
            continue

        if proc.returncode != 0:
            continue

        out = proc.stdout.strip()
        if not out:
            continue

        # Try to find JSON in the output (lenient)
        m = re.search(r'(\{[\s\S]*\})', out)
        candidate = m.group(1) if m else out

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and 'pain_area' in parsed:
                return {'pain_area': parsed.get('pain_area'), 'matched_keyword': parsed.get('matched_keyword')}
        except Exception:
            # not valid JSON — skip to next
            continue

    # If CLI didn't yield a result, try HTTP fallback
    return ollama_http_intent_extractor(text, model=model)


def _extract_json_from_text(text: str) -> Optional[dict]:
    m = re.search(r'(\{[\s\S]*\})', text)
    candidate = m.group(1) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        return None


def ollama_http_intent_extractor(text: str, model: str = 'ergo-intent', host: str = 'http://127.0.0.1:11434') -> Optional[Dict[str, Optional[str]]]:
    """Call Ollama HTTP API as a fallback. Returns parsed intent dict or None."""
    if requests is None:
        return None

    # Safely embed the user's message using JSON encoding to avoid quoting issues
    user_msg = json.dumps(text)
    prompt = (
        "RESPOND WITH STRICT JSON ONLY. Return a JSON object with keys: \"pain_area\" and \"matched_keyword\". "
        "pain_area must be one of: 'neck','wrist','lower_back','shoulder','elbow','environment', or null. "
        "matched_keyword must be a short string or null.\n\n"
        f"User message: {user_msg}\n\nRespond now with JSON ONLY."
    )

    url = host.rstrip('/') + '/api/generate'
    try:
        resp = requests.post(url, json={'model': model, 'prompt': prompt}, timeout=6)
    except Exception:
        return None

    if resp.status_code != 200:
        # try to parse JSON body anyway
        try:
            parsed = resp.json()
            text_blob = json.dumps(parsed)
        except Exception:
            return None
    else:
        # prefer raw text if available
        text_blob = resp.text or ''

    # Attempt to extract JSON from the response blob
    parsed = _extract_json_from_text(text_blob)
    if parsed and isinstance(parsed, dict) and 'pain_area' in parsed:
        return {'pain_area': parsed.get('pain_area'), 'matched_keyword': parsed.get('matched_keyword')}
    # As a last resort, try parsing the JSON body
    try:
        parsed_body = resp.json()
        parsed = _extract_json_from_text(json.dumps(parsed_body))
        if parsed and 'pain_area' in parsed:
            return {'pain_area': parsed.get('pain_area'), 'matched_keyword': parsed.get('matched_keyword')}
    except Exception:
        pass

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
        model = st.session_state.get('selected_ollama_model', 'llama3.2:1b')
        intent = ollama_intent_extractor(msg, model=model)
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

    # Detect available Ollama models (if `ollama` CLI is present)
    def get_ollama_models() -> list:
        try:
            proc = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=4)
        except Exception:
            return []
        if proc.returncode != 0:
            return []
        out = proc.stdout.strip().splitlines()
        names = []
        for line in out:
            parts = line.split()
            if parts:
                names.append(parts[0])
        return names

    available_models = get_ollama_models()

    # Ollama settings in sidebar
    if 'use_ollama' not in st.session_state:
        st.session_state.use_ollama = True
    if 'selected_ollama_model' not in st.session_state:
        default_model = 'llama3.2:1b' if 'llama3.2:1b' in available_models else (available_models[0] if available_models else 'ergo-intent')
        st.session_state.selected_ollama_model = default_model

    with st.sidebar:
        st.header('Settings')
        st.session_state.use_ollama = st.checkbox('Use Ollama for intent extraction', value=st.session_state.use_ollama)
        if available_models:
            st.session_state.selected_ollama_model = st.selectbox('Ollama model', available_models, index=available_models.index(st.session_state.selected_ollama_model))
        else:
            st.info('Ollama CLI not found or no local models. Keyword fallback will be used.')
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

    # Neural Diagnostics Dashboard (semantic search + comfort map)
    from semantic import build_kb_from_dir, top_match_and_coords

    # Try to load a markdown KB from `kb/` and use an on-disk cache; fall back to small built-in KB
    try:
        kb = build_kb_from_dir('kb', cache_path='data/kb_cache.json')
    except Exception:
        kb = []
    if not kb:
        # minimal in-memory KB fallback
        KB_DOCS = [
            {'id': 'wrist', 'title': 'Wrist Setup', 'content': 'Advice for wrist pain and ulnar soreness when typing.'},
            {'id': 'neck', 'title': 'Neck Relief', 'content': 'Cervical stretch and neck pain relief guidance.'},
            {'id': 'lumbar', 'title': 'Lower Back', 'content': 'Lumbar support and lower back strain exercises.'},
        ]
        from semantic import build_kb
        kb = build_kb(KB_DOCS)

    import plotly.express as px
    from semantic import compute_comfort_coords

    with st.expander('Neural Diagnostics Dashboard'):
        q = st.text_input('Query for neural diagnostics (or paste conversation text)')
        # Prepare KB scatter data
        nodes = []
        for d in kb:
            x, y = compute_comfort_coords(d.get('content', ''))
            nodes.append({'id': d.get('id'), 'title': d.get('title'), 'x': x, 'y': y, 'content': d.get('content')})

        # Run diagnostics when user clicks; compute query point and top match
        top_match = None
        query_coords = None
        if q:
            out = top_match_and_coords(q, kb)
            query_coords = out['coords']
            top_match = out['top']

        # Build plotly figure
        if nodes:
            df = nodes
            fig = px.scatter(df, x='x', y='y', hover_name='title', hover_data=['id'], text='title')
            # highlight top match if available
            if top_match:
                # add top match marker
                tx, ty = compute_comfort_coords(top_match.get('content', ''))
                fig.add_scatter(x=[tx], y=[ty], mode='markers+text', marker=dict(size=14, color='red'), name='Top match')
            # add query point
            if query_coords:
                qx, qy = query_coords
                fig.add_scatter(x=[qx], y=[qy], mode='markers', marker=dict(size=12, color='green'), name='Query')

            fig.update_layout(title='Comfort Map (posture_balance x tension_level)', xaxis_title='Posture (left=-1 right=1)', yaxis_title='Tension (0-1)')
            st.plotly_chart(fig, use_container_width=True)

        # Node selector (click events require extra package; use selector as fallback)
        titles = [n['title'] for n in nodes]
        default_idx = 0
        if top_match:
            try:
                default_idx = titles.index(top_match.get('title'))
            except Exception:
                default_idx = 0

        sel = st.selectbox('Select a KB node to view', ['(none)'] + titles, index=(default_idx + 1 if titles else 0))
        if sel and sel != '(none)':
            sel_idx = titles.index(sel)
            sel_doc = nodes[sel_idx]
            st.subheader(sel_doc['title'])
            st.write(sel_doc['content'])
        elif top_match and sel == '(none)':
            # show top match by default when available
            st.subheader(top_match.get('title'))
            st.write(top_match.get('content'))


if __name__ == '__main__':
    main()
