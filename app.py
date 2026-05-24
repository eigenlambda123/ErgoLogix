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

from environmental import analyze_environment, compute_thermal_fatigue_multiplier, fetch_user_location, met_for_workspace_mode, normalize_workspace_mode


def init_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_tool' not in st.session_state:
        st.session_state.last_tool = 'none'
    if 'pain_area' not in st.session_state:
        st.session_state.pain_area = None
    if 'extracted_params' not in st.session_state:
        st.session_state.extracted_params = {}
    if 'latitude' not in st.session_state:
        st.session_state.latitude = 37.7749
    if 'longitude' not in st.session_state:
        st.session_state.longitude = -122.4194
    if 'workspace_mode' not in st.session_state:
        st.session_state.workspace_mode = 'sitting'
    if 'body_weight_kg' not in st.session_state:
        st.session_state.body_weight_kg = 72.0
    if 'session_duration_min' not in st.session_state:
        st.session_state.session_duration_min = 60.0
    if 'environment_metrics' not in st.session_state:
        st.session_state.environment_metrics = {}
    if 'thermal_fatigue_multiplier' not in st.session_state:
        st.session_state.thermal_fatigue_multiplier = 1.0
    if 'auto_detect_location' not in st.session_state:
        st.session_state.auto_detect_location = True
    if 'location_auto_detected' not in st.session_state:
        st.session_state.location_auto_detected = False
    if 'location_source' not in st.session_state:
        st.session_state.location_source = 'manual'
    if 'location_label' not in st.session_state:
        st.session_state.location_label = ''


def refresh_user_location(force: bool = False):
    if not force and not st.session_state.get('auto_detect_location', True):
        return None

    location = fetch_user_location()
    if not location:
        st.session_state.location_auto_detected = False
        st.session_state.location_source = 'manual'
        return None

    st.session_state.latitude = float(location['latitude'])
    st.session_state.longitude = float(location['longitude'])
    st.session_state.location_auto_detected = True
    st.session_state.location_source = location.get('source', 'ipapi')
    city = location.get('city', '')
    region = location.get('region', '')
    country = location.get('country', '')
    parts = [part for part in [city, region, country] if part]
    st.session_state.location_label = ', '.join(parts) if parts else 'Auto-detected location'
    return location


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
    if tool == 'process_environmental_metabolic_metrics':
        process_environmental_metabolic_metrics()


def process_environmental_metabolic_metrics(force_refresh: bool = True):
    latitude = float(st.session_state.get('latitude', 37.7749))
    longitude = float(st.session_state.get('longitude', -122.4194))
    workspace_mode = normalize_workspace_mode(st.session_state.get('workspace_mode', 'sitting'))
    body_weight_kg = float(st.session_state.get('body_weight_kg', 72.0))
    session_duration_min = float(st.session_state.get('session_duration_min', 60.0))

    try:
        if force_refresh or not st.session_state.get('environment_metrics'):
            metrics = analyze_environment(
                latitude=latitude,
                longitude=longitude,
                workspace_mode=workspace_mode,
                weight_kg=body_weight_kg,
                duration_minutes=session_duration_min,
            )
            st.session_state.environment_metrics = metrics
        else:
            metrics = st.session_state.environment_metrics

        st.session_state.thermal_fatigue_multiplier = float(metrics.get('thermal_fatigue_multiplier', 1.0))
        st.session_state.environment_error = None
        st.session_state.extracted_params.update({
            'latitude': latitude,
            'longitude': longitude,
            'workspace_mode': workspace_mode,
            'body_weight_kg': body_weight_kg,
            'session_duration_min': session_duration_min,
        })
        return metrics
    except Exception as e:
        fallback = {
            'latitude': latitude,
            'longitude': longitude,
            'temperature_c': 30.0,
            'humidity_percent': 50.0,
            'cloud_cover_percent': 0.0,
            'rain_probability_percent': 0.0,
            'wind_speed_kph': 0.0,
            'apparent_temperature_c': 30.0,
            'workspace_mode': workspace_mode,
            'met': round(met_for_workspace_mode(workspace_mode), 2),
            'calories_burned': round((met_for_workspace_mode(workspace_mode) * 3.5 * body_weight_kg * session_duration_min) / 200.0, 2),
            'thermal_fatigue_multiplier': compute_thermal_fatigue_multiplier(30.0, 50.0),
        }
        st.session_state.environment_metrics = fallback
        st.session_state.thermal_fatigue_multiplier = fallback['thermal_fatigue_multiplier']
        st.session_state.environment_error = str(e)
        return fallback


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
        st.markdown('---')
        st.subheader('Environmental inputs')
        st.session_state.auto_detect_location = st.checkbox(
            'Auto-detect my location',
            value=st.session_state.auto_detect_location,
            help='Uses your public IP to estimate latitude and longitude automatically.',
        )
        if st.button('Use my current location'):
            refresh_user_location(force=True)
        if st.session_state.auto_detect_location and not st.session_state.location_auto_detected:
            refresh_user_location(force=True)
        st.session_state.latitude = st.number_input('Latitude', value=float(st.session_state.latitude), format='%.4f')
        st.session_state.longitude = st.number_input('Longitude', value=float(st.session_state.longitude), format='%.4f')
        st.session_state.workspace_mode = st.selectbox('Workspace mode', ['sitting', 'standing', 'walking_pad'], index=['sitting', 'standing', 'walking_pad'].index(normalize_workspace_mode(st.session_state.workspace_mode)))
        st.session_state.body_weight_kg = st.number_input('Weight (kg)', min_value=20.0, max_value=250.0, value=float(st.session_state.body_weight_kg), step=0.5)
        st.session_state.session_duration_min = st.number_input('Session duration (min)', min_value=1.0, max_value=1440.0, value=float(st.session_state.session_duration_min), step=5.0)
        if st.session_state.location_label:
            st.caption(f"Location: {st.session_state.location_label} ({st.session_state.location_source})")
        if st.button('Refresh environmental data'):
            process_environmental_metabolic_metrics(force_refresh=True)
        # Rebuild KB cache on demand (calls semantic.build_kb_from_dir)
        if st.button('Rebuild KB cache'):
            try:
                from semantic import build_kb_from_dir
                kb_new = build_kb_from_dir('kb', cache_path='data/kb_cache.json')
                st.success(f'Rebuilt KB cache ({len(kb_new)} docs)')
            except Exception as e:
                st.error(f'Failed to rebuild KB cache: {e}')
        st.markdown('---')
        st.subheader('Live status')
        st.write('**Last tool:**', st.session_state.last_tool)
        st.write('**Pain area:**', st.session_state.pain_area)
        st.write('**Thermal fatigue:**', f"{st.session_state.thermal_fatigue_multiplier:.2f}x")

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
        st.write('**Thermal fatigue:**', f"{st.session_state.thermal_fatigue_multiplier:.2f}x")
        st.json(st.session_state.extracted_params)
    # Neural Diagnostics Dashboard (semantic search + comfort map)
    from semantic import build_kb_from_dir, rank_kb, project_kb_layout

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

    with st.expander('Neural Diagnostics Dashboard'):
        q = st.text_input('Query for neural diagnostics (or paste conversation text)')
        projection_choice = st.selectbox('Map projection', ['Auto', 'PCA', 'UMAP', 'Heuristic'], index=0)
        projection_method = projection_choice.lower()

        layout = project_kb_layout(kb, query=q or None, method=projection_method)
        layout_method = str(layout.get('method', 'heuristic')).upper()
        layout_source = str(layout.get('source', 'heuristic')).upper()
        node_coords = layout.get('coords', {}) if isinstance(layout.get('coords', {}), dict) else {}
        query_coords = layout.get('query_coords')

        # Prepare KB scatter data
        nodes = []
        for d in kb:
            content = d.get('content', '')
            x, y = node_coords.get(d.get('id'), (0.0, 0.0))
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                x, y = 0.0, 0.0
            # build a short preview/snippet (first paragraph or first 120 chars)
            para = ''
            for line in content.splitlines():
                s = line.strip()
                if s:
                    para = s
                    break
            if not para:
                para = content[:120]
            snippet = (para[:120] + '...') if len(para) > 120 else para
            nodes.append({'id': d.get('id'), 'title': d.get('title'), 'x': x, 'y': y, 'content': content, 'snippet': snippet, 'score': 0.0})

        # selected_doc tracks a doc chosen via click; initialize to None
        selected_doc = None

        # Run diagnostics when user clicks; compute query point and top match(s)
        top_match = None
        top_matches = []
        score_by_title = {}
        if q:
            ranked = rank_kb(q, kb, top_k=min(5, len(kb)))
            top_match = ranked[0] if ranked else None
            # keep the semantic search ranking, but attach actual similarity scores from the computed nodes
        
        map_title = 'Comfort Map (heuristic fallback)'
        if layout_method != 'HEURISTIC' or layout_source != 'HEURISTIC':
            map_title = f'Semantic Map ({layout_method} on {layout_source})'

        # Build plotly figure
        if nodes:
            if q:
                top_matches = []
                for m in ranked:
                    title = m.get('title')
                    score = round(float(m.get('score', 0.0)), 4)
                    score_by_title[title] = score
                    top_matches.append({
                        'id': m.get('id'),
                        'title': title,
                        'content': m.get('content', ''),
                        'score': score,
                    })

                # update every plotted node with the same ranking score used above
                for n in nodes:
                    n['score'] = round(float(score_by_title.get(n.get('title'), 0.0)), 4)

            df = nodes
            # include snippet and score in hover data
            fig = px.scatter(df, x='x', y='y', hover_name='title', hover_data=['snippet', 'score'], text='title')
            # highlight top matches if available (first one is the strongest)
            if top_matches:
                top_ids = set()
                for m in top_matches:
                    match_title = m.get('title')
                    if match_title:
                        top_ids.add(match_title)

                # add a trace for the additional matching docs so the plot shows more than one result
                match_points = [n for n in nodes if n.get('title') in top_ids]
                if match_points:
                    fig.add_scatter(
                        x=[n['x'] for n in match_points],
                        y=[n['y'] for n in match_points],
                        mode='markers',
                        marker=dict(size=10, color='orange'),
                        hoverinfo='text',
                        hovertext=[f"{n['title']}<br>{n.get('snippet','')}<br>score={n.get('score', 0.0)}" for n in match_points],
                        name='Matching docs'
                    )

            # highlight the single top match if available
            if top_match:
                # add top match marker and label with document title (avoid showing raw numeric value)
                tx, ty = node_coords.get(top_match.get('id'), (0.0, 0.0))
                if not isinstance(tx, (int, float)) or not isinstance(ty, (int, float)):
                    tx, ty = 0.0, 0.0
                tm_snip = (top_match.get('content', '') or '')[:120]
                tm_score = None
                if q:
                    tm_score = round(float(score_by_title.get(top_match.get('title'), 0.0)), 4)
                hovertext = f"{top_match.get('title','')}<br>{tm_snip}" + (f"<br>score={tm_score}" if tm_score is not None else '')
                fig.add_scatter(
                    x=[tx],
                    y=[ty],
                    mode='markers+text',
                    marker=dict(size=14, color='red'),
                    text=[top_match.get('title', 'Top match')],
                    textposition='top center',
                    textfont=dict(color='white', size=12),
                    hoverinfo='text',
                    hovertext=hovertext,
                    name='Top match'
                )
            # add query point
            if query_coords:
                qx, qy = query_coords
                q_snip = (q or '')[:120]
                fig.add_scatter(x=[qx], y=[qy], mode='markers', marker=dict(size=12, color='green'), name='Query', hovertext=q_snip, hoverinfo='text')

                fig.update_layout(
                    title=map_title,
                    xaxis_title='Projection X',
                    yaxis_title='Projection Y',
                    template='plotly_dark',
                    autosize=True,
                    height=420,
                )

                # Prefer streamlit-plotly-events for click interactivity; fall back to st.plotly_chart.
                try:
                    from streamlit_plotly_events import plotly_events
                except Exception:
                    plotly_events = None

                selected_doc = None
                if plotly_events is None:
                    st.plotly_chart(fig, use_container_width=True)
                    st.info('Install `streamlit-plotly-events` to enable clicking nodes on the chart.')
                else:
                    selected = plotly_events(fig, click_event=True, key='plot_click')
                    if selected:
                        p = selected[0]
                        # Prefer pointNumber when available
                        curve = p.get('curveNumber')
                        point = p.get('pointNumber')
                        sel_doc = None
                        if curve == 0 and point is not None and 0 <= point < len(nodes):
                            sel_doc = nodes[int(point)]
                        else:
                            # fallback: use x/y coordinates to find nearest point
                            px = p.get('x')
                            py = p.get('y')
                            if px is not None and py is not None:
                                best = None
                                bestd = None
                                for n in nodes:
                                    dx = n['x'] - float(px)
                                    dy = n['y'] - float(py)
                                    d = dx * dx + dy * dy
                                    if best is None or d < bestd:
                                        best = n
                                        bestd = d
                                sel_doc = best
                        if sel_doc:
                            # store selection in session_state so UI reflects click on rerun
                            try:
                                st.session_state['kb_selector'] = sel_doc['title']
                            except Exception:
                                pass
                            selected_doc = sel_doc

        # Node selector (click events require extra package; use selector as fallback)
        titles = [n['title'] for n in nodes]
        default_title = None
        if 'kb_selector' in st.session_state:
            default_title = st.session_state.get('kb_selector')
        elif top_match:
            default_title = top_match.get('title')

        # Build options and determine selected value
        options = ['(none)'] + titles
        if default_title and default_title in titles:
            sel_value = default_title
        else:
            sel_value = '(none)'

        sel = st.selectbox('Select a KB node to view (click a point on the map or pick here)', options, index=options.index(sel_value))
        st.caption('This selector is a fallback to view KB documents when you prefer not to click the map. Clicking a node on the comfort map will also select it here.')

        # Determine which doc to display: priority -> click-selected (selected_doc), selectbox, top_match
        display_doc = None
        if selected_doc:
            display_doc = selected_doc
        elif sel and sel != '(none)':
            sel_idx = titles.index(sel)
            display_doc = nodes[sel_idx]
        elif top_match:
            display_doc = top_match

        if display_doc:
            with st.expander(display_doc.get('title', 'Document')):
                st.markdown(display_doc.get('content', ''))

        if q and top_matches:
            st.markdown('**Top matching KB documents**')
            for i, m in enumerate(top_matches, start=1):
                score = m.get('score', 0.0) if isinstance(m, dict) else 0.0
                st.write(f"{i}. {m.get('title', 'Untitled')} — score: {score:.4f}")

    with st.expander('Environmental Dashboard'):
        if not st.session_state.get('environment_metrics'):
            process_environmental_metabolic_metrics(force_refresh=True)

        env = st.session_state.get('environment_metrics', {}) or {}
        if st.session_state.get('environment_error'):
            st.warning(f"Using fallback environmental values: {st.session_state.environment_error}")

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric('Temperature (°C)', f"{env.get('temperature_c', 30.0):.1f}")
        col_b.metric('Humidity (%)', f"{env.get('humidity_percent', 50.0):.1f}")
        col_c.metric('Cloud Cover (%)', f"{env.get('cloud_cover_percent', 0.0):.1f}")
        col_d.metric('Thermal Fatigue (x)', f"{env.get('thermal_fatigue_multiplier', 1.0):.2f}")
        col_e.metric('Calorie Burn (kcal)', f"{env.get('calories_burned', 0.0):.2f}")

        st.caption(
            f"Location: {env.get('latitude', st.session_state.latitude):.4f}, {env.get('longitude', st.session_state.longitude):.4f} | "
            f"Workspace: {env.get('workspace_mode', st.session_state.workspace_mode)} | "
            f"MET: {env.get('met', met_for_workspace_mode(st.session_state.workspace_mode)):.2f}"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric('Feels like', f"{env.get('apparent_temperature_c', env.get('temperature_c', 30.0)):.1f} °C")
        c2.metric('Wind speed', f"{env.get('wind_speed_kph', 0.0):.1f} kph")
        c3.metric('Rain probability', f"{env.get('rain_probability_percent', 0.0):.0f}%")

        temp = float(env.get('temperature_c', 30.0))
        humidity = float(env.get('humidity_percent', 50.0))
        fatigue = float(env.get('thermal_fatigue_multiplier', 1.0))
        if temp > 30.0 or humidity > 70.0:
            if fatigue >= 1.1:
                st.error('Heat and humidity are both elevated. Take a cooling break and reduce exertion if possible.')
            else:
                st.warning('Conditions are warmer or more humid than ideal. Keep hydration and break cadence steady.')
        else:
            st.success('Environmental conditions are within a comfortable baseline.')


if __name__ == '__main__':
    main()
