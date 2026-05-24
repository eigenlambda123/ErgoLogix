import streamlit as st
from typing import Optional, Dict


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

    col1, col2 = st.columns([3, 1])

    with col1:
        msg = st.text_area('Describe your discomfort, posture, or environment', height=120)
        if st.button('Send') and msg.strip():
            try:
                process_message(msg)
                st.experimental_rerun()
            except Exception as e:
                st.error(f'Error processing message: {e}')

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
