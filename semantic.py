import re
import math
from collections import Counter
from typing import List, Dict, Tuple


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


def build_kb(docs: List[Dict[str, str]]) -> List[Dict]:
    """Build a simple in-memory KB. Each doc is dict with `id`, `title`, `content`."""
    kb = []
    for i, d in enumerate(docs):
        content = d.get('content', '')
        tokens = tokenize(content)
        vect = Counter(tokens)
        kb.append({
            'id': d.get('id', i),
            'title': d.get('title', f'doc-{i}'),
            'content': content,
            'tokens': vect,
        })
    return kb


def cosine_sim(c1: Counter, c2: Counter) -> float:
    dot = 0.0
    for k, v in c1.items():
        dot += v * c2.get(k, 0)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def search_kb(query: str, kb: List[Dict], top_k: int = 3) -> List[Dict]:
    q_tokens = Counter(tokenize(query))
    scored = []
    for d in kb:
        score = cosine_sim(q_tokens, d['tokens'])
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k]]


# Simple keyword maps to compute a 2D comfort map (posture_balance, tension_level)
TENSION_KEYWORDS = {
    'pain': 1.0, 'sore': 1.0, 'soreness': 1.0, 'ache': 0.8, 'tight': 0.9, 'stiff': 0.9, 'fatigue': 1.0, 'tired': 0.8
}

POSTURE_KEYWORDS = {
    'wrist': 0.6, 'ulnar': 0.6, 'hand': 0.5,
    'neck': -0.6, 'cervical': -0.6,
    'lower': 0.4, 'back': 0.4, 'lumbar': 0.5, 'shoulder': -0.2, 'elbow': 0.0
}


def compute_comfort_coords(text: str) -> Tuple[float, float]:
    """Returns (posture_balance, tension_level) where posture_balance roughly in [-1,1] and tension_level in [0,1]."""
    tokens = tokenize(text)
    posture = 0.0
    tension = 0.0
    for t in tokens:
        posture += POSTURE_KEYWORDS.get(t, 0.0)
        tension += TENSION_KEYWORDS.get(t, 0.0)
    # normalize
    if tokens:
        posture = max(-1.0, min(1.0, posture / max(1.0, len(tokens) * 0.5)))
        tension = max(0.0, min(1.0, tension / max(1.0, len(tokens) * 0.5)))
    else:
        posture = 0.0
        tension = 0.0
    return posture, tension


def top_match_and_coords(query: str, kb: List[Dict]) -> Dict:
    results = search_kb(query, kb, top_k=1)
    coords = compute_comfort_coords(query)
    top = results[0] if results else None
    return {'top': top, 'coords': coords}
