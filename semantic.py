import re
import math
import os
import glob
import json
import hashlib
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


def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_cache(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(path: str, data: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_markdown_kb(kb_dir: str) -> List[Dict[str, str]]:
    """Scan a directory for .md files and return list of docs with id,title,content."""
    docs = []
    if not os.path.isdir(kb_dir):
        return docs
    patterns = [os.path.join(kb_dir, '**', '*.md')]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    for path in sorted(files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                txt = f.read()
        except Exception:
            continue
        # simple title extraction: first H1 '# '
        title = None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith('# '):
                title = line.lstrip('# ').strip()
                break
            if line:
                # fallback: first non-empty line as title
                if title is None:
                    title = line
        if not title:
            title = os.path.splitext(os.path.basename(path))[0]
        docs.append({'id': os.path.relpath(path, kb_dir), 'title': title, 'content': txt, 'path': path})
    return docs


def build_kb_from_dir(kb_dir: str = 'kb', cache_path: str = 'data/kb_cache.json') -> List[Dict]:
    """Build KB from markdown files with a simple on-disk cache of token vectors.

    Cache format: { relative_path: { 'hash': sha1, 'title':..., 'content':..., 'vector': {token:count} } }
    """
    docs = load_markdown_kb(kb_dir)
    cache = _load_cache(cache_path)
    updated = False
    kb = []
    for d in docs:
        rel = d['id']
        path = d.get('path')
        file_hash = _file_hash(path) if path and os.path.exists(path) else None
        cached = cache.get(rel)
        if cached and file_hash and cached.get('hash') == file_hash:
            vect = Counter(cached.get('vector', {}))
            kb.append({'id': rel, 'title': cached.get('title', d.get('title')), 'content': cached.get('content', d.get('content')), 'tokens': vect})
            continue
        # compute
        content = d.get('content', '')
        tokens = tokenize(content)
        vect = Counter(tokens)
        kb.append({'id': rel, 'title': d.get('title'), 'content': content, 'tokens': vect})
        # update cache
        cache[rel] = {'hash': file_hash, 'title': d.get('title'), 'content': content, 'vector': dict(vect)}
        updated = True
    if updated:
        try:
            _save_cache(cache_path, cache)
        except Exception:
            pass
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


def _doc_frequencies(kb: List[Dict]) -> Dict[str, int]:
    """Count in how many documents each token appears."""
    df: Dict[str, int] = {}
    for d in kb:
        tokens = set(d.get('tokens', Counter()).keys())
        for t in tokens:
            df[t] = df.get(t, 0) + 1
    return df


def _tfidf_score(query_tokens: Counter, doc_tokens: Counter, df: Dict[str, int], n_docs: int) -> float:
    """Compute a simple cosine-like TF-IDF score between query and doc token counters."""
    if not query_tokens or not doc_tokens or n_docs <= 0:
        return 0.0

    def tfidf(counter: Counter) -> Dict[str, float]:
        total = sum(counter.values()) or 1
        vec: Dict[str, float] = {}
        for term, count in counter.items():
            idf = math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1.0
            vec[term] = (count / total) * idf
        return vec

    q_vec = tfidf(query_tokens)
    d_vec = tfidf(doc_tokens)
    dot = 0.0
    for term, qv in q_vec.items():
        dot += qv * d_vec.get(term, 0.0)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
    if q_norm == 0 or d_norm == 0:
        return 0.0
    return dot / (q_norm * d_norm)


def search_kb(query: str, kb: List[Dict], top_k: int = 3) -> List[Dict]:
    """Rank KB docs for a query using a lightweight TF-IDF cosine score.

    This improves over raw token-count cosine by down-weighting common terms and
    giving more signal to rarer tokens across the KB.
    """
    q_tokens = Counter(tokenize(query))
    if not kb:
        return []

    df = _doc_frequencies(kb)
    n_docs = len(kb)
    scored = []
    for d in kb:
        score = _tfidf_score(q_tokens, d.get('tokens', Counter()), df, n_docs)
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for s, d in scored[:top_k]]


def rank_kb(query: str, kb: List[Dict], top_k: int = 3) -> List[Dict]:
    """Return the top matching docs with their TF-IDF scores attached.

    Each result is a shallow copy of the KB doc with a `score` field.
    """
    q_tokens = Counter(tokenize(query))
    if not kb:
        return []

    df = _doc_frequencies(kb)
    n_docs = len(kb)
    scored = []
    for d in kb:
        score = _tfidf_score(q_tokens, d.get('tokens', Counter()), df, n_docs)
        doc = dict(d)
        doc['score'] = score
        scored.append((score, doc))
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
