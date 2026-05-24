import re
import math
import os
import glob
import json
import hashlib
from collections import Counter
from typing import List, Dict, Tuple

try:
    import requests
except Exception:
    requests = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import umap as umap_module
except Exception:
    umap_module = None


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


def _vector_dot(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def _vector_norm(v: List[float]) -> float:
    return math.sqrt(sum(a * a for a in v))


def cosine_sim_vectors(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    dot = _vector_dot(v1, v2)
    n1 = _vector_norm(v1)
    n2 = _vector_norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _embed_text_ollama(text: str, model: str = 'nomic-embed-text', host: str = 'http://127.0.0.1:11434') -> List[float] | None:
    """Try to embed text using Ollama's local HTTP API.

    Returns a float vector or None if the endpoint/model is unavailable.
    """
    if requests is None:
        return None

    payloads = [
        (host.rstrip('/') + '/api/embeddings', {'model': model, 'prompt': text}),
        (host.rstrip('/') + '/api/embed', {'model': model, 'input': text}),
    ]

    for url, payload in payloads:
        try:
            resp = requests.post(url, json=payload, timeout=10)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        try:
            data = resp.json()
        except Exception:
            continue

        emb = data.get('embedding')
        if emb is None and isinstance(data.get('embeddings'), list):
            emb = data['embeddings']
            if emb and isinstance(emb[0], list):
                emb = emb[0]

        if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
            return [float(x) for x in emb]

    return None


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

    Cache format: { relative_path: { 'hash': sha1, 'title':..., 'content':..., 'vector': {token:count}, 'embedding': [floats] } }
    """
    docs = load_markdown_kb(kb_dir)
    cache = _load_cache(cache_path)
    updated = False
    kb = []
    embed_model = os.environ.get('ERGLOGIX_EMBED_MODEL', 'nomic-embed-text')
    embed_host = os.environ.get('ERGLOGIX_EMBED_HOST', 'http://127.0.0.1:11434')
    for d in docs:
        rel = d['id']
        path = d.get('path')
        file_hash = _file_hash(path) if path and os.path.exists(path) else None
        cached = cache.get(rel)
        if cached and file_hash and cached.get('hash') == file_hash:
            vect = Counter(cached.get('vector', {}))
            doc = {'id': rel, 'title': cached.get('title', d.get('title')), 'content': cached.get('content', d.get('content')), 'tokens': vect}

            cached_embedding = cached.get('embedding')
            cached_model = cached.get('embedding_model')
            if isinstance(cached_embedding, list) and cached_model == embed_model:
                doc['embedding'] = [float(x) for x in cached_embedding if isinstance(x, (int, float))]
                doc['embedding_model'] = cached_model
            else:
                embedding = _embed_text_ollama(doc['content'], model=embed_model, host=embed_host)
                if embedding is not None:
                    doc['embedding'] = embedding
                    doc['embedding_model'] = embed_model
                    cache[rel] = {
                        'hash': file_hash,
                        'title': doc['title'],
                        'content': doc['content'],
                        'vector': dict(vect),
                        'embedding': embedding,
                        'embedding_model': embed_model,
                    }
                    updated = True

            kb.append(doc)
            continue
        # compute
        content = d.get('content', '')
        tokens = tokenize(content)
        vect = Counter(tokens)
        doc = {'id': rel, 'title': d.get('title'), 'content': content, 'tokens': vect}

        embedding = _embed_text_ollama(content, model=embed_model, host=embed_host)
        if embedding is not None:
            doc['embedding'] = embedding
            doc['embedding_model'] = embed_model

        kb.append(doc)
        # update cache
        cache_entry = {'hash': file_hash, 'title': d.get('title'), 'content': content, 'vector': dict(vect)}
        if embedding is not None:
            cache_entry['embedding'] = embedding
            cache_entry['embedding_model'] = embed_model
        cache[rel] = cache_entry
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


def _embedding_rows(kb: List[Dict]) -> List[List[float]] | None:
    rows: List[List[float]] = []
    for d in kb:
        emb = d.get('embedding')
        if not isinstance(emb, list) or not emb:
            return None
        row: List[float] = []
        for value in emb:
            if not isinstance(value, (int, float)):
                return None
            row.append(float(value))
        rows.append(row)

    if not rows:
        return None
    dims = {len(row) for row in rows}
    if len(dims) != 1:
        return None
    return rows


def _tfidf_matrix_and_vocab(kb: List[Dict]) -> Tuple[List[List[float]], List[str], Dict[str, int]] | None:
    vocab = sorted({term for d in kb for term in d.get('tokens', Counter()).keys()})
    if not vocab:
        return None

    df = _doc_frequencies(kb)
    n_docs = len(kb)
    index = {term: i for i, term in enumerate(vocab)}
    matrix: List[List[float]] = []
    for d in kb:
        row = [0.0] * len(vocab)
        tokens = d.get('tokens', Counter())
        total = sum(tokens.values()) or 1
        for term, count in tokens.items():
            idx = index.get(term)
            if idx is None:
                continue
            idf = math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1.0
            row[idx] = (count / total) * idf
        matrix.append(row)

    return matrix, vocab, df


def _tfidf_vector_for_query(query: str, vocab: List[str], df: Dict[str, int], n_docs: int) -> List[float]:
    q_tokens = Counter(tokenize(query))
    index = {term: i for i, term in enumerate(vocab)}
    total = sum(q_tokens.values()) or 1
    row = [0.0] * len(vocab)
    for term, count in q_tokens.items():
        idx = index.get(term)
        if idx is None:
            continue
        idf = math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1.0
        row[idx] = (count / total) * idf
    return row


def _project_with_pca(matrix: List[List[float]], query_vector: List[float] | None = None) -> Dict[str, object] | None:
    if np is None:
        return None

    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return None

    mean = X.mean(axis=0)
    centered = X - mean
    if centered.shape[0] == 1:
        coords = np.zeros((1, 2), dtype=float)
        query_coords = None
        if query_vector is not None:
            query_coords = np.zeros(2, dtype=float)
        return {'coords': coords, 'query_coords': query_coords, 'method': 'pca'}

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    if components.shape[0] == 1:
        components = np.vstack([components, np.zeros_like(components)])

    for i in range(components.shape[0]):
        row = components[i]
        if not np.any(row):
            continue
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            components[i] = -row

    coords = centered @ components.T
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=float)])

    query_coords = None
    if query_vector is not None:
        q = np.asarray(query_vector, dtype=float) - mean
        q_coords = q @ components.T
        if np.ndim(q_coords) == 0:
            q_coords = np.asarray([float(q_coords), 0.0], dtype=float)
        elif len(q_coords) == 1:
            q_coords = np.asarray([float(q_coords[0]), 0.0], dtype=float)
        query_coords = q_coords

    return {'coords': coords, 'query_coords': query_coords, 'method': 'pca'}


def _project_with_umap(matrix: List[List[float]], query_vector: List[float] | None = None) -> Dict[str, object] | None:
    if np is None or umap_module is None:
        return None
    if len(matrix) < 2:
        return None

    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        return None

    try:
        mapper = umap_module.UMAP(n_components=2, random_state=42)
        coords = mapper.fit_transform(X)
        query_coords = None
        if query_vector is not None:
            q = np.asarray([query_vector], dtype=float)
            q_coords = mapper.transform(q)
            query_coords = q_coords[0]
        return {'coords': coords, 'query_coords': query_coords, 'method': 'umap'}
    except Exception:
        return None


def project_kb_layout(kb: List[Dict], query: str | None = None, method: str = 'auto') -> Dict[str, object]:
    """Project KB documents into 2D space for visualization.

    Prefers embeddings when every document has them and the query can also be embedded.
    Falls back to TF-IDF vectors, and then to the heuristic comfort map if no vector path
    is available.
    """
    method = (method or 'auto').lower()
    layout: Dict[str, object] = {
        'method': 'heuristic',
        'source': 'heuristic',
        'coords': {},
        'query_coords': None,
    }

    if not kb:
        if query:
            layout['query_coords'] = compute_comfort_coords(query)
        return layout

    embed_rows = _embedding_rows(kb)
    vector_source = None
    doc_matrix: List[List[float]] | None = None
    query_vector: List[float] | None = None
    tfidf_vocab: List[str] | None = None
    tfidf_df: Dict[str, int] | None = None

    embed_model = os.environ.get('ERGLOGIX_EMBED_MODEL', 'nomic-embed-text')
    embed_host = os.environ.get('ERGLOGIX_EMBED_HOST', 'http://127.0.0.1:11434')

    if embed_rows is not None:
        vector_source = 'embeddings'
        doc_matrix = embed_rows
        if query:
            query_vector = _embed_text_ollama(query, model=embed_model, host=embed_host)
            if query_vector is None or len(query_vector) != len(doc_matrix[0]):
                vector_source = None
                doc_matrix = None
                query_vector = None

    if doc_matrix is None:
        tfidf = _tfidf_matrix_and_vocab(kb)
        if tfidf is not None:
            doc_matrix, tfidf_vocab, tfidf_df = tfidf
            vector_source = 'tfidf'
            if query:
                query_vector = _tfidf_vector_for_query(query, tfidf_vocab, tfidf_df, len(kb))

    if doc_matrix is None:
        for d in kb:
            layout['coords'][d.get('id')] = compute_comfort_coords(d.get('content', ''))
        if query:
            layout['query_coords'] = compute_comfort_coords(query)
        return layout

    projection = None
    if method in {'heuristic', 'comfort', 'fallback'}:
        projection = None
    elif method == 'umap':
        projection = _project_with_umap(doc_matrix, query_vector=query_vector)
        if projection is None:
            projection = _project_with_pca(doc_matrix, query_vector=query_vector)
    elif method == 'pca':
        projection = _project_with_pca(doc_matrix, query_vector=query_vector)
    else:
        projection = _project_with_umap(doc_matrix, query_vector=query_vector)
        if projection is None:
            projection = _project_with_pca(doc_matrix, query_vector=query_vector)

    if projection is None:
        for d in kb:
            layout['coords'][d.get('id')] = compute_comfort_coords(d.get('content', ''))
        if query:
            layout['query_coords'] = compute_comfort_coords(query)
        return layout

    coords = projection['coords']
    layout['method'] = projection['method']
    layout['source'] = vector_source or 'heuristic'
    for doc, point in zip(kb, coords):
        layout['coords'][doc.get('id')] = (float(point[0]), float(point[1]))

    q_coords = projection.get('query_coords')
    if q_coords is not None:
        layout['query_coords'] = (float(q_coords[0]), float(q_coords[1]))
    elif query:
        layout['query_coords'] = compute_comfort_coords(query)

    return layout


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
    q_embedding = _embed_text_ollama(query)
    scored = []
    for d in kb:
        score = 0.0
        doc_embedding = d.get('embedding')
        if q_embedding is not None and isinstance(doc_embedding, list) and doc_embedding:
            score = cosine_sim_vectors(q_embedding, doc_embedding)
        if score == 0.0:
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
    q_embedding = _embed_text_ollama(query)
    scored = []
    for d in kb:
        score = 0.0
        doc_embedding = d.get('embedding')
        if q_embedding is not None and isinstance(doc_embedding, list) and doc_embedding:
            score = cosine_sim_vectors(q_embedding, doc_embedding)
        if score == 0.0:
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
    results = rank_kb(query, kb, top_k=1)
    coords = compute_comfort_coords(query)
    top = results[0] if results else None
    return {'top': top, 'coords': coords}
