from semantic import build_kb, rank_kb


def test_rank_kb_uses_embeddings_when_available(monkeypatch):
    docs = [
        {'id': 'back', 'title': 'Back Relief', 'content': 'lower back support and lumbar soreness'},
        {'id': 'wrist', 'title': 'Wrist Relief', 'content': 'wrist pain and hand soreness'},
    ]

    kb = build_kb(docs)
    kb[0]['embedding'] = [1.0, 0.0]
    kb[1]['embedding'] = [0.0, 1.0]

    def fake_embed(text, model='nomic-embed-text', host='http://127.0.0.1:11434'):
        if 'back' in text or 'lumbar' in text:
            return [1.0, 0.0]
        if 'wrist' in text:
            return [0.0, 1.0]
        return [0.0, 0.0]

    monkeypatch.setattr('semantic._embed_text_ollama', fake_embed)

    ranked = rank_kb('my back hurts', kb, top_k=2)

    assert ranked[0]['id'] == 'back'
    assert ranked[0]['score'] > ranked[1]['score']
