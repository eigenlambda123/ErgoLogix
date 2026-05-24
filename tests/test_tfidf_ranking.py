from semantic import build_kb, rank_kb


def test_tfidf_ranking_prefers_specific_match():
    docs = [
        {'id': 'generic', 'title': 'Generic Back', 'content': 'back pain back pain posture'},
        {'id': 'specific', 'title': 'Lumbar Support', 'content': 'lumbar support and lower back soreness'},
        {'id': 'other', 'title': 'Wrist Relief', 'content': 'wrist pain and hand soreness'},
    ]

    kb = build_kb(docs)
    ranked = rank_kb('lower back lumbar pain', kb, top_k=3)

    assert ranked[0]['id'] == 'specific'
    assert ranked[0]['score'] >= ranked[1]['score']
    assert ranked[0]['score'] > 0.0
