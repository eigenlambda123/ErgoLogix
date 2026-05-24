from semantic import build_kb, search_kb, compute_comfort_coords, top_match_and_coords


SAMPLE_DOCS = [
    {'id': 'd1', 'title': 'Wrist Setup', 'content': 'Advice for wrist pain and ulnar soreness when typing.'},
    {'id': 'd2', 'title': 'Neck Relief', 'content': 'Cervical stretch and neck pain relief guidance.'},
    {'id': 'd3', 'title': 'Lower Back', 'content': 'Lumbar support and lower back strain exercises.'},
]


def test_search_kb_basic():
    kb = build_kb(SAMPLE_DOCS)
    results = search_kb('my wrist hurts when typing', kb, top_k=1)
    assert len(results) == 1
    assert results[0]['id'] == 'd1'


def test_compute_comfort_coords():
    x, y = compute_comfort_coords('my wrist feels sore and tight')
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert -1.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0


def test_top_match_and_coords():
    kb = build_kb(SAMPLE_DOCS)
    out = top_match_and_coords('I have lumbar pain in my lower back', kb)
    assert out['top'] is not None
    assert out['top']['id'] == 'd3'
    assert isinstance(out['coords'], tuple)
