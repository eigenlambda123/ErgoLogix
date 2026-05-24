import pytest

pytest.importorskip('numpy')

from semantic import build_kb, project_kb_layout


def test_project_kb_layout_pca_is_deterministic():
    docs = [
        {'id': 'wrist', 'title': 'Wrist Relief', 'content': 'wrist pain typing and hand soreness'},
        {'id': 'neck', 'title': 'Neck Relief', 'content': 'neck strain cervical posture and stretch breaks'},
        {'id': 'back', 'title': 'Lower Back', 'content': 'lumbar support lower back pain and chair setup'},
    ]

    kb = build_kb(docs)

    layout1 = project_kb_layout(kb, query='my lower back hurts while typing', method='pca')
    layout2 = project_kb_layout(kb, query='my lower back hurts while typing', method='pca')

    assert layout1['method'] == 'pca'
    assert layout1['source'] in {'tfidf', 'embeddings'}
    assert layout1['coords'] == layout2['coords']
    assert layout1['query_coords'] == layout2['query_coords']

    coords = list(layout1['coords'].values())
    assert all(len(point) == 2 for point in coords)
    assert len({tuple(round(value, 6) for value in point) for point in coords}) > 1
    assert isinstance(layout1['query_coords'], tuple)
    assert len(layout1['query_coords']) == 2