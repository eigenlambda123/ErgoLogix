from semantic import build_kb_from_dir


def test_build_kb_from_dir_persists_embeddings(tmp_path, monkeypatch):
    kb_dir = tmp_path / 'kb'
    kb_dir.mkdir()
    (kb_dir / 'back.md').write_text('# Back\n\nlower back lumbar support\n')

    cache_path = tmp_path / 'cache.json'

    monkeypatch.setattr('semantic._embed_text_ollama', lambda text, model='nomic-embed-text', host='http://127.0.0.1:11434': [0.9, 0.1])

    kb1 = build_kb_from_dir(str(kb_dir), cache_path=str(cache_path))
    assert kb1[0]['embedding'] == [0.9, 0.1]

    # second load should reuse the cached embedding (no server call needed)
    kb2 = build_kb_from_dir(str(kb_dir), cache_path=str(cache_path))
    assert kb2[0]['embedding'] == [0.9, 0.1]
