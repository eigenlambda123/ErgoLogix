import json
import time
from pathlib import Path

import pytest

from semantic import build_kb_from_dir


def write_md(path: Path, title: str, body: str):
    path.write_text(f"# {title}\n\n{body}\n")


def load_cache(cache_path: Path):
    with cache_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def test_build_cache_creates_file(tmp_path):
    kb_dir = tmp_path / 'kb'
    kb_dir.mkdir()
    write_md(kb_dir / 'a.md', 'A', 'content one')
    write_md(kb_dir / 'b.md', 'B', 'content two two')

    cache_path = tmp_path / 'kb_cache.json'
    assert not cache_path.exists()

    kb = build_kb_from_dir(str(kb_dir), cache_path=str(cache_path))

    assert cache_path.exists()
    data = load_cache(cache_path)
    # expect two entries (relative paths)
    assert len(data) == 2
    # check one vector exists and has tokens
    entry = data.get('a.md')
    assert entry is not None
    assert 'vector' in entry and isinstance(entry['vector'], dict)


def test_cache_invalidates_on_file_change(tmp_path):
    kb_dir = tmp_path / 'kb'
    kb_dir.mkdir()
    file_path = kb_dir / 'c.md'
    write_md(file_path, 'C', 'alpha beta')

    cache_path = tmp_path / 'kb_cache.json'
    build_kb_from_dir(str(kb_dir), cache_path=str(cache_path))
    data1 = load_cache(cache_path)
    h1 = data1['c.md']['hash']
    vec1 = data1['c.md']['vector']

    # modify file
    time.sleep(0.01)
    write_md(file_path, 'C', 'alpha beta gamma delta')

    build_kb_from_dir(str(kb_dir), cache_path=str(cache_path))
    data2 = load_cache(cache_path)
    h2 = data2['c.md']['hash']
    vec2 = data2['c.md']['vector']

    assert h1 != h2
    assert vec1 != vec2
