import json
import subprocess
import sys
from pathlib import Path


def run_sync(args, cwd):
    cmd = [sys.executable, 'scripts/sync_kb.py'] + args
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sync_kb failed: {proc.returncode}\nstderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")


def test_sync_writes_cache(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    kb_dir = tmp_path / 'kb'
    kb_dir.mkdir()
    (kb_dir / 'a.md').write_text('# A\n\nhello world\n')
    cache_path = tmp_path / 'cache.json'

    run_sync(['--kb', str(kb_dir), '--cache', str(cache_path)], cwd=str(project_root))

    assert cache_path.exists()
    data = json.loads(cache_path.read_text(encoding='utf-8'))
    assert 'a.md' in data


def test_sync_force_rebuild(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    kb_dir = tmp_path / 'kb'
    kb_dir.mkdir()
    p = kb_dir / 'b.md'
    p.write_text('# B\n\none two\n')
    cache_path = tmp_path / 'cache.json'

    # initial build
    run_sync(['--kb', str(kb_dir), '--cache', str(cache_path)], cwd=str(project_root))
    assert cache_path.exists()

    # corrupt cache and force rebuild
    cache_path.write_text('{}')
    run_sync(['--kb', str(kb_dir), '--cache', str(cache_path), '--force'], cwd=str(project_root))

    data = json.loads(cache_path.read_text(encoding='utf-8'))
    assert 'b.md' in data
