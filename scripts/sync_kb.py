#!/usr/bin/env python3
"""CLI to rebuild the KB cache from Markdown files.

Usage:
  python scripts/sync_kb.py --kb kb --cache data/kb_cache.json [--force] [--verbose]

This calls `semantic.build_kb_from_dir` and writes the cache file. Use `--force`
to remove the existing cache before rebuilding.
"""
import argparse
import sys
import os

# Ensure the project root is on sys.path so we can import top-level modules
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from semantic import build_kb_from_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Sync KB markdown files into on-disk cache')
    p.add_argument('--kb', default='kb', help='KB directory containing markdown files')
    p.add_argument('--cache', default='data/kb_cache.json', help='Path to write cache JSON')
    p.add_argument('--force', action='store_true', help='Remove existing cache before rebuilding')
    p.add_argument('--verbose', action='store_true', help='Verbose output')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kb_dir = args.kb
    cache_path = args.cache

    if args.force and os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            if args.verbose:
                print(f'Removed existing cache: {cache_path}')
        except Exception as e:
            print(f'Failed to remove cache {cache_path}: {e}', file=sys.stderr)
            return 2

    if args.verbose:
        print(f'Building KB from: {kb_dir}\nCache path: {cache_path}')

    try:
        kb = build_kb_from_dir(kb_dir, cache_path=cache_path)
    except Exception as e:
        print(f'Error building KB: {e}', file=sys.stderr)
        return 1

    print(f'Built KB with {len(kb)} documents. Cache written to {cache_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
