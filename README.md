# ErgoLogix

Lightweight prototype for conversational ergonomics tooling.

## What it does

- Streamlit conversational router UI in `app.py`.
- Local LLM intent fallback via Ollama CLI with HTTP fallback.
- Markdown knowledge base loader with on-disk cache in `semantic.py`.
- Neural Diagnostics Dashboard with semantic ranking and 2D projection.

## Setup

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Add Markdown knowledge-base files under `kb/`.

Each file should use the first H1 as the title:

```md
# Wrist Setup

Advice for wrist pain and typing posture.
```

3. Rebuild the KB cache if you want to precompute vectors.

```bash
python scripts/sync_kb.py --kb kb --cache data/kb_cache.json --verbose
```

4. Start the app.

```bash
streamlit run app.py
```

## Dashboard

- Use the projection selector to switch between `Auto`, `PCA`, `UMAP`, and `Heuristic`.
- `Auto` prefers UMAP when available and falls back to PCA.
- UMAP is optional; install `umap-learn` if you want that path enabled locally.
- The map uses embeddings when available, otherwise TF-IDF, and falls back to the heuristic comfort map.
- Hover points to preview snippets and scores.
- Click a point or use the selector to inspect the full document.

## Testing

```bash
pytest -q
```

## Release / Packaging

- See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the release checklist and packaging options.
- The app currently ships as a source-based Streamlit project; release artifacts are primarily the repo, pinned dependencies, and the KB cache.

## Notes

- Cache file: `data/kb_cache.json`
- Sample KB: `kb/`
- CI: `.github/workflows/ci.yml`