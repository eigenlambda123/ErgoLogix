

# ErgoLogix

Lightweight prototype for conversational ergonomics tooling. This project demonstrates:

- A Streamlit conversational router UI (`app.py`).
- Local LLM intent fallback (Ollama CLI + HTTP fallback) for intent extraction.
- A lightweight semantic KB built from Markdown files with an on-disk cache (`semantic.py`).
- A Neural Diagnostics Dashboard showing a comfort-map visualization of KB nodes.

Getting started
--------------

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Add knowledge-base Markdown files under the `kb/` directory (one `.md` per document). The loader
	 extracts the first H1 as the title and uses the rest as content. Example structure:

```
kb/
	wrist.md        # starts with '# Wrist Setup'
	neck.md         # starts with '# Neck Relief'
	lower_back.md   # starts with '# Lower Back'
```

3. Rebuild the on-disk KB cache (optional — the Streamlit app will rebuild automatically on request):

```bash
python scripts/sync_kb.py --kb kb --cache data/kb_cache.json --verbose
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

Using the app
-------------

- The sidebar contains Ollama settings and a **Rebuild KB cache** button to regenerate the cache from `kb/`.
- The Neural Diagnostics Dashboard provides a comfort-map of KB documents and lets you query text to see the
	top matching document and coordinates.

Testing
-------

Run tests with `pytest`:

```bash
pytest -q
```

Development notes
-----------------
- KB caching: tokens/vectors are stored in `data/kb_cache.json`. The cache stores per-file SHA1 hashes so only
	changed files are recomputed on rebuilds.
- CLI: `scripts/sync_kb.py` is a small utility to rebuild the cache from the command line and is used by tests.
- Plotly visualization: The comfort-map uses a simple heuristic projection from token keywords; consider adding
	PCA/UMAP or real embeddings later for better topology.

Next steps
----------
- Add click-to-select Plotly interactivity in the app (optional dependency: `streamlit-plotly-events`).
- Add CI to run tests on push (GitHub Actions).

