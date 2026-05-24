

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
 Streamlit conversational router UI (`app.py`) with a simple intent-extraction pipeline and Ollama fallback (CLI + HTTP).
 Semantic KB loaded from Markdown files in `kb/` with an on-disk token/vector cache at `data/kb_cache.json` and a `scripts/sync_kb.py` CLI to rebuild it.
 Semantic ranking now prefers **real embeddings** when an Ollama embedding model is available, with TF-IDF fallback when embeddings are unavailable.
 Neural Diagnostics Dashboard: Plotly comfort-map visualization showing KB nodes, query points, multiple matching docs, and a highlighted top match.
- Streamlit conversational router UI (`app.py`) with a simple intent-extraction pipeline and Ollama fallback (CLI + HTTP).
- Semantic KB loaded from Markdown files in `kb/` with an on-disk token/vector cache at `data/kb_cache.json` and a `scripts/sync_kb.py` CLI to rebuild it.
- Neural Diagnostics Dashboard: Plotly comfort-map visualization showing KB nodes, query points, and a highlighted top match.
- Click-to-select interactivity using `streamlit-plotly-events` (optional): clicking a point selects the KB node and displays its document.
- Hover snippets and similarity scores are shown on points to help users preview matches.
- Unit tests for extractors, KB loader, and the sync CLI are included under `tests/` and run in CI.
- Minimal GitHub Actions CI workflow added to run `pytest` on push/PR (`.github/workflows/ci.yml`).
- `requirements.txt` is pinned to tested package versions for reproducible installs.

How to run (quick)
------------------
 Embeddings: if Ollama embeddings are available, `semantic.py` also stores float embeddings in the same cache and uses them for ranking; otherwise it falls back to TF-IDF.
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Add or edit KB Markdown files under `kb/` (one file per document). The loader uses the first H1 as the title.

3. (Optional) Rebuild the on-disk KB cache (or use the app sidebar button):

```bash
python scripts/sync_kb.py --kb kb --cache data/kb_cache.json --verbose
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Interact with the Neural Diagnostics Dashboard:
- Type a query to see the query point and top-matching KB node on the comfort-map.
- Hover nodes to see a short snippet and similarity score.
- Click a node (requires `streamlit-plotly-events`) or use the selector below the chart to view the document content.

Testing
-------

Run unit tests locally:

```bash
pytest -q
```

CI
--
We added a basic GitHub Actions workflow at `.github/workflows/ci.yml` that installs dependencies and runs `pytest` on push/PR.

Developer notes
---------------
- KB caching: token vectors and per-file hashes are stored in `data/kb_cache.json`. The cache prevents recomputing vectors for unchanged files.
- Interactivity: `streamlit-plotly-events` is optional. Install it to enable click-to-select on the comfort-map. If missing, the app falls back to a static Plotly chart and a selector UI.
- Hover previews: we compute a short snippet (first paragraph or first 120 chars) for each doc and display it plus a cosine-similarity score when a query is present.
- Defaults & fallbacks: the app uses a lightweight token-count approach for semantic matching and a keyword-based 2D comfort-map projection. These are simple and transparent — consider switching to embeddings for higher accuracy.

Next steps (recommended)
------------------------
 1. Acceptance / UI tests: add a smoke test and browser-based UI tests to verify the interactive flows in CI.
 2. Optional: add PCA/UMAP projection for improved map layout (requires `numpy` + `scikit-learn` or `umap-learn`).
 3. Polish: pin any remaining dependencies, add release notes, and consider a small tutorial notebook or demo script.

If you want, I can implement any of the next steps — tell me which and I'll start.

