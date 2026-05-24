# ErgoLogix

Lightweight prototype for conversational ergonomics tooling.

## What it does

- Streamlit conversational router UI in `app.py`.
- Local LLM intent fallback via Ollama CLI with HTTP fallback.
- Markdown knowledge base loader with on-disk cache in `semantic.py`.
- Neural Diagnostics Dashboard with semantic ranking and 2D projection.
- Environmental Dashboard with Open-Meteo weather, thermal fatigue, and calorie burn.

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
- Environmental telemetry can auto-detect your approximate location from your public IP, with a manual override in the sidebar.
	- New: Browser geolocation — use the sidebar "Detect my location (browser)" button to request high-accuracy coordinates from your browser (calls `navigator.geolocation`). This requires the optional dependency `streamlit-javascript` (already pinned in `requirements.txt`) and HTTPS in production (localhost works for development). The app only requests geolocation when you explicitly click the button; if the browser API is unavailable, permission is denied, or parsing fails, the app falls back to the IP-based lookup (`ipapi.co`). See the Notes section for privacy and deployment caveats.
- Hover points to preview snippets and scores.
- Click a point or use the selector to inspect the full document.
- Edit the sidebar environmental inputs to refresh weather and metabolic metrics.

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

**Browser Geolocation:**
- **Dependency:** `streamlit-javascript` (included in `requirements.txt`).
- **Usage:** Click "Detect my location (browser)" in the sidebar to allow the browser to share precise coordinates. The app will update the environmental metrics on success.
- **Fallback:** If the browser denies permission, the helper is not installed, or the call fails, the app will fall back to the server-side IP lookup.
- **Deployment:** Browsers require HTTPS for `navigator.geolocation` on deployed sites. Local development (localhost) is allowed over HTTP.
- **Privacy:** Geolocation is requested only on explicit user action and never sent to third parties by the app itself; the IP geolocation provider (`ipapi.co`) is used only for approximate location and may be subject to their terms.