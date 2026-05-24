# ErgoLogix — Implemented Features (Actual Implementation)

This document records what was implemented in the repository (source files and behavior), matching the high-level features in `features.md`.

## 1. Conversational Intent Router and Parameter Extraction

- Implemented in [app.py](app.py).
- Conversation UI: Streamlit text input and `Send` button drive `process_message()` which extracts intent and routes to tools.
- Intent extraction:
  - `ollama_intent_extractor()` attempts the local `ollama` CLI and falls back to `ollama_http_intent_extractor()` which calls a local Ollama HTTP endpoint. See implementation in `app.py`.
  - A lightweight `keyword_intent_extractor()` is used when Ollama is unavailable or returns no result.
- Extracted parameters and routing state persist in `st.session_state` keys such as `pain_area`, `extracted_params`, and `last_tool`.

## 2. Tool Engine A: Predictive Health Risk Classifier

- The feature is scaffolded and referenced in `features.md` but the repo contains a placeholder implementation and a mathematical fallback. The risk classifier code lives in the `scripts/` helpers and in `app.py` where the router calls the appropriate tool. See `scripts/` for training/CLI helpers (if present).

## 3. Semantic Search Engine & Latent Space Comfort Map

- Implemented in `semantic.py`.
  - KB loader: `build_kb_from_dir(kb_dir, cache_path)` reads markdown docs and optionally uses an on-disk cache (`data/kb_cache.json`).
  - Ranking: TF‑IDF ranking is implemented and tested (`rank_kb()` and tests in `tests/test_tfidf_ranking.py`). Dense embedding hooks exist for Ollama-based embeddings with safe fallbacks.
  - Projection: PCA implemented (SVD with deterministic sign stabilization) in `_project_with_pca()`; optional UMAP path in `_project_with_umap()` (requires `umap-learn`). The public helper `project_kb_layout()` returns per-doc coords and a query coord when available.
  - The Neural Diagnostics Dashboard in [app.py](app.py) uses `project_kb_layout()` and `rank_kb()` to render a Plotly comfort map with hover snippets and click handling (uses `streamlit-plotly-events` if available).

## 4. Tool Engine C: Live Environmental and Metabolic Analyzer

- Implemented in `environmental.py` and wired into `app.py`.
  - Weather: `fetch_open_meteo(latitude, longitude)` calls the Open‑Meteo API to retrieve temperature, humidity, cloud cover, wind, and apparent temperature.
  - IP-based location: `fetch_user_location()` uses `https://ipapi.co/json/` to return an approximate `latitude`/`longitude`, `city`, `region`, and `country` and is used for server-side auto-detect.
  - MET & calories: `met_for_workspace_mode()` and `compute_calorie_burn()` compute MET values and calories burned using the clinical formula:

  $$\text{Calories Burned} = \frac{\text{MET} \times 3.5 \times \text{Weight (kg)} \times \text{Duration (minutes)}}{200}$$

  - Thermal fatigue multiplier: `compute_thermal_fatigue_multiplier(temperature_c, humidity_percent)` applies the implemented accumulation penalties: penalty = max(0, temperature - 30.0) * 0.03 and +0.05 when humidity > 70%. The final multiplier is truncated to two decimal places (test‑aligned behavior).
  - Error handling: The entire environmental flow is wrapped with try/except fallbacks; when API calls fail, conservative fallback values are written into `st.session_state.environment_metrics` and `st.session_state.thermal_fatigue_multiplier`.

## 5. Browser Geolocation + IP Fallback

- Implemented in [app.py](app.py):
  - Optional dependency: `streamlit-javascript==0.0.4` is pinned in `requirements.txt`.
  - Sidebar button: "Detect my location (browser)" uses `st_javascript` to run a short JS snippet calling `navigator.geolocation.getCurrentPosition()` and returns `{lat, lon}`.
  - Result handler: `handle_browser_location_result(loc)` applies the browser coordinates to `st.session_state.latitude/longitude`, sets `location_source='browser'`, and refreshes environmental metrics. If the result is missing or parsing fails, it falls back to `refresh_user_location()` (IP lookup).
  - Session-state flags: `auto_detect_location`, `location_auto_detected`, `location_label`, and `location_source` are used to show status and caption text in the sidebar.

## 6. Testing and CI

- Unit tests: Added and passing. Notable tests include:
  - `tests/test_environmental.py` — environmental calculations and fallback behavior.
  - `tests/test_projection.py` — PCA/UMAP projection checks.
  - `tests/test_geolocation.py` — tests for `handle_browser_location_result` (success and fallback mocked).
  - Full suite: `pytest -q` reports all tests passing in the maintained environment (29 passed).
- CI: GitHub Actions workflow present at `.github/workflows/ci.yml` to run tests on push.

## 7. Dependencies and Docs

- `requirements.txt` pins `streamlit==1.26.0`, `streamlit-javascript==0.0.4`, `streamlit-plotly-events==0.0.6`, `pytest`, `requests`, and `plotly`.
- `README.md` updated with a Browser Geolocation note, dependency guidance, HTTPS caveat, and privacy/fallback behavior.

## 8. Files of Interest (implementation locations)

- Main app: [app.py](app.py)
- Environmental engine: [environmental.py](environmental.py)
- Semantic engine: [semantic.py](semantic.py)
- Tests: `tests/` — e.g. [tests/test_geolocation.py](tests/test_geolocation.py), [tests/test_environmental.py](tests/test_environmental.py), [tests/test_projection.py](tests/test_projection.py)
- Requirements: [requirements.txt](requirements.txt)
- README: [README.md](README.md)

## 9. Background analysis (async) and Muscular Fatigue Index

- Background execution: `environmental.py` exports `analyze_environment_async()` which schedules the full fetch-and-analyze pipeline on a module-level `ThreadPoolExecutor` (implemented as `_EXECUTOR`). The Streamlit app (`app.py`) stores the returned `Future` in `st.session_state['env_future']` and polls it on reruns; results are applied when the Future completes.
- Muscular Fatigue Index (MFI): Implemented in `environmental.analyze_environment()` and computed as MET-hours (MET × hours) adjusted by the thermal fatigue multiplier. The result is rounded/truncated to two decimals and stored in `st.session_state['muscular_fatigue_index']` for UI display and short-term comparisons.
- UI behavior: environmental refresh actions submit background jobs so the UI remains responsive; failures fall back to conservative defaults and are surfaced via `st.session_state.environment_metrics`.

## 10. Changelog

- See [CHANGELOG.md](CHANGELOG.md) for the most recent commit-level notes.
