# Release / Packaging Notes

## Current packaging shape

ErgoLogix is packaged as a source-based Streamlit app.
There is no wheel, installer, or PyPI package yet.

## What to include in a release

- `app.py`
- `semantic.py`
- `scripts/sync_kb.py`
- `kb/` markdown docs
- `requirements.txt`
- `.github/workflows/ci.yml`
- `README.md`

## Release checklist

1. Run the test suite.
2. Rebuild the KB cache if KB files changed.
3. Verify the Streamlit app starts locally.
4. Commit the KB cache only if you want a prebuilt snapshot.
5. Tag the release in git.

Example:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## Packaging options

### Option 1: Source release

Best for now.
Distribute the repository plus `requirements.txt`.
Users install dependencies and run:

```bash
streamlit run app.py
```

### Option 2: Frozen dependency snapshot

If you want reproducible installs, generate a lock snapshot:

```bash
pip freeze > requirements-lock.txt
```

### Option 3: Python package later

If the project grows, add a `pyproject.toml` and package metadata, then ship a wheel.

## Windows notes

Use PowerShell to create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Suggested versioning

Use semantic versioning:

- `0.1.0` for the current prototype
- `0.2.0` when packaging or UI automation improves
- `1.0.0` only when the app is stable and release-ready
