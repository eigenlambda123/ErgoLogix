import subprocess
import json
import re
from typing import Optional, Dict, Any

try:
    import requests
except Exception:
    requests = None


def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r'(\{[\s\S]*\})', text)
    candidate = m.group(1) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _call_ollama_cli(prompt: str, model: str = 'ergo-orchestrator') -> Optional[str]:
    cmds = [
        ['ollama', 'run', model, prompt],
        ['ollama', 'query', model, prompt],
        ['ollama', 'generate', model, prompt],
    ]
    for cmd in cmds:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=12)
        except FileNotFoundError:
            return None
        except Exception:
            continue
        if proc.returncode != 0:
            continue
        out = proc.stdout.strip()
        if out:
            return out
    return None


def _call_ollama_http(prompt: str, model: str = 'ergo-orchestrator', host: str = 'http://127.0.0.1:11434') -> Optional[str]:
    if requests is None:
        return None
    url = host.rstrip('/') + '/api/generate'
    try:
        resp = requests.post(url, json={'model': model, 'prompt': prompt}, timeout=10)
    except Exception:
        return None
    try:
        return resp.text or json.dumps(resp.json())
    except Exception:
        return None


def llm_orchestrate(message: str, model: str = 'ergo-orchestrator', host: str = 'http://127.0.0.1:11434') -> Optional[Dict[str, Any]]:
    """Ask the LLM which tool to call and with what params.

    The model is asked to return STRICT JSON with keys: `tool`, `params`, `assistant_response`.
    `tool` should be the tool function name to call (or "no_tool"), `params` is an object,
    and `assistant_response` is a short string the assistant should present to the user. If
    the LLM is not available, return None so the caller can fallback to keyword routing.
    """
    # Build a clear instruction prompting for JSON only
    safe_message = json.dumps(message)
    prompt = (
        "RESPOND WITH STRICT JSON ONLY. Return an object with keys: 'tool', 'params', and 'assistant_response'. "
        "'tool' must be the name of a tool function to call (one of: 'process_environmental_metabolic_metrics', "
        "'process_wrist_assessment','process_posture_neck_metrics','process_lumbar_metrics','process_shoulder_assessment', "
        "'process_elbow_assessment') or the string 'no_tool' if no external tool should be called.\n\n"
        f"User message: {safe_message}\n\nReturn JSON ONLY."
    )

    out = _call_ollama_cli(prompt, model=model)
    if out is None:
        out = _call_ollama_http(prompt, model=model, host=host)
    if out is None:
        return None

    parsed = _extract_json(out)
    if not parsed:
        return None
    # Normalize keys
    tool = parsed.get('tool')
    params = parsed.get('params') if isinstance(parsed.get('params'), dict) else {}
    assistant_response = parsed.get('assistant_response') or parsed.get('reply') or parsed.get('message')
    return {'tool': tool, 'params': params, 'assistant_response': assistant_response}
