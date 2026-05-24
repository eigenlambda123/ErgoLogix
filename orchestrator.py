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

    The model is asked to return STRICT JSON with keys: `tools`, `params`, `assistant_response`.
    `tools` should be an ordered list of tool names to call (or an empty list), `params` is
    an object of shared extracted parameters, and `assistant_response` is a short string the
    assistant should present to the user. Backward-compatible single `tool` responses are
    normalized into a single-item `tools` list.
    """
    # Build a clear instruction prompting for JSON only
    safe_message = json.dumps(message)
    prompt = (
        "RESPOND WITH STRICT JSON ONLY. Return an object with keys: 'tools', 'params', and 'assistant_response'. "
        "'tools' must be an ordered array of tool function names to call. Valid tools are: 'process_environmental_metabolic_metrics', "
        "'process_wrist_assessment','process_posture_neck_metrics','process_lumbar_metrics','process_shoulder_assessment', "
        "'process_elbow_assessment','execute_semantic_search'. Use an empty array when no tool call is needed.\n\n"
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
    # Normalize keys (supports legacy single-tool responses)
    tools = []
    raw_tools = parsed.get('tools')
    if isinstance(raw_tools, list):
        for item in raw_tools:
            if isinstance(item, str) and item:
                tools.append(item)
            elif isinstance(item, dict):
                name = item.get('name')
                if isinstance(name, str) and name:
                    tools.append(name)

    if not tools:
        tool = parsed.get('tool')
        if isinstance(tool, str) and tool and tool != 'no_tool':
            tools = [tool]

    params = parsed.get('params') if isinstance(parsed.get('params'), dict) else {}
    assistant_response = parsed.get('assistant_response') or parsed.get('reply') or parsed.get('message')
    return {'tools': tools, 'params': params, 'assistant_response': assistant_response}
