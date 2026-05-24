import json
from unittest.mock import patch, MagicMock

import pytest

from app import keyword_intent_extractor, ollama_intent_extractor


def test_keyword_intent_extractor_neck():
    res = keyword_intent_extractor("My neck has been sore since morning")
    assert res["pain_area"] == "neck"
    assert res["matched_keyword"] == "neck"


def test_keyword_intent_extractor_wrist():
    res = keyword_intent_extractor("I feel ulnar side soreness near my wrist when typing")
    assert res["pain_area"] == "wrist"
    assert res["matched_keyword"] in ("ulnar", "wrist")


def test_keyword_intent_extractor_environment():
    res = keyword_intent_extractor("The room is hot and humid, I feel exhausted")
    assert res["pain_area"] == "environment"


def test_keyword_intent_extractor_none():
    res = keyword_intent_extractor("Just saying hello, no pain")
    assert res["pain_area"] is None


@patch("subprocess.run")
def test_ollama_intent_extractor_valid_json(mock_run):
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = '{"pain_area":"wrist","matched_keyword":"ulnar"}\n'
    mock_run.return_value = proc

    res = ollama_intent_extractor("My wrist hurts", model="llama3.2:1b")
    assert res == {"pain_area": "wrist", "matched_keyword": "ulnar"}


@patch("subprocess.run", side_effect=FileNotFoundError)
def test_ollama_intent_extractor_missing_cli(mock_run):
    res = ollama_intent_extractor("test message")
    assert res is None


@patch("subprocess.run")
def test_ollama_intent_extractor_noisy_output(mock_run):
    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = 'Some informational banner\n{"pain_area":"neck","matched_keyword":"nape"}\n(EOF)'
    mock_run.return_value = proc

    res = ollama_intent_extractor("My neck hurts when I turn it", model="llama3.2:1b")
    assert res == {"pain_area": "neck", "matched_keyword": "nape"}