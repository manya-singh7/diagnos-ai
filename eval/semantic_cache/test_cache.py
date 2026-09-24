"""
Tests for Person B's semantic cache (backend/cache).

No Gemini and no network, apart from the one-time MiniLM model download.

Verifies:
1. The veto layer classifies every labelled pair in cache_pairs.json correctly.
   Pairs are checked at threshold 0 so every pair reaches the veto; the
   similarity threshold itself is tuned separately with eval/tune_cache.py.
2. no_match / empty / error responses are never stored.
3. Storing a query and looking it up again is a hit.
4. Lookup on an empty cache is a clean miss.
5. The cache never raises, even on bad input or an embedding failure.
6. /v1/troubleshoot serves hits in both RESPONSE_SHAPEs without calling Gemini.
"""

import json
import sys
from pathlib import Path

import pytest

backend_dir = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(backend_dir))

import cache
from cache import cache_clear, cache_lookup, cache_stats, cache_store, is_cache_ready
from schema import AppendixBResponse, ContextDeeplinkResponse, ResponseMeta

PAIRS = json.loads((Path(__file__).resolve().parent / "cache_pairs.json").read_text())

_GOAL = {
    "goal": "Follow these steps to perform this Bluetooth Troubleshooting",
    "title": "Bluetooth settings",
    "score": 0.9,
    "actions": [
        {
            "actionName": "Open Bluetooth Settings",
            "description": "It will open Bluetooth settings",
            "category": "auto",
            "stepGroups": [{"steps": ["Open Settings.", "Tap Connections.", "Tap Bluetooth."]}],
        }
    ],
}


def _meta() -> ResponseMeta:
    return ResponseMeta(latency_ms=900, cache_hit=False, model="gemini-3.6-flash", cost_usd=0.0003)


def _ok_response() -> ContextDeeplinkResponse:
    return ContextDeeplinkResponse(contexts=[_GOAL], fallback=None, meta=_meta())


@pytest.fixture(autouse=True)
def _fresh_cache():
    cache_clear()
    yield
    cache_clear()


# ---------------------------------------------------------------------------
# Readiness & labelled pairs
# ---------------------------------------------------------------------------

def test_is_cache_ready():
    assert is_cache_ready() is True


def test_pair_file_has_enough_pairs():
    assert len(PAIRS) >= 40
    assert all(p["expected"] in ("hit", "veto") for p in PAIRS)


@pytest.mark.parametrize("pair", PAIRS, ids=[f"{p['a']} | {p['b']}" for p in PAIRS])
def test_veto_layer_on_labelled_pairs(pair):
    cache_store(pair["a"], _ok_response(), [])
    response, info = cache_lookup(pair["b"], threshold=0.0)
    if pair["expected"] == "hit":
        assert info["decision"] == "hit", info
        assert response is not None
    else:
        assert info["decision"].startswith("veto_"), info
        assert response is None


# ---------------------------------------------------------------------------
# Store rules
# ---------------------------------------------------------------------------

def test_no_match_response_is_never_stored():
    no_match = ContextDeeplinkResponse(contexts=[], fallback="no_match", meta=_meta())
    cache_store("turn bluetooth on", no_match, ["enable bluetooth", "switch bluetooth on"])
    response, info = cache_lookup("turn bluetooth on", threshold=0.0)
    assert response is None
    assert cache_stats()["entries"] == 0
    assert cache_stats()["stores_skipped"] == 1


def test_no_match_appendix_b_response_is_never_stored():
    no_match = AppendixBResponse(
        query="turn bluetooth on",
        query_variations=[],
        response={"contexts": [], "fallback": "no_match"},
        meta=_meta(),
    )
    cache_store("turn bluetooth on", no_match, [])
    assert cache_stats()["entries"] == 0


def test_empty_contexts_are_never_stored():
    cache_store("turn bluetooth on", ContextDeeplinkResponse(contexts=[], fallback=None, meta=_meta()), [])
    assert cache_stats()["entries"] == 0


def test_store_then_exact_lookup_is_hit():
    cache_store("turn bluetooth on", _ok_response(), ["enable bluetooth"])
    response, info = cache_lookup("turn bluetooth on")
    assert info["decision"] == "hit"
    assert info["similarity"] >= 0.999
    assert info["matched_query"] == "turn bluetooth on"
    assert response["contexts"][0]["title"] == "Bluetooth settings"


def test_variation_vector_points_back_to_original_entry():
    cache_store("turn bluetooth on", _ok_response(), ["please enable bluetooth"])
    response, info = cache_lookup("please enable bluetooth")
    assert info["decision"] == "hit"
    assert info["matched_query"] == "turn bluetooth on"
    assert cache_stats()["entries"] == 1
    assert cache_stats()["vectors"] == 2


def test_empty_cache_lookup_is_miss():
    response, info = cache_lookup("turn bluetooth on")
    assert response is None
    assert info["decision"] == "miss_low_sim"
    assert "lookup_ms" in info


def test_opposite_polarity_above_threshold_is_vetoed():
    cache_store("turn bluetooth on", _ok_response(), [])
    response, info = cache_lookup("turn bluetooth off")
    assert response is None
    assert info["decision"] in ("veto_polarity", "miss_low_sim")


# ---------------------------------------------------------------------------
# Never breaks the request
# ---------------------------------------------------------------------------

def test_store_never_raises_on_bad_input():
    cache_store(None, object(), None)
    cache_store("", _ok_response(), [])
    cache_store("q", {"not": "a response"}, None)
    assert cache_stats()["entries"] == 0


def test_lookup_error_returns_error_decision(monkeypatch):
    cache_store("turn bluetooth on", _ok_response(), [])

    def boom(texts):
        raise RuntimeError("onnx exploded")

    monkeypatch.setattr(cache, "_embed", boom)
    response, info = cache_lookup("turn bluetooth on")
    assert response is None
    assert info["decision"] == "error"
    assert cache_stats()["errors"] == 1


def test_stats_counts():
    cache_store("turn bluetooth on", _ok_response(), [])
    cache_lookup("turn bluetooth on")
    cache_lookup("turn bluetooth off", threshold=0.0)
    cache_lookup("camera photos are blurry", threshold=0.99)
    stats = cache_stats()
    assert stats["lookups"] == 3
    assert stats["hits"] == 1
    assert stats["veto_polarity"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == round(1 / 3, 4)
    assert stats["entries"] == 1


# ---------------------------------------------------------------------------
# main.py integration (Gemini replaced with a spy)
# ---------------------------------------------------------------------------

@pytest.fixture
def api(monkeypatch):
    import main
    from fastapi.testclient import TestClient

    gemini_calls = []

    def fake_extract_goals(**kwargs):
        gemini_calls.append(kwargs)
        return [], {"prompt_tokens": 0, "candidates_tokens": 0}

    monkeypatch.setattr(main, "extract_goals", fake_extract_goals)
    return main, TestClient(main.app), gemini_calls


@pytest.mark.parametrize("shape", ["flat", "appendix_b"])
def test_troubleshoot_serves_hit_without_gemini(api, monkeypatch, shape):
    main, client, gemini_calls = api
    monkeypatch.setattr(main, "RESPONSE_SHAPE", shape)
    cache_store("turn bluetooth on", _ok_response(), ["enable bluetooth"])

    body = client.post("/v1/troubleshoot", json={"query": "turn bluetooth on"}).json()

    assert gemini_calls == []
    assert body["meta"]["cache_hit"] is True
    assert body["meta"]["cost_usd"] == 0.0
    assert body["meta"]["latency_ms"] < 900  # real elapsed time, not the stored 900
    contexts = body["contexts"] if shape == "flat" else body["response"]["contexts"]
    assert contexts[0]["title"] == "Bluetooth settings"
    if shape == "appendix_b":
        assert body["query"] == "turn bluetooth on"


def test_troubleshoot_vetoed_query_goes_to_gemini_and_no_match_is_not_cached(api):
    main, client, gemini_calls = api
    cache_store("turn bluetooth on", _ok_response(), [])

    body = client.post("/v1/troubleshoot", json={"query": "turn bluetooth off"}).json()

    assert len(gemini_calls) == 1
    assert body["meta"]["cache_hit"] is False
    assert body["fallback"] == "no_match"
    assert cache_stats()["entries"] == 1  # the no_match was not added


def test_cache_stats_endpoint(api):
    main, client, _ = api
    body = client.get("/v1/cache/stats").json()
    for key in ("hits", "misses", "veto_polarity", "veto_entity", "veto_domain", "hit_rate", "avg_lookup_ms", "entries"):
        assert key in body
