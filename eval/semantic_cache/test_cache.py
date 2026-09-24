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


def test_cache_decision_header_on_hit_veto_and_miss(api):
    main, client, _ = api
    cache_store("wifi won't connect", _ok_response(), [])
    cache_store("enable dark mode", _ok_response(), [])

    hit = client.post("/v1/troubleshoot", json={"query": "wifi is not connecting"})
    veto = client.post("/v1/troubleshoot", json={"query": "disable dark mode"})
    miss = client.post("/v1/troubleshoot", json={"query": "camera photos are blurry"})

    assert hit.headers["X-Cache-Decision"].startswith("hit; sim=")
    assert hit.headers["X-Cache-Decision"].endswith("; matched=wifi won't connect")
    assert veto.headers["X-Cache-Decision"].startswith("veto_polarity; sim=")
    assert veto.headers["X-Cache-Decision"].endswith("; matched=enable dark mode")
    assert miss.headers["X-Cache-Decision"].startswith("miss_low_sim; sim=")
    assert "matched=" not in miss.headers["X-Cache-Decision"]
    assert "X-Cache-Decision" not in json.dumps(hit.json())  # body untouched


def test_cache_decision_header_survives_non_latin1_query(api):
    main, client, _ = api
    query = "wifi 📶 “won’t” connect 100%"
    cache_store(query, _ok_response(), [])
    r = client.post("/v1/troubleshoot", json={"query": query})
    assert r.status_code == 200
    assert r.headers["X-Cache-Decision"].startswith("hit; ")
    assert "100%25" in r.headers["X-Cache-Decision"]


def test_cache_decision_header_is_exposed_to_browsers(api):
    main, client, _ = api
    r = client.post("/v1/troubleshoot", json={"query": "anything"}, headers={"Origin": "http://localhost:3000"})
    assert "X-Cache-Decision" in r.headers.get("access-control-expose-headers", "")


def test_clarify_skips_cache_lookup(api):
    main, client, gemini_calls = api
    original = "screen gestures not responding"
    answer = "it started after the latest update"
    cache_store(original, _ok_response(), [])

    body = client.post(
        "/v1/clarify",
        json={"query": original, "hypotheses": [], "clarification_answer": answer},
    ).json()

    assert body["meta"]["cache_hit"] is False
    assert len(gemini_calls) == 1  # the pipeline re-ran on the clarified query
    assert cache_stats()["lookups"] == 0

    # Guard against this test passing vacuously: without the skip, the clarified
    # query really would be served the pre-clarification answer.
    _, info = cache_lookup(f"{original}. User clarification: {answer}.")  # main.clarify()'s format
    assert info["decision"] == "hit", info


def test_internal_call_without_flag_still_uses_cache(api):
    # /v1/troubleshoot-image calls troubleshoot() directly without skip_cache_lookup,
    # so it receives the Query() default object, which is truthy.
    main, _, gemini_calls = api
    cache_store("turn bluetooth on", _ok_response(), [])
    result = main.troubleshoot(main.TroubleshootRequest(query="turn bluetooth on"))
    assert result.meta.cache_hit is True
    assert gemini_calls == []


def test_live_sequence_where_gemini_failures_are_never_stored(monkeypatch):
    """
    Replays a live debugging session after a fresh restart. Gemini returned no_match
    (8s SLA fallback) for "enable dark mode" and "screen brightness is too high", so
    neither was stored. That is why "too low" missed at 0.39 (best match: "disable
    dark mode") instead of being vetoed against "too high" (0.944), and why the only
    veto was the second "enable dark mode" against the stored "disable dark mode".
    """
    import main
    from fastapi.testclient import TestClient
    from schema import Goal

    gemini_fails_for = {main.enrich_query(q) for q in ("enable dark mode", "screen brightness is too high")}

    def fake_extract_goals(query, **kwargs):
        if query in gemini_fails_for:
            return [], {"prompt_tokens": 0, "candidates_tokens": 0}
        return [Goal(**_GOAL)], {"prompt_tokens": 0, "candidates_tokens": 0}

    monkeypatch.setattr(main, "extract_goals", fake_extract_goals)
    client = TestClient(main.app)

    # (query, expected decision, expected similarity, expected best match)
    sequence = [
        ("wifi is not connecting", "miss_low_sim", 0.00, None),
        ("wifi is not connecting", "hit", 1.00, "wifi is not connecting"),
        ("wifi won't connect", "hit", 0.93, "wifi is not connecting"),
        ("enable dark mode", "miss_low_sim", 0.11, None),
        ("disable dark mode", "miss_low_sim", 0.07, None),
        ("enable dark mode", "veto_polarity", 0.94, "disable dark mode"),
        ("screen brightness is too high", "miss_low_sim", 0.37, None),
        ("screen brightness is too low", "miss_low_sim", 0.39, None),
    ]
    for query, decision, sim, matched in sequence:
        header = client.post("/v1/troubleshoot", json={"query": query}).headers["X-Cache-Decision"]
        expected = f"{decision}; sim={sim:.2f}" + (f"; matched={matched}" if matched else "")
        assert header == expected, (query, header)

    stats = cache_stats()
    assert {k: stats[k] for k in ("lookups", "hits", "misses", "veto_polarity", "entries", "stores", "stores_skipped")} == {
        "lookups": 8, "hits": 2, "misses": 5, "veto_polarity": 1, "entries": 3, "stores": 3, "stores_skipped": 3,
    }

    debug = cache.cache_debug()
    assert [e["query"] for e in debug["entries"]] == [
        "wifi is not connecting", "disable dark mode", "screen brightness is too low",
    ]
    skipped = [s["query"] for s in debug["recent_stores"] if s["result"].startswith("skipped")]
    assert skipped == ["enable dark mode", "enable dark mode", "screen brightness is too high"]
    vetoes = [(l["query"], l["matched_query"]) for l in debug["recent_lookups"] if l["decision"] == "veto_polarity"]
    assert vetoes == [("enable dark mode", "disable dark mode")]


def test_cache_entries_endpoint_is_gated_by_cache_debug(api, monkeypatch):
    main, client, _ = api
    cache_store("turn bluetooth on", _ok_response(), ["enable bluetooth"])

    monkeypatch.delenv("CACHE_DEBUG", raising=False)
    assert client.get("/v1/cache/entries").status_code == 404

    monkeypatch.setenv("CACHE_DEBUG", "true")
    body = client.get("/v1/cache/entries").json()
    assert body["entries"][0]["query"] == "turn bluetooth on"
    assert body["entries"][0]["variations"] == ["enable bluetooth"]
    assert body["recent_stores"][-1]["result"] == "stored (2 vectors)"
    assert "/v1/cache/entries" not in client.get("/openapi.json").json()["paths"]


def test_restoring_same_query_refreshes_without_deadlock():
    import threading

    def store_twice():
        cache_store("turn bluetooth on", _ok_response(), [])
        cache_store("Turn Bluetooth On", _ok_response(), [])

    t = threading.Thread(target=store_twice, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive(), "cache_store deadlocked on the refresh path"
    assert cache_stats()["entries"] == 1
    assert [s["result"] for s in cache.cache_debug()["recent_stores"]] == ["stored (1 vector)", "refreshed"]


# ---------------------------------------------------------------------------
# Self-critique (ENABLE_SELF_CRITIQUE) interaction
# ---------------------------------------------------------------------------

def _second_goal() -> dict:
    return {
        "goal": "Follow these steps to perform this Network Reset Troubleshooting",
        "title": "Network reset",
        "score": 0.8,
        "actions": [
            {
                "actionName": "Reset Network Settings",
                "description": "It will reset all network settings",
                "category": "critical",
                "stepGroups": [{"steps": ["Open Settings.", "Tap General management.", "Tap Reset."]}],
            }
        ],
    }


@pytest.fixture
def critique_api(monkeypatch):
    """Self-critique enabled, with a mock client (no network) so the critique branch can run."""
    from unittest.mock import MagicMock

    import main
    from fastapi.testclient import TestClient
    from schema import Goal

    calls = {"extract": 0, "critique": 0}

    def fake_extract_goals(**kwargs):
        calls["extract"] += 1
        return [Goal(**_GOAL), Goal(**_second_goal())], {"prompt_tokens": 0, "candidates_tokens": 0}

    def fake_critique(goals, query, **kwargs):
        calls["critique"] += 1
        # Keep the Bluetooth plan at half relevance, reject the network reset.
        return [(True, 0.5), (False, 0.1)], {"prompt_tokens": 0, "candidates_tokens": 0}

    monkeypatch.setenv("ENABLE_SELF_CRITIQUE", "true")
    monkeypatch.setattr(main, "gemini_client", MagicMock(name="mock_gemini_client"))
    monkeypatch.setattr(main, "extract_goals", fake_extract_goals)
    monkeypatch.setattr(main, "critique_goals_combined", fake_critique)
    return main, TestClient(main.app), calls


def test_cache_hit_returns_before_self_critique(critique_api):
    main, client, calls = critique_api
    cache_store("turn bluetooth on", _ok_response(), [])

    body = client.post("/v1/troubleshoot", json={"query": "turn bluetooth on"}).json()

    assert body["meta"]["cache_hit"] is True
    assert calls == {"extract": 0, "critique": 0}


def test_cache_miss_stores_the_post_critique_response(critique_api):
    main, client, calls = critique_api

    first = client.post("/v1/troubleshoot", json={"query": "turn bluetooth on"}).json()

    assert calls == {"extract": 1, "critique": 1}  # the critique branch really ran
    assert [g["title"] for g in first["contexts"]] == ["Bluetooth settings"]  # rejected plan dropped

    stored, info = cache_lookup("turn bluetooth on")
    assert info["decision"] == "hit"
    assert stored["contexts"] == first["contexts"]  # final response, incl. critique-scaled score

    second = client.post("/v1/troubleshoot", json={"query": "turn bluetooth on"}).json()
    assert second["meta"]["cache_hit"] is True
    assert second["contexts"] == first["contexts"]
    assert calls == {"extract": 1, "critique": 1}  # the hit made no further calls


def test_cache_stats_endpoint(api):
    main, client, _ = api
    body = client.get("/v1/cache/stats").json()
    for key in ("hits", "misses", "veto_polarity", "veto_entity", "veto_domain", "hit_rate", "avg_lookup_ms", "entries"):
        assert key in body
