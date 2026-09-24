"""
Semantic cache for Diagnos AI (Person B).

- Embeddings: all-MiniLM-L6-v2 via fastembed (ONNX, no torch), loaded once, lazily.
- Storage: in-memory numpy matrix guarded by a lock. The original query and each
  query variation get their own row, all pointing back to one entry.
- Lookup: best cosine match, then a deterministic rule-based VETO (polarity /
  entity / domain) so "turn bluetooth on" never serves "turn bluetooth off".

Every public function swallows its own errors: the cache must never break a request.
"""

import logging
import os
import re
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("diagnos_ai.cache")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
# Picked with eval/semantic_cache/tune_cache.py: best accuracy with zero wrong serves.
DEFAULT_THRESHOLD = 0.75
# Keep the model next to the app instead of the OS temp dir, which free tiers wipe.
MODEL_CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".fastembed_cache"
)
# Bounded so the matrix fits a 512MB box: 2000 entries x ~11 vectors x 384 x 4B ~= 34MB worst case.
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "2000"))
# After a failed model load, don't retry on every /health probe.
_LOAD_RETRY_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Embedding model (lazy, loaded once)
# ---------------------------------------------------------------------------

_model: Any = None
_model_lock = threading.Lock()
_model_last_failure: float = 0.0


def _get_model() -> Any:
    global _model, _model_last_failure
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if _model_last_failure and time.monotonic() - _model_last_failure < _LOAD_RETRY_SECONDS:
            raise RuntimeError("embedding model load failed recently; backing off")
        try:
            from fastembed import TextEmbedding

            _model = TextEmbedding(model_name=MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
            _model_last_failure = 0.0
            logger.info("Semantic cache model loaded: %s", MODEL_NAME)
        except Exception:
            _model_last_failure = time.monotonic()
            raise
    return _model


def _embed(texts: List[str]) -> np.ndarray:
    """Returns an (n, 384) float32 matrix of L2-normalised embeddings."""
    vectors = np.asarray(list(_get_model().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def is_cache_ready() -> bool:
    """True once the embedding model is usable. Never raises."""
    try:
        return _get_model() is not None
    except Exception as e:
        logger.warning("Semantic cache not ready: %s", e)
        return False


# ---------------------------------------------------------------------------
# In-memory storage
# ---------------------------------------------------------------------------

_store_lock = threading.Lock()
_matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
_row_entry: List[int] = []  # row index -> entry id
_row_text: List[str] = []  # row index -> the text that was embedded (query or variation)
_entries: Dict[int, Dict[str, Any]] = {}  # entry id -> entry
_query_index: Dict[str, int] = {}  # normalised query -> entry id
_next_entry_id = 0

_stats: Dict[str, Any] = {}


def _reset_stats() -> None:
    _stats.clear()
    _stats.update({
        "lookups": 0,
        "hits": 0,
        "miss_low_sim": 0,
        "veto_polarity": 0,
        "veto_entity": 0,
        "veto_domain": 0,
        "errors": 0,
        "total_lookup_ms": 0.0,
        "stores": 0,
        "stores_skipped": 0,
        "evictions": 0,
    })


_reset_stats()

# Recent activity for cache_debug(): what was looked up / stored, and why not.
_RECENT_LOG_SIZE = 50
_recent_lookups: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_LOG_SIZE)
_recent_stores: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_LOG_SIZE)


def _log_store(query: Any, result: str) -> None:
    with _store_lock:
        _recent_stores.append({"at": time.time(), "query": query, "result": result})


def cache_clear() -> None:
    """Drops every entry and resets stats. Intended for tests and tuning."""
    global _matrix, _row_entry, _row_text, _entries, _query_index, _next_entry_id
    with _store_lock:
        _matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
        _row_entry, _row_text = [], []
        _entries, _query_index = {}, {}
        _next_entry_id = 0
        _reset_stats()
        _recent_lookups.clear()
        _recent_stores.clear()


def _norm_key(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _response_parts(response: Any) -> Tuple[Dict[str, Any], List[Any], Optional[str], Optional[str]]:
    """Returns (dump, contexts, fallback, error) for either response shape, model or dict."""
    if hasattr(response, "model_dump"):
        dump = response.model_dump(mode="json")
    elif isinstance(response, dict):
        dump = response
    else:
        raise TypeError(f"unsupported response type: {type(response).__name__}")
    inner = dump.get("response") if isinstance(dump.get("response"), dict) else dump
    return dump, inner.get("contexts") or [], inner.get("fallback"), inner.get("error")


def _evict_oldest_locked() -> None:
    global _matrix, _row_entry, _row_text
    oldest = min(_entries)
    entry = _entries.pop(oldest)
    _query_index.pop(_norm_key(entry["query"]), None)
    keep = [i for i, eid in enumerate(_row_entry) if eid != oldest]
    _matrix = _matrix[keep]
    _row_entry = [_row_entry[i] for i in keep]
    _row_text = [_row_text[i] for i in keep]
    _stats["evictions"] += 1


def cache_store(query: str, response: Any, variations: List[str]) -> None:
    """
    Stores a successful response under the query and each of its variations.
    Skips no_match / empty / error responses so a transient Gemini failure is never cached.
    Never raises.
    """
    global _matrix, _next_entry_id
    try:
        if not query or not query.strip():
            return
        dump, contexts, fallback, error = _response_parts(response)
        if fallback == "no_match" or not contexts or error:
            with _store_lock:
                _stats["stores_skipped"] += 1
            _log_store(query, f"skipped: {'fallback=' + fallback if fallback else 'error=' + error if error else 'empty contexts'}")
            return

        key = _norm_key(query)
        texts = [query.strip()]
        seen = {key}
        for v in variations or []:
            if isinstance(v, str) and v.strip() and _norm_key(v) not in seen:
                seen.add(_norm_key(v))
                texts.append(v.strip())

        entry = {
            "query": query.strip(),
            "response": dump,
            "variations": texts[1:],
            "features": extract_features(query),
            "stored_at": time.time(),
        }

        with _store_lock:
            existing = _query_index.get(key)
            if existing is not None:
                # Same query again: refresh the response, keep the existing vectors.
                _entries[existing].update(response=dump, stored_at=entry["stored_at"])
                _stats["stores"] += 1
        if existing is not None:
            _log_store(query, "refreshed")
            return

        # Embed outside the lock; onnxruntime inference is thread-safe.
        vectors = _embed(texts)

        with _store_lock:
            if key in _query_index:
                return
            while _entries and len(_entries) >= MAX_ENTRIES:
                _evict_oldest_locked()
            entry_id = _next_entry_id
            _next_entry_id += 1
            _entries[entry_id] = entry
            _query_index[key] = entry_id
            _matrix = np.vstack([_matrix, vectors])
            _row_entry.extend([entry_id] * len(texts))
            _row_text.extend(texts)
            _stats["stores"] += 1
        _log_store(query, f"stored ({len(texts)} vector{'s' if len(texts) != 1 else ''})")
    except Exception as e:
        logger.warning("cache_store failed (ignored): %s", e)
        try:
            _log_store(query if isinstance(query, str) else repr(query), f"failed: {e}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Rule-based veto features (deterministic, no LLM)
# ---------------------------------------------------------------------------

_NEGATIONS = {
    "not", "no", "never", "without", "stop", "stops", "stopped", "stopping",
    "won't", "wont", "can't", "cant", "cannot", "unable", "doesn't", "doesnt",
    "isn't", "isnt", "don't", "dont", "didn't", "didnt", "aren't", "arent",
}
_NEGATION_WINDOW = 3  # tokens before a polarity cue that can flip it

# Multi-word names collapsed to one token first, so e.g. the "on" in
# "always on display" or the "dark" in "dark mode" is not read as a polarity cue.
_PHRASE_TOKENS = [
    (r"\balways[\s-]*on(?:\s+display)?\b", "aod"),
    (r"\b(?:dark|night)\s+(?:mode|theme)\b", "darkmode"),
    (r"\bbattery\s+saver\b|\bpower\s+sav(?:ing|er)(?:\s+mode)?\b|\blow\s+power\s+mode\b", "batterysaver"),
    (r"\bscreen\s+time\s*-?\s*out\b|\bscreen\s+(?:turns|goes|shuts)\s+off\s+(?:too\s+)?(?:fast|quickly|quick|soon)\b", "screentimeout"),
    (r"\bwi\s*-?\s*fi\b|\bwify\b", "wifi"),
    (r"\bhot\s+spot\b", "hotspot"),
]

_WORD = r"(?:(?!on\b|off\b|up\b|down\b)[a-z0-9'-]+\s+)"
# axis -> list of (regex, sign)
_POLARITY_RULES: Dict[str, List[Tuple[str, int]]] = {
    "power": [
        (rf"\b(?:turn|turns|turned|turning|switch|switches|switched|switching)\s+{_WORD}{{0,3}}?on\b", +1),
        (rf"\b(?:turn|turns|turned|turning|switch|switches|switched|switching)\s+{_WORD}{{0,3}}?off\b", -1),
        (r"\b(?:is|are|was|stays|stay|stuck|keeps)\s+on\b", +1),
        (r"\b(?:is|are|was|stays|stay|stuck|keeps)\s+off\b", -1),
        (r"\b(?:enable|enables|enabled|enabling|activate|activates|activated|activating)\b", +1),
        (r"\b(?:disable|disables|disabled|disabling|deactivate|deactivates|deactivated|deactivating)\b", -1),
        (r"\b(?:connect|connects|connected|connecting|pair|pairs|paired|pairing)\b", +1),
        (r"\b(?:disconnect|disconnects|disconnected|disconnecting|unpair|unpaired|shuts?\s+(?:off|down)|shutting\s+(?:off|down))\b", -1),
    ],
    "level": [
        (r"\b(?:increase|increases|increased|increasing|raise|higher|brighter|louder)\b", +1),
        (r"\b(?:decrease|decreases|decreased|decreasing|lower|dimmer|quieter)\b", -1),
        (r"\b(?:too|very|so|really|super)\s+(?:high|bright|loud)\b", +1),
        (r"\b(?:too|very|so|really|super)\s+(?:low|dim|dark|quiet)\b", -1),
        (r"\b(?:dim|dims|dimmed|dimming|dark)\b", -1),
        (rf"\bturn(?:s|ed|ing)?\s+{_WORD}{{0,3}}?up\b", +1),
        (rf"\bturn(?:s|ed|ing)?\s+{_WORD}{{0,3}}?down\b", -1),
    ],
    "speed": [
        (r"\b(?:fast|faster|quick|quickly|rapidly)\b", +1),
        (r"\b(?:slow|slower|slowly|sluggish|lag|lags|laggy|lagging|forever)\b", -1),
    ],
}

_ENTITY_RULES: Dict[str, str] = {
    "bluetooth": r"\b(?:bluetooth|blutooth|bluetoth|bt)\b",
    "wifi": r"\b(?:wifi|wlan|wireless)\b",
    "mobile data": r"\b(?:mobile\s+data|cellular|cell\s+data|data|lte|5g|4g)\b",
    "hotspot": r"\b(?:hotspot|tether\w*)\b",
    "nfc": r"\b(?:nfc|contactless|tap\s+to\s+pay)\b",
    "location": r"\b(?:gps|location)\b",
    "brightness": r"\b(?:brightness|bright|brighter|dim|dims|dimmer|dimmed|dark)\b",
    "dark mode": r"\bdarkmode\b",
    "refresh rate": r"\b(?:refresh\s+rate|\d+\s*hz|hz)\b",
    "screen timeout": r"\b(?:screentimeout|timeout|auto\s*-?lock)\b",
    "always on display": r"\baod\b",
    "battery saver": r"\bbatterysaver\b",
    "charging": r"\b(?:charg\w*)\b",
    "camera": r"\b(?:camera|cam|selfie)\b",
    "flash": r"\b(?:flash|flashlight|torch)\b",
    "storage": r"\b(?:storage|space|disk)\b",
    "memory": r"\b(?:ram|memory)\b",
    "apps": r"\b(?:apps?|application\w*)\b",
    "notifications": r"\b(?:notification\w*|notifs?)\b",
    "volume": r"\b(?:volume|sound|audio|speaker|loud\w*|quiet\w*|ringtone|ringer)\b",
}

_DOMAIN_RULES: Dict[str, str] = {
    "battery": r"\b(?:battery|battry|batery|drain\w*|charg\w*|batterysaver|dies|dying)\b",
    "display": r"\b(?:screen|display|brightness|bright\w*|dim\w*|dark|darkmode|refresh|hz|aod|screentimeout)\b",
    "camera": r"\b(?:camera|cam|selfie|photo\w*|picture\w*|flash\w*|torch|lens)\b",
    "performance": r"\b(?:lag\w*|freez\w*|froze|hang\w*|stutter\w*|sluggish|ram|memory|performance|crash\w*)\b",
}


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s-]", " ", text)
    for pattern, token in _PHRASE_TOKENS:
        text = re.sub(pattern, token, text)
    return re.sub(r"\s+", " ", text).strip()


def extract_features(text: str) -> Dict[str, Any]:
    """
    polarity: {axis: +1 | -1} (axes with mixed signals are dropped)
    entities: sorted list of target entities
    domains:  sorted list of battery / display / camera / performance
    """
    norm = _normalise(text or "")

    polarity: Dict[str, int] = {}
    for axis, rules in _POLARITY_RULES.items():
        signs: Set[int] = set()
        for pattern, sign in rules:
            for m in re.finditer(pattern, norm):
                preceding = norm[: m.start()].split()[-_NEGATION_WINDOW:]
                signs.add(-sign if any(w in _NEGATIONS for w in preceding) else sign)
        if len(signs) == 1:
            polarity[axis] = signs.pop()

    entities = {name for name, pattern in _ENTITY_RULES.items() if re.search(pattern, norm)}
    if "dark mode" in entities:
        entities.discard("brightness")
    if len(entities) > 1:
        # "camera app crashes" is about the camera; "apps" only counts on its own.
        entities.discard("apps")

    domains = {name for name, pattern in _DOMAIN_RULES.items() if re.search(pattern, norm)}

    return {"polarity": polarity, "entities": sorted(entities), "domains": sorted(domains)}


def veto_reason(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[str]:
    """Returns 'veto_polarity' / 'veto_entity' / 'veto_domain', or None if the match may be served."""
    for axis, sign in a["polarity"].items():
        if axis in b["polarity"] and b["polarity"][axis] != sign:
            return "veto_polarity"
    if set(a["entities"]) != set(b["entities"]):
        return "veto_entity"
    if a["domains"] and b["domains"] and not set(a["domains"]) & set(b["domains"]):
        return "veto_domain"
    return None


# ---------------------------------------------------------------------------
# Lookup & stats
# ---------------------------------------------------------------------------

def _threshold() -> float:
    try:
        return float(os.getenv("CACHE_SIM_THRESHOLD", str(DEFAULT_THRESHOLD)))
    except ValueError:
        return DEFAULT_THRESHOLD


def _record(info: Dict[str, Any], query: Any) -> None:
    with _store_lock:
        _stats["lookups"] += 1
        key = {"hit": "hits", "error": "errors"}.get(info["decision"], info["decision"])
        if key in _stats:
            _stats[key] += 1
        _stats["total_lookup_ms"] += info.get("lookup_ms", 0.0)
        _recent_lookups.append({
            "at": time.time(),
            "query": query,
            "decision": info["decision"],
            "similarity": info.get("similarity"),
            "matched_query": info.get("matched_query"),
        })


def cache_lookup(query: str, threshold: Optional[float] = None) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (response_dict, info) on a hit, (None, info) otherwise.
    info = {decision, similarity, matched_query, lookup_ms, threshold, ...}
    decision is one of: hit | miss_low_sim | veto_polarity | veto_entity | veto_domain | error.
    Never raises.
    """
    start = time.perf_counter()
    thr = _threshold() if threshold is None else threshold
    info: Dict[str, Any] = {"decision": "miss_low_sim", "similarity": 0.0, "matched_query": None, "threshold": thr}
    try:
        with _store_lock:
            empty = _matrix.shape[0] == 0
        if not query or not query.strip() or empty:
            info["lookup_ms"] = round((time.perf_counter() - start) * 1000, 2)
            _record(info, query)
            return None, info

        q_vec = _embed([query.strip()])[0]
        with _store_lock:
            sims = _matrix @ q_vec
            best = int(np.argmax(sims))
            similarity = float(sims[best])
            entry = _entries[_row_entry[best]]
            matched_text = _row_text[best]

        info.update(
            similarity=round(similarity, 4),
            matched_query=entry["query"],
            matched_text=matched_text,
        )

        response: Optional[Dict[str, Any]] = None
        if similarity >= thr:
            q_feat = extract_features(query)
            reason = veto_reason(q_feat, entry["features"])
            if reason:
                info.update(decision=reason, query_features=q_feat, matched_features=entry["features"])
            else:
                info["decision"] = "hit"
                info["variations"] = list(entry["variations"])
                response = entry["response"]

        info["lookup_ms"] = round((time.perf_counter() - start) * 1000, 2)
        _record(info, query)
        return response, info
    except Exception as e:
        logger.warning("cache_lookup failed (treated as miss): %s", e)
        info.update(decision="error", error=str(e), lookup_ms=round((time.perf_counter() - start) * 1000, 2))
        try:
            _record(info, query)
        except Exception:
            pass
        return None, info


def cache_stats() -> Dict[str, Any]:
    """Counters for hits, misses and each veto type, plus hit_rate, avg_lookup_ms and entry count."""
    try:
        with _store_lock:
            s = dict(_stats)
            entries, vectors = len(_entries), int(_matrix.shape[0])
        lookups = s["lookups"]
        return {
            "lookups": lookups,
            "hits": s["hits"],
            "misses": s["miss_low_sim"],
            "veto_polarity": s["veto_polarity"],
            "veto_entity": s["veto_entity"],
            "veto_domain": s["veto_domain"],
            "errors": s["errors"],
            "hit_rate": round(s["hits"] / lookups, 4) if lookups else 0.0,
            "avg_lookup_ms": round(s["total_lookup_ms"] / lookups, 2) if lookups else 0.0,
            "entries": entries,
            "vectors": vectors,
            "stores": s["stores"],
            "stores_skipped": s["stores_skipped"],
            "evictions": s["evictions"],
            "threshold": _threshold(),
            "model_loaded": _model is not None,
        }
    except Exception as e:
        return {"error": str(e)}


def cache_debug() -> Dict[str, Any]:
    """
    What is actually cached, plus the last 50 lookups and store attempts (with skip reasons).
    Contains users' raw queries: only expose behind a debug flag.
    """
    try:
        with _store_lock:
            entries = [
                {
                    "id": eid,
                    "query": e["query"],
                    "variations": list(e["variations"]),
                    "features": e["features"],
                    "stored_at": e["stored_at"],
                }
                for eid, e in sorted(_entries.items())
            ]
            return {
                "entries": entries,
                "recent_lookups": list(_recent_lookups),
                "recent_stores": list(_recent_stores),
            }
    except Exception as e:
        return {"error": str(e)}


__all__ = [
    "cache_store",
    "cache_lookup",
    "cache_stats",
    "cache_debug",
    "is_cache_ready",
    "cache_clear",
    "extract_features",
    "veto_reason",
]
