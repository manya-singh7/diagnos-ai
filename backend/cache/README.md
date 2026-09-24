# Semantic cache

Serves a stored `/v1/troubleshoot` response when a new query means the same thing as one already answered, skipping Gemini. A hit returns in about 1–2 ms with `meta.cache_hit=true` and `meta.cost_usd=0.0`.

Embeddings come from `all-MiniLM-L6-v2` via fastembed (ONNX, no torch). Storage is an in-memory numpy matrix. The original query and each of its variations get their own vector, and all of them point back to one entry.

## How a lookup decides

1. Embed the query and take the best cosine match across all stored vectors.
2. If the similarity is below `CACHE_SIM_THRESHOLD` (default **0.75**), it's a **miss**.
3. Otherwise run the **veto**, a set of deterministic rules with no LLM. The same features are extracted from the new query and the matched entry's original query:
   - **Polarity**, on three axes: power (on/off, enable/disable, connect/disconnect), level (high/low, brighter/dimmer, louder/quieter), speed (fast/slow). A negation (`not`, `won't`, `can't`, `stop`, …) within the 3 words before a cue flips it. The veto fires if both sides have a sign on the same axis and the signs differ.
   - **Entity**: bluetooth, wifi, mobile data, hotspot, nfc, location, brightness, dark mode, refresh rate, screen timeout, AOD, battery saver, charging, camera, flash, storage, memory, apps, notifications, volume. The veto fires if the two sets differ. `apps` is ignored when a more specific entity is present.
   - **Domain** (battery / display / camera / performance): the veto fires only if both sides have domains and they don't overlap.

The veto is strict on purpose. A false veto costs one Gemini call. A false hit shows the user the wrong fix.

Responses with `fallback: "no_match"` or empty `contexts` are never stored, so a transient Gemini failure can't be cached. Every function catches its own errors: a cache failure turns into a miss and never breaks a request.

## Why 0.75

MiniLM scores real paraphrases lower than you'd expect. "screen too dim" vs "display is too dark" is only 0.62. Opposite meanings can score very high: "enable dark mode" vs "disable dark mode" is 0.94. A threshold on its own can't separate these. The veto makes a low threshold safe.

From `python eval/semantic_cache/tune_cache.py` (50 labelled pairs in `eval/semantic_cache/cache_pairs.json`: 22 should hit, 28 should not):

| threshold | accuracy | paraphrases served | wrong serves | blocked only by veto |
|---|---|---|---|---|
| **0.75** | **80.0%** | 12/22 | **0** | 15 |
| 0.80 | 72.0% | 8/22 | 0 | 13 |
| 0.85 | 66.0% | 5/22 | 0 | 11 |
| 0.90 | 60.0% | 2/22 | 0 | 7 |

0.75 serves the most correct paraphrases with zero wrong serves. At that threshold, 15 pairs that should not hit clear the similarity bar and are stopped only by the veto.

## Live demo

Send the first query of each pair (it goes to Gemini and gets stored), then the second. The demo page reads the `X-Cache-Decision` response header and shows the result next to the HIT/MISS badge and in the reasoning trace.

| Shows | 1st query | 2nd query | Similarity | Demo page shows |
|---|---|---|---|---|
| Clean hit | `wifi is not connecting` | `wifi won't connect` | 0.93 | green "Served from cache — matched …" |
| Polarity veto | `enable dark mode` | `disable dark mode` | 0.94 | red "Cache VETOED — … but means the opposite" |

`storage almost full` / `memory almost full` (0.78, entity veto) stays in the test set but is **not for the live demo**. Many users say "memory" when they mean storage, so an audience may not see the veto as correct.

**Check the first query was stored before sending the second.** If Gemini fails on it (the 8s SLA fallback returns `no_match`), it is correctly *not* cached, and the second query then shows a plain miss instead of the veto. `stores` in `/v1/cache/stats` should go up by one after the first query.

Each `/v1/troubleshoot` response carries `X-Cache-Decision`, e.g. `hit; sim=0.93; matched=wifi won't connect` or `miss_low_sim; sim=0.41`. The matched query is percent-encoded outside printable ASCII. `GET /v1/cache/stats` has the running counts.

## Deploy notes

- **Build step:** from `backend/`, run `python -m cache`. It downloads the model (~87 MB, about 1 minute) into `backend/.fastembed_cache` (git-ignored; override with `FASTEMBED_CACHE_PATH`). Without it, the first `/health` call waits for the download and returns 503 until the model is ready.
- **Single uvicorn worker only.** Each worker would load its own copy of the model (~230 MB) and keep its own separate cache. One process peaks at about 311 MB, so two won't fit in 512 MB, and their caches wouldn't be shared anyway.
- **The cache lives in memory and resets on every restart or redeploy.** There is no database. It holds at most `CACHE_MAX_ENTRIES` (default 2000) entries; when full, the oldest is dropped.
- **Debugging:** with `CACHE_DEBUG=true`, `GET /v1/cache/entries` lists what is stored, plus the last 50 lookups (decision, similarity, match) and store attempts (stored / refreshed / skipped with the reason). It returns 404 otherwise, because it exposes users' queries, and it isn't listed in `/docs`.
- **Hit rate:** set `ENABLE_QUERY_VARIATIONS=true` to store 8–10 paraphrases per answer. This raises the hit rate, but adds a parallel Gemini call on every miss. No code change is needed; it's read per request.

## Files

- `__init__.py`: `cache_store`, `cache_lookup`, `cache_stats`, `is_cache_ready`
- `__main__.py`: `python -m cache`, the model warm-up
- `eval/semantic_cache/`: `test_cache.py` (pytest), `tune_cache.py`, `cache_pairs.json`
