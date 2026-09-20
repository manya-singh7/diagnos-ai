import re
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from schema import ActionCategory, ContextDeeplinkResponse, Deeplink, Goal

app = FastAPI()


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Step 1: query enrichment
# ---------------------------------------------------------------------------

_FILLER_WORDS = {
    "um", "umm", "uh", "uhh", "like", "basically", "actually", "kinda",
    "sorta", "so", "well", "just", "hey", "please", "ok", "okay",
}

_NORMALIZATION_MAP = {
    "wifi": "wi-fi",
    "wont": "won't",
    "doesnt": "doesn't",
    "isnt": "isn't",
    "cant": "can't",
    "dont": "don't",
    "ur": "your",
    "u": "you",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "tv": "TV",
}


def enrich_query(raw_query: str) -> str:
    """
    Normalize a raw, casually-phrased complaint into a cleaner technical query.

    - lowercases and strips punctuation noise
    - expands common typos/shorthand (e.g. "wont" -> "won't", "u" -> "you")
    - drops filler words ("um", "like", "just", ...) that add no signal
    - collapses whitespace

    This is intentionally rule-based (no LLM call) since it's a cheap,
    deterministic cleanup pass that runs before the LLM extraction step.
    """
    text = raw_query.strip().lower()
    text = re.sub(r"[^\w\s'-]", " ", text)

    words = []
    for word in text.split():
        word = _NORMALIZATION_MAP.get(word, word)
        if word in _FILLER_WORDS:
            continue
        words.append(word)

    return re.sub(r"\s+", " ", " ".join(words)).strip()


# ---------------------------------------------------------------------------
# Step 2: structured extraction (LLM stub)
# ---------------------------------------------------------------------------

def extract_goal(query: str) -> Goal:
    """
    Call an LLM to turn a normalized troubleshooting query into a structured
    Goal. STUB: wire up the real API call here (e.g. Anthropic Messages API),
    parse/validate the JSON response, and construct a Goal from it.

    Suggested prompt (system + user), designed to satisfy the schema's
    constraints exactly:

        SYSTEM:
        You are a device-troubleshooting assistant. Given a customer's
        troubleshooting query, output a single JSON object (no markdown
        fences, no prose) with exactly this shape:

        {
          "goal": "<one-sentence restatement of the user's underlying goal>",
          "title": "<2-3 words, Sentence case>",
          "actions": [
            {
              "actionName": "<short snake_case identifier, e.g. 'restart_router'>",
              "description": "<MUST start with the literal words 'It will', 5-7 words total>",
              "category": "<one of exactly: 'auto', 'manual', 'critical'>",
              "stepGroups": [
                {"steps": ["<imperative step 1>", "<imperative step 2>"]}
              ]
            }
          ],
          "score": <float between 0 and 1, your confidence in this plan>
        }

        Hard constraints (validate before returning):
        - title: exactly 2-3 words, Sentence case (first word capitalized,
          rest lowercase unless a proper noun/acronym).
        - description: starts with "It will" and is 5-7 words total,
          counting "It will" as the first two words.
        - category: must be exactly "auto", "manual", or "critical" — no
          other values, no synonyms.
        - Return ONLY the JSON object, nothing else.

        USER:
        Troubleshooting query: "{query}"

    Returns a placeholder Goal until the real call is wired up.
    """
    return Goal(
        goal=f"Resolve the issue described as: {query}",
        title="Fix issue",
        actions=[],
        score=0.0,
    )


# ---------------------------------------------------------------------------
# Step 3: deeplink retrieval (stub) + ordering
# ---------------------------------------------------------------------------

_SAMPLE_DEEPLINKS: Dict[str, Deeplink] = {
    "restart_router": Deeplink(
        deeplink="app://settings/network/restart",
        description="Restart the router",
    ),
    "reset_wifi_password": Deeplink(
        deeplink="app://settings/network/wifi/password",
        description="Reset the Wi-Fi password",
    ),
    "check_firmware_update": Deeplink(
        deeplink="app://settings/system/firmware",
        description="Check for a firmware update",
    ),
}

_DEFAULT_DEEPLINK = Deeplink(
    deeplink="app://settings/general",
    description="Open general settings",
)


def get_deeplink(action_name: str) -> Deeplink:
    """
    STUB: returns a hardcoded example deeplink keyed by action name.

    A teammate is building the real retrieval system separately; swap this
    out for their module once it's ready.
    """
    return _SAMPLE_DEEPLINKS.get(action_name, _DEFAULT_DEEPLINK)


_CATEGORY_ORDER = {
    ActionCategory.auto: 0,
    ActionCategory.manual: 1,
    ActionCategory.critical: 2,
}


@app.post("/v1/troubleshoot", response_model=ContextDeeplinkResponse)
def troubleshoot(payload: dict):
    """
    Takes a customer complaint and returns an actionable plan.
    Expected input: {"query": "...", "siis_response": "<optional>"}
    """
    raw_query = payload.get("query", "")

    query = enrich_query(raw_query)
    goal = extract_goal(query)

    for action in goal.actions:
        deeplink = get_deeplink(action.actionName)
        for step_group in action.stepGroups:
            step_group.actionableDeeplink = deeplink

    goal.actions.sort(key=lambda a: _CATEGORY_ORDER.get(a.category, 1))

    return ContextDeeplinkResponse(contexts=[goal])


@app.post("/v1/clarify", response_model=ContextDeeplinkResponse)
def clarify(payload: dict):
    """
    Takes the original query + a clarifying answer, re-ranks the result.
    Expected input: {"query": "...", "clarification_answer": "..."}
    """
    # TODO: re-rank contexts using the extra detail
    return ContextDeeplinkResponse(contexts=[])