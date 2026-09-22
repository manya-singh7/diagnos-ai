import re
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Zero URL Leaks: Scrubbing & Detection
# ---------------------------------------------------------------------------

_MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]]+)\]\((?:https?://|www\.|\S+\.(?:com|org|net|edu|gov|io|co|ai|info|biz|dev))[^\)]*\)",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(
    r"(?:https?://\S+|www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?|\b(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|ai|info|biz|dev)(?:/[^\s,.]*)?)",
    re.IGNORECASE,
)


def contains_url(text: str) -> bool:
    """Check if text contains any web URLs (http, https, www, markdown links, or domains)."""
    if not text:
        return False
    return bool(_MARKDOWN_LINK_PATTERN.search(text) or _URL_PATTERN.search(text))


def scrub_urls(text: str) -> str:
    """
    Programmatically scrub web URLs from text (including markdown links, http/https, www).
    Preserves bixby:// URIs and cleans dangling whitespace.
    """
    if not text:
        return text
    # 1. Convert markdown links [text](url) -> text
    cleaned = _MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    # 2. Remove phrases like 'visit http...', 'see www...', 'go to samsung.com'
    cleaned = re.sub(
        r"(?i)\b(?:please\s+)?(?:visit|see|go to|check|browse)\s+(?:https?://\S+|www\.\S+|(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|ai)[^\s]*)",
        "",
        cleaned,
    )
    # 3. Remove raw URLs
    cleaned = _URL_PATTERN.sub("", cleaned)
    # 4. Clean up punctuation and whitespace
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Condition(str, Enum):
    greater = "greater"
    equal = "equal"
    less = "less"


class ResultTypes(str, Enum):
    boolean = "boolean"
    intNum = "integer"
    string = "str"
    floatNum = "float"


class ActionCategory(str, Enum):
    auto = "auto"
    manual = "manual"
    critical = "critical"


# ---------------------------------------------------------------------------
# Deeplink Models
# ---------------------------------------------------------------------------

class BaseDeeplink(BaseModel):
    deeplink: str

    @field_validator("deeplink")
    @classmethod
    def validate_deeplink(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("deeplink URI cannot be empty")
        v = v.strip()
        if contains_url(v):
            raise ValueError(f"deeplink cannot contain web URLs: {v}")
        if not (v.startswith("bixby://") or v.startswith("app://")):
            raise ValueError(f"deeplink must be a valid Bixby URI (starting with bixby://): {v}")
        return v


class Deeplink(BaseDeeplink):
    description: str
    message: Optional[str] = ""
    classes: Optional[Dict[str, str]] = None
    originalType: Optional[str] = None

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if contains_url(v):
            raise ValueError(f"description cannot contain web URLs: {v}")
        return v


class ValidationDeepLink(BaseDeeplink):
    key: str
    resultType: Optional[ResultTypes] = None
    condition: Optional[Condition] = None
    value: Optional[str] = None


# ---------------------------------------------------------------------------
# StepGroup & Action Models
# ---------------------------------------------------------------------------

_ALLOWED_MINOR_WORDS = {
    "and", "or", "to", "in", "of", "for", "on", "at", "by", "a", "an", "the", "with"
}


class StepGroup(BaseModel):
    steps: List[str]
    validationDeeplink: Optional[ValidationDeepLink] = None
    actionableDeeplink: Optional[Deeplink] = None

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, steps: List[str]) -> List[str]:
        if not steps:
            raise ValueError("stepGroup must contain at least one step")
        cleaned_steps = []
        for i, step in enumerate(steps):
            cleaned = scrub_urls(step)
            if contains_url(cleaned):
                raise ValueError(f"Step {i+1} contains forbidden web URL: '{step}'")
            if not cleaned:
                raise ValueError(f"Step {i+1} is empty after scrubbing URLs: '{step}'")
            cleaned_steps.append(cleaned)
        return cleaned_steps


class Action(BaseModel):
    actionName: str
    description: str
    stepGroups: List[StepGroup]
    category: Optional[ActionCategory] = ActionCategory.manual

    @field_validator("actionName")
    @classmethod
    def validate_action_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("actionName cannot be empty")
        v = v.strip()
        if contains_url(v):
            raise ValueError(f"actionName cannot contain web URLs: '{v}'")
        words = v.split()
        if not words[0][0].isupper():
            raise ValueError(f"actionName must be Title Case (first word capitalized): '{v}'")
        for w in words[1:]:
            clean_w = w.strip(".,!?:;\"'()[]")
            if clean_w.isupper() or clean_w[0].isupper() or clean_w.lower() in _ALLOWED_MINOR_WORDS:
                continue
            raise ValueError(f"actionName must be Title Case: '{v}' (word '{w}' is not capitalized)")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("description cannot be empty")
        v = v.strip()
        if contains_url(v):
            raise ValueError(f"description cannot contain web URLs: '{v}'")
        if not v.startswith("It will"):
            raise ValueError(f"description must start with 'It will', got: '{v}'")
        words = v.split()
        if not (5 <= len(words) <= 7):
            raise ValueError(
                f"description must be exactly 5 to 7 words (counting 'It will'), got {len(words)}: '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_category_constraints(self) -> "Action":
        if self.category == ActionCategory.manual:
            for i, sg in enumerate(self.stepGroups):
                if sg.actionableDeeplink is not None:
                    raise ValueError(
                        f"Manual action '{self.actionName}' stepGroup[{i}] cannot carry an actionableDeeplink"
                    )
        return self


# ---------------------------------------------------------------------------
# Goal Model
# ---------------------------------------------------------------------------

_GOAL_SYNTAX_PATTERN = re.compile(
    r"^Follow these steps to perform this (.+) (Troubleshooting|Configuration)$"
)
_COMMON_ACRONYMS_AND_PROPER = {
    # Wireless & Connectivity
    "wi-fi", "wifi", "bluetooth", "ble", "nfc", "uwb", "gps", "glonass", "cellular",
    "lte", "5g", "4g", "3g", "volte", "vowifi", "apn", "sim", "esim", "hotspot",
    # Hardware, Displays & Ports
    "usb", "hdmi", "otg", "microsd", "sd", "cpu", "gpu", "ram", "rom", "npu", "soc",
    "oled", "amoled", "lcd", "led", "hdr", "fps", "hz",
    # Ecosystem & Samsung Features
    "samsung", "galaxy", "android", "bixby", "google", "knox", "dex", "oneui",
    "smart", "view", "switch", "routines", "routine", "share", "pen",
    # Audio, Media & Sensors
    "dolby", "atmos", "codec", "dac", "anc", "eq", "pin", "sos", "biometrics",
    # System & General Tech
    "os", "ui", "battery", "tv", "pc", "mac", "app", "id", "ip", "dns", "qr"
}


def normalize_title(title: str) -> str:
    """
    Programmatically normalizes a 2-3 word title into strict sentence case.
    Capitalizes the first word, and lowercases subsequent words unless they are
    all-caps acronyms or recognized proper nouns.
    """
    words = title.strip().split()
    if not words:
        return title
    normalized = [words[0].capitalize()]
    for w in words[1:]:
        clean_w = w.strip(".,!?:;\"'()[]")
        if clean_w.isupper() or clean_w.lower() in _COMMON_ACRONYMS_AND_PROPER:
            normalized.append(w)
        else:
            normalized.append(w.lower())
    return " ".join(normalized)


class Goal(BaseModel):
    goal: str
    title: str
    actions: List[Action]
    score: float

    @field_validator("goal")
    @classmethod
    def validate_goal_syntax(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("goal cannot be empty")
        v = v.strip()
        if contains_url(v):
            raise ValueError(f"goal cannot contain web URLs: '{v}'")
        if not _GOAL_SYNTAX_PATTERN.match(v):
            raise ValueError(
                f"goal must follow exact syntax: 'Follow these steps to perform this <Topic> Troubleshooting' "
                f"or '... <Topic> Configuration', got: '{v}'"
            )
        return v

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title cannot be empty")
        v = v.strip()
        if contains_url(v):
            raise ValueError(f"title cannot contain web URLs: '{v}'")
        words = v.split()
        if not (2 <= len(words) <= 3):
            raise ValueError(f"title must be exactly 2 to 3 words, got {len(words)}: '{v}'")
        if not words[0][0].isupper():
            raise ValueError(f"title must be sentence case (first word capitalized): '{v}'")
        for w in words[1:]:
            clean_w = w.strip(".,!?:;\"'()[]")
            if clean_w.isupper() or clean_w.lower() in _COMMON_ACRONYMS_AND_PROPER:
                continue
            if clean_w[0].isupper():
                raise ValueError(
                    f"title must be sentence case (word '{w}' should be lowercase unless acronym/proper noun): '{v}'"
                )
        return v

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be a float between 0.0 and 1.0, got: {v}")
        return v

    @model_validator(mode="after")
    def validate_action_ordering(self) -> "Goal":
        """Validate that critical actions are ordered last."""
        seen_critical = False
        for action in self.actions:
            if action.category == ActionCategory.critical:
                seen_critical = True
            elif seen_critical:
                raise ValueError(
                    f"Critical actions must be ordered last. Found {action.category} action "
                    f"'{action.actionName}' after a critical action."
                )
        return self


# ---------------------------------------------------------------------------
# API Request & Response Models
# ---------------------------------------------------------------------------

class ResponseMeta(BaseModel):
    latency_ms: int = Field(..., ge=0, description="Latency of request processing in milliseconds")
    cache_hit: bool = Field(..., description="Whether response was served from cache")
    model: Optional[str] = Field(None, description="Model identifier used for inference")
    cost_usd: float = Field(0.0, ge=0.0, description="Inference cost in USD")


class ContextDeeplinkResponse(BaseModel):
    """RAG response containing a list of Goal objects, optional fallback, and operational metadata."""
    contexts: List[Goal] = Field(default_factory=list)
    fallback: Optional[str] = Field(None, description="Fallback flag when no match is found, e.g. 'no_match'")
    meta: Optional[ResponseMeta] = Field(None, description="Operational metadata")

    @model_validator(mode="after")
    def validate_fallback_and_contexts(self) -> "ContextDeeplinkResponse":
        if self.fallback == "no_match" and len(self.contexts) > 0:
            raise ValueError("When fallback is 'no_match', contexts must be empty ([])")
        return self


class TroubleshootRequest(BaseModel):
    query: str
    siis_response: Optional[str] = None

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return scrub_urls(v)

    @field_validator("siis_response")
    @classmethod
    def sanitize_siis(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return scrub_urls(v)


class HypothesisItem(BaseModel):
    title: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClarifyRequest(BaseModel):
    query: str
    hypotheses: List[HypothesisItem] = Field(default_factory=list)
    clarification_answer: Optional[str] = None
    gap_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return scrub_urls(v)

    @field_validator("clarification_answer")
    @classmethod
    def sanitize_answer(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return scrub_urls(v)


class ClarifyResponse(ContextDeeplinkResponse):
    needs_clarification: bool = Field(False, description="Whether clarification question is needed")
    question: Optional[str] = Field(None, description="One short question distinguishing the top two hypotheses")