import os
import sys
from unittest.mock import MagicMock

# Ensure backend and root are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import backend.main as bmain
from schema import Action, ActionCategory, Goal, StepGroup, TroubleshootRequest


def test_offdomain_enforcement():
    print("=" * 75)
    print("TESTING OFF-DOMAIN BACKSTOP ENFORCEMENT & DISCARD LOGIC")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Scenario 1: Model complies with SYSTEM_PROMPT and returns {"goals": [], "no_match": true}
    # -------------------------------------------------------------------------
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"goals": [], "no_match": true}'
    mock_resp.usage_metadata.prompt_token_count = 10
    mock_resp.usage_metadata.candidates_token_count = 5
    mock_client.models.generate_content.return_value = mock_resp
    bmain.gemini_client = mock_client

    resp1 = bmain.troubleshoot(
        TroubleshootRequest(query="book me a flight to Paris"),
        skip_cache_lookup=True,
    )
    print("Scenario 1 (Model declared no_match):")
    print(f"  contexts: {resp1.contexts}")
    print(f"  fallback: {resp1.fallback}")
    assert resp1.contexts == [], "Contexts must be empty for off-domain query"
    assert resp1.fallback == "no_match", "Expected genuine 'no_match' fallback from LLM"
    print("[PASS] Scenario 1: Compliant LLM no-match correctly routes to fallback 'no_match'")

    # -------------------------------------------------------------------------
    # Scenario 2: Model hallucinates a plan for off-domain request
    # -------------------------------------------------------------------------
    hallucinated_goal = Goal(
        goal="Follow these steps to perform this Travel Troubleshooting",
        title="Flight travel booking",
        score=0.9,
        actions=[
            Action(
                actionName="Open Travel App",
                description="It will let you book flight tickets",
                category=ActionCategory.auto,
                stepGroups=[StepGroup(steps=["Open travel application."])],
            )
        ],
    )
    # Simulate extraction returning the hallucinated goal
    bmain.extract_goals = MagicMock(
        return_value=([hallucinated_goal], {"prompt_tokens": 50, "candidates_tokens": 30})
    )

    resp2 = bmain.troubleshoot(
        TroubleshootRequest(query="book me a flight to Paris"),
        skip_cache_lookup=True,
    )
    print("\nScenario 2 (Model hallucinated a troubleshooting plan):")
    print(f"  contexts: {resp2.contexts}")
    print(f"  fallback: {resp2.fallback}")
    assert resp2.contexts == [], "Hallucinated goals MUST be discarded by deterministic backstop"
    assert resp2.fallback == "no_match_offdomain_heuristic", (
        f"Expected fallback 'no_match_offdomain_heuristic', got '{resp2.fallback}'"
    )
    print("[PASS] Scenario 2: Hallucinated goals safely discarded; fallback set to 'no_match_offdomain_heuristic'")

    # -------------------------------------------------------------------------
    # Scenario 3: Device query (e.g. battery drain) must NOT be discarded
    # -------------------------------------------------------------------------
    valid_goal = Goal(
        goal="Follow these steps to perform this Battery Troubleshooting",
        title="Battery fast drain",
        score=0.92,
        actions=[
            Action(
                actionName="Configure Battery Settings",
                description="It will optimize battery usage settings",
                category=ActionCategory.auto,
                stepGroups=[StepGroup(steps=["Open Settings.", "Tap Battery."])],
            )
        ],
    )
    bmain.extract_goals = MagicMock(
        return_value=([valid_goal], {"prompt_tokens": 60, "candidates_tokens": 40})
    )
    resp3 = bmain.troubleshoot(
        TroubleshootRequest(query="battery drains fast overnight"),
        skip_cache_lookup=True,
    )
    print("\nScenario 3 (Valid on-domain device query):")
    print(f"  contexts count: {len(resp3.contexts)}")
    print(f"  fallback: {resp3.fallback}")
    assert len(resp3.contexts) == 1, "Legitimate device query goals must not be discarded"
    assert resp3.fallback is None, "Legitimate device query must not trigger fallback"
    print("[PASS] Scenario 3: Valid device complaint preserves goals without false triggering")

    print("\n" + "=" * 75)
    print("ALL OFF-DOMAIN ENFORCEMENT TESTS PASSED!")
    print("=" * 75)


if __name__ == "__main__":
    test_offdomain_enforcement()
