import os
import sys
from unittest.mock import MagicMock

# Add project root and backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema import Goal, Action, ActionCategory, StepGroup, TroubleshootRequest
import backend.main as bmain

# Test 1: critique_goal_relevance helper function logic directly
goal = Goal(
    goal="Follow these steps to perform this Battery Troubleshooting",
    title="Battery optimization",
    score=0.9,
    actions=[
        Action(
            actionName="Optimize Battery",
            description="It will optimize battery usage",
            category=ActionCategory.auto,
            stepGroups=[StepGroup(steps=["Tap battery"])],
        )
    ],
)

# Test 1a: mock client returning relevant critique
mock_client = MagicMock()
mock_resp = MagicMock()
mock_resp.text = '{"is_relevant": true, "relevance_score": 0.95, "critique": "Plan directly addresses battery drain."}'
mock_resp.usage_metadata.prompt_token_count = 50
mock_resp.usage_metadata.candidates_token_count = 20
mock_client.models.generate_content.return_value = mock_resp

is_rel, score, tokens = bmain.critique_goal_relevance(goal, "battery dies fast", client=mock_client)
assert is_rel is True, f"Expected True, got {is_rel}"
assert score == 0.95, f"Expected 0.95, got {score}"
assert tokens["prompt_tokens"] == 50
print("[PASS] Critique helper relevant test passed")

# Test 1b: mock client returning irrelevant critique
mock_resp.text = '{"is_relevant": false, "relevance_score": 0.2, "critique": "Plan addresses battery but complaint was about camera."}'
is_rel, score, tokens = bmain.critique_goal_relevance(goal, "camera blurry", client=mock_client)
assert is_rel is False, f"Expected False, got {is_rel}"
assert score == 0.2, f"Expected 0.2, got {score}"
print("[PASS] Critique helper irrelevant test passed")

# Test 1c: exception handling fallback (safe failure)
mock_client.models.generate_content.side_effect = RuntimeError("API connection timeout")
is_rel, score, tokens = bmain.critique_goal_relevance(goal, "battery dies fast", client=mock_client)
assert is_rel is True, "Fallback on error should accept goal"
assert score == 1.0, "Fallback score should be 1.0"
print("[PASS] Critique helper safe error fallback passed")
mock_client.models.generate_content.side_effect = None

# Test 2: Full troubleshoot endpoint with ENABLE_SELF_CRITIQUE='false' vs 'true'
# Test 2a: 'false'
os.environ["ENABLE_SELF_CRITIQUE"] = "false"
bmain.gemini_client = mock_client
bmain.extract_goals = MagicMock(return_value=([goal.model_copy(deep=True)], {"prompt_tokens": 100, "candidates_tokens": 50}))
bmain.critique_goal_relevance = MagicMock(wraps=bmain.critique_goal_relevance)

resp = bmain.troubleshoot(TroubleshootRequest(query="battery dies fast"))
assert bmain.critique_goal_relevance.call_count == 0, "Critique should NOT be called when flag is false"
print("[PASS] ENABLE_SELF_CRITIQUE=false: Zero calls made, zero overhead")

# Test 2b: 'true'
os.environ["ENABLE_SELF_CRITIQUE"] = "true"
bmain.critique_goal_relevance.reset_mock()
bmain.critique_goal_relevance.return_value = (True, 0.9, {"prompt_tokens": 40, "candidates_tokens": 15})

resp = bmain.troubleshoot(TroubleshootRequest(query="battery dies fast"))
assert bmain.critique_goal_relevance.call_count == 1, "Critique should be called when flag is true"
print("[PASS] ENABLE_SELF_CRITIQUE=true: Called and modulated goal")

# Test 2c: 'true' and critique rejects goal (relevance < 0.5)
bmain.critique_goal_relevance.reset_mock()
bmain.critique_goal_relevance.return_value = (False, 0.2, {"prompt_tokens": 40, "candidates_tokens": 15})

resp_rejected = bmain.troubleshoot(TroubleshootRequest(query="battery dies fast"))
assert bmain.critique_goal_relevance.call_count == 1
assert resp_rejected.fallback == "no_match", f"Expected fallback 'no_match', got {resp_rejected.fallback}"
assert len(resp_rejected.contexts) == 0
print("[PASS] ENABLE_SELF_CRITIQUE=true rejection: Discarded irrelevant goal and fell back cleanly")

print("\nALL SELF-CRITIQUE VERIFICATION TESTS PASSED SUCCESSFULLY!")
