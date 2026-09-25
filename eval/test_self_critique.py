import os
import sys
import time
from unittest.mock import MagicMock, patch

# Add project root and backend to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema import Goal, Action, ActionCategory, StepGroup, TroubleshootRequest
import backend.main as bmain

goal1 = Goal(
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

goal2 = Goal(
    goal="Follow these steps to perform this Display Troubleshooting",
    title="Screen brightness",
    score=0.85,
    actions=[
        Action(
            actionName="Adjust Brightness",
            description="It will lower display power draw",
            category=ActionCategory.auto,
            stepGroups=[StepGroup(steps=["Tap display"])],
        )
    ],
)

# Test 1: Combined critique with 2 goals in a single call
mock_client = MagicMock()
mock_resp = MagicMock()
mock_resp.text = '{"evaluations": [{"index": 0, "is_relevant": true, "relevance_score": 0.95, "critique": "Directly targets battery drain."}, {"index": 1, "is_relevant": true, "relevance_score": 0.8, "critique": "Screen brightness affects battery."}]}'
mock_resp.usage_metadata.prompt_token_count = 80
mock_resp.usage_metadata.candidates_token_count = 45
mock_client.models.generate_content.return_value = mock_resp

evals, tokens = bmain.critique_goals_combined([goal1, goal2], "battery dies fast", client=mock_client)
assert len(evals) == 2
assert evals[0] == (True, 0.95, "Directly targets battery drain.")
assert evals[1] == (True, 0.8, "Screen brightness affects battery.")
assert tokens["prompt_tokens"] == 80
assert mock_client.models.generate_content.call_count == 1
print("[PASS] Combined critique: 2 goals evaluated in a single API call with critique text")

# Test 2: Single-goal critique helper wrapper
is_rel, score, critique_text, tokens = bmain.critique_goal_relevance(goal1, "battery dies fast", client=mock_client)
assert is_rel is True
assert score == 0.95
assert critique_text == "Directly targets battery drain."
print("[PASS] Single-goal wrapper delegates to combined critique cleanly and extracts critique text")

# Test 3: Safe failure on API error
mock_client.models.generate_content.side_effect = RuntimeError("API timeout")
evals, tokens = bmain.critique_goals_combined([goal1, goal2], "battery dies fast", client=mock_client)
assert evals == [(True, 1.0, None), (True, 1.0, None)]
print("[PASS] Safe fallback on API exception: Accepts goals with default scores and None critique")
mock_client.models.generate_content.side_effect = None

# Test 4: Full troubleshoot pipeline with ENABLE_SELF_CRITIQUE='false'
os.environ["ENABLE_SELF_CRITIQUE"] = "false"
bmain.gemini_client = mock_client
bmain.extract_goals = MagicMock(return_value=([goal1.model_copy(deep=True)], {"prompt_tokens": 100, "candidates_tokens": 50}))
bmain.critique_goals_combined = MagicMock(wraps=bmain.critique_goals_combined)

resp = bmain.troubleshoot(TroubleshootRequest(query="battery dies fast"), skip_cache_lookup=True)
assert bmain.critique_goals_combined.call_count == 0
assert resp.contexts[0].self_critique is None
print("[PASS] ENABLE_SELF_CRITIQUE=false: Zero critique calls made, self_critique is None")

# Test 5: Full troubleshoot pipeline with ENABLE_SELF_CRITIQUE='true' within SLA budget
os.environ["ENABLE_SELF_CRITIQUE"] = "true"
bmain.critique_goals_combined.reset_mock()
bmain.critique_goals_combined.return_value = ([(True, 0.9, "Plan directly targets battery drain.")], {"prompt_tokens": 50, "candidates_tokens": 20})

resp = bmain.troubleshoot(TroubleshootRequest(query="battery dies fast"), skip_cache_lookup=True)
assert bmain.critique_goals_combined.call_count == 1
assert resp.contexts[0].self_critique == "Plan directly targets battery drain."
print("[PASS] ENABLE_SELF_CRITIQUE=true (<= 4000ms): Critique executed and self_critique surfaced on Goal")

# Test 6: SLA 4000ms Threshold Guard Bypass
bmain.critique_goals_combined.reset_mock()
def slow_extract(*args, **kwargs):
    time.sleep(0.01)
    return [goal1.model_copy(deep=True)], {"prompt_tokens": 100, "candidates_tokens": 50}

bmain.extract_goals = slow_extract

# Simulate start_time that occurred 4500ms ago
original_perf_counter = time.perf_counter
call_counts = [0]
def mock_perf_counter():
    call_counts[0] += 1
    # On first call (start_time), return 0.0. On subsequent calls, return 4.5s (4500ms)
    if call_counts[0] == 1:
        return 0.0
    return 4.5

with patch("time.perf_counter", side_effect=mock_perf_counter):
    resp_bypassed = bmain.troubleshoot(TroubleshootRequest(query="battery dies fast"))

assert bmain.critique_goals_combined.call_count == 0
print("[PASS] 4000ms Threshold Guard: Critique safely bypassed when elapsed time exceeds 4000ms")

# Test 7: 2000ms Hard Timeout Guard via concurrent.futures
bmain.critique_goals_combined.reset_mock()
def hanging_critique(*args, **kwargs):
    time.sleep(2.5)  # exceeds 2.0s timeout
    return [(True, 0.5, None)], {"prompt_tokens": 0, "candidates_tokens": 0}

bmain.critique_goals_combined = hanging_critique
bmain.extract_goals = MagicMock(return_value=([goal1.model_copy(deep=True)], {"prompt_tokens": 100, "candidates_tokens": 50}))

t0 = time.perf_counter()
resp_timeout = bmain.troubleshoot(TroubleshootRequest(query="battery dies fast"), skip_cache_lookup=True)
total_duration = time.perf_counter() - t0

# Verify it timed out in ~2.0s rather than waiting 2.5s+
assert total_duration < 2.3, f"Expected timeout around 2.0s, took {total_duration:.2f}s"
assert len(resp_timeout.contexts) == 1, "Should safely retain extracted goals on timeout"
print(f"[PASS] 2000ms Hard Timeout: Aborted critique after {total_duration:.2f}s and returned extracted goals safely")

print("\nALL COMBINED CRITIQUE & SLA GUARD TESTS PASSED SUCCESSFULLY!")
