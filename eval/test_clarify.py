import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from main import clarify, troubleshoot
from schema import (
    Action,
    ActionCategory,
    ClarifyRequest,
    ClarifyResponse,
    ContextDeeplinkResponse,
    Goal,
    HypothesisItem,
    StepGroup,
    TroubleshootRequest,
    contains_url,
    scrub_urls,
)


def run_clarify_tests():
    print("=" * 75)
    print("RUNNING POST /v1/clarify TESTS")
    print("=" * 75)

    results = []

    def record_test(name: str, passed: bool, detail: str = ""):
        results.append((name, passed, detail))
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {name}")
        if detail:
            print(f"       Detail: {detail}")

    # -----------------------------------------------------------------------
    # 1. Close scores produce a question (gap < 0.15)
    # -----------------------------------------------------------------------
    try:
        req = ClarifyRequest(
            query="Phone screen navigation gestures are erratic",
            hypotheses=[
                HypothesisItem(title="Swipe navigation settings", score=0.85),
                HypothesisItem(title="Touch screen sensitivity", score=0.80),
            ],
            clarification_answer=None,
            gap_threshold=0.15,
        )
        resp = clarify(req)
        passed = resp.needs_clarification is True and resp.question is not None and len(resp.question) > 0
        record_test(
            "Close scores produce question (gap=0.05 < 0.15)",
            passed,
            f"needs_clarification={resp.needs_clarification} | question='{resp.question}'",
        )
    except Exception as e:
        record_test("Close scores produce question", False, str(e))

    # -----------------------------------------------------------------------
    # 2. Wide gap produces no question (gap >= 0.15)
    # -----------------------------------------------------------------------
    try:
        req = ClarifyRequest(
            query="Phone screen navigation gestures are erratic",
            hypotheses=[
                HypothesisItem(title="Swipe navigation settings", score=0.92),
                HypothesisItem(title="Display touch sensitivity", score=0.65),
            ],
            clarification_answer=None,
            gap_threshold=0.15,
        )
        resp = clarify(req)
        passed = resp.needs_clarification is False and resp.question is None
        record_test(
            "Wide gap produces no question (gap=0.27 >= 0.15)",
            passed,
            f"needs_clarification={resp.needs_clarification} | question={resp.question}",
        )
    except Exception as e:
        record_test("Wide gap produces no question", False, str(e))

    # -----------------------------------------------------------------------
    # 3. An answer changes ranking / re-runs pipeline
    # -----------------------------------------------------------------------
    try:
        # Mock troubleshoot response to simulate re-ranked pipeline output
        mock_goal_reranked = Goal(
            goal="Follow these steps to perform this Third Party App Troubleshooting",
            title="Uninstall rogue app",
            score=0.96,
            actions=[
                Action(
                    actionName="Uninstall Problematic Application",
                    description="It will remove recently installed apps",
                    category=ActionCategory.auto,
                    stepGroups=[StepGroup(steps=["Open Settings.", "Tap Apps.", "Tap Uninstall."])],
                )
            ],
        )

        with patch("main.extract_goals") as mock_extract:
            mock_extract.return_value = ([mock_goal_reranked], {"prompt_tokens": 100, "candidates_tokens": 50})

            req = ClarifyRequest(
                query="Phone screen navigation gestures are erratic",
                hypotheses=[
                    HypothesisItem(title="Swipe navigation settings", score=0.85),
                    HypothesisItem(title="Touch screen sensitivity", score=0.80),
                ],
                clarification_answer="The problem only happens in a launcher app I downloaded yesterday",
            )
            resp = clarify(req)

            passed = (
                resp.needs_clarification is False
                and len(resp.contexts) > 0
                and resp.contexts[0].title == "Uninstall rogue app"
                and resp.contexts[0].score == 0.96
            )
            record_test(
                "Clarification answer re-runs pipeline and updates ranking",
                passed,
                f"Updated top hypothesis: '{resp.contexts[0].title}' (score: {resp.contexts[0].score})",
            )
    except Exception as e:
        record_test("Clarification answer re-runs pipeline and updates ranking", False, str(e))

    # -----------------------------------------------------------------------
    # 4. A URL in an answer is scrubbed
    # -----------------------------------------------------------------------
    try:
        raw_answer = "Check out https://malicious-forum.com/hack and configure buttons"
        req = ClarifyRequest(
            query="Navigation gestures not working",
            hypotheses=[HypothesisItem(title="Swipe navigation settings", score=0.85)],
            clarification_answer=raw_answer,
        )

        # Confirm Pydantic validator scrubbed input on intake
        clean_in_req = req.clarification_answer
        passed = "malicious-forum" not in clean_in_req and not contains_url(clean_in_req)
        record_test(
            "URL in clarification answer is scrubbed",
            passed,
            f"Original: '{raw_answer}' -> Sanitized: '{clean_in_req}'",
        )
    except Exception as e:
        record_test("URL in clarification answer is scrubbed", False, str(e))

    # -----------------------------------------------------------------------
    # 5. Bad input returns clean error (no 500)
    # -----------------------------------------------------------------------
    try:
        # User provides an answer that is ONLY a URL, which becomes empty after scrubbing
        req = ClarifyRequest(
            query="Navigation gestures not working",
            clarification_answer="http://phishing.com/leak",
        )
        resp = clarify(req)

        passed = (
            resp.fallback == "no_match"
            and len(resp.contexts) == 0
            and resp.needs_clarification is False
        )
        record_test(
            "Bad/garbage input returns clean fallback (no 500 error)",
            passed,
            f"Returned fallback='{resp.fallback}', contexts={resp.contexts}",
        )
    except Exception as e:
        record_test("Bad/garbage input returns clean fallback", False, str(e))

    # -----------------------------------------------------------------------
    # 6. Optional Live Call (Behind --live flag)
    # -----------------------------------------------------------------------
    if "--live" in sys.argv:
        print("\n--- Running Live /v1/clarify Call ---")
        try:
            live_req = ClarifyRequest(
                query="Phone gestures erratic",
                hypotheses=[
                    HypothesisItem(title="Swipe navigation settings", score=0.85),
                    HypothesisItem(title="Display touch sensitivity", score=0.80),
                ],
                clarification_answer="It only happens after I downloaded a game app yesterday",
            )
            live_resp = clarify(live_req)
            passed = len(live_resp.contexts) > 0 and live_resp.meta.latency_ms > 0
            record_test(
                "Live /v1/clarify Execution",
                passed,
                f"Top Goal: '{live_resp.contexts[0].title}' | Latency: {live_resp.meta.latency_ms}ms",
            )
        except Exception as e:
            record_test("Live /v1/clarify Execution", False, str(e))
    else:
        print("\n--- [INFO] Live /v1/clarify call skipped (pass '--live' to run against live API) ---")

    print("\n" + "=" * 75)
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    print(f"POST /v1/clarify TEST SUMMARY: {passed_count}/{total_count} PASSED")
    print("=" * 75)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_clarify_tests()
    sys.exit(0 if success else 1)
