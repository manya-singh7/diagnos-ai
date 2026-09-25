import os
import sys
from unittest.mock import MagicMock, patch

# Ensure root and backend directory are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import backend.main as bmain
from google.genai import errors


def run_failover_tests():
    print("=" * 75)
    print("RUNNING GEMINI DUAL-KEY FAILOVER TESTS")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Test 1: Quota Detection Predicate (_is_quota_exhausted)
    # -------------------------------------------------------------------------
    err_429 = errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Resource has been exhausted."}})
    err_503 = errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "Service unavailable."}})
    err_timeout = TimeoutError("Connection timed out after 30s")
    err_val = ValueError("Invalid schema: score must be between 0.0 and 1.0")
    err_custom_429 = RuntimeError("429 Too Many Requests: quota exceeded")

    assert bmain._is_quota_exhausted(err_429) is True, "429 ClientError must be recognized as quota exhausted"
    assert bmain._is_quota_exhausted(err_custom_429) is True, "Custom 429 message must be recognized as quota exhausted"
    assert bmain._is_quota_exhausted(err_503) is False, "503 ServerError must NOT trigger quota failover"
    assert bmain._is_quota_exhausted(err_timeout) is False, "TimeoutError must NOT trigger quota failover"
    assert bmain._is_quota_exhausted(err_val) is False, "ValueError must NOT trigger quota failover"
    print("[PASS] Test 1: _is_quota_exhausted cleanly identifies 429 quota exhaustion and ignores 503/timeout/validation")

    # -------------------------------------------------------------------------
    # Test 2: Primary succeeds -> No failover, primary serves request
    # -------------------------------------------------------------------------
    mock_primary = MagicMock()
    mock_backup = MagicMock()
    mock_resp_primary = MagicMock()
    mock_resp_primary.text = "Primary Response"
    mock_primary.models.generate_content.return_value = mock_resp_primary

    bmain.gemini_client_primary = mock_primary
    bmain.gemini_client_backup = mock_backup

    resp = bmain.generate_content_with_failover(
        client=mock_primary,
        model="gemini-3.6-flash",
        contents="test prompt",
        call_name="test_primary_success",
    )
    assert resp.text == "Primary Response"
    assert mock_primary.models.generate_content.call_count == 1
    assert mock_backup.models.generate_content.call_count == 0
    print("[PASS] Test 2: Primary success serves request directly without calling backup client")

    # -------------------------------------------------------------------------
    # Test 3: Primary fails with 503 -> Re-raises immediately, backup NOT called
    # -------------------------------------------------------------------------
    mock_primary.reset_mock()
    mock_backup.reset_mock()
    mock_primary.models.generate_content.side_effect = err_503

    raised_503 = False
    try:
        bmain.generate_content_with_failover(
            client=mock_primary,
            model="gemini-3.6-flash",
            contents="test prompt",
            call_name="test_503_bypass",
        )
    except Exception as e:
        if getattr(e, "code", None) == 503 or "503" in str(e):
            raised_503 = True

    assert raised_503 is True, "503 error must be re-raised immediately"
    assert mock_primary.models.generate_content.call_count == 1
    assert mock_backup.models.generate_content.call_count == 0, "Backup must NOT be called on 503"
    print("[PASS] Test 3: 503 UNAVAILABLE bypasses failover and raises immediately for transient handler")

    # -------------------------------------------------------------------------
    # Test 4: Primary hits 429 RESOURCE_EXHAUSTED -> Fails over to backup
    # -------------------------------------------------------------------------
    mock_primary.reset_mock()
    mock_backup.reset_mock()
    mock_primary.models.generate_content.side_effect = err_429
    mock_resp_backup = MagicMock()
    mock_resp_backup.text = "Backup Response"
    mock_backup.models.generate_content.return_value = mock_resp_backup

    resp = bmain.generate_content_with_failover(
        client=mock_primary,
        model="gemini-3.6-flash",
        contents="test prompt",
        call_name="test_failover_success",
    )
    assert resp.text == "Backup Response"
    assert mock_primary.models.generate_content.call_count == 1
    assert mock_backup.models.generate_content.call_count == 1
    print("[PASS] Test 4: 429 RESOURCE_EXHAUSTED triggers immediate single-retry failover to backup client")

    # -------------------------------------------------------------------------
    # Test 5: Single-key setup (gemini_client_backup is None) -> 429 raises as-is
    # -------------------------------------------------------------------------
    mock_primary.reset_mock()
    bmain.gemini_client_backup = None
    mock_primary.models.generate_content.side_effect = err_429

    raised_429 = False
    try:
        bmain.generate_content_with_failover(
            client=mock_primary,
            model="gemini-3.6-flash",
            contents="test prompt",
            call_name="test_no_backup",
        )
    except Exception as e:
        if getattr(e, "code", None) == 429:
            raised_429 = True

    assert raised_429 is True, "Without backup client, 429 must raise normally for backward compatibility"
    assert mock_primary.models.generate_content.call_count == 1
    print("[PASS] Test 5: Backward compatibility verified — without backup client, 429 raises unchanged")

    # Restore backup client for subsequent tests
    bmain.gemini_client_backup = mock_backup

    # -------------------------------------------------------------------------
    # Test 6: Failover across retry attempts inside extract_goals
    # -------------------------------------------------------------------------
    # Attempt 0: Primary returns schema error (invalid JSON)
    # Attempt 1: Schema retry runs; primary hits 429; backup serves valid Goal JSON
    mock_primary.reset_mock()
    mock_backup.reset_mock()

    mock_resp_bad_json = MagicMock()
    mock_resp_bad_json.text = "NOT JSON"
    mock_resp_bad_json.usage_metadata.prompt_token_count = 10
    mock_resp_bad_json.usage_metadata.candidates_token_count = 10

    mock_resp_valid_json = MagicMock()
    mock_resp_valid_json.text = (
        '{"goals": [{"goal": "Follow these steps to perform this Display Troubleshooting", '
        '"title": "Swipe navigation settings", "score": 0.95, '
        '"actions": [{"actionName": "Configure Navigation Bar", '
        '"description": "It will let you choose navigation type", "category": "auto", '
        '"stepGroups": [{"steps": ["Tap Settings"]}]}]}]}'
    )
    mock_resp_valid_json.usage_metadata.prompt_token_count = 20
    mock_resp_valid_json.usage_metadata.candidates_token_count = 20

    # Primary returns bad JSON on attempt 1, then on attempt 2 raises 429
    mock_primary.models.generate_content.side_effect = [mock_resp_bad_json, err_429]
    mock_backup.models.generate_content.return_value = mock_resp_valid_json

    goals, usage = bmain.extract_goals(
        query="swipe navigation is broken",
        client=mock_primary,
        max_retries=2,
    )

    assert len(goals) == 1, "extract_goals should recover and return valid Goal"
    assert goals[0].title == "Swipe navigation settings"
    assert mock_primary.models.generate_content.call_count == 2, "Primary attempted on attempt 0 and attempt 1"
    assert mock_backup.models.generate_content.call_count == 1, "Backup served on attempt 1 when primary hit 429"
    print("[PASS] Test 6: Failover successfully covers schema-validation retry attempts inside extract_goals")

    print("\n" + "=" * 75)
    print("ALL DUAL-KEY FAILOVER TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_failover_tests()
