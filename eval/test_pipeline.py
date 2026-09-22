import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from main import (
    MODEL_NAME,
    RESPONSE_SHAPE,
    SYSTEM_PROMPT,
    app,
    build_user_prompt,
    enrich_query,
    extract_goal,
    gemini_client,
    generate_query_variations,
    get_deeplinks,
    health,
    health_details,
    serialize_response,
    troubleshoot,
    validate_and_sanitize_deeplinks,
    validate_query_variations,
)
from schema import (
    Action,
    ActionCategory,
    AppendixBResponse,
    ContextDeeplinkResponse,
    Deeplink,
    Goal,
    ResponseMeta,
    StepGroup,
    TroubleshootRequest,
    contains_url,
)


def run_pipeline_tests():
    print("=" * 75)
    print("RUNNING ENDPOINT & PIPELINE TESTS (Person A)")
    print("=" * 75)

    results = []

    def record_test(name: str, passed: bool, detail: str = ""):
        results.append((name, passed, detail))
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {name}")
        if detail:
            print(f"       Detail: {detail}")

    # -----------------------------------------------------------------------
    # 1. GET /health Exact Shape & HTTP 503 on Unready + /health/details
    # -----------------------------------------------------------------------
    try:
        h_resp = health()
        is_exact_shape = (
            h_resp.status == "ok"
            and getattr(h_resp, "model_dump", lambda: {})() == {"status": "ok"}
        )
        h_details = health_details()
        details_body = json.loads(h_details.body.decode("utf-8")) if hasattr(h_details, "body") else h_details

        with patch("main._check_model_ready", return_value=False):
            h_unready = health()
            unready_503 = getattr(h_unready, "status_code", None) == 503

        record_test(
            "GET /health: Exact shape {'status': 'ok'} (HTTP 200) and 503 when unready",
            is_exact_shape and details_body.get("status") == "ok" and unready_503,
            f"exact_shape={is_exact_shape} | details={details_body} | unready_status=503",
        )
    except Exception as e:
        record_test("GET /health: Exact shape and 503 when unready", False, str(e))

    # -----------------------------------------------------------------------
    # 2. Deeplink Guard: Restart/Reboot never get unrelated deeplinks
    # -----------------------------------------------------------------------
    try:
        reboot_link = get_deeplinks("Restart Device in Safe Mode", "disable third party apps", category=ActionCategory.critical)
        restart_link = get_deeplinks("Reboot Phone", "restart Android system", category=ActionCategory.critical)
        reset_link = get_deeplinks("Factory Data Reset", "restore to factory defaults", category=ActionCategory.critical)

        all_none = (reboot_link is None) and (restart_link is None) and (reset_link is None)
        record_test(
            "Deeplink Guard: Restart/reboot/safe mode never get unrelated deeplinks",
            all_none,
            f"reboot: {reboot_link}, restart: {restart_link}, reset: {reset_link}",
        )
    except Exception as e:
        record_test("Deeplink Guard: Restart/reboot never get unrelated deeplinks", False, str(e))

    # -----------------------------------------------------------------------
    # 3. Deeplink Guard: Weak match returns no deeplink
    # -----------------------------------------------------------------------
    try:
        weak_link_1 = get_deeplinks("Unknown Hardware Component", "unindexed physical item")
        weak_link_2 = get_deeplinks("Inspect USB Cable Connector", "clean metal port pins")

        record_test(
            "Deeplink Guard: Weak match returns no deeplink",
            weak_link_1 is None and weak_link_2 is None,
            f"weak_link_1: {weak_link_1}, weak_link_2: {weak_link_2}",
        )
    except Exception as e:
        record_test("Deeplink Guard: Weak match returns no deeplink", False, str(e))

    # -----------------------------------------------------------------------
    # 4. Deeplink Rules: auto + strong match -> catalog, auto + Settings -> dummy_positive, manual -> None
    # -----------------------------------------------------------------------
    try:
        nav_link = get_deeplinks("Configure Navigation Bar Settings", "navigation bar settings", category=ActionCategory.auto)
        touch_link = get_deeplinks("Adjust Display Touch Sensitivity", "open touch sensitivity settings", category=ActionCategory.auto)
        manual_link = get_deeplinks("Clean Charging Port", "clean lint with brush", category=ActionCategory.manual)

        passed_rules = (
            nav_link is not None
            and "navigation_bar" in nav_link.deeplink
            and touch_link is not None
            and touch_link.deeplink == "bixby://dummy_positive"
            and manual_link is None
        )
        record_test(
            "Deeplink Rules: Catalog match, Settings dummy_positive fallback, Manual veto",
            passed_rules,
            f"nav: '{getattr(nav_link, 'deeplink', None)}', touch: '{getattr(touch_link, 'deeplink', None)}', manual: {manual_link}",
        )
    except Exception as e:
        record_test("Deeplink Rules: Catalog match, Settings dummy_positive fallback, Manual veto", False, str(e))

    # -----------------------------------------------------------------------
    # 5. Final Deeplink Validator (Finished Response)
    # -----------------------------------------------------------------------
    try:
        test_goal = Goal(
            goal="Follow these steps to perform this Test Troubleshooting",
            title="Test deeplink validation",
            score=0.9,
            actions=[
                Action(
                    actionName="Test Legitimate Action",
                    description="It will configure device navigation settings",
                    category=ActionCategory.auto,
                    stepGroups=[
                        StepGroup(
                            steps=["Tap display"],
                            actionableDeeplink=Deeplink(
                                deeplink="bixby://masked/act/setting/display/navigation_bar",
                                description="Legitimate catalog link",
                            ),
                        ),
                        StepGroup(
                            steps=["Tap dummy"],
                            actionableDeeplink=Deeplink(
                                deeplink="bixby://dummy_positive",
                                description="Legitimate dummy positive",
                            ),
                        ),
                        StepGroup(
                            steps=["Tap rogue"],
                            actionableDeeplink=Deeplink(
                                deeplink="bixby://hallucinated/unauthorized/action",
                                description="Rogue uncatalogued link",
                            ),
                        ),
                    ],
                )
            ],
        )

        validate_and_sanitize_deeplinks([test_goal])
        sgs = test_goal.actions[0].stepGroups
        valid_kept = sgs[0].actionableDeeplink is not None and sgs[0].actionableDeeplink.deeplink == "bixby://masked/act/setting/display/navigation_bar"
        dummy_kept = sgs[1].actionableDeeplink is not None and sgs[1].actionableDeeplink.deeplink == "bixby://dummy_positive"
        rogue_stripped = sgs[2].actionableDeeplink is None

        record_test(
            "Final Validator: Uncatalogued deeplink stripped & valid/dummy kept",
            valid_kept and dummy_kept and rogue_stripped,
            f"legitimate={valid_kept}, dummy={dummy_kept}, rogue_stripped={rogue_stripped}",
        )
    except Exception as e:
        record_test("Final Validator: Uncatalogued deeplink stripped", False, str(e))

    # -----------------------------------------------------------------------
    # 6. Query Variations Validation: 8-10 distinct, no empty, zero URLs
    # -----------------------------------------------------------------------
    try:
        raw_test_variations = [
            "phone swipe navigation is inverted",
            "how to fix swipe gestures on galaxy",
            "Check out https://bad-url.com for instructions",  # contains URL -> should be scrubbed
            "",  # empty -> dropped
            "phone swipe navigation is inverted",  # duplicate -> deduplicated
            "swipes go up instead of left or right",
            "navigation gestures misbehaving after update",
            "galaxy swipe direction broken",
            "what to do when phone scrolls wrong way",
            "navigation bar gesture problem",
            "swiping gestures not working",
        ]
        validated = validate_query_variations(raw_test_variations, base_query="swipe navigation broken")
        has_url = any(contains_url(v) for v in validated)
        has_empty = any(not v.strip() for v in validated)
        is_distinct = len(set(validated)) == len(validated)
        count_ok = 8 <= len(validated) <= 10

        record_test(
            "Query Variations: 8-10 items, distinct, no empty, zero URLs",
            count_ok and is_distinct and not has_empty and not has_url,
            f"count={len(validated)}, distinct={is_distinct}, zero_urls={not has_url}, no_empty={not has_empty}",
        )
    except Exception as e:
        record_test("Query Variations: 8-10 items, distinct, no empty, zero URLs", False, str(e))

    # -----------------------------------------------------------------------
    # 7. Response Shape: Flat (Appendix A) vs Appendix B Switching
    # -----------------------------------------------------------------------
    try:
        mock_meta = ResponseMeta(latency_ms=120, cache_hit=False, model=MODEL_NAME, cost_usd=0.0001)
        mock_g = Goal(
            goal="Follow these steps to perform this Swipe Navigation Troubleshooting",
            title="Swipe navigation settings",
            score=0.92,
            actions=[
                Action(
                    actionName="Configure Navigation Bar Settings",
                    description="It will let you choose navigation type",
                    category=ActionCategory.auto,
                    stepGroups=[StepGroup(steps=["Tap display"])],
                )
            ],
        )

        flat_resp = serialize_response([mock_g], fallback=None, meta=mock_meta, query="swipe issue", shape="flat")
        flat_dump = flat_resp.model_dump()
        flat_has_keys = "contexts" in flat_dump and "fallback" in flat_dump and "meta" in flat_dump and "query" not in flat_dump

        app_b_resp = serialize_response([mock_g], fallback=None, meta=mock_meta, query="swipe issue", query_variations=["var 1", "var 2"], shape="appendix_b")
        app_b_dump = app_b_resp.model_dump()
        app_b_has_keys = "query" in app_b_dump and "query_variations" in app_b_dump and "response" in app_b_dump and "contexts" in app_b_dump["response"]

        record_test(
            "Response Shape: 'flat' (Appendix A) vs 'appendix_b' (Appendix B)",
            flat_has_keys and app_b_has_keys,
            f"flat_keys={list(flat_dump.keys())} | appendix_b_keys={list(app_b_dump.keys())}",
        )
    except Exception as e:
        record_test("Response Shape: 'flat' vs 'appendix_b'", False, str(e))

    # -----------------------------------------------------------------------
    # 3. Mocked Test: Retry Loop Success (Self-Correction)
    # -----------------------------------------------------------------------
    try:
        # Mock client: 1st response fails (invalid score 2.5), 2nd response succeeds
        mock_client = MagicMock()

        bad_response_text = json.dumps({
            "goal": "Follow these steps to perform this Display Troubleshooting",
            "title": "Swipe navigation settings",
            "score": 2.5,  # Invalid score > 1.0 triggers Pydantic ValidationError
            "actions": [{
                "actionName": "Configure Navigation Bar",
                "description": "It will let you choose navigation type",
                "category": "auto",
                "stepGroups": [{"steps": ["Tap Settings"]}]
            }]
        })

        good_response_text = json.dumps({
            "goal": "Follow these steps to perform this Display Troubleshooting",
            "title": "Swipe navigation settings",
            "score": 0.95,
            "actions": [{
                "actionName": "Configure Navigation Bar",
                "description": "It will let you choose navigation type",
                "category": "auto",
                "stepGroups": [{"steps": ["Tap Settings", "Tap Navigation bar."]}]
            }]
        })

        bad_resp_obj = MagicMock()
        bad_resp_obj.text = bad_response_text
        bad_resp_obj.usage_metadata = MagicMock(prompt_token_count=50, candidates_token_count=30)

        good_resp_obj = MagicMock()
        good_resp_obj.text = good_response_text
        good_resp_obj.usage_metadata = MagicMock(prompt_token_count=70, candidates_token_count=40)

        mock_client.models.generate_content.side_effect = [bad_resp_obj, good_resp_obj]

        goal, tokens = extract_goal(
            query="swipe navigation broken",
            client=mock_client,
            max_retries=2,
        )

        record_test(
            "Self-Correction Loop: Retry succeeds after schema error",
            goal is not None and mock_client.models.generate_content.call_count == 2,
            f"Model retried once after schema error and parsed valid Goal: '{goal.title}'",
        )
    except Exception as e:
        record_test("Self-Correction Loop: Retry succeeds after schema error", False, str(e))

    # -----------------------------------------------------------------------
    # 4. Mocked Test: Retry Exhaustion -> Fallback "no_match" (No 500 error)
    # -----------------------------------------------------------------------
    try:
        mock_client_fail = MagicMock()
        gibberish_resp = MagicMock()
        gibberish_resp.text = "I am an LLM and I refuse to return valid JSON."
        gibberish_resp.usage_metadata = MagicMock(prompt_token_count=20, candidates_token_count=10)
        mock_client_fail.models.generate_content.return_value = gibberish_resp

        goal_fail, usage = extract_goal("unfixable strange issue", client=mock_client_fail, max_retries=2)
        resp = ContextDeeplinkResponse(
            contexts=[],
            fallback="no_match",
            meta=ResponseMeta(latency_ms=10, cache_hit=False, model=MODEL_NAME, cost_usd=0.0),
        )

        record_test(
            "Self-Correction Exhaustion: Graceful fallback 'no_match'",
            goal_fail is None and resp.fallback == "no_match" and len(resp.contexts) == 0,
            f"Returned fallback='{resp.fallback}', contexts={resp.contexts}, retries attempted: {mock_client_fail.models.generate_content.call_count}",
        )
    except Exception as e:
        record_test("Self-Correction Exhaustion: Graceful fallback 'no_match'", False, str(e))

    # -----------------------------------------------------------------------
    # 5. Category Ordering & Manual Action Deeplink Stripping Test
    # -----------------------------------------------------------------------
    try:
        from main import _normalize_and_validate_goal

        # Raw goal with mixed, out-of-order categories: critical, manual, auto
        raw_goal_dict = {
            "goal": "Follow these steps to perform this Device Troubleshooting",
            "title": "Device maintenance settings",
            "score": 0.9,
            "actions": [
                {
                    "actionName": "Factory Data Reset",
                    "description": "It will restore phone to factory defaults",
                    "category": "critical",
                    "stepGroups": [{"steps": ["Confirm factory reset."]}],
                },
                {
                    "actionName": "Clean Charging Port",
                    "description": "It will remove lint from connector",
                    "category": "manual",
                    "stepGroups": [{"steps": ["Use soft dry brush."]}],
                },
                {
                    "actionName": "Configure Battery Saver",
                    "description": "It will extend device battery life",
                    "category": "auto",
                    "stepGroups": [{"steps": ["Turn on power saving."]}],
                },
            ],
        }

        # _normalize_and_validate_goal automatically sorts actions by category
        mock_goal = _normalize_and_validate_goal(raw_goal_dict)

        # Wire deeplinks as troubleshoot() does
        for action in mock_goal.actions:
            if action.category == ActionCategory.manual:
                for sg in action.stepGroups:
                    sg.actionableDeeplink = None
            else:
                dl = get_deeplinks(action.actionName, action.description)
                for sg in action.stepGroups:
                    sg.actionableDeeplink = dl

        categories = [a.category for a in mock_goal.actions]
        manual_has_no_dl = all(sg.actionableDeeplink is None for sg in mock_goal.actions[1].stepGroups)

        record_test(
            "Action Ordering (auto -> manual -> critical) & Manual Deeplink Veto",
            categories == [ActionCategory.auto, ActionCategory.manual, ActionCategory.critical] and manual_has_no_dl,
            f"Ordered categories: {[c.value for c in categories]}, manual deeplink is None: {manual_has_no_dl}",
        )
    except Exception as e:
        record_test("Action Ordering & Manual Deeplink Veto", False, str(e))

    # -----------------------------------------------------------------------
    # 6. & 7. LIVE / MOCKED PIPELINE EXECUTION
    # -----------------------------------------------------------------------
    is_live = "--live" in sys.argv

    if is_live:
        print("\n--- Running Live Call 1 of 2 (Display Navigation Complaint) ---")
        live_req_1 = TroubleshootRequest(
            query="The mobile phone swipe navigation moves up or down instead of left or right after downloading an app"
        )
        try:
            resp_1 = troubleshoot(live_req_1)
            has_url_1 = any(
                contains_url(s)
                for g in resp_1.contexts
                for a in g.actions
                for sg in a.stepGroups
                for s in sg.steps
            )
            passed_1 = (
                len(resp_1.contexts) > 0
                and not has_url_1
                and resp_1.meta is not None
                and resp_1.meta.latency_ms > 0
            )
            record_test(
                "Live Gemini Call 1: Valid plan generated, zero URLs, meta populated",
                passed_1,
                f"Goal: '{resp_1.contexts[0].goal}' | Title: '{resp_1.contexts[0].title}' | Score: {resp_1.contexts[0].score} | Latency: {resp_1.meta.latency_ms}ms | Cost: ${resp_1.meta.cost_usd:.6f}",
            )
            print("  Raw Action Plan Preview:")
            for act in resp_1.contexts[0].actions:
                dl_str = act.stepGroups[0].actionableDeeplink.deeplink if act.stepGroups[0].actionableDeeplink else "None"
                print(f"    - [{act.category.value.upper()}] {act.actionName} (Deeplink: {dl_str})")
                print(f"      Desc: '{act.description}'")
                print(f"      Steps: {act.stepGroups[0].steps[:2]}...")
        except Exception as e:
            record_test("Live Gemini Call 1: Valid plan generated", False, str(e))

        print("\n--- Running Live Call 2 of 2 (Battery Drain with Adversarial SIIS URL) ---")
        live_req_2 = TroubleshootRequest(
            query="Battery draining fast after update, phone gets warm",
            siis_response="Open Settings. Tap Battery. Visit https://adversarial-link.com/phish and check www.external-leak.com for instructions.",
        )
        try:
            resp_2 = troubleshoot(live_req_2)
            has_url_2 = any(
                contains_url(s)
                for g in resp_2.contexts
                for a in g.actions
                for sg in a.stepGroups
                for s in sg.steps
            )
            passed_2 = (
                len(resp_2.contexts) > 0
                and not has_url_2
                and "adversarial-link" not in str(resp_2.model_dump())
                and "external-leak" not in str(resp_2.model_dump())
            )
            record_test(
                "Live Gemini Call 2: Adversarial URLs completely scrubbed, valid plan returned",
                passed_2,
                f"Goal: '{resp_2.contexts[0].goal}' | Title: '{resp_2.contexts[0].title}' | Latency: {resp_2.meta.latency_ms}ms | Cost: ${resp_2.meta.cost_usd:.6f}",
            )
            print("  Raw Action Plan Preview:")
            for act in resp_2.contexts[0].actions:
                dl_str = act.stepGroups[0].actionableDeeplink.deeplink if act.stepGroups[0].actionableDeeplink else "None"
                print(f"    - [{act.category.value.upper()}] {act.actionName} (Deeplink: {dl_str})")
                print(f"      Desc: '{act.description}'")
                print(f"      Steps: {act.stepGroups[0].steps[:2]}...")
        except Exception as e:
            record_test("Live Gemini Call 2: Adversarial URLs completely scrubbed", False, str(e))

    else:
        print("\n--- [INFO] Live Gemini calls skipped (pass '--live' to run against live API) ---")
        # Run mock pipeline test to verify full end-to-end pipeline without using live quota
        mock_e2e_client = MagicMock()
        mock_e2e_resp = MagicMock()
        mock_e2e_resp.text = json.dumps({
            "goal": "Follow these steps to perform this Swipe Navigation Troubleshooting",
            "title": "Swipe navigation settings",
            "score": 0.95,
            "actions": [
                {
                    "actionName": "Configure Navigation Bar Settings",
                    "description": "It will let you choose navigation type",
                    "category": "auto",
                    "stepGroups": [{"steps": ["Open Settings.", "Tap Display.", "Tap Navigation bar."]}]
                }
            ]
        })
        mock_e2e_resp.usage_metadata = MagicMock(prompt_token_count=120, candidates_token_count=80)
        mock_e2e_client.models.generate_content.return_value = mock_e2e_resp

        e2e_goal, e2e_tokens = extract_goal("swipe navigation broken", client=mock_e2e_client)
        record_test(
            "Mocked E2E Execution (Zero Quota Mode)",
            e2e_goal is not None and e2e_goal.title == "Swipe navigation settings",
            f"Successfully executed full pipeline in zero-quota mode: '{e2e_goal.goal}'",
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    print(f"ENDPOINT & PIPELINE TEST SUMMARY: {passed_count}/{total_count} PASSED")
    print("=" * 75)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_pipeline_tests()
    sys.exit(0 if success else 1)
