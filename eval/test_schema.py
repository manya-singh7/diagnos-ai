import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from pydantic import ValidationError
from schema import (
    Action,
    ActionCategory,
    ContextDeeplinkResponse,
    Deeplink,
    Goal,
    ResponseMeta,
    StepGroup,
    TroubleshootRequest,
    contains_url,
    normalize_title,
    scrub_urls,
)


def run_tests():
    print("=" * 70)
    print("RUNNING SCHEMA & VALIDATOR TESTS (Priority 2)")
    print("=" * 70)

    results = []

    def record_test(name: str, passed: bool, detail: str = ""):
        results.append((name, passed, detail))
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {name}")
        if detail:
            print(f"       Detail: {detail}")

    # -----------------------------------------------------------------------
    # 1. Valid End-to-End Example (matches Appendix B)
    # -----------------------------------------------------------------------
    try:
        valid_goal = Goal(
            goal="Follow these steps to perform this Swipe Navigation Troubleshooting",
            title="Swipe navigation settings",
            score=0.93,
            actions=[
                Action(
                    actionName="Configure Navigation Bar Settings",
                    description="It will let you choose navigation type",
                    category=ActionCategory.auto,
                    stepGroups=[
                        StepGroup(
                            steps=[
                                "Navigate to and open Settings.",
                                "Tap on Display.",
                                "Tap on Navigation bar.",
                                "Select your preferred navigation type between Buttons and Swipe gestures.",
                                "Optionally toggle on Gesture hint to display guidance lines at the bottom of the screen.",
                            ],
                            actionableDeeplink=Deeplink(
                                deeplink="bixby://dummy_positive",
                                description="Open navigation bar settings under Display",
                                message="choose navigation type in Display settings",
                            ),
                        )
                    ],
                )
            ],
        )
        response = ContextDeeplinkResponse(
            contexts=[valid_goal],
            meta=ResponseMeta(
                latency_ms=212,
                cache_hit=True,
                model="gemini-2.5-flash",
                cost_usd=0.0,
            ),
        )
        record_test("Valid Appendix B Payload", True, f"Parsed 1 goal with score {valid_goal.score}")
    except Exception as e:
        record_test("Valid Appendix B Payload", False, str(e))

    # -----------------------------------------------------------------------
    # 2. Valid Fallback "no_match" Response
    # -----------------------------------------------------------------------
    try:
        fallback_resp = ContextDeeplinkResponse(
            contexts=[],
            fallback="no_match",
            meta=ResponseMeta(latency_ms=45, cache_hit=False, cost_usd=0.0),
        )
        record_test("Valid Fallback 'no_match' Payload", True, f"contexts: [], fallback: '{fallback_resp.fallback}'")
    except Exception as e:
        record_test("Valid Fallback 'no_match' Payload", False, str(e))

    # -----------------------------------------------------------------------
    # 3. Invalid: Fallback "no_match" with non-empty contexts
    # -----------------------------------------------------------------------
    try:
        ContextDeeplinkResponse(
            contexts=[valid_goal],
            fallback="no_match",
            meta=ResponseMeta(latency_ms=50, cache_hit=False, cost_usd=0.0),
        )
        record_test("Invalid: fallback 'no_match' with non-empty contexts", False, "Should have failed validation")
    except ValidationError as e:
        record_test("Invalid: fallback 'no_match' with non-empty contexts", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # -----------------------------------------------------------------------
    # 4. Description Validation: Must start with "It will", 5-7 words
    # -----------------------------------------------------------------------
    # (a) Does not start with "It will"
    try:
        Action(
            actionName="Battery Saver",
            description="This will extend device battery life",
            category=ActionCategory.auto,
            stepGroups=[StepGroup(steps=["Turn on battery saver"])]
        )
        record_test("Invalid: Description without 'It will'", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Description without 'It will'", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # (b) Too short (< 5 words)
    try:
        Action(
            actionName="Battery Saver",
            description="It will save battery",  # 4 words
            category=ActionCategory.auto,
            stepGroups=[StepGroup(steps=["Turn on battery saver"])]
        )
        record_test("Invalid: Description too short (4 words)", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Description too short (4 words)", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # (c) Too long (> 7 words)
    try:
        Action(
            actionName="Battery Saver",
            description="It will greatly extend the overall battery life of device",  # 10 words
            category=ActionCategory.auto,
            stepGroups=[StepGroup(steps=["Turn on battery saver"])]
        )
        record_test("Invalid: Description too long (10 words)", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Description too long (10 words)", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # -----------------------------------------------------------------------
    # 5. Title Validation: 2-3 words, Sentence case
    # -----------------------------------------------------------------------
    # (a) Title Case instead of Sentence case
    try:
        Goal(
            goal="Follow these steps to perform this Display Troubleshooting",
            title="Swipe Navigation Settings",  # Title Case (Navigation and Settings capitalized)
            score=0.9,
            actions=[],
        )
        record_test("Invalid: Title in Title Case instead of sentence case", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Title in Title Case instead of sentence case", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # (b) Too few words (< 2)
    try:
        Goal(
            goal="Follow these steps to perform this Display Troubleshooting",
            title="Navigation",
            score=0.9,
            actions=[],
        )
        record_test("Invalid: Title has 1 word", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Title has 1 word", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # (c) Too many words (> 3)
    try:
        Goal(
            goal="Follow these steps to perform this Display Troubleshooting",
            title="Swipe navigation bar settings",  # 4 words
            score=0.9,
            actions=[],
        )
        record_test("Invalid: Title has 4 words", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Title has 4 words", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # (d) Valid acronym in title
    try:
        g = Goal(
            goal="Follow these steps to perform this Wi-Fi Troubleshooting",
            title="Reset Wi-Fi settings",  # 3 words, Wi-Fi is proper acronym
            score=0.85,
            actions=[],
        )
        record_test("Valid: Title with acronym 'Wi-Fi'", True, f"Accepted: '{g.title}'")
    except Exception as e:
        record_test("Valid: Title with acronym 'Wi-Fi'", False, str(e))

    # (e) Valid proper nouns in title (Bluetooth, Bixby, Knox, Dolby, NFC)
    proper_titles = [
        "Reset Bluetooth settings",
        "Adjust Bixby routines",
        "Configure Knox security",
        "Check Dolby Atmos",
        "Toggle NFC reader",
    ]
    all_passed = True
    err_msg = ""
    for pt in proper_titles:
        try:
            Goal(
                goal="Follow these steps to perform this Device Troubleshooting",
                title=pt,
                score=0.9,
                actions=[],
            )
        except Exception as ex:
            all_passed = False
            err_msg = f"Failed on '{pt}': {ex}"
            break
    record_test("Valid: Titles with expanded proper nouns (Bluetooth, Bixby, Knox, etc.)", all_passed, err_msg or "All passed")

    # (f) normalize_title auto-corrects Title Case to sentence case
    raw_title = "Reset Bluetooth Settings"
    normalized = normalize_title(raw_title)
    record_test(
        "normalize_title: Auto-corrects Title Case to sentence case",
        normalized == "Reset Bluetooth settings",
        f"'{raw_title}' -> '{normalized}'",
    )

    # -----------------------------------------------------------------------
    # 6. Goal Syntax Validation
    # -----------------------------------------------------------------------
    try:
        Goal(
            goal="Fix the display flickering issue on Samsung phone",
            title="Display flickering fix",
            score=0.9,
            actions=[],
        )
        record_test("Invalid: Goal syntax not following template", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Goal syntax not following template", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # -----------------------------------------------------------------------
    # 7. ActionName Validation: Title Case required
    # -----------------------------------------------------------------------
    try:
        Action(
            actionName="configure navigation bar settings",  # all lowercase
            description="It will let you choose navigation type",
            category=ActionCategory.auto,
            stepGroups=[StepGroup(steps=["Tap display"])],
        )
        record_test("Invalid: actionName all lowercase", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: actionName all lowercase", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # -----------------------------------------------------------------------
    # 8. Category Constraints
    # -----------------------------------------------------------------------
    # (a) Manual action cannot carry actionableDeeplink
    try:
        Action(
            actionName="Clean Charging Port",
            description="It will remove dirt from connector",
            category=ActionCategory.manual,
            stepGroups=[
                StepGroup(
                    steps=["Gently use a wooden toothpick to remove lint."],
                    actionableDeeplink=Deeplink(
                        deeplink="bixby://dummy_positive",
                        description="Invalid deeplink on manual action",
                    ),
                )
            ],
        )
        record_test("Invalid: Manual action carrying actionableDeeplink", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Manual action carrying actionableDeeplink", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # (b) Critical action must be ordered last
    try:
        Goal(
            goal="Follow these steps to perform this System Troubleshooting",
            title="System performance reset",
            score=0.9,
            actions=[
                Action(
                    actionName="Factory Data Reset",
                    description="It will restore phone to defaults",
                    category=ActionCategory.critical,
                    stepGroups=[StepGroup(steps=["Confirm factory data reset."])],
                ),
                Action(
                    actionName="Clear App Cache",
                    description="It will free up cached memory",
                    category=ActionCategory.auto,
                    stepGroups=[StepGroup(steps=["Clear cache in settings."])],
                ),
            ],
        )
        record_test("Invalid: Critical action before auto action", False, "Should have failed")
    except ValidationError as e:
        record_test("Invalid: Critical action before auto action", True, f"Correctly caught: {e.errors()[0]['msg']}")

    # -----------------------------------------------------------------------
    # 9. Zero-URL Scrubbing & Detection
    # -----------------------------------------------------------------------
    # (a) Raw URL in step is scrubbed automatically
    sg = StepGroup(
        steps=[
            "Visit https://www.samsung.com/support for firmware download.",
            "Tap on Software update in Settings.",
        ]
    )
    has_url = any(contains_url(s) for s in sg.steps)
    record_test(
        "Zero-URL: Step scrubbed of web URL",
        not has_url and "https://" not in sg.steps[0],
        f"Scrubbed step: '{sg.steps[0]}'",
    )

    # (b) Markdown link [anchor](url) converted to anchor text
    sg_md = StepGroup(
        steps=["Check out [Samsung Care](https://samsung.com/help) for guide."]
    )
    record_test(
        "Zero-URL: Markdown link scrubbed to anchor text",
        "https://" not in sg_md.steps[0] and "Samsung Care" in sg_md.steps[0],
        f"Scrubbed step: '{sg_md.steps[0]}'",
    )

    # (c) TroubleshootRequest sanitizes siis_response and query
    req = TroubleshootRequest(
        query="Phone lags, check https://badlink.com for help",
        siis_response="Open Settings. Visit http://samsung.com/support and www.google.com for more info.",
    )
    record_test(
        "Zero-URL: TroubleshootRequest sanitizes query & siis_response",
        not contains_url(req.query) and not contains_url(req.siis_response),
        f"Cleaned siis_response: '{req.siis_response}' | query: '{req.query}'",
    )

    # (d) Harmless text preservation: ensure Wi-Fi, e.g., and Settings > Battery are untouched
    harmless_inputs = [
        "Wi-Fi",
        "e.g.",
        "Settings > Battery",
        "Version 3.0",
        "Follow step 1. Then step 2.",
        "Tap Display > Navigation bar.",
        "Adjust screen timeout e.g. 30 seconds",
        "Connect to Wi-Fi network and check speed",
    ]
    harmless_preserved = True
    harmless_err = ""
    for text in harmless_inputs:
        scrubbed = scrub_urls(text)
        has_url = contains_url(text)
        if scrubbed != text or has_url:
            harmless_preserved = False
            harmless_err = f"Mangled '{text}' -> '{scrubbed}' (contains_url={has_url})"
            break

    record_test(
        "Zero-URL: Harmless text preservation (Wi-Fi, e.g., Settings > Battery)",
        harmless_preserved,
        harmless_err or "All harmless strings left exactly untouched without false-positive URL detection",
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    print(f"TEST SUMMARY: {passed_count}/{total_count} PASSED")
    print("=" * 70)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
