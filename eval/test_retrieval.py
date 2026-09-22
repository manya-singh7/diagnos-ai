"""
Tests for Person D's BM25 Retrieval System (backend/retrieval).
Verifies:
1. Obvious queries map to the correct catalog top matches.
2. Sample data fallback warning is logged/printed when deeplinks.json is absent.
3. Top-k ranking returns correct count and descending order.
4. Weak/unrelated queries yield scores below threshold or 0.0.
5. get_deeplinks() integration preserves schema rules and safety guards.
"""

import logging
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from main import ActionCategory, get_deeplinks
from retrieval.bm25_retriever import BM25Retriever, get_retriever

logger = logging.getLogger("test_retrieval")

def run_retrieval_tests():
    print("=" * 75)
    print("RUNNING RETRIEVAL TESTS (Person D - BM25)")
    print("=" * 75)

    passed_count = 0
    total_count = 0

    def record(name: str, passed: bool, detail: str = ""):
        nonlocal passed_count, total_count
        total_count += 1
        status = "[PASS]" if passed else "[FAIL]"
        if passed:
            passed_count += 1
        print(f"{status} {name}")
        if detail:
            print(f"       Detail: {detail}")

    # -----------------------------------------------------------------------
    # Test 1: Sample Data Detection & Warning
    # -----------------------------------------------------------------------
    try:
        retriever = get_retriever(force_reload=True)
        catalog_is_sample = retriever.is_sample
        record(
            "Sample Data Detection: Correctly flags sample data when deeplinks.json absent",
            catalog_is_sample,
            f"is_sample={catalog_is_sample}, path={retriever.catalog_path.name}",
        )
    except Exception as e:
        record("Sample Data Detection", False, str(e))

    # -----------------------------------------------------------------------
    # Test 2: Obvious Queries Map to Correct Catalog Entries (Top Matches)
    # -----------------------------------------------------------------------
    ground_truth_test_cases = [
        (
            "Display Navigation Bar",
            "Configure Navigation Bar Settings",
            "navigation bar swipe gestures settings",
            "bixby://masked/act/setting/display/navigation_bar",
        ),
        (
            "Battery Power Saving",
            "Enable Power Saving Mode",
            "turn on battery power saving mode",
            "bixby://masked/act/setting/battery/power_saving",
        ),
        (
            "Battery App Usage",
            "Check Battery Usage Details",
            "view battery usage by application to see which app is draining battery",
            "bixby://masked/act/setting/battery/usage",
        ),
        (
            "Device Care Memory Clean",
            "Clean Memory in Device Care",
            "free up RAM in device care memory and clear background memory",
            "bixby://masked/act/setting/device_care/memory_clean",
        ),
        (
            "Display Adaptive Brightness",
            "Toggle Adaptive Brightness",
            "turn off adaptive brightness in Display settings",
            "bixby://masked/act/setting/display/adaptive_brightness",
        ),
        (
            "Camera Settings Reset",
            "Reset Camera Settings",
            "reset camera app preferences to default",
            "bixby://masked/act/setting/camera/reset",
        ),
    ]

    all_matched = True
    match_details = []
    for test_label, action_name, desc, expected_uri in ground_truth_test_cases:
        top_item, score = retriever.get_top_match(f"{action_name} {desc}")
        top_uri = top_item.get("deeplink") if top_item else None
        is_ok = (top_uri == expected_uri) and (score >= 0.5)
        if not is_ok:
            all_matched = False
        match_details.append(f"{test_label}: expected={expected_uri} got={top_uri} (score={score})")

    record(
        "Obvious Query Accuracy: All 6 domain actions retrieve exact catalog deeplink",
        all_matched,
        " | ".join(match_details),
    )

    # -----------------------------------------------------------------------
    # Test 3: Top-K Ranking & Order
    # -----------------------------------------------------------------------
    try:
        top_3 = retriever.retrieve("battery power saving and battery usage details", top_k=3)
        count_ok = len(top_3) <= 3 and len(top_3) > 1
        scores_descending = all(top_3[i][1] >= top_3[i + 1][1] for i in range(len(top_3) - 1))
        record(
            "Top-K Ranking: Returns top-k items in descending order of relevance score",
            count_ok and scores_descending,
            f"count={len(top_3)}, scores={[s for _, s in top_3]}",
        )
    except Exception as e:
        record("Top-K Ranking", False, str(e))

    # -----------------------------------------------------------------------
    # Test 4: Unrelated Query Scored Below Threshold / Zero
    # -----------------------------------------------------------------------
    try:
        top_unrelated, score_unrelated = retriever.get_top_match("Completely unrelated query about kitchen refrigerator")
        below_thresh = (top_unrelated is None) or (score_unrelated < 0.5)
        record(
            "Threshold Guard: Unrelated / noisy queries score below 0.5 threshold",
            below_thresh,
            f"score={score_unrelated}",
        )
    except Exception as e:
        record("Threshold Guard", False, str(e))

    # -----------------------------------------------------------------------
    # Test 5: Integration with get_deeplinks() Signature & Safety Guards
    # -----------------------------------------------------------------------
    try:
        # Strong match -> catalog deeplink
        dl_nav = get_deeplinks("Configure Navigation Bar Settings", "navigation bar", category=ActionCategory.auto)
        nav_ok = dl_nav is not None and dl_nav.deeplink == "bixby://masked/act/setting/display/navigation_bar"

        # Auto + real settings screen without exact catalog match -> dummy_positive
        dl_touch = get_deeplinks("Adjust Touch Sensitivity", "open settings touch sensitivity", category=ActionCategory.auto)
        touch_ok = dl_touch is not None and dl_touch.deeplink == "bixby://dummy_positive"

        # Manual veto -> None
        dl_manual = get_deeplinks("Clean Charging Port", "clean lint from port", category=ActionCategory.manual)
        manual_ok = dl_manual is None

        # Critical non-settings -> None
        dl_reboot = get_deeplinks("Reboot Galaxy Device", "restart system processes", category=ActionCategory.critical)
        reboot_ok = dl_reboot is None

        integrated_ok = nav_ok and touch_ok and manual_ok and reboot_ok
        record(
            "Integration with get_deeplinks(): Preserves auto/catalog, dummy_positive, manual veto, and critical guard",
            integrated_ok,
            f"nav={getattr(dl_nav, 'deeplink', None)}, touch={getattr(dl_touch, 'deeplink', None)}, manual={dl_manual}, reboot={dl_reboot}",
        )
    except Exception as e:
        record("Integration with get_deeplinks()", False, str(e))

    print("=" * 75)
    print(f"RETRIEVAL TEST SUMMARY: {passed_count}/{total_count} PASSED")
    print("=" * 75)
    return passed_count == total_count

if __name__ == "__main__":
    success = run_retrieval_tests()
    sys.exit(0 if success else 1)
