"""
Live End-to-End System Smoke Test (eval/smoke_test.py)

Assumes the Diagnos AI server is running locally (default: http://127.0.0.1:8000).
Makes real HTTP requests via `requests` to verify the full live system end-to-end:
1. GET /health — Expects 200 OK and {"status": "ok"}
2. POST /v1/troubleshoot — 4 real queries across domains (battery, display, camera, performance)
   Expects HTTP 200, non-empty contexts, and compliant schema shape.
3. POST /v1/troubleshoot — Off-domain/nonsense query ("book me a flight to Paris")
   Expects fallback: "no_match" and empty contexts [].
4. POST /v1/clarify — Ambiguous close-scored hypotheses with no answer
   Expects needs_clarification=True and a non-empty question.
5. POST /v1/clarify — Clarification answer provided
   Expects needs_clarification=False and a valid re-ranked plan.
6. POST /v1/troubleshoot-image — Multipart test image upload
   Expects HTTP 200 and a valid actionable troubleshooting plan.

Output matches the style of eval/test_retrieval.py ([PASS]/[FAIL] and final summary).
"""

import io
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

# Enable ANSI escape sequence processing on Windows consoles
os.system("")
# Ensure proper Unicode display on Windows consoles (e.g. UTF-8 characters)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 30


def create_test_image_bytes() -> bytes:
    """Generates a small in-memory test JPEG image for vision testing."""
    try:
        from PIL import Image

        img = Image.new("RGB", (100, 100), color=(40, 40, 40))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        # Minimal valid 1x1 standalone JPEG byte sequence fallback
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08"
            b"\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13"
            b"\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff"
            b"\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01"
            b"\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
            b"\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
        )


def extract_contexts_and_fallback(resp_json: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Normalizes extraction across both 'flat' (Appendix A) and 'appendix_b' shapes."""
    if "response" in resp_json and isinstance(resp_json["response"], dict):
        return resp_json["response"].get("contexts", []), resp_json["response"].get("fallback")
    return resp_json.get("contexts", []), resp_json.get("fallback")


def run_smoke_tests(base_url: str = DEFAULT_BASE_URL) -> bool:
    print("=" * 75)
    print("RUNNING LIVE END-TO-END SYSTEM SMOKE TESTS")
    print(f"Target Server: {base_url}")
    print("=" * 75)

    passed_count = 0
    total_count = 0

    category_counts = {
        "Core Pipeline": [0, 0],
        "Edge Cases": [0, 0],
        "Multimodal": [0, 0],
    }

    def record(name: str, passed: bool, detail: str = "", category: str = "Core Pipeline"):
        nonlocal passed_count, total_count
        total_count += 1
        if category in category_counts:
            category_counts[category][1] += 1
        status = "[PASS]" if passed else "[FAIL]"
        if passed:
            passed_count += 1
            if category in category_counts:
                category_counts[category][0] += 1
        print(f"{status} {name}")
        if detail:
            print(f"       Detail: {detail}")

    # -----------------------------------------------------------------------
    # Test 1: GET /health
    # -----------------------------------------------------------------------
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        status_ok = r.status_code == 200
        body = r.json() if status_ok else {}
        body_ok = body.get("status") == "ok"
        record(
            "GET /health: Exact shape {'status': 'ok'} (HTTP 200)",
            status_ok and body_ok,
            f"http_status={r.status_code}, response={body}",
        )
    except Exception as e:
        record(
            "GET /health: Exact shape {'status': 'ok'} (HTTP 200)",
            False,
            f"Connection failed to {base_url}. Is the server running? ({e})",
        )
        print("\n[ABORT] Cannot reach server. Please start the server and rerun.\n")
        return False

    # -----------------------------------------------------------------------
    # Test 2: POST /v1/troubleshoot with 4 real queries across domains
    # -----------------------------------------------------------------------
    domain_queries = [
        ("Battery", "my battery drains fast overnight"),
        ("Display", "screen flickers with horizontal lines"),
        ("Camera", "camera app crashes with black preview"),
        ("Performance", "phone is lagging and apps freeze"),
    ]

    for idx, (domain, query) in enumerate(domain_queries):
        if idx > 0:
            time.sleep(3)
        try:
            start_t = time.perf_counter()
            r = requests.post(
                f"{base_url}/v1/troubleshoot",
                json={"query": query},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            elapsed_ms = int((time.perf_counter() - start_t) * 1000)

            if r.status_code != 200:
                record(
                    f"POST /v1/troubleshoot [{domain}]: '{query}'",
                    False,
                    f"http_status={r.status_code}, body={r.text[:200]}",
                )
                continue

            resp_json = r.json()
            contexts, fallback = extract_contexts_and_fallback(resp_json)

            has_contexts = isinstance(contexts, list) and len(contexts) >= 1
            valid_fallback = fallback is None
            top_goal = contexts[0] if has_contexts else {}
            title = top_goal.get("title", "")
            score = top_goal.get("score", 0.0)
            actions = top_goal.get("actions", [])
            has_actions = len(actions) > 0

            # Verify deeplinks attached to actions
            first_deeplink = None
            if has_actions and actions[0].get("stepGroups"):
                first_dl_obj = actions[0]["stepGroups"][0].get("actionableDeeplink")
                if first_dl_obj:
                    first_deeplink = first_dl_obj.get("deeplink")

            success = has_contexts and valid_fallback and has_actions and (0.0 <= score <= 1.0)
            detail = (
                f"title='{title}' | score={score} | actions={len(actions)} | "
                f"top_deeplink={first_deeplink} | latency={elapsed_ms}ms"
            )
            record(f"POST /v1/troubleshoot [{domain}]: '{query}'", success, detail)

        except Exception as e:
            record(f"POST /v1/troubleshoot [{domain}]: '{query}'", False, str(e))

    # -----------------------------------------------------------------------
    # Test 3: POST /v1/troubleshoot with off-domain/nonsense query
    # -----------------------------------------------------------------------
    time.sleep(3)
    nonsense_query = "book me a flight to Paris"
    try:
        start_t = time.perf_counter()
        r = requests.post(
            f"{base_url}/v1/troubleshoot",
            json={"query": nonsense_query},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        elapsed_ms = int((time.perf_counter() - start_t) * 1000)

        if r.status_code != 200:
            record(
                f"POST /v1/troubleshoot [Off-Domain]: '{nonsense_query}'",
                False,
                f"http_status={r.status_code}, body={r.text[:200]}",
                category="Edge Cases",
            )
        else:
            resp_json = r.json()
            contexts, fallback = extract_contexts_and_fallback(resp_json)
            is_fallback = fallback in ("no_match", "no_match_offdomain_heuristic")
            empty_contexts = len(contexts) == 0
            success = is_fallback and empty_contexts

            record(
                f"POST /v1/troubleshoot [Off-Domain]: '{nonsense_query}' triggers fallback",
                success,
                f"fallback='{fallback}', contexts_len={len(contexts)} | latency={elapsed_ms}ms",
                category="Edge Cases",
            )
    except Exception as e:
        record(f"POST /v1/troubleshoot [Off-Domain]: '{nonsense_query}'", False, str(e), category="Edge Cases")

    # -----------------------------------------------------------------------
    # Test 4: POST /v1/clarify without answer (close-scored hypotheses)
    # -----------------------------------------------------------------------
    time.sleep(3)
    ambiguous_payload = {
        "query": "phone is acting up and draining battery",
        "hypotheses": [
            {"title": "Battery fast drain", "score": 0.85},
            {"title": "Screen brightness high", "score": 0.80},
        ],
        "clarification_answer": None,
        "gap_threshold": 0.15,
    }
    try:
        r = requests.post(
            f"{base_url}/v1/clarify",
            json=ambiguous_payload,
            timeout=5,
        )
        if r.status_code != 200:
            record(
                "POST /v1/clarify [Ambiguous]: Expects needs_clarification=True",
                False,
                f"http_status={r.status_code}, body={r.text[:200]}",
                category="Edge Cases",
            )
        else:
            body = r.json()
            needs_clar = body.get("needs_clarification") is True
            question = body.get("question") or ""
            success = needs_clar and len(question.strip()) > 0
            record(
                "POST /v1/clarify [Ambiguous]: Returns needs_clarification=True and question",
                success,
                f"needs_clarification={needs_clar} | question='{question}'",
                category="Edge Cases",
            )
    except Exception as e:
        record("POST /v1/clarify [Ambiguous]: Expects needs_clarification=True", False, str(e), category="Edge Cases")

    # -----------------------------------------------------------------------
    # Test 5: POST /v1/clarify with answer provided
    # -----------------------------------------------------------------------
    time.sleep(3)
    resolved_payload = {
        "query": "phone is acting up and draining battery",
        "hypotheses": [
            {"title": "Battery fast drain", "score": 0.85},
            {"title": "Screen brightness high", "score": 0.80},
        ],
        "clarification_answer": "It drains overnight even when the screen is completely locked and dark",
        "gap_threshold": 0.15,
    }
    try:
        start_t = time.perf_counter()
        r = requests.post(
            f"{base_url}/v1/clarify",
            json=resolved_payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        elapsed_ms = int((time.perf_counter() - start_t) * 1000)

        if r.status_code != 200:
            record(
                "POST /v1/clarify [Answered]: Folds answer into query and re-ranks",
                False,
                f"http_status={r.status_code}, body={r.text[:200]}",
                category="Edge Cases",
            )
        else:
            body = r.json()
            needs_clar = body.get("needs_clarification") is False
            contexts, fallback = extract_contexts_and_fallback(body)
            has_contexts = len(contexts) > 0
            top_title = contexts[0].get("title", "") if has_contexts else ""
            success = needs_clar and has_contexts and fallback is None
            record(
                "POST /v1/clarify [Answered]: Folds answer into query and re-ranks plan",
                success,
                f"needs_clarification={needs_clar} | top_title='{top_title}' | "
                f"contexts={len(contexts)} | latency={elapsed_ms}ms",
                category="Edge Cases",
            )
    except Exception as e:
        record("POST /v1/clarify [Answered]: Folds answer into query and re-ranks", False, str(e), category="Edge Cases")

    # -----------------------------------------------------------------------
    # Test 6: POST /v1/troubleshoot-image with test image file
    # -----------------------------------------------------------------------
    time.sleep(3)
    try:
        img_bytes = create_test_image_bytes()
        files = {"file": ("device_photo.jpg", img_bytes, "image/jpeg")}

        start_t = time.perf_counter()
        r = requests.post(
            f"{base_url}/v1/troubleshoot-image",
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        elapsed_ms = int((time.perf_counter() - start_t) * 1000)

        if r.status_code != 200:
            record(
                "POST /v1/troubleshoot-image: Processes test image and returns plan",
                False,
                f"http_status={r.status_code}, body={r.text[:200]}",
                category="Multimodal",
            )
        else:
            body = r.json()
            contexts, fallback = extract_contexts_and_fallback(body)
            has_contexts = len(contexts) > 0
            top_title = contexts[0].get("title", "") if has_contexts else ""
            success = has_contexts and fallback is None
            record(
                "POST /v1/troubleshoot-image: Processes test image and returns actionable plan",
                success,
                f"top_title='{top_title}' | contexts={len(contexts)} | latency={elapsed_ms}ms",
                category="Multimodal",
            )
    except Exception as e:
        record("POST /v1/troubleshoot-image: Processes test image and returns plan", False, str(e), category="Multimodal")

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    print("=" * 75)
    print(f"SMOKE TEST SUMMARY: {passed_count}/{total_count} PASSED")
    print("=" * 75)

    COLOR_GREEN = "\033[92m\033[1m"
    COLOR_RED = "\033[91m\033[1m"
    COLOR_RESET = "\033[0m"

    is_all_passed = (passed_count == total_count and total_count > 0)
    status_line = (
        f"{COLOR_GREEN}SYSTEM STATUS: READY FOR DEMO{COLOR_RESET}"
        if is_all_passed
        else f"{COLOR_RED}SYSTEM STATUS: ISSUES DETECTED — SEE ABOVE{COLOR_RESET}"
    )

    core_desc = f"Core Pipeline ({category_counts['Core Pipeline'][0]}/{category_counts['Core Pipeline'][1]}: Health, 4 Domains)"
    edge_desc = f"Edge Cases ({category_counts['Edge Cases'][0]}/{category_counts['Edge Cases'][1]}: Off-Domain, Clarify)"
    multi_desc = f"Multimodal ({category_counts['Multimodal'][0]}/{category_counts['Multimodal'][1]}: Vision Upload)"

    print(status_line)
    print(f"Categories: {core_desc} | {edge_desc} | {multi_desc}")
    print("=" * 75)

    return passed_count == total_count


if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    success = run_smoke_tests(base_url=target_url)
    sys.exit(0 if success else 1)
