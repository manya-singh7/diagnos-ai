"""
Evaluation script (eval/run_eval.py) for Diagnos AI.
- Loads queries.json (fallback queries.sample.json with clear warning).
- Calls /v1/troubleshoot for each query.
- Validates the response against schema.py.
- Verifies zero URL leaks.
- Serializes results to results.jsonl in the Appendix B shape.
- Computes and prints summary: pass rate, P50 and P95 latency.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

# Enforce Appendix B shape for evaluation output serialization
os.environ["RESPONSE_SHAPE"] = "appendix_b"

from main import (
    MODEL_NAME,
    app,
    gemini_client,
    troubleshoot,
)
from schema import (
    AppendixBResponse,
    Goal,
    TroubleshootRequest,
    contains_url,
)

logger = logging.getLogger("diagnos_ai.eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def find_queries_path(explicit_path: str = "") -> Tuple[Path, bool]:
    """Resolves queries.json or falls back to queries.sample.json with a warning."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p, "sample" in p.name.lower()

    root_dir = Path(__file__).resolve().parent.parent
    real_path = root_dir / "queries.json"
    if real_path.exists():
        return real_path, False

    sample_path = root_dir / "queries.sample.json"
    if sample_path.exists():
        return sample_path, True

    cwd_real = Path.cwd() / "queries.json"
    if cwd_real.exists():
        return cwd_real, False

    cwd_sample = Path.cwd() / "queries.sample.json"
    if cwd_sample.exists():
        return cwd_sample, True

    raise FileNotFoundError("Could not find 'queries.json' or 'queries.sample.json'.")


def load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "queries" in data:
        return data["queries"]
    return []


def check_url_leaks_in_obj(obj: Any) -> List[str]:
    """Recursively checks for any URL leak in strings across nested dicts/lists."""
    leaks = []
    if isinstance(obj, str):
        if contains_url(obj):
            leaks.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            leaks.extend(check_url_leaks_in_obj(v))
    elif isinstance(obj, list):
        for item in obj:
            leaks.extend(check_url_leaks_in_obj(item))
    return leaks


# Realistic schema-compliant mocked goal templates for offline zero-quota testing
OFFLINE_DOMAIN_TEMPLATES = {
    "Display": {
        "goal": "Follow these steps to perform this Swipe Navigation Troubleshooting",
        "title": "Swipe navigation settings",
        "score": 0.94,
        "actions": [
            {
                "actionName": "Configure Navigation Bar Settings",
                "description": "It will let you choose navigation type",
                "category": "auto",
                "stepGroups": [
                    {"steps": ["Open Settings.", "Tap Display.", "Tap Navigation bar."]}
                ],
            },
            {
                "actionName": "Reboot Phone in Safe Mode",
                "description": "It will isolate problem third party apps",
                "category": "critical",
                "stepGroups": [
                    {"steps": ["Hold Power off.", "Tap Safe mode."]}
                ],
            },
        ],
    },
    "Battery": {
        "goal": "Follow these steps to perform this Battery Fast Drain Troubleshooting",
        "title": "Battery fast drain",
        "score": 0.92,
        "actions": [
            {
                "actionName": "Enable Power Saving Mode",
                "description": "It will help reduce total battery consumption",
                "category": "auto",
                "stepGroups": [
                    {"steps": ["Open Settings.", "Tap Battery.", "Turn on Power saving."]}
                ],
            },
            {
                "actionName": "Clean Charging Port",
                "description": "It will clear lint from connector pins",
                "category": "manual",
                "stepGroups": [
                    {"steps": ["Inspect connector.", "Use soft dry brush."]}
                ],
            },
        ],
    },
    "Camera": {
        "goal": "Follow these steps to perform this Camera Reset Configuration",
        "title": "Camera app reset",
        "score": 0.89,
        "actions": [
            {
                "actionName": "Reset Camera App Settings",
                "description": "It will restore default camera options cleanly",
                "category": "auto",
                "stepGroups": [
                    {"steps": ["Open Camera.", "Tap Camera Settings icon.", "Tap Reset settings."]}
                ],
            }
        ],
    },
    "Performance": {
        "goal": "Follow these steps to perform this Memory Clean Troubleshooting",
        "title": "Memory clean optimization",
        "score": 0.91,
        "actions": [
            {
                "actionName": "Clean Memory in Device Care",
                "description": "It will free background memory and ram",
                "category": "auto",
                "stepGroups": [
                    {"steps": ["Open Settings.", "Tap Device care.", "Tap Memory.", "Tap Clean now."]}
                ],
            }
        ],
    },
}


def create_offline_mock_response(query: str, domain: str = "") -> str:
    template = OFFLINE_DOMAIN_TEMPLATES.get(domain)
    if not template:
        # Fallback to Display template
        template = OFFLINE_DOMAIN_TEMPLATES["Display"]
    return json.dumps({"goals": [template]})


class RateLimitMonitor(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rate_limits: List[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            self.rate_limits.append(msg)


def run_evaluation(
    queries_path_str: str = "",
    output_path_str: str = "results.jsonl",
    live: bool = False,
    max_queries: int = 10,
    delay: float = 15.0,
) -> Dict[str, Any]:
    queries_path, is_sample = find_queries_path(queries_path_str)

    if is_sample:
        warn_msg = (
            "[WARNING] 'queries.json' not found in project root. "
            "Falling back to 'queries.sample.json'. "
            "Sample data is in use — results are not final numbers."
        )
        logger.warning(warn_msg)
        print(warn_msg)

    queries_data = load_queries(queries_path)
    if not queries_data:
        print("[ERROR] No queries loaded for evaluation.")
        return {"pass_rate": 0.0, "total": 0}

    queries = queries_data[:max_queries]
    total_queries = len(queries)

    print("\n" + "=" * 75)
    print(f"DIAGNOS AI — PIPELINE EVALUATION ({'LIVE API' if live else 'OFFLINE ZERO-QUOTA'})")
    print(f"Dataset: {queries_path.name} ({total_queries} queries) | Output: {output_path_str}")
    if live and delay > 0:
        print(f"Pacing delay: {delay:.1f}s between queries to respect free-tier rate limits (<= 5 RPM)")
    print("=" * 75)

    if live and gemini_client is None:
        print("[ERROR] Live evaluation requested via --live, but GEMINI_API_KEY is not configured.")
        sys.exit(1)

    # Attach rate limit monitor to loggers
    rate_monitor = RateLimitMonitor()
    logging.getLogger().addHandler(rate_monitor)
    logging.getLogger("diagnos_ai").addHandler(rate_monitor)

    output_path = Path(output_path_str)
    # Ensure parent dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_records: List[Dict[str, Any]] = []
    latencies: List[int] = []
    passed_validations = 0
    total_url_leaks = 0
    cache_hits = 0

    for idx, q_entry in enumerate(queries):
        if live and idx > 0 and delay > 0:
            print(f"--- Pacing query {idx+1}/{total_queries}: waiting {delay:.1f}s to respect free-tier rate limit ---")
            time.sleep(delay)

        qid = q_entry.get("id", f"q{idx+1}")
        q_text = q_entry.get("query", "")
        domain = q_entry.get("domain", "")

        req = TroubleshootRequest(query=q_text)

        t_start = time.perf_counter()
        if live:
            resp_obj = troubleshoot(req)
        else:
            # Mock Gemini generation with domain-accurate compliant JSON
            mock_resp = MagicMock()
            mock_resp.text = create_offline_mock_response(q_text, domain)
            mock_resp.usage_metadata = MagicMock(prompt_token_count=120, candidates_token_count=65)

            with patch("main.gemini_client.models.generate_content", return_value=mock_resp):
                resp_obj = troubleshoot(req)

        dur_ms = int((time.perf_counter() - t_start) * 1000)

        # Convert to dict and validate Appendix B schema
        dumped = resp_obj.model_dump()

        # 1. Appendix B Schema Validation
        try:
            validated_b = AppendixBResponse.model_validate(dumped)
            schema_ok = True
        except Exception as e:
            schema_ok = False
            logger.error(f"Schema validation failed for {qid}: {e}")

        # 2. Zero URL Leaks Check
        url_leaks = check_url_leaks_in_obj(dumped)
        total_url_leaks += len(url_leaks)
        urls_ok = len(url_leaks) == 0

        # 3. Overall query pass
        query_passed = schema_ok and urls_ok
        if query_passed:
            passed_validations += 1

        effective_latency = dumped.get("meta", {}).get("latency_ms", dur_ms)
        latencies.append(effective_latency)
        if dumped.get("meta", {}).get("cache_hit", False):
            cache_hits += 1

        results_records.append(dumped)

        status_str = "[PASS]" if query_passed else "[FAIL]"
        contexts_cnt = len(dumped.get("response", {}).get("contexts", []))
        variations_cnt = len(dumped.get("query_variations", []))
        hit_flag = "HIT" if dumped.get("meta", {}).get("cache_hit", False) else "MISS"
        print(
            f"{status_str} [{qid}] ({domain or 'General'}) {q_text[:45]}... "
            f"| Latency: {effective_latency}ms | Cache: {hit_flag} | Contexts: {contexts_cnt} | Vars: {variations_cnt}"
        )

    # Clean up rate monitor
    logging.getLogger().removeHandler(rate_monitor)
    logging.getLogger("diagnos_ai").removeHandler(rate_monitor)

    # Write results.jsonl in Appendix B shape (one JSON object per line)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Compute Summary Statistics
    latencies.sort()
    n = len(latencies)
    p50_latency = latencies[int(n * 0.50)] if n > 0 else 0
    p95_latency = latencies[min(n - 1, int(n * 0.95))] if n > 0 else 0
    pass_rate = (passed_validations / total_queries) * 100.0 if total_queries > 0 else 0.0
    cache_hit_rate = (cache_hits / total_queries) * 100.0 if total_queries > 0 else 0.0
    rate_limit_occurred = len(rate_monitor.rate_limits) > 0

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Queries Evaluated:    {total_queries}")
    print(f"Schema & Gate Pass Rate:    {pass_rate:.1f}% ({passed_validations}/{total_queries})")
    print(f"P50 Latency:                {p50_latency} ms")
    print(f"P95 Latency:                {p95_latency} ms (Target <= 8000 ms)")
    print(f"Cache Hit Rate:             {cache_hit_rate:.1f}% ({cache_hits}/{total_queries})")
    print(f"Total URL Leaks Detected:   {total_url_leaks}")
    print(f"HTTP 429 Rate Limits:       {'YES (' + str(len(rate_monitor.rate_limits)) + ' detected)' if rate_limit_occurred else 'None (0 detected)'}")
    print(f"Results Written:            {output_path.resolve()}")
    if is_sample:
        print("[NOTE] Evaluation was performed on sample queries and sample deeplinks.")
    print("=" * 75)

    return {
        "pass_rate": pass_rate,
        "passed": passed_validations,
        "total": total_queries,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "cache_hit_rate": cache_hit_rate,
        "rate_limits_detected": len(rate_monitor.rate_limits),
        "url_leaks": total_url_leaks,
        "is_sample": is_sample,
        "output_file": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on Diagnos AI troubleshooting pipeline")
    parser.add_argument("--queries", type=str, default="", help="Path to queries.json (defaults to auto-detect)")
    parser.add_argument("--output", type=str, default="results.jsonl", help="Output results.jsonl file")
    parser.add_argument("--live", action="store_true", help="Execute live calls against Gemini API (max 6 calls)")
    parser.add_argument("--max-queries", type=int, default=6, help="Maximum number of queries to evaluate")
    parser.add_argument("--delay", type=float, default=15.0, help="Delay in seconds between live queries (default: 15.0)")
    args = parser.parse_args()

    run_evaluation(
        queries_path_str=args.queries,
        output_path_str=args.output,
        live=args.live,
        max_queries=args.max_queries,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
