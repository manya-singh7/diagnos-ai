"""
Tunes CACHE_SIM_THRESHOLD against eval/semantic_cache/cache_pairs.json.

For every pair: store `a`, look up `b` through the real cache_lookup, and print
similarity, decision, expected and PASS/FAIL at the current threshold
(CACHE_SIM_THRESHOLD, default 0.85). Then prints accuracy at 0.75 / 0.80 / 0.85 / 0.90.

A pair expected "veto" passes if it is not served (vetoed or below threshold).
A pair expected "hit" passes only if it is served.

Usage: python eval/semantic_cache/tune_cache.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))

from cache import DEFAULT_THRESHOLD, _threshold, cache_clear, cache_lookup, cache_store  # noqa: E402
from schema import ContextDeeplinkResponse, ResponseMeta  # noqa: E402

THRESHOLDS = [0.75, 0.80, 0.85, 0.90]

_DUMMY = ContextDeeplinkResponse(
    contexts=[{
        "goal": "Follow these steps to perform this Cache Tuning Troubleshooting",
        "title": "Cache tuning",
        "score": 0.9,
        "actions": [{
            "actionName": "Open Device Settings",
            "description": "It will open device settings",
            "category": "auto",
            "stepGroups": [{"steps": ["Open Settings."]}],
        }],
    }],
    fallback=None,
    meta=ResponseMeta(latency_ms=0, cache_hit=False, model="tune", cost_usd=0.0),
)


def decide(similarity: float, rule_decision: str, threshold: float) -> str:
    return "miss_low_sim" if similarity < threshold else rule_decision


def main() -> None:
    pairs = json.loads((HERE / "cache_pairs.json").read_text())
    current = _threshold()

    # Threshold 0 makes every pair reach the veto, giving the rule decision on its own.
    # The decision at any threshold is then derived from (similarity, rule decision).
    results = []
    for p in pairs:
        cache_clear()
        cache_store(p["a"], _DUMMY, [])
        _, info = cache_lookup(p["b"], threshold=0.0)
        results.append((p, info["similarity"], info["decision"]))
    cache_clear()

    print(f"Per-pair results at threshold {current:.2f}"
          f"{'' if current == DEFAULT_THRESHOLD else f' (default {DEFAULT_THRESHOLD})'}\n")
    print(f"{'sim':>6}  {'decision':<14} {'expected':<8} {'result':<6}  pair")
    print("-" * 110)
    for p, sim, rule in results:
        decision = decide(sim, rule, current)
        ok = (decision == "hit") == (p["expected"] == "hit")
        print(f"{sim:6.3f}  {decision:<14} {p['expected']:<8} {'PASS' if ok else 'FAIL':<6}  "
              f"{p['a']!r} vs {p['b']!r}")

    n_hit = sum(p["expected"] == "hit" for p, _, _ in results)
    n_veto = len(results) - n_hit
    print(f"\nAccuracy by threshold ({len(results)} pairs: {n_hit} hit, {n_veto} veto)\n")
    print(f"{'threshold':>9}  {'accuracy':>8}  {'hits served':>11}  {'wrong serves':>12}  {'rescued by veto':>15}")
    print("-" * 66)
    for t in THRESHOLDS:
        correct = served_ok = wrong_served = rescued = 0
        for p, sim, rule in results:
            decision = decide(sim, rule, t)
            served = decision == "hit"
            correct += served == (p["expected"] == "hit")
            served_ok += served and p["expected"] == "hit"
            wrong_served += served and p["expected"] == "veto"
            rescued += p["expected"] == "veto" and sim >= t and rule.startswith("veto_")
        marker = "  <- current" if abs(t - current) < 1e-9 else ""
        print(f"{t:9.2f}  {correct / len(results):8.1%}  {served_ok:>5}/{n_hit:<5}  {wrong_served:>12}  "
              f"{rescued:>15}{marker}")
    print("\nwrong serves = veto-labelled pairs that would be served from cache (the dangerous error).")
    print("rescued by veto = veto-labelled pairs above threshold that only the rule layer blocked.")


if __name__ == "__main__":
    main()
