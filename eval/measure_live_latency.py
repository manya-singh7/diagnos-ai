import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from main import clarify, gemini_client, troubleshoot
from schema import ClarifyRequest, TroubleshootRequest

print("=" * 75)
print("REAL ENDPOINT LATENCY MEASUREMENT (Max 3 live calls)")
print("=" * 75)

if gemini_client is None:
    print("Gemini client not initialized. Cannot run live measurement.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Call 1: Real /v1/troubleshoot (2 Ranked Hypotheses, Real Prompt)
# ---------------------------------------------------------------------------
req1 = TroubleshootRequest(
    query="The mobile phone swipe navigation moves up or down instead of left or right after downloading an app"
)

print("\n[Executing Live Call 1] POST /v1/troubleshoot (real prompt, max 2 ranked goals)...")
start1 = time.perf_counter()
resp1 = troubleshoot(req1)
dur1_ms = int((time.perf_counter() - start1) * 1000)

print(f"-> Call 1 Completed: {dur1_ms} ms (meta.latency_ms: {resp1.meta.latency_ms} ms)")
print(f"   Returned {len(resp1.contexts)} goals:")
for i, g in enumerate(resp1.contexts):
    print(f"     Goal {i+1}: '{g.title}' (score: {g.score}) - '{g.goal}'")
    print(f"       Actions: {[a.actionName for a in g.actions]}")
print(f"   Cost: ${resp1.meta.cost_usd:.6f}")

# ---------------------------------------------------------------------------
# Call 2: Forced Retry Case through /v1/troubleshoot
# ---------------------------------------------------------------------------
print("\n[Executing Live Call 2] Forced Retry via /v1/troubleshoot...")
# We wrap active_client.models.generate_content so that call 1 injects a schema flaw,
# forcing the retry loop, and call 2 executes live against Gemini!
real_generate_content = gemini_client.models.generate_content
attempt_counter = [0]


def wrapped_generate_content(*args, **kwargs):
    attempt_counter[0] += 1
    if attempt_counter[0] == 1:
        # Deliberately return invalid JSON to force the retry loop
        mock_bad = MagicMock()
        mock_bad.text = json.dumps({
            "goals": [
                {
                    "goal": "Follow these steps to perform this Display Troubleshooting",
                    "title": "Swipe navigation settings",
                    "score": 2.5,  # Invalid score > 1.0 forces Pydantic ValidationError
                    "actions": []
                }
            ]
        })
        mock_bad.usage_metadata = MagicMock(prompt_token_count=100, candidates_token_count=50)
        return mock_bad
    else:
        # Call real live Gemini model on retry!
        return real_generate_content(*args, **kwargs)


start2 = time.perf_counter()
with patch.object(gemini_client.models, "generate_content", side_effect=wrapped_generate_content):
    req2 = TroubleshootRequest(query="Screen flickers and battery dies fast")
    resp2 = troubleshoot(req2)
dur2_ms = int((time.perf_counter() - start2) * 1000)

print(f"-> Call 2 (Forced Retry) Completed: {dur2_ms} ms (meta.latency_ms: {resp2.meta.latency_ms} ms)")
print(f"   Total attempts made: {attempt_counter[0]}")
print(f"   Returned {len(resp2.contexts)} goals after self-correction retry:")
for i, g in enumerate(resp2.contexts):
    print(f"     Goal {i+1}: '{g.title}' (score: {g.score})")
print(f"   Cost: ${resp2.meta.cost_usd:.6f}")

# ---------------------------------------------------------------------------
# Call 3: Real /v1/clarify (Clarification Answer re-ranking live pipeline)
# ---------------------------------------------------------------------------
print("\n[Executing Live Call 3] POST /v1/clarify with answer folded into query...")
req3 = ClarifyRequest(
    query="Screen flickers and battery dies fast",
    clarification_answer="The flickering only happens when adaptive brightness is turned on",
)

start3 = time.perf_counter()
resp3 = clarify(req3)
dur3_ms = int((time.perf_counter() - start3) * 1000)

print(f"-> Call 3 Completed: {dur3_ms} ms (meta.latency_ms: {resp3.meta.latency_ms} ms)")
print(f"   needs_clarification: {resp3.needs_clarification}")
print(f"   Returned {len(resp3.contexts)} re-ranked goals:")
for i, g in enumerate(resp3.contexts):
    print(f"     Goal {i+1}: '{g.title}' (score: {g.score})")
print(f"   Cost: ${resp3.meta.cost_usd:.6f}")

print("\n" + "=" * 75)
print("LATENCY MEASUREMENT SUMMARY")
print(f"Call 1 (/v1/troubleshoot 2-Goal cold path): {dur1_ms} ms (Spec P95 Target: <= 8000 ms)")
print(f"Call 2 (Forced Retry self-correction):     {dur2_ms} ms")
print(f"Call 3 (/v1/clarify live re-ranking):       {dur3_ms} ms")
print("=" * 75)
