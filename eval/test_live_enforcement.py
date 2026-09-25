import json
import os
import sys
import time

# Ensure repo root and backend directory are in path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_queries():
    print("=" * 75)
    print("TESTING LIVE OFF-DOMAIN ENFORCEMENT & UNUSUAL DOMAIN COMPLAINTS")
    print("=" * 75)

    # 1. Off-domain test: "book me a flight to Paris"
    off_domain_query = "book me a flight to Paris"
    print(f"\n[Test 1] Off-Domain Query: '{off_domain_query}'")
    resp = client.post("/v1/troubleshoot", json={"query": off_domain_query})
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    contexts = data.get("contexts", [])
    fallback = data.get("fallback")
    print(f"Response: fallback='{fallback}', len(contexts)={len(contexts)}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(contexts) == 0, f"Expected empty contexts, got {len(contexts)}"
    assert fallback in ("no_match", "no_match_offdomain_heuristic"), f"Unexpected fallback: {fallback}"
    print(f"PASS: Off-domain query returned empty contexts and no-match fallback ('{fallback}')")

    # Sleep 3.5s to respect free-tier RPM limit
    time.sleep(3.5)

    # 2. Battery query phrased unusually
    battery_query = "battery is dying super quickly after the new update"
    print(f"\n[Test 2] Unusual Battery Query: '{battery_query}'")
    resp = client.post("/v1/troubleshoot", json={"query": battery_query})
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    contexts = data.get("contexts", [])
    fallback = data.get("fallback")
    print(f"Response: fallback='{fallback}', len(contexts)={len(contexts)}")
    if contexts:
        print(f"Goal 1: {contexts[0].get('goal')} (deeplink: {contexts[0].get('deeplink', {}).get('action')})")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(contexts) > 0, f"Expected legitimate query to return contexts, got fallback={fallback}"
    assert fallback is None, f"Expected no fallback, got {fallback}"
    print("PASS: Legitimate battery query was recognized and returned troubleshooting contexts")

    # Sleep 3.5s
    time.sleep(3.5)

    # 3. Camera query phrased unusually
    camera_query = "taking photos looks super blurry and distorted when zooming in"
    print(f"\n[Test 3] Unusual Camera Query: '{camera_query}'")
    resp = client.post("/v1/troubleshoot", json={"query": camera_query})
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    contexts = data.get("contexts", [])
    fallback = data.get("fallback")
    print(f"Response: fallback='{fallback}', len(contexts)={len(contexts)}")
    if contexts:
        print(f"Goal 1: {contexts[0].get('goal')} (deeplink: {contexts[0].get('deeplink', {}).get('action')})")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(contexts) > 0, f"Expected legitimate query to return contexts, got fallback={fallback}"
    assert fallback is None, f"Expected no fallback, got {fallback}"
    print("PASS: Legitimate camera query was recognized and returned troubleshooting contexts")

    # Sleep 3.5s
    time.sleep(3.5)

    # 4. Another camera query phrased unusually
    camera_query_2 = "camera preview is completely black and shutter button freezes"
    print(f"\n[Test 4] Unusual Camera Query 2: '{camera_query_2}'")
    resp = client.post("/v1/troubleshoot", json={"query": camera_query_2})
    print(f"Status Code: {resp.status_code}")
    data = resp.json()
    contexts = data.get("contexts", [])
    fallback = data.get("fallback")
    print(f"Response: fallback='{fallback}', len(contexts)={len(contexts)}")
    if contexts:
        print(f"Goal 1: {contexts[0].get('goal')} (deeplink: {contexts[0].get('deeplink', {}).get('action')})")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert len(contexts) > 0, f"Expected legitimate query to return contexts, got fallback={fallback}"
    assert fallback is None, f"Expected no fallback, got {fallback}"
    print("PASS: Legitimate camera query was recognized and returned troubleshooting contexts")

    print("\n" + "=" * 75)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    test_queries()
