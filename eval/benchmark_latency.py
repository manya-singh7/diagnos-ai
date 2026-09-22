import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client()

prompt = """You are an expert Samsung Galaxy device troubleshooting AI.
Output a JSON Goal object adhering to:
- goal: "Follow these steps to perform this Swipe Navigation Troubleshooting"
- title: "Swipe navigation settings"
- score: 0.95
- actions: list of actions with actionName in Title Case, description starting with "It will" (5-7 words), category ("auto"|"manual"|"critical"), stepGroups with steps. No web URLs.

Customer complaint: The mobile phone swipe navigation moves up or down instead of left or right after downloading an app."""

print("=" * 60)
print("BENCHMARKING CALL 3 (gemini-3.5-flash-lite vs gemini-3.6-flash)")
print("=" * 60)

try:
    config3 = types.GenerateContentConfig(
        temperature=0.0,
        seed=42,
        max_output_tokens=450,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
    start3 = time.perf_counter()
    resp3 = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=config3,
    )
    lat3 = int((time.perf_counter() - start3) * 1000)
    print(f"Call 3 [gemini-3.5-flash-lite, thinking=0, seed=42]: {lat3} ms")
    if hasattr(resp3, "usage_metadata") and resp3.usage_metadata:
        print(f"       Tokens: prompt={resp3.usage_metadata.prompt_token_count}, candidates={resp3.usage_metadata.candidates_token_count}")
        print("Output:", resp3.text[:150])
except Exception as e:
    print(f"Call 3 Failed: {e}")

print("=" * 60)
