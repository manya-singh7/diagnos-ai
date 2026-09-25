"""
Downloads and loads the embedding model ahead of time.

Run as a deploy build step (from backend/) so the first request and /health
don't wait on the model download:

    python -m cache
"""

import sys
import time

from cache import MODEL_CACHE_DIR, is_cache_ready

start = time.perf_counter()
ok = is_cache_ready()
print(f"semantic cache model {'ready' if ok else 'FAILED'} in {time.perf_counter() - start:.1f}s ({MODEL_CACHE_DIR})")
sys.exit(0 if ok else 1)
