FROM python:3.11-slim

WORKDIR /app

# Install curl for container HEALTHCHECK and judge inspection
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and catalog datasets
COPY . .

# Pre-download the semantic cache embedding model (~87 MB) at build time so /health
# is ready at start-up instead of waiting on the download; fails the build if it can't load
RUN cd backend && python -m cache

# Ensure backend directory is in PYTHONPATH so internal modules resolve cleanly
ENV PYTHONPATH="/app/backend:/app"
ENV PYTHONUNBUFFERED=1

# Expose FastAPI service port
EXPOSE 8000

# Built-in healthcheck pointing at /health readiness endpoint
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI app with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
