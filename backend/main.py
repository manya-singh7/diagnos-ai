from fastapi import FastAPI
from pydantic import BaseModel
from schema import ContextDeeplinkResponse

app = FastAPI()


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/v1/troubleshoot", response_model=ContextDeeplinkResponse)
def troubleshoot(payload: dict):
    """
    Takes a customer complaint and returns an actionable plan.
    Expected input: {"query": "...", "siis_response": "<optional>"}
    """
    # TODO: real pipeline goes here —
    # query enrichment -> structure extraction -> deeplink mapping -> ordering
    return ContextDeeplinkResponse(contexts=[])


@app.post("/v1/clarify", response_model=ContextDeeplinkResponse)
def clarify(payload: dict):
    """
    Takes the original query + a clarifying answer, re-ranks the result.
    Expected input: {"query": "...", "clarification_answer": "..."}
    """
    # TODO: re-rank contexts using the extra detail
    return ContextDeeplinkResponse(contexts=[])