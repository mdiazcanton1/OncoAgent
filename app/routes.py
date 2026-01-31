"""API routes for OncoAgent."""

from __future__ import annotations

import base64
import uuid
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from src.oncoagent.graph import compile_graph

router = APIRouter()

# Compile graph ONCE at module load - preserves checkpointer state across requests
_graph = compile_graph()


class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None
    cancer_type: str | None = None
    images: List[str] | None = None  # base64 encoded, optional data URI


class QueryResponse(BaseModel):
    response: str
    confidence: str
    thread_id: str
    citations: list[dict]
    clinical_trials: list[dict]


def _infer_query_type(query: str) -> str:
    q = query.lower()
    if "trial" in q or "clinical trial" in q:
        return "trial"
    if "guideline" in q:
        return "guideline"
    if "drug" in q or "dose" in q or "dosing" in q:
        return "drug"
    if "treatment" in q or "therapy" in q:
        return "treatment"
    return "general"


def _decode_image(image_str: str, index: int) -> dict:
    mime_type = "image/jpeg"
    payload = image_str

    if image_str.startswith("data:"):
        header, encoded = image_str.split(",", 1)
        payload = encoded
        if ";" in header:
            mime_type = header.split(";")[0].split(":")[1]

    image_bytes = base64.b64decode(payload)
    return {
        "path": f"uploaded-{index}",
        "type": "uploaded",
        "data": image_bytes,
        "mime_type": mime_type,
    }


def _extract_citations(state: dict) -> list[dict]:
    claims = state.get("claims", [])
    citations = []
    for claim in claims:
        for citation in claim.citations:
            citations.append(citation.model_dump())
    return citations


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
async def query_oncoagent(request: QueryRequest) -> QueryResponse:
    thread_id = request.thread_id or str(uuid.uuid4())

    images = []
    if request.images:
        images = [_decode_image(img, idx) for idx, img in enumerate(request.images)]

    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "original_query": request.query,
        "query_type": _infer_query_type(request.query),
        "cancer_type": request.cancer_type,
        "images": images,
        "evidence": [],
        "clinical_trials": [],
        "claims": [],
        "current_agent": "",
        "agents_completed": [],
        "needs_cross_validation": True,
        "response": None,
        "confidence_overall": None,
    }

    result = await _graph.ainvoke(initial_state, {"configurable": {"thread_id": thread_id}})

    return QueryResponse(
        response=result["response"],
        confidence=result["confidence_overall"],
        thread_id=thread_id,
        citations=_extract_citations(result),
        clinical_trials=result.get("clinical_trials", []),
    )

