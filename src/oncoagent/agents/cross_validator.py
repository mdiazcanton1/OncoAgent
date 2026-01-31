"""Cross-validation agent using a secondary LLM."""

from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..state import Claim, OncoAgentState


async def cross_validator(state: OncoAgentState) -> dict:
    """Verify claims using a secondary LLM (GPT-4 class)."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    llm = ChatOpenAI(model="gpt-4-turbo", api_key=settings.openai_api_key)
    validated_claims: list[Claim] = []

    for claim in state.get("claims", []):
        messages = [
            SystemMessage(
                content=(
                    "You are a medical fact-checker. Verify if this claim is supported "
                    "by the provided evidence. Be strict - if uncertain, say so."
                )
            ),
            HumanMessage(
                content=(
                    "Claim:\n"
                    f"{claim.statement}\n\n"
                    "Evidence:\n"
                    f"{json.dumps([c.model_dump() for c in claim.citations], indent=2)}\n\n"
                    "Is this claim:\n"
                    "1. SUPPORTED by the evidence?\n"
                    "2. CONTRADICTED by the evidence?\n"
                    "3. UNCERTAIN - not enough evidence?\n\n"
                    "Provide reasoning."
                )
            ),
        ]

        validation = await llm.ainvoke(messages)
        claim.cross_validated = True
        claim.validation_notes = validation.content

        if "UNCERTAIN" in validation.content.upper():
            claim.confidence = "UNCERTAIN"

        validated_claims.append(claim)

    return {"claims": validated_claims, "needs_cross_validation": False}

