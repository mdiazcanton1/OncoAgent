"""Aggregator node for parallel agent outputs."""

from __future__ import annotations

from ..state import OncoAgentState


async def aggregator(state: OncoAgentState) -> dict:
    """Consolidate evidence collected by parallel agents."""
    return {"agents_completed": ["aggregator"]}
