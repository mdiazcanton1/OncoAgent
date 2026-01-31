"""Vision agent for multimodal image analysis."""

from __future__ import annotations

import asyncio

from ..state import OncoAgentState
from ..tools.gemini_tools import analyze_medical_image


async def vision_agent(state: OncoAgentState) -> dict:
    """Analyze medical images with Gemini."""
    analyses = []

    for image in state.get("images", []):
        image_bytes = image.get("data")
        mime_type = image.get("mime_type", "image/jpeg")
        if not image_bytes:
            continue

        analysis = await asyncio.to_thread(
            analyze_medical_image,
            image_bytes=image_bytes,
            mime_type=mime_type,
        )
        analyses.append(
            {
                "path": image.get("path"),
                "type": image.get("type"),
                "analysis": analysis,
            }
        )

    return {"images": analyses, "agents_completed": ["vision"]}

