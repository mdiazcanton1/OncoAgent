"""Gemini multimodal helpers for medical image analysis."""

from __future__ import annotations

from google import genai
from google.genai import types
from pydantic import BaseModel

from ..config import get_settings


class MedicalImageAnalysis(BaseModel):
    description: str
    findings: list[str]
    relevant_features: list[str]
    limitations: str
    disclaimer: str = "This analysis is for informational purposes only."


def analyze_medical_image(
    image_bytes: bytes,
    mime_type: str,
    prompt: str | None = None,
    model: str = "gemini-2.5-pro",
) -> MedicalImageAnalysis:
    """Analyze a medical image with Gemini and return structured output."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=settings.gemini_api_key)
    prompt_text = (
        prompt
        or """Analyze this medical image. Describe visible features objectively.
IMPORTANT:
- Only describe what you can actually observe
- Do not make diagnoses
- Note any limitations in image quality
- This is for informational purposes only."""
    )

    response = client.models.generate_content(
        model=model,
        contents=[
            prompt_text,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MedicalImageAnalysis,
        ),
    )
    return MedicalImageAnalysis.model_validate_json(response.text)

