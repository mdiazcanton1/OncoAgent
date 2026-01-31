"""Exa search wrapper using exa-py client."""

from __future__ import annotations

from typing import Iterable

from exa_py import Exa

from ..config import get_settings


def _get_client() -> Exa:
    settings = get_settings()
    if not settings.exa_api_key:
        raise ValueError("EXA_API_KEY is not configured.")
    return Exa(settings.exa_api_key)


def search_medical_sources(
    query: str,
    include_domains: Iterable[str] | None = None,
    num_results: int = 10,
    category: str = "research paper",
) -> list[dict]:
    """Search medical sources via Exa and return structured results."""
    client = _get_client()
    results = client.search_and_contents(
        query=query,
        num_results=num_results,
        category=category,
        include_domains=list(include_domains or []),
        text=True,
        highlights=True,
        summary=True,
    )

    structured = []
    for item in results.results:
        structured.append(
            {
                "url": item.url,
                "title": item.title,
                "highlights": item.highlights or [],
                "summary": item.summary,
                "text": item.text,
                "id": item.id,
            }
        )
    return structured

