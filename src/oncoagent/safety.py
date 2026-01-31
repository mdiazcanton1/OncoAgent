"""Safety utilities for claim validation and confidence scoring."""

from __future__ import annotations

import re
from typing import Literal

from .state import Claim


def validate_claim(claim: str, evidence: list[dict]) -> tuple[bool, str]:
    """Validate that numeric data in a claim exists in evidence."""
    numbers = re.findall(r"\d+(?:\.\d+)?%?", claim)
    if not numbers:
        return True, "OK"

    for num in numbers:
        found = any(num in str(item.get("snippet", "")) for item in evidence)
        if not found:
            return False, f"Number {num} not found in evidence"

    return True, "OK"


def calculate_confidence(claim: Claim) -> Literal["HIGH", "MEDIUM", "LOW", "UNCERTAIN"]:
    """Compute confidence based on number and quality of citations."""
    if not claim.citations:
        return "UNCERTAIN"

    high_quality = {"pubmed", "fda", "nccn"}
    n_sources = len(claim.citations)
    quality_sources = sum(1 for c in claim.citations if c.source_type in high_quality)

    if n_sources >= 3 and quality_sources >= 2:
        return "HIGH"
    if n_sources >= 2 and quality_sources >= 1:
        return "MEDIUM"
    if n_sources >= 1:
        return "LOW"
    return "UNCERTAIN"


def calculate_overall_confidence(
    claims: list[Claim],
) -> Literal["HIGH", "MEDIUM", "LOW", "UNCERTAIN"]:
    """Aggregate confidence across claims."""
    if not claims:
        return "UNCERTAIN"

    levels = [claim.confidence for claim in claims]
    if all(level == "HIGH" for level in levels):
        return "HIGH"
    if any(level == "UNCERTAIN" for level in levels):
        return "LOW"
    if any(level == "LOW" for level in levels):
        return "LOW"
    return "MEDIUM"


def calculate_confidence_from_evidence(
    evidence: list[dict],
) -> Literal["HIGH", "MEDIUM", "LOW", "UNCERTAIN"]:
    """Calculate confidence based on evidence sources collected."""
    if not evidence:
        return "UNCERTAIN"

    # High quality sources
    high_quality_domains = {
        "fda", "nccn", "asco", "esmo", "nejm", "lancet", "pubmed", "cochrane"
    }

    n_sources = len(evidence)
    high_quality_count = sum(
        1 for e in evidence
        if e.get("source_type", "other") in high_quality_domains
    )

    # Determine confidence
    if n_sources >= 5 and high_quality_count >= 3:
        return "HIGH"
    if n_sources >= 3 and high_quality_count >= 2:
        return "HIGH"
    if n_sources >= 2 and high_quality_count >= 1:
        return "MEDIUM"
    if n_sources >= 1:
        return "LOW"
    return "UNCERTAIN"

