"""Response builder agent for final output."""

from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..safety import calculate_confidence_from_evidence, calculate_overall_confidence
from ..state import OncoAgentState


async def response_builder(state: OncoAgentState) -> dict:
    """Construct a structured, cited response for clinicians."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not configured.")

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
    )

    system_prompt = (
        "You are a medical research assistant for oncologists.\n\n"
        "CRITICAL RULES:\n"
        "1. RESPOND IN THE SAME LANGUAGE AS THE USER'S QUERY (if Spanish, respond in Spanish; if English, respond in English, etc.)\n"
        "2. NEVER invent information - only use provided evidence\n"
        "3. EVERY claim must have an inline citation [1], [2], etc.\n"
        "4. If you're uncertain about something, SAY SO explicitly\n"
        "5. NEVER invent dosages, percentages, or numerical data UNLESS they appear in the evidence\n"
        "6. Include confidence level for each major claim\n"
        "7. When asked for treatment recommendations, provide evidence-based guidance but ALWAYS include a disclaimer\n\n"
        "FOCUS AND RELEVANCE - VERY IMPORTANT:\n"
        "8. Answer SPECIFICALLY what the user asked - do NOT include tangential information\n"
        "   - If asked about 'side effects of osimertinib', focus on side effects, NOT treatment algorithms\n"
        "   - If asked about 'dosing', focus on dosing, NOT diagnostic requirements\n"
        "   - Keep the response focused and concise - clinicians are busy\n"
        "9. Only include sections that are relevant to the specific question\n"
        "   - Skip 'Diagnostic prerequisites' if asking about side effects\n"
        "   - Skip 'Clinical trials' if not directly relevant to the question\n\n"
        "TREATMENT-SPECIFIC REQUIREMENTS (when asking about treatments):\n"
        "10. DOSING: Include standard dosing regimens if available:\n"
        "    - Drug name, dose, route, frequency (e.g., 'Osimertinib 80mg oral daily')\n"
        "    - Dose modifications for toxicity if mentioned\n"
        "11. DIAGNOSTIC PREREQUISITES: Specify required testing before targeted therapies\n\n"
        "REFERENCES FORMAT - STANDARDIZED:\n"
        "12. EVERY reference MUST include a clickable link when available. Format:\n"
        "    [#] Author et al. Title. Journal Year. PMID: XXXXX. URL: https://pubmed.ncbi.nlm.nih.gov/XXXXX/\n"
        "    - For PubMed: https://pubmed.ncbi.nlm.nih.gov/{PMID}/\n"
        "    - For PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/{PMCID}/\n"
        "    - For FDA: Include full FDA.gov URL\n"
        "    - For ClinicalTrials.gov: https://clinicaltrials.gov/study/{NCT_NUMBER}\n"
        "    - For guidelines (NCCN, ASCO, ESMO): Include direct URL if in evidence\n"
        "13. If URL is not available, at minimum include PMID or DOI\n\n"
        "CONFIDENCE ASSESSMENT:\n"
        "14. Assign confidence based on evidence quality:\n"
        "    - HIGH: FDA labels, Phase III trials, major guidelines (NCCN, ASCO, ESMO)\n"
        "    - MEDIUM: Phase II trials, systematic reviews, expert consensus\n"
        "    - LOW: Case reports, retrospective studies, limited data\n"
        "    - UNCERTAIN: Conflicting evidence or insufficient data\n\n"
        "Format your response with ONLY relevant sections:\n"
        "- Summary (always)\n"
        "- [Relevant content sections based on question]\n"
        "- Limitations and uncertainties (always)\n"
        "- References with links (always)\n"
        "- DISCLAIMER (at the end): 'Esta información está basada en evidencia científica publicada y guías clínicas. "
        "La decisión terapéutica final debe ser tomada por el médico tratante considerando las características "
        "individuales del paciente, comorbilidades, preferencias y contexto clínico específico.'\n"
        "  (Translate this disclaimer to match the language of the response)\n"
    )

    user_payload = {
        "query": state["original_query"],
        "evidence": state.get("evidence", []),
        "clinical_trials": state.get("clinical_trials", []),
        "image_analyses": [
            img.get("analysis") for img in state.get("images", []) if img.get("analysis")
        ],
    }

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                "Use the following evidence to answer the query.\n\n"
                f"{json.dumps(user_payload, indent=2)}"
            )
        ),
    ]

    response = await llm.ainvoke(messages)

    # Calculate confidence from evidence quality, fallback to claims-based
    claims = state.get("claims", [])
    evidence = state.get("evidence", [])

    if evidence:
        confidence = calculate_confidence_from_evidence(evidence)
    elif claims:
        confidence = calculate_overall_confidence(claims)
    else:
        confidence = "UNCERTAIN"

    return {
        "response": response.content,
        "confidence_overall": confidence,
        "agents_completed": ["response_builder"],
    }

