"""Research agent using Exa search."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import get_settings
from ..state import OncoAgentState
from ..tools.exa_tools import search_medical_sources


# Expanded medical domains for better coverage
MEDICAL_DOMAINS = [
    # PubMed / NIH
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    # Guidelines organizations
    "nccn.org",
    "asco.org",
    "esmo.org",
    # Government / regulatory
    "fda.gov",
    "cancer.gov",
    "clinicaltrials.gov",
    # Major journals
    "nejm.org",
    "thelancet.com",
    "jnccn.org",
    "ascopubs.org",
    "nature.com",
    "link.springer.com",
    # Evidence resources
    "medicinesresources.nhs.uk",
    "cochranelibrary.com",
]


def _classify_source(url: str) -> str:
    if "pubmed.ncbi.nlm.nih.gov" in url or "pmc.ncbi.nlm.nih.gov" in url:
        return "pubmed"
    if "asco.org" in url or "ascopubs.org" in url:
        return "asco"
    if "fda.gov" in url:
        return "fda"
    if "esmo.org" in url:
        return "esmo"
    if "nccn.org" in url or "jnccn.org" in url:
        return "nccn"
    if "nejm.org" in url:
        return "nejm"
    if "thelancet.com" in url:
        return "lancet"
    if "cochranelibrary.com" in url:
        return "cochrane"
    return "other"


async def _generate_search_queries(
    original_query: str,
    api_key: str,
    conversation_history: list | None = None,
) -> list[str]:
    """Generate optimized English search queries with conversation context."""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=api_key)

    system_prompt = """You are a medical search query optimizer. Given a user's oncology question (in any language) and conversation history, generate 3-4 optimized English search queries for medical literature databases.

Rules:
1. ALWAYS output in English (medical literature is primarily in English)
2. Use standard medical terminology (e.g., "first-line treatment" not "initial therapy")
3. Include specific terms: drug names, biomarkers (HER2, EGFR, etc.), cancer types
4. Add "guidelines" or "NCCN" or "standard of care" when asking about treatments
5. Include recent year (2024, 2025 or 2026) for guideline queries
6. Keep queries concise but specific
7. CRITICAL: If the user asks a follow-up question (e.g., "what are the doses for THIS treatment?"), you MUST resolve references using the conversation history. Extract the EXACT drug names mentioned previously.
   - Example: If previous answer mentioned "pertuzumab + trastuzumab + docetaxel" and user asks "doses for this treatment", your queries should include those exact drug names.

OPTION SELECTION - VERY IMPORTANT:
8. If the assistant's last message offered options (a, b, c or numbered choices) and the user responds with a short selection like "todas", "all", "a", "b", "c", "la primera", "all of them", etc.:
   - You MUST look at what options were offered and generate queries for ALL selected topics
   - Example: If assistant offered "a) side effects of osimertinib, b) drug interactions, c) dosing" and user says "todas":
     → Generate queries for side effects, drug interactions, AND dosing of osimertinib
   - Extract the drug names and cancer type from the conversation context

TREATMENT QUERIES - When the question is about treatment recommendations:
9. ALWAYS include a query for DOSING: "[drug name] dosing regimen mg dose schedule"
10. ALWAYS include a query for DIAGNOSTIC REQUIREMENTS: "[cancer type] [biomarker] testing requirements NGS PCR before treatment"
11. Include FDA label or prescribing information when asking about specific drugs: "[drug name] FDA prescribing information dosage"

Examples:
- Input: "todas" (after assistant offered options about EGFR inhibitor side effects)
  Output queries:
  - osimertinib side effects adverse events toxicity profile
  - osimertinib drug interactions CYP3A4
  - EGFR TKI side effects comparison erlotinib gefitinib osimertinib
  - osimertinib safety profile FDA label

Output ONLY the queries, one per line. No explanations or numbering."""

    # Build context from conversation history
    context_parts = []
    if conversation_history:
        for msg in conversation_history[-6:]:  # Last 3 exchanges max
            role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
            content = getattr(msg, "content", str(msg))
            # Truncate long responses but keep drug names visible
            if len(content) > 800:
                content = content[:800] + "..."
            context_parts.append(f"{role}: {content}")

    context_str = "\n".join(context_parts) if context_parts else "No previous context."

    user_message = f"""Conversation history:
{context_str}

Current user query: {original_query}

IMPORTANT: 
- If the user's query is a short selection like "todas", "a", "b", "all", etc., look at the assistant's LAST message to see what options were offered and generate queries for those specific topics.
- Resolve any references (like "this treatment", "those drugs", "estas dosis") using the conversation history.
- Include specific drug names from the history.

Generate search queries:"""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])

    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return queries[:4]  # Max 4 queries (treatment + dosing + testing + FDA)


async def _search_single_query(query: str, num_results: int = 8) -> list[dict]:
    """Execute a single search query."""
    return await asyncio.to_thread(
        search_medical_sources,
        query,
        include_domains=MEDICAL_DOMAINS,
        num_results=num_results,
        category="research paper",
    )


async def research_agent(state: OncoAgentState) -> dict:
    """Search medical literature with optimized multi-query strategy."""
    settings = get_settings()
    original_query = state["original_query"]
    conversation_history = state.get("messages", [])
    retrieved_date = datetime.now(timezone.utc).isoformat()

    # Generate optimized search queries with conversation context
    if settings.anthropic_api_key:
        try:
            queries = await _generate_search_queries(
                original_query,
                settings.anthropic_api_key,
                conversation_history,
            )
        except Exception:
            # Fallback to original query if LLM fails
            queries = [original_query]
    else:
        queries = [original_query]

    # Execute searches in parallel
    search_tasks = [_search_single_query(q) for q in queries]
    all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Deduplicate results by URL
    seen_urls = set()
    evidence = []

    for results in all_results:
        if isinstance(results, Exception):
            continue
        for result in results:
            url = result["url"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            evidence.append({
                "source": url,
                "title": result["title"],
                "snippet": result.get("highlights") or [],
                "source_type": _classify_source(url),
                "retrieved_date": retrieved_date,
            })

    return {"evidence": evidence, "agents_completed": ["research"]}

