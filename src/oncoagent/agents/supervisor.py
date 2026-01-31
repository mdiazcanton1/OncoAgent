"""Supervisor agent for routing."""

from __future__ import annotations

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from ..config import get_settings
from ..state import OncoAgentState


# Intent categories:
# - greeting: Simple greetings, no medical content
# - followup_with_context: Follow-up that can be answered from conversation history
# - repeated_question: Same question already answered - ask user preference
# - option_selection: User selecting from options offered by assistant (a, b, c, todas, etc.)
# - research_required: New question that needs fresh search
DIRECT_INTENTS = {"greeting", "followup_with_context", "repeated_question"}
# option_selection goes to research but with context resolution


async def _classify_intent_with_llm(query: str, messages: list, api_key: str) -> str:
    """Use LLM to classify intent based on query and conversation history."""
    if not messages or len(messages) <= 1:
        # No history, must be a new question
        q = query.lower().strip()
        greetings = ["hola", "hello", "hi", "buenos dias", "buenas tardes", "como estas", "hey"]
        if any(g in q for g in greetings) and len(q.split()) < 5:
            return "greeting"
        return "research_required"

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", api_key=api_key)

    # Build conversation summary
    history_parts = []
    for msg in messages[-8:]:  # Last 4 exchanges
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        content = getattr(msg, "content", str(msg))
        if len(content) > 500:
            content = content[:500] + "..."
        history_parts.append(f"{role}: {content}")

    history_str = "\n".join(history_parts)

    system_prompt = """You are an intent classifier for a medical research assistant. Classify the user's query into ONE of these categories:

1. "greeting" - Simple greetings like "hola", "hello", "cómo estás", with no medical content

2. "followup_with_context" - A follow-up question that CAN be fully answered from conversation history because the info IS ALREADY THERE. Examples:
   - Asking to clarify something that was already explained
   - Asking for a summary of what was discussed
   - "¿Puedes repetir eso?" / "Can you explain that again?"
   
3. "repeated_question" - The EXACT same question that was already fully answered.

4. "option_selection" - The user is selecting from options that the assistant previously offered (a, b, c, todas, all, etc.)
   - IMPORTANT: Look at the assistant's last message - if it offered numbered/lettered options and user responds with a selection, this is option_selection

5. "research_required" - USE THIS FOR:
   - New medical questions
   - Follow-up questions asking for SPECIFIC DETAILS not in history (dosing, side effects, interactions, etc.)
   - "¿Y las dosis?" → research_required (needs to search for dosing info)
   - "¿Y los efectos secundarios?" → research_required (needs to search for side effects)
   - "¿Y para pacientes mayores?" → research_required (needs specific data)
   - ANY question that requires NEW information not explicitly stated in previous answers

CRITICAL RULE: If the user asks a follow-up about specific medical details (doses, side effects, interactions, contraindications, etc.) that were NOT already provided in detail → ALWAYS classify as "research_required". Do NOT classify as "followup_with_context" unless the answer is FULLY available in the history.

Respond with ONLY the category name, nothing else."""

    user_message = f"""Conversation history:
{history_str}

New user query: {query}

Classify this query:"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        intent = response.content.strip().lower().replace('"', '').replace("'", "")
        
        valid_intents = {"greeting", "followup_with_context", "repeated_question", "option_selection", "research_required"}
        if intent in valid_intents:
            return intent
        return "research_required"
    except Exception:
        return "research_required"


def _classify_intent_simple(query: str) -> str:
    """Simple fallback classification without LLM."""
    q = query.lower().strip()

    greetings = ["hola", "hello", "hi", "buenos dias", "buenas tardes", "como estas"]
    if any(g in q for g in greetings) and len(q.split()) < 5:
        return "greeting"

    return "research_required"


async def supervisor(state: OncoAgentState) -> dict:
    """Classify intent using LLM for smart routing."""
    settings = get_settings()
    query = state.get("original_query", "")
    messages = state.get("messages", [])

    if settings.anthropic_api_key:
        intent = await _classify_intent_with_llm(query, messages, settings.anthropic_api_key)
    else:
        intent = _classify_intent_simple(query)

    return {"query_type": intent}


def route_after_supervisor(
    state: OncoAgentState,
) -> Literal["direct_chat", "context_responder"] | list[Send]:
    """Route based on classified intent. Used with add_conditional_edges."""
    intent = state.get("query_type", "research_required")

    # Greetings go to simple direct chat
    if intent == "greeting":
        return "direct_chat"

    # Follow-ups and repeated questions use context responder (no new search)
    if intent in {"followup_with_context", "repeated_question"}:
        return "context_responder"

    # Everything else needs research
    sends: list[Send] = [Send("research", state)]

    if intent in ["treatment", "trial"]:
        sends.append(Send("clinical_trials", state))

    if state.get("images"):
        sends.append(Send("vision", state))

    return sends

