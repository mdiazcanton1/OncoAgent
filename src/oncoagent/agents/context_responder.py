"""Context responder agent - answers from conversation history without new search."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import get_settings
from ..state import OncoAgentState


async def context_responder(state: OncoAgentState) -> dict:
    """Respond using conversation history without searching for new evidence."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not configured.")

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
    )

    query = state.get("original_query", "")
    intent = state.get("query_type", "followup_with_context")
    messages = state.get("messages", [])

    # Build conversation history for context
    history_parts = []
    for msg in messages:
        role = "User" if getattr(msg, "type", "") == "human" else "Assistant"
        content = getattr(msg, "content", str(msg))
        history_parts.append(f"{role}: {content}")

    history_str = "\n\n".join(history_parts)

    if intent == "repeated_question":
        # User is asking something already answered - offer options
        system_prompt = """You are a medical research assistant. The user is asking a question that was already answered in this conversation.

Your task:
1. Acknowledge that this topic was already discussed
2. Provide a brief summary of the key points from the previous answer
3. Ask if they want:
   a) More details on a specific aspect
   b) Updated information (new search)
   c) The same information presented differently

ALWAYS respond in the same language as the user's query.
Be helpful and non-judgmental - it's normal to want clarification."""

    else:  # followup_with_context
        # Follow-up question - answer from history
        system_prompt = """You are a medical research assistant. The user is asking a follow-up question about something already discussed.

Your task:
1. Answer the follow-up question using ONLY the information from the conversation history
2. If the specific information requested is NOT in the history, say so and offer to search for it
3. Maintain the same citation style if referencing previous information
4. Be concise and direct

ALWAYS respond in the same language as the user's query.
NEVER invent medical information - only use what was previously discussed."""

    prompt_content = f"""Conversation history:
{history_str}

Current user query: {query}

Respond appropriately based on the conversation context."""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content),
    ])

    return {
        "messages": [AIMessage(content=response.content)],
        "response": response.content,
        "confidence_overall": "HIGH" if intent == "followup_with_context" else "MEDIUM",
    }
