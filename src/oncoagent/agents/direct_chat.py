"""Direct chat agent for conversational replies without search."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import get_settings
from ..state import OncoAgentState


async def direct_chat_agent(state: OncoAgentState) -> dict:
    """Respond conversationally without external search."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not configured.")

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
    )

    system_prompt = (
        "You are a conversational assistant for oncologists.\n"
        "ALWAYS respond in the same language the user writes in.\n"
        "Be concise, friendly, and clear. If you are unsure, say so.\n"
        "Do not fabricate medical facts or numbers.\n"
        "For medical questions, you may provide general knowledge but remind the user that "
        "for evidence-based recommendations, they should ask a specific clinical question."
    )

    messages = [SystemMessage(content=system_prompt)]
    history = state.get("messages", [])

    if history:
        messages.extend(history)
    else:
        messages.append(HumanMessage(content=state.get("original_query", "")))

    response = await llm.ainvoke(messages)

    return {
        "messages": [AIMessage(content=response.content)],
        "response": response.content,
        "confidence_overall": "UNCERTAIN",
    }
