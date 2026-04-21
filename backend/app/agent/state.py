"""AgentState — LangGraph state definition."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State shared across all LangGraph nodes.

    Attributes:
        messages: Conversation history (managed by add_messages reducer).
        intent: Classified user intent — "qa" | "search" | "detail".
        retrieved_docs: Documents retrieved from Qdrant.
        tool_output: Output from tool execution (AniList API).
        final_answer: Synthesized response to user.
    """

    messages: Annotated[list, add_messages]
    intent: str
    retrieved_docs: list
    tool_output: dict
    final_answer: str
