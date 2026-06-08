from __future__ import annotations

from typing import List, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def _get_last_user_message(messages: List[BaseMessage]) -> str:
    """Return the most recent human message content, or empty string."""

    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return str(getattr(m, "content", "") or "").strip()
    return ""


def _build_history_context_from_messages(
    messages: List[BaseMessage], max_pairs: int = 3
) -> str:
    """Convert recent message history into a compact text context.

    We reconstruct user/assistant pairs from LangChain messages.
    """

    if not messages:
        return ""

    pairs: List[tuple[str, str]] = []
    last_user: Optional[str] = None
    for m in messages:
        if isinstance(m, HumanMessage):
            if last_user is not None:
                pairs.append((last_user, ""))
            last_user = str(getattr(m, "content", "") or "").strip()
        elif isinstance(m, AIMessage):
            content = str(getattr(m, "content", "") or "").strip()
            if last_user is None:
                pairs.append(("", content))
            else:
                pairs.append((last_user, content))
                last_user = None
    if last_user:
        pairs.append((last_user, ""))

    tail = pairs[-max_pairs:]
    lines: List[str] = []
    for u, a in tail:
        if u:
            lines.append(f"User: {u}")
        if a:
            lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def _get_last_turn_from_messages(
    messages: List[BaseMessage],
) -> Optional[Dict[str, str]]:
    """Return the last user/assistant pair from messages, if any."""

    if not messages:
        return None
    last_user: Optional[str] = None
    last_ai: Optional[str] = None
    for m in messages:
        if isinstance(m, HumanMessage):
            last_user = str(getattr(m, "content", "") or "").strip()
        elif isinstance(m, AIMessage):
            last_ai = str(getattr(m, "content", "") or "").strip()
    if last_user is None and last_ai is None:
        return None
    return {"user": last_user or "", "assistant": last_ai or ""}
