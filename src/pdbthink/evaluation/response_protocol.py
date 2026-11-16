"""Separate served reasoning delimiters from answers without changing the scoring contract."""

from __future__ import annotations

from ..taskgen.protocol import tool_events

MARIN_THINKING = "marin-thinking-v1"
THINK_START = "<|start_think|>"
THINK_END = "<|end_think|>"


def answer_content(message: dict, protocol: str = "plain") -> str:
    """Never promote an example FINAL field inside an unfinished reasoning trace to an answer."""
    content = message.get("content") or message.get("refusal") or ""
    if protocol == "plain":
        return content
    if protocol != MARIN_THINKING:
        raise ValueError(f"Unknown response protocol: {protocol}")
    if THINK_START not in content and THINK_END not in content:
        return content
    if content.count(THINK_START) != 1 or content.count(THINK_END) != 1:
        return ""
    before, thought_and_answer = content.split(THINK_START, 1)
    if before.strip() or THINK_END not in thought_and_answer:
        return ""
    return thought_and_answer.split(THINK_END, 1)[1].strip()


def response_tool_events(response: dict, protocol: str = "plain") -> list[str]:
    """Marin may emit delimited calls as text instead of structured API tool events."""
    events = tool_events(response)
    if protocol == MARIN_THINKING:
        for index, choice in enumerate(response.get("choices", [])):
            content = choice.get("message", {}).get("content") or ""
            if "<tool_call>" in content or "</tool_call>" in content:
                events.append(f"response.choices[{index}].message.content.tool_call_delimiter")
    return events
