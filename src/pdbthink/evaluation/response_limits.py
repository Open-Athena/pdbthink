"""Recognise provider cutoffs before admitting a response as a completed answer."""

from __future__ import annotations


def provider_clamped_output(response: dict, requested: int) -> bool:
    choices = response.get("choices", [])
    if len(choices) != 1:
        return False
    choice = choices[0]
    usage = response.get("usage", {})
    used = usage.get("completion_tokens", requested)
    if used >= requested:
        return False
    if choice.get("finish_reason") == "length":
        return True
    # The November report observed GLM stop mid-reasoning at exactly
    # the same 128K boundary as Parasail, but misreport termination as ``stop``.
    message = choice.get("message", {})
    return (
        "glm-5.3" in response.get("model", "")
        and used == 131072
        and bool(message.get("reasoning") or message.get("reasoning_content"))
        and not message.get("content")
        and not message.get("refusal")
    )
