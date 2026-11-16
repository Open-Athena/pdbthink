"""Require explicit tool-free requests without rewriting historical run identities."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


def require_tool_free(payload: dict[str, Any], provider: str, endpoint: str) -> None:
    """Fail before inference if a configuration can expose tools to a model.

    Historical configs remain loadable for scoring and cache-only recovery.
    New network requests must carry the controls in their saved configuration,
    so a stricter rerun cannot silently reuse a historical response cache key.
    """
    # vLLM rejects an empty array but accepts explicit null (no definitions).
    # Keep the explicit key so absence in historical configs still fails closed.
    if "tools" not in payload or not (
        payload["tools"] == [] or (provider == "openai_chat" and payload["tools"] is None)
    ):
        raise ValueError(
            "Tool-free evaluation requires explicit tools: [] (or null for OpenAI-compatible servers)"
        )
    expected = {"type": "none"} if provider == "anthropic_messages" else "none"
    if provider != "ollama_chat" and payload.get("tool_choice") != expected:
        raise ValueError(f"Tool-free evaluation requires tool_choice: {expected!r}")
    if payload.get("functions") or payload.get("function_call") not in (None, "none"):
        raise ValueError("Legacy function calling is forbidden in tool-free evaluation")
    if any(
        m.get("role") in ("tool", "function") or m.get("tool_calls") or m.get("function_call")
        for m in payload.get("messages", [])
    ):
        raise ValueError("Tool messages are forbidden in benchmark inputs")
    plugins = payload.get("plugins", [])
    if not isinstance(plugins, list) or any(p.get("enabled") is not False for p in plugins):
        raise ValueError("Provider plugins must be explicitly disabled")
    if "online" in payload.get("model", "").split(":")[1:] or payload.get("web_search_options") is not None:
        raise ValueError("Web search is forbidden in tool-free evaluation")
    if urlsplit(endpoint).hostname == "openrouter.ai":
        if not any(p.get("id") == "web" and p.get("enabled") is False for p in plugins):
            raise ValueError("OpenRouter evaluation requires the web plugin to be disabled")
