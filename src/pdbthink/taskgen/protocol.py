"""Request controls shared by the tool-free runner and its offline tests."""

from __future__ import annotations


def request_payload(model: str, messages: list[dict], budget: dict, *, extra: dict | None = None) -> dict:
    required = ("input_tokens", "context_window", "native_output_limit")
    if not model or not isinstance(model, str):
        raise ValueError("an explicit model identifier is required")
    if not all(type(budget.get(key)) is int and budget[key] > 0 for key in required):
        raise ValueError(f"budget must specify positive integer {required}")
    if not budget.get("tokenizer_revision") or not budget.get("endpoint_limit_source"):
        raise ValueError("budget must document its exact tokenizer and endpoint limits")
    maximum = min(budget["native_output_limit"], budget["context_window"] - budget["input_tokens"])
    if maximum <= 0:
        raise ValueError("context exclusion: exact prompt leaves no output capacity")
    extra = extra or {}
    protected = {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "plugins",
        "max_tokens",
        "max_completion_tokens",
        "functions",
        "function_call",
        "web_search_options",
        "stream",
        "n",
    }
    if protected & extra.keys():
        raise ValueError("extra request fields may not override protocol or budget controls")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": maximum,
        **extra,
        "tools": [],
        "tool_choice": "none",
        "plugins": [{"id": "web", "enabled": False}],
    }


def tool_events(value, path="response") -> list[str]:
    events = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("tool_calls", "function_call", "annotations") and item:
                events.append(f"{path}.{key}")
            if key == "type" and item in ("tool_use", "tool_result", "function_call", "web_search_call"):
                events.append(f"{path}.type:{item}")
            events.extend(tool_events(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            events.extend(tool_events(item, f"{path}[{index}]"))
    return events
