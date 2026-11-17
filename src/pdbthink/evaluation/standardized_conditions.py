"""Keep the user-approved comparison separate from provider sampling defaults."""

from __future__ import annotations


def require_standardized_condition(endpoint: dict) -> None:
    if endpoint.get("benchmark_condition") != "standardized-v1":
        return
    if endpoint.get("context_window") != 262144:
        raise ValueError("Standardized conditions require a 262144-token total budget")
    extra = endpoint.get("extra", {})
    if "seed" in extra or "seed_base" in endpoint:
        raise ValueError("The standardized comparison omits seed for every provider")
    exception = endpoint.get("sampling_exception")
    fixed = endpoint["model_id"] in ("openai/gpt-6-astra", "anthropic/claude-opus-5")
    if bool(exception) != fixed:
        raise ValueError("Only Astra and Opus have approved fixed-sampling exceptions")
    sampling = {
        "temperature": 0.7, "top_p": 1.0, "presence_penalty": 0.0,
        "frequency_penalty": 0.0, "repetition_penalty": 1.0,
    }
    if fixed:
        if (sampling.keys() | {"top_k", "min_p"}) & extra.keys():
            raise ValueError("Fixed-sampling endpoints must not claim adjustable sampling")
    else:
        if any(extra.get(key) != value for key, value in sampling.items()):
            raise ValueError("Standardized sampling settings differ from the agreed condition")
        disabled = -1 if endpoint.get("tokenize") else 0
        if extra.get("top_k") != disabled:
            raise ValueError("Standardized top-k filtering must be explicitly disabled")
        if endpoint.get("tokenize") and extra.get("min_p") != 0.0:
            raise ValueError("Standardized local min-p filtering must be explicitly disabled")
    if endpoint.get("tokenize"):
        if extra.get("chat_template_kwargs") != {"enable_thinking": True}:
            raise ValueError("Standardized local models require thinking enabled")
    elif extra.get("reasoning_effort") != "high":
        raise ValueError("Standardized hosted models require high reasoning effort")
    if "openrouter.ai" in endpoint["base_url"]:
        provider = extra.get("provider", {})
        if not provider.get("require_parameters") or provider.get("allow_fallbacks") is not False:
            raise ValueError("Standardized hosted routing must reject unsupported parameters")
        if len(provider.get("only", [])) != 1:
            raise ValueError("Standardized hosted routing requires one recorded provider")
