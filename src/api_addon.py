"""
API report middleware: turns an upstream shitty, undetailed /v1/models report into a proper one.

Report conventions follow OpenRouter's /v1/models shape where the convention exists.
"""

from __future__ import annotations

import configparser
import time
from typing import Any

# Standard effort vocabulary, least to most intensive. `options` must be a subset of this list
STANDARD_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")

# Disable keywords: how a client expresses thinking-off on the wire.
# "none" -> reasoning_effort "none"
# "lowest" -> the lowest available level
# "qwen" -> chat_template_kwargs.enable_thinking false
DISABLE_KEYWORDS: tuple[str, ...] = ("none", "lowest", "qwen")


def _warn(model: str, what: str) -> None:
    print(f"[ROUTER] {model}: ignoring {what} in reasoning_effort config", flush=True)


# ---------------------------------------------------------------------------
# Preset facts from the API-exposed preset INI
# ---------------------------------------------------------------------------


def _preset_value(parser: configparser.ConfigParser, keys: tuple[str, ...]) -> str | None:
    """
    Looks up a preset key across the parser's sections

    Args:
        parser: The loaded preset INI parser
        keys: The key spellings to try, in priority order

    Returns:
        The value, or None when unset
    """
    for section in parser.sections():
        for key in keys:
            if parser.has_option(section, key):
                return parser.get(section, key)
    return None


def _preset_facts(preset_ini: str | None) -> dict[str, int]:
    """
    Extracts the per-request facts the addon reports from a preset INI string

    Args:
        preset_ini: The preset INI string from the row's status.preset

    Returns:
        A dict with "context_window" and/or "max_output_tokens"
    """
    if not preset_ini:
        return {}
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), strict=False)
    try:
        parser.read_string(preset_ini)
    except configparser.Error:
        return {}
    facts: dict[str, int] = {}
    raw_c = _preset_value(parser, ("c", "ctx-size"))
    if raw_c is not None:
        raw_parallel = _preset_value(parser, ("np", "parallel"))
        try:
            ctx = int(raw_c)
            parallel = int(raw_parallel) if raw_parallel is not None else 1
        except ValueError:
            pass
        else:
            facts["context_window"] = ctx // max(parallel, 1)
    raw_predict = _preset_value(parser, ("n-predict",))
    if raw_predict is not None:
        try:
            facts["max_output_tokens"] = int(raw_predict)
        except ValueError:
            pass
    return facts


# ---------------------------------------------------------------------------
# Config-owned report fields
# ---------------------------------------------------------------------------


def reasoning_effort_report(model: str, cfg_entry: dict[str, Any]) -> dict[str, Any] | None:
    """
    Builds the reasoning_effort payload for one model

    Args:
        model (str): The model id, for warning messages
        cfg_entry: The model's entry from the router config's LLM section

    Returns:
        The reasoning_effort dict, or None when the model has no reasoning surface
    """
    raw = cfg_entry.get("reasoning_effort")
    if not isinstance(raw, dict):
        if raw is not None:
            _warn(model, f"reasoning_effort must be an object with an 'options' key, got {type(raw).__name__}")
        return None
    options = raw.get("options")
    if not isinstance(options, list) or not options:
        _warn(model, "reasoning_effort.options must be a non-empty list of effort levels")
        return None
    levels: list[str] = []
    for opt in options:
        if isinstance(opt, str) and opt in STANDARD_EFFORTS:
            levels.append(opt)
        else:
            _warn(model, f"non-standard effort level {opt!r}")
    if not levels:
        return None
    report: dict[str, Any] = {"levels": levels}
    disable = raw.get("disable")
    if disable is not None and disable is not False:
        if isinstance(disable, str) and disable in DISABLE_KEYWORDS:
            report["disable"] = disable
        else:
            _warn(model, f"disable {disable!r} is not one of {list(DISABLE_KEYWORDS)}")
    default = raw.get("default")
    if isinstance(default, str) and default in levels:
        report["default"] = default
    elif default is not None:
        _warn(model, f"default {default!r} is not one of the configured options")
    return report


def pricing_report(cfg: Any) -> dict[str, Any] | None:
    """
    Maps the config's cost entry onto the OpenAI-style pricing fields

    Args:
        cfg: The raw cost value from the model config ($/1M tokens)

    Returns:
        {"input", "output", "cache_read", "cache_write"}, or None when unset
    """
    if not isinstance(cfg, dict):
        return None
    return {
        "input": cfg.get("input", 0),
        "output": cfg.get("output", 0),
        "cache_read": cfg.get("cached_input", 0),
        "cache_write": cfg.get("cache_write", 0),
    }


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------


def _augment_row(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Overlays the addon fields onto one model row

    Args:
        row: The upstream (or synthesized) model row
        cfg: The model's entry from the router config's LLM section

    Returns:
        A new row with the addon fields applied
    """
    row = dict(row)
    status = row.get("status")
    preset_ini = status.get("preset") if isinstance(status, dict) else None
    facts = _preset_facts(preset_ini if isinstance(preset_ini, str) else None)
    row.pop("status", None)
    row.pop("source", None)
    row.pop("can_remove", None)
    # context_length: preset wins, then models.json, then upstream.
    # If models.json declares it, it's the mandatory floor.
    cfg_ctx = cfg.get("context_length")
    upstream_ctx = row.get("context_length")
    if "context_window" in facts:
        row["context_length"] = facts["context_window"]
    elif isinstance(cfg_ctx, int) and cfg_ctx > 0:
        row["context_length"] = cfg_ctx
    elif isinstance(upstream_ctx, int) and upstream_ctx > 0:
        row["context_length"] = upstream_ctx
    # max_output_tokens: preset wins, then models.json, then upstream.
    cfg_out = cfg.get("max_output_tokens")
    upstream_out = row.get("max_output_tokens")
    if "max_output_tokens" in facts:
        row["max_output_tokens"] = facts["max_output_tokens"]
    elif isinstance(cfg_out, int) and cfg_out > 0:
        row["max_output_tokens"] = cfg_out
    elif isinstance(upstream_out, int) and upstream_out > 0:
        row["max_output_tokens"] = upstream_out
    # Architecture from models.json (tokenizer, modalities, etc.)
    arch_cfg = cfg.get("architecture")
    if isinstance(arch_cfg, dict) and arch_cfg:
        arch_row = dict(row.get("architecture") or {})
        arch_row.update(arch_cfg)
        row["architecture"] = arch_row
    # Reasoning-effort
    effort = reasoning_effort_report(row.get("id", "?"), cfg)
    if effort is not None:
        row["reasoning_effort"] = effort
    # Supported parameters: explicit list from models.json wins, else auto-derive.
    params_cfg = cfg.get("parameters")
    if isinstance(params_cfg, list) and params_cfg:
        row["supported_parameters"] = params_cfg
    else:
        params: list[str] = ["tools"]
        if effort is not None:
            params.append("reasoning")
        row["supported_parameters"] = params
    pricing = pricing_report(cfg.get("pricing"))
    if pricing is not None:
        row["pricing"] = pricing
    return row


def build_report(upstream: dict[str, Any] | None, models_config: dict[str, Any]) -> dict[str, Any]:
    """
    Builds the addon /v1/models report (the module's single entry point)

    Args:
        upstream: The upstream {"object": "list", "data": [...]} response, or None
        models_config: The models.json content, mapping model id to its API addon entry

    Returns:
        An OpenAI-style {"object": "list", "data": [...]} dict
    """
    upstream_rows: dict[str, dict[str, Any]] = {}
    if isinstance(upstream, dict):
        for row in upstream.get("data", []):
            if isinstance(row, dict) and "id" in row:
                upstream_rows[row["id"]] = row
    created = int(time.time())
    data = [
        _augment_row(
            upstream_rows.get(model_id)
            or {"id": model_id, "object": "model", "created": created, "owned_by": "llama-router"},
            cfg if isinstance(cfg, dict) else {},
        )
        for model_id, cfg in models_config.items()
    ]
    return {"object": "list", "data": data}
