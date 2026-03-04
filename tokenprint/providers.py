from __future__ import annotations

from dataclasses import dataclass

from tokenprint.constants import (
    CLAUDE_RATE_CACHE_READ_MULTIPLIER,
    CLAUDE_RATE_CACHE_WRITE_MULTIPLIER,
    CLAUDE_RATE_CACHED_PER_TOKEN,
    CLAUDE_RATE_INPUT_PER_TOKEN,
    CLAUDE_RATE_OUTPUT_PER_TOKEN,
    CODEX_RATE_CACHED_PER_TOKEN,
    CODEX_RATE_INPUT_PER_TOKEN,
    CODEX_RATE_OUTPUT_PER_TOKEN,
    GEMINI_RATE_CACHED_PER_TOKEN,
    GEMINI_RATE_INPUT_PER_TOKEN,
    GEMINI_RATE_OUTPUT_PER_TOKEN,
)


@dataclass(frozen=True)
class ProviderConfig:
    name: str           # "claude" — internal key
    display_name: str   # "Claude Code" — shown in UI
    key: str            # "c" — compact key in raw data JSON
    color: str          # "#6366f1" — chart/legend color
    collect_fn: str     # "collect_claude_data" — function name (string for mockability)
    label: str          # "Claude Code (ccusage)" — status message during collection
    rates: tuple[float, float, float]  # (input, output, cached) per-token USD rates


PROVIDERS: tuple[ProviderConfig, ...] = (
    ProviderConfig(
        "claude",
        "Claude Code",
        "c",
        "#6366f1",
        "collect_claude_data",
        "Claude Code (ccusage)",
        (CLAUDE_RATE_INPUT_PER_TOKEN, CLAUDE_RATE_OUTPUT_PER_TOKEN, CLAUDE_RATE_CACHED_PER_TOKEN),
    ),
    ProviderConfig(
        "codex",
        "Codex CLI",
        "x",
        "#22c55e",
        "collect_codex_data",
        "Codex CLI (@ccusage/codex)",
        (CODEX_RATE_INPUT_PER_TOKEN, CODEX_RATE_OUTPUT_PER_TOKEN, CODEX_RATE_CACHED_PER_TOKEN),
    ),
    ProviderConfig(
        "gemini",
        "Gemini CLI",
        "g",
        "#f59e0b",
        "collect_gemini_data",
        "Gemini CLI (telemetry)",
        (GEMINI_RATE_INPUT_PER_TOKEN, GEMINI_RATE_OUTPUT_PER_TOKEN, GEMINI_RATE_CACHED_PER_TOKEN),
    ),
)

_PROVIDER_BY_NAME: dict[str, ProviderConfig] = {provider.name: provider for provider in PROVIDERS}
_PROVIDER_NAME_SET: frozenset[str] = frozenset(_PROVIDER_BY_NAME)
_PROVIDER_BY_KEY: dict[str, ProviderConfig] = {provider.key: provider for provider in PROVIDERS}


def provider_by_name(name: str) -> ProviderConfig | None:
    """Return provider configuration by internal name."""
    return _PROVIDER_BY_NAME.get(name)


def provider_by_key(key: str) -> ProviderConfig | None:
    """Return provider configuration by output key."""
    return _PROVIDER_BY_KEY.get(key)


def resolve_provider(identifier: str) -> ProviderConfig | None:
    """Resolve provider config by internal name or compact key."""
    if not identifier:
        return None
    by_name = provider_by_name(identifier)
    if by_name is not None:
        return by_name
    return provider_by_key(identifier)


def provider_name_set() -> set[str]:
    """Return provider names as a set."""
    return set(_PROVIDER_NAME_SET)


def provider_names() -> tuple[str, ...]:
    """Return provider names in configured registry order."""
    return tuple(provider.name for provider in PROVIDERS)
