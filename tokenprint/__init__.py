#!/usr/bin/env python3
"""
TokenPrint Dashboard Generator

Collects token usage data from Claude Code (ccusage), Codex CLI (@ccusage/codex),
and Gemini CLI (OpenTelemetry logs), then generates an interactive HTML dashboard
showing usage trends, costs, and estimated environmental impact.

Usage:
    tokenprint [--since YYYYMMDD|YYYY-MM-DD] [--until YYYYMMDD|YYYY-MM-DD]
    [--no-cache] [--no-open] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tokenprint.constants import (
    CARBON_INTENSITY,
    CLAUDE_RATE_BY_MODEL_PREFIX,
    CLAUDE_RATE_CACHE_READ_MULTIPLIER,
    CLAUDE_RATE_CACHE_WRITE_MULTIPLIER,
    CLAUDE_RATE_CACHED_PER_TOKEN,
    CLAUDE_RATE_INPUT_PER_TOKEN,
    CLAUDE_RATE_OUTPUT_PER_TOKEN,
    CODEX_RATE_CACHED_PER_TOKEN,
    CODEX_RATE_INPUT_PER_TOKEN,
    CODEX_RATE_OUTPUT_PER_TOKEN,
    ELECTRICITY_COST_KWH,
    EMBODIED_CARBON_FACTOR,
    ENERGY_PER_CACHED_TOKEN_WH,
    ENERGY_PER_INPUT_TOKEN_WH,
    ENERGY_PER_OUTPUT_TOKEN_WH,
    GEMINI_RATE_CACHED_PER_TOKEN,
    GEMINI_RATE_INPUT_PER_TOKEN,
    GEMINI_RATE_OUTPUT_PER_TOKEN,
    GEMINI_TELEMETRY_LOG_PATH_ENV_VAR,
    GRID_LOSS_FACTOR,
    PROVIDER_CACHE_FILENAME,
    PROVIDER_CACHE_SCHEMA_VERSION,
    PUE,
    TOKENPRINT_CACHE_PATH_ENV_VAR,
    WATER_USE_EFFICIENCY,
)
from tokenprint.providers import (
    PROVIDERS,
    ProviderConfig,
    provider_name_set,
    provider_names,
    resolve_provider,
)

# Keep legacy rate constants on the module surface for existing imports/tests.
_REEXPORTED_RATE_CONSTANTS = (
    CLAUDE_RATE_CACHED_PER_TOKEN,
    CLAUDE_RATE_INPUT_PER_TOKEN,
    CLAUDE_RATE_OUTPUT_PER_TOKEN,
    CODEX_RATE_CACHED_PER_TOKEN,
    CODEX_RATE_INPUT_PER_TOKEN,
    CODEX_RATE_OUTPUT_PER_TOKEN,
)


def _tokenprint_version() -> str:
    """Return version from package metadata with safe fallback."""
    try:
        return _pkg_version("tokenprint")
    except PackageNotFoundError:
        return "0.0.0-dev"
    except Exception:
        return "0.0.0"


# Claude pricing fallback for models that may appear unpriced in ccusage output.
# Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-02-20).


def run_command(cmd: list[str], timeout: int = 60) -> str | None:
    """Run a command (list of args) and return stdout, or None on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def _warn(message: str) -> None:
    """Emit a warning to stderr."""
    print(f"  [warn] {message}", file=sys.stderr)


def _info(message: str) -> None:
    """Emit an info message to stderr."""
    print(f"  [info] {message}", file=sys.stderr)


def _command_exists(name: str) -> bool:
    """Return whether a command exists in PATH."""
    return shutil.which(name) is not None


def _run_cli_check() -> bool:
    """Run a quick CLI preflight check. Returns True on success."""
    checks: dict[str, bool] = {
        "template": (Path(__file__).resolve().parent / "template.html").exists(),
        "ccusage": _command_exists("ccusage"),
        "ccusage-codex": _command_exists("ccusage-codex"),
    }
    ok = True
    print("tokenprint checks:")
    for name, result in checks.items():
        if result:
            print(f"  [ok] {name}")
        else:
            print(f"  [missing] {name}")
            if name in {"template", "ccusage"}:
                ok = False
    return ok


def _safe_float(val: Any) -> float:
    """Convert a value to float safely."""
    if isinstance(val, bool):
        return 0.0
    try:
        value = float(val)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def detect_github_username() -> str:
    """Detect GitHub username from gh CLI or git config, prompt if neither available."""
    username = run_command(["gh", "api", "user", "--jq", ".login"], timeout=5)
    if username:
        return username
    username = run_command(["git", "config", "user.name"], timeout=5)
    if username:
        return username
    try:
        username = input("Enter your name (for share image): ").strip()
        if username:
            return username
    except (EOFError, KeyboardInterrupt):
        return "dev"
    return "dev"


def _claude_rates_for_model(model_name: str | None) -> tuple[float, float] | None:
    """Return (input_rate, output_rate) for a Claude model, or None if unknown."""
    if not model_name:
        return None
    lowered = model_name.lower()
    for prefix, rates in CLAUDE_RATE_BY_MODEL_PREFIX:
        if lowered.startswith(prefix):
            return rates
    return None


def _estimate_claude_model_cost(model_row: dict[str, Any]) -> float:
    """Estimate Claude model cost from token counts using published rates."""
    model_name = str(model_row.get("modelName") or model_row.get("model") or "")
    rates = _claude_rates_for_model(model_name)
    if not rates:
        return 0.0

    input_rate, output_rate = rates
    # Claude prompt caching published multipliers:
    # 5-minute write tokens are 1.25x input, cache reads are 0.1x input.
    cache_write_rate = input_rate * CLAUDE_RATE_CACHE_WRITE_MULTIPLIER
    cache_read_rate = input_rate * CLAUDE_RATE_CACHE_READ_MULTIPLIER

    input_tok = _safe_int(model_row.get("inputTokens", model_row.get("input_tokens", 0)))
    output_tok = _safe_int(model_row.get("outputTokens", model_row.get("output_tokens", 0)))
    cache_creation_tok = _safe_int(model_row.get("cacheCreationTokens", model_row.get("cache_creation_tokens", 0)))
    cache_read_tok = _safe_int(model_row.get("cacheReadTokens", model_row.get("cache_read_tokens", 0)))

    return (
        input_tok * input_rate
        + output_tok * output_rate
        + cache_creation_tok * cache_write_rate
        + cache_read_tok * cache_read_rate
    )


def _rates_for_provider(name: str) -> tuple[float, float, float]:
    """Return configured per-token rates for a provider, or zeros when unknown."""
    provider = resolve_provider(name)
    if provider is None:
        return 0.0, 0.0, 0.0
    return provider.rates


def collect_claude_data(since: str | None = None, until: str | None = None) -> dict[str, dict[str, Any]]:
    """Collect Claude Code usage via ccusage."""
    cmd = ["ccusage", "daily", "--json"]
    if since:
        cmd.extend(["--since", since])
    if until:
        cmd.extend(["--until", until])

    output = run_command(cmd, timeout=120)
    if not output:
        _warn("ccusage not available or returned no data")
        return {}

    try:
        raw = json.loads(output)
    except json.JSONDecodeError:
        _warn("ccusage returned invalid JSON")
        return {}

    # ccusage wraps data in {"daily": [...]}
    data = raw.get("daily", raw) if isinstance(raw, dict) else raw
    if not isinstance(data, list):
        _warn("ccusage returned unexpected JSON shape")
        return {}

    daily: dict[str, dict[str, Any]] = {}
    estimated_missing_cost_total = 0.0
    estimated_models: set[str] = set()
    for entry in data:
        if not isinstance(entry, dict):
            continue
        date = _parse_date_flexible(entry.get("date", ""))
        if not date:
            continue
        day_cost = _safe_float(entry.get("totalCost", 0) or 0)

        # ccusage may report modelBreakdowns with cost=0 for newer Claude models.
        # If totalCost equals known breakdown cost, estimate only the missing zero-cost models.
        model_breakdowns = entry.get("modelBreakdowns")
        if isinstance(model_breakdowns, list) and model_breakdowns:
            known_cost = 0.0
            zero_cost_models: list[dict[str, Any]] = []
            for row in model_breakdowns:
                if not isinstance(row, dict):
                    continue
                row_cost = _safe_float(row.get("cost", 0) or 0)
                if row_cost > 0:
                    known_cost += row_cost
                else:
                    zero_cost_models.append(row)

            if day_cost <= known_cost + 1e-9:
                for row in zero_cost_models:
                    est = _estimate_claude_model_cost(row)
                    if est > 0:
                        day_cost += est
                        estimated_missing_cost_total += est
                        model_name = str(row.get("modelName") or row.get("model") or "").strip()
                        if model_name:
                            estimated_models.add(model_name)

        if date not in daily:
            daily[date] = {
                "provider": "claude",
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0,
            }
        daily[date]["input_tokens"] += _safe_int(entry.get("inputTokens", 0))
        daily[date]["output_tokens"] += _safe_int(entry.get("outputTokens", 0))
        daily[date]["cache_read_tokens"] += _safe_int(entry.get("cacheReadTokens", 0))
        daily[date]["cache_write_tokens"] += _safe_int(entry.get("cacheCreationTokens", 0))
        daily[date]["cost"] += day_cost

    if estimated_missing_cost_total > 0:
        models_str = ", ".join(sorted(estimated_models)) if estimated_models else "unknown models"
        _info(f"Added ${estimated_missing_cost_total:.2f} estimated Claude cost for unpriced models: {models_str}")
    return daily


def _parse_date_flexible(date_str: str | None) -> str | None:
    """Parse dates in ISO (2026-01-07) or human (Jan 7, 2026) format to YYYY-MM-DD."""
    if not date_str:
        return None
    date_str = date_str.strip()
    if not date_str:
        return None
    # Try ISO format first (validate calendar correctness)
    if len(date_str) >= 10 and date_str[4] == "-":
        candidate = date_str[:10]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            return None
    # Support compact YYYYMMDD (for provider/cache rows)
    if len(date_str) == 8 and date_str.isdigit():
        try:
            return datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return None
    # Try human-readable formats
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_cache_date_key(date_key: str) -> str | None:
    """Normalize cache date keys to YYYY-MM-DD."""
    iso = _parse_date_flexible(date_key)
    return iso


def _normalize_timezone_name(timezone_name: str | None) -> str:
    """Return a validated IANA timezone name (or UTC for blank input)."""
    candidate = (timezone_name or "UTC").strip() or "UTC"
    try:
        ZoneInfo(candidate)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {candidate}") from exc
    return candidate


def _normalize_cli_date_arg(date_arg: str | None) -> str | None:
    """Normalize CLI date args to YYYYMMDD strings."""
    if not date_arg:
        return None
    date_arg = date_arg.strip()
    if not date_arg:
        return None
    if len(date_arg) == 8 and date_arg.isdigit():
        try:
            return datetime.strptime(date_arg, "%Y%m%d").strftime("%Y%m%d")
        except ValueError:
            return None
    if len(date_arg) >= 10 and date_arg[4] == "-" and date_arg[7] == "-":
        if len(date_arg) > 10 and date_arg[10] not in {"T", " "}:
            return None
        try:
            return datetime.strptime(date_arg[:10], "%Y-%m-%d").strftime("%Y%m%d")
        except ValueError:
            return None
    return None


def _parse_gemini_timestamp(ts: Any, timezone_name: str = "UTC") -> str | None:
    """Parse Gemini timestamp values into YYYY-MM-DD."""
    try:
        tz = ZoneInfo(_normalize_timezone_name(timezone_name))
    except ValueError:
        return None
    if ts is None:
        return None
    if isinstance(ts, bool):
        return None

    if isinstance(ts, str):
        raw_ts = ts.strip()
        if not raw_ts:
            return None
        if raw_ts.endswith("Z"):
            raw_ts = raw_ts[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw_ts)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(tz).strftime("%Y-%m-%d")
        except ValueError:
            pass
        try:
            return _parse_gemini_timestamp(float(raw_ts), timezone_name=timezone_name)
        except (TypeError, ValueError):
            return None

    if isinstance(ts, (int, float)):
        try:
            ts_float = float(ts)
        except (TypeError, ValueError):
            return None
        if ts_float <= 0 or ts_float != ts_float:
            return None
        if not math.isfinite(ts_float):
            return None
        if ts_float >= 10**18:
            ts_float /= 1_000_000_000
        elif ts_float >= 10**15:
            ts_float /= 1_000_000
        elif ts_float >= 10**12:
            ts_float /= 1_000
        try:
            return datetime.fromtimestamp(ts_float, tz=timezone.utc).astimezone(tz).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return None

    if not isinstance(ts, str):
        ts = str(ts).strip()
        if not ts:
            return None

    date_prefix = _parse_date_flexible(ts[:10]) if len(ts) >= 10 else None
    if date_prefix:
        return date_prefix
    return None


def _normalize_gemini_attributes(raw_attrs: Any) -> dict[str, Any] | None:
    """Normalize Gemini attribute payloads into a key/value dict."""
    if isinstance(raw_attrs, dict):
        attrs: dict[str, Any] = {}
        for key, val in raw_attrs.items():
            if not isinstance(key, str) or not key:
                continue
            if isinstance(val, dict):
                attrs[key] = val.get("intValue", val.get("Int64Value", val.get("stringValue", 0)))
            else:
                attrs[key] = val
        return attrs
    if not isinstance(raw_attrs, list):
        return None

    attrs: dict[str, Any] = {}
    for item in raw_attrs:
        if not isinstance(item, dict):
            continue
        key = item.get("Key", item.get("key"))
        if not isinstance(key, str) or not key:
            continue
        val = item.get("Value", item.get("value", 0))
        if isinstance(val, dict):
            val = val.get("intValue", val.get("Int64Value", val.get("stringValue", 0)))
        attrs[key] = val
    return attrs


def _run_codex_json_command(since: str | None = None, until: str | None = None) -> Any:
    """Run Codex usage command with local/cached fallbacks and return parsed JSON."""
    date_args: list[str] = []
    if since:
        date_args.extend(["--since", since])
    if until:
        date_args.extend(["--until", until])

    attempts: list[tuple[list[str], int]] = [
        (["ccusage-codex", "daily", "--json", *date_args], 90),
        (["npx", "--no-install", "@ccusage/codex@18", "daily", "--json", *date_args], 5),
        (["npx", "--no-install", "@ccusage/codex@latest", "daily", "--json", *date_args], 5),
    ]

    for cmd, timeout in attempts:
        output = run_command(cmd, timeout=timeout)
        if not output:
            continue
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            continue
    return None


def collect_codex_data(since: str | None = None, until: str | None = None) -> dict[str, dict[str, Any]]:
    """Collect Codex CLI usage via @ccusage/codex."""
    raw = _run_codex_json_command(since, until)
    if raw is None:
        _warn("@ccusage/codex unavailable (missing install or invalid JSON)")
        _warn("       Install once: npm i -g @ccusage/codex@18")
        return {}

    # @ccusage/codex wraps data in {"daily": [...]}
    data = raw.get("daily", raw) if isinstance(raw, dict) else raw
    if not isinstance(data, list):
        _warn("@ccusage/codex returned unexpected JSON shape")
        return {}

    # First pass: collect all entries
    entries: list[tuple[str, int, int, int, float]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        raw_date = entry.get("date", "")
        if not raw_date:
            continue
        date = _parse_date_flexible(raw_date)
        if not date:
            continue
        raw_input = _safe_int(entry.get("inputTokens", 0))
        output_tok = _safe_int(entry.get("outputTokens", 0))
        cached_tok = _safe_int(entry.get("cachedInputTokens", 0))
        # Codex inputTokens includes cached — subtract to get non-cached input
        input_tok = max(0, raw_input - cached_tok)
        cost = _safe_float(entry.get("costUSD", 0))
        entries.append((date, input_tok, output_tok, cached_tok, cost))

    # Use codex provider rates for all unpriced days (including gpt-5.3-codex)
    rate_input, rate_output, rate_cached = _rates_for_provider("codex")

    daily: dict[str, dict[str, Any]] = {}
    for date, input_tok, output_tok, cached_tok, cost in entries:
        if not cost and (input_tok or output_tok):
            cost = input_tok * rate_input + output_tok * rate_output + cached_tok * rate_cached
        if date not in daily:
            daily[date] = {
                "provider": "codex",
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost": 0,
            }
        daily[date]["input_tokens"] += input_tok
        daily[date]["output_tokens"] += output_tok
        daily[date]["cache_read_tokens"] += cached_tok
        daily[date]["cost"] += cost
    return daily


def collect_gemini_data(
    since: str | None = None,
    until: str | None = None,
    timezone_name: str = "UTC",
    gemini_log_path: str | None = None,
    *,
    log_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect Gemini CLI usage from OpenTelemetry log."""
    if log_path is not None and gemini_log_path is None:
        gemini_log_path = log_path
    log_path = _resolve_gemini_log_path() if gemini_log_path is None else _resolve_gemini_log_path(gemini_log_path)
    if not log_path.exists():
        _warn(f"Gemini telemetry log not found: {log_path}")
        _warn("       Run: bash install.sh (or bash setup-gemini-telemetry.sh)")
        return {}

    daily: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "provider": "gemini",
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost": 0,
        }
    )

    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue

                # Extract timestamp for date grouping
                raw_ts = record.get("timestamp") or record.get("time") or record.get("Timestamp")
                date = _parse_gemini_timestamp(raw_ts, timezone_name)  # supports ISO, epoch, and millis
                if not date:
                    continue

                # Apply date filters
                date_compact = date.replace("-", "")
                if since and date_compact < since:
                    continue
                if until and date_compact > until:
                    continue

                # Look for token usage in attributes or body
                raw_attrs = record.get("attributes", record.get("Attributes", {}))
                attrs = _normalize_gemini_attributes(raw_attrs)
                if not isinstance(attrs, dict):
                    continue

                raw_input = _safe_int(attrs.get("input_token_count", attrs.get("gen_ai.usage.input_tokens", 0)))
                output_tok = _safe_int(attrs.get("output_token_count", attrs.get("gen_ai.usage.output_tokens", 0)))
                cached_tok = _safe_int(
                    attrs.get("cached_content_token_count", attrs.get("gen_ai.usage.cached_tokens", 0))
                )
                # Gemini input_token_count includes cached — subtract to get non-cached input
                input_tok = max(0, raw_input - cached_tok)
                has_token_fields = any(
                    key in attrs
                    for key in (
                        "input_token_count",
                        "output_token_count",
                        "cached_content_token_count",
                        "gen_ai.usage.input_tokens",
                        "gen_ai.usage.output_tokens",
                        "gen_ai.usage.cached_tokens",
                    )
                )

                if input_tok or output_tok or cached_tok or has_token_fields:
                    daily[date]["input_tokens"] += input_tok
                    daily[date]["output_tokens"] += output_tok
                    daily[date]["cache_read_tokens"] += cached_tok
    except (OSError, PermissionError):
        _warn("Could not read Gemini telemetry log (permission or I/O error)")
        return {}

    # Estimate Gemini costs (Gemini 2.5 Pro pricing, cached = 10% of input rate)
    for d in daily.values():
        input_cost = d["input_tokens"] * GEMINI_RATE_INPUT_PER_TOKEN
        output_cost = d["output_tokens"] * GEMINI_RATE_OUTPUT_PER_TOKEN
        cached_cost = d["cache_read_tokens"] * GEMINI_RATE_CACHED_PER_TOKEN
        d["cost"] = input_cost + output_cost + cached_cost

    return dict(daily)


def _safe_int(val: Any) -> int:
    """Convert a value to int safely."""
    if isinstance(val, bool):
        return 0
    try:
        value = int(val)
    except (TypeError, ValueError, OverflowError):
        return 0
    if isinstance(val, float) and not math.isfinite(val):
        return 0
    return value


def _provider_cache_path() -> Path:
    """Return the provider cache location."""
    return Path(tempfile.gettempdir()) / PROVIDER_CACHE_FILENAME


def _empty_provider_cache() -> dict[str, dict[str, dict[str, Any]]]:
    """Return a zero-value provider cache payload."""
    return {name: {} for name in provider_names()}


def _warn_cache(message: str) -> None:
    _warn(message)


def _coerce_cache_schema_version(version: Any) -> int | None:
    """Normalize a cache schema version value, rejecting invalid or non-integer values."""
    if isinstance(version, bool):
        return None

    if isinstance(version, int):
        return version

    if isinstance(version, float):
        if not math.isfinite(version) or not version.is_integer():
            return None
        return int(version)

    if isinstance(version, str):
        cleaned = version.strip()
        if not cleaned:
            return None
        try:
            return int(cleaned)
        except (TypeError, ValueError):
            try:
                version_float = float(cleaned)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(version_float) or not version_float.is_integer():
                return None
            return int(version_float)

    return None


def _extract_provider_payload(
    raw: Any,
) -> tuple[dict[str, Any] | None, int]:
    """Extract provider payload and schema version from raw cache JSON."""
    if not isinstance(raw, dict):
        return None, 0

    known_provider_names = provider_name_set()
    if "version" in raw:
        known_version = _coerce_cache_schema_version(raw.get("version"))
        if known_version is None:
            return None, 0
    else:
        known_version = 1

    if "providers" in raw:
        providers = raw.get("providers")
        if not isinstance(providers, dict):
            return None, 0
        if not isinstance(known_version, int):
            return None, 0
        return providers, known_version

    if not isinstance(known_version, int):
        return None, 0

    if any(provider_name in raw for provider_name in known_provider_names):
        return raw, known_version
    return None, 0


def _migrate_provider_payload(
    provider_payload: dict[str, Any],
    cache_version: int,
) -> dict[str, Any]:
    """Migrate cache payloads from older schema versions."""
    if cache_version <= 0:
        _warn_cache(f"Invalid cache schema version: {cache_version}; ignoring cache")
        return {}

    if cache_version > PROVIDER_CACHE_SCHEMA_VERSION:
        _warn_cache(
            f"Cache schema version {cache_version} is newer than "
            f"supported ({PROVIDER_CACHE_SCHEMA_VERSION}); ignoring cache"
        )
        return {}

    # current migration is additive and schema-compatible.
    return provider_payload


def _resolve_cache_path(cache_path: str | None) -> Path | None:
    """Resolve user-provided cache path into a concrete file path."""
    normalized_cache_path = (cache_path or "").strip()
    resolved_path = normalized_cache_path or os.environ.get(TOKENPRINT_CACHE_PATH_ENV_VAR, "").strip()
    if not resolved_path:
        return None
    path = Path(resolved_path).expanduser()
    if path.is_dir():
        return path / PROVIDER_CACHE_FILENAME
    return path


def _resolve_gemini_log_path(log_path: str | None = None) -> Path:
    """Resolve Gemini telemetry log path from explicit value or environment."""
    resolved_path = (log_path or os.environ.get(GEMINI_TELEMETRY_LOG_PATH_ENV_VAR, "")).strip()
    if not resolved_path:
        return Path.home() / ".gemini" / "telemetry.log"

    path = Path(resolved_path).expanduser()
    if path.is_dir():
        return path / "telemetry.log"
    return path


def _next_day_compact(date_iso: str) -> str | None:
    """Return YYYYMMDD for the day after YYYY-MM-DD, or None if invalid."""
    try:
        return (datetime.strptime(date_iso, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d")
    except ValueError:
        return None


def _normalize_provider_day(provider_name: str, raw: Any) -> dict[str, Any] | None:
    """Normalize a cached provider day payload to the expected shape."""
    if not isinstance(raw, dict):
        return None
    cost = _safe_float(raw.get("cost", 0))
    return {
        "provider": provider_name,
        "input_tokens": max(0, _safe_int(raw.get("input_tokens", 0))),
        "output_tokens": max(0, _safe_int(raw.get("output_tokens", 0))),
        "cache_read_tokens": max(0, _safe_int(raw.get("cache_read_tokens", 0))),
        "cache_write_tokens": max(0, _safe_int(raw.get("cache_write_tokens", 0))),
        "cost": cost,
    }


def _load_provider_cache(cache_path: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """Load provider cache from disk; returns empty maps on any issue."""
    cache_file = cache_path or _provider_cache_path()
    empty = _empty_provider_cache()
    if not cache_file.exists():
        return empty

    try:
        with open(cache_file, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        _warn_cache("Could not read provider cache file")
        return empty

    if not isinstance(raw, dict):
        _warn_cache("Provider cache has invalid top-level structure")
        return empty

    providers_blob, cache_version = _extract_provider_payload(raw)
    if providers_blob is None:
        _warn_cache("Provider cache has unrecognized format")
        return empty

    migrated_payload = _migrate_provider_payload(providers_blob, cache_version)
    if not migrated_payload:
        return empty

    normalized: dict[str, dict[str, dict[str, Any]]] = _empty_provider_cache()
    for name in provider_names():
        raw_provider = migrated_payload.get(name, {})
        if not isinstance(raw_provider, dict):
            _warn_cache(f"Ignoring invalid provider payload for '{name}'")
            continue
        provider_days: dict[str, dict[str, Any]] = {}
        for date_key, entry in raw_provider.items():
            if not isinstance(date_key, str):
                _warn_cache(f"Ignoring non-string cache date key for '{name}'")
                continue
            iso_date = _normalize_cache_date_key(date_key)
            if not iso_date:
                _warn_cache(f"Ignoring invalid cache date key '{date_key}' for '{name}'")
                continue
            normalized_entry = _normalize_provider_day(name, entry)
            if normalized_entry:
                provider_days[iso_date] = normalized_entry
        normalized[name] = provider_days

    return normalized


def _save_provider_cache(
    provider_data: dict[str, dict[str, dict[str, Any]]],
    cache_path: Path | None = None,
) -> None:
    """Persist provider cache to disk."""
    cache_file = cache_path or _provider_cache_path()
    payload = {
        "version": PROVIDER_CACHE_SCHEMA_VERSION,
        "updatedAt": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "providers": provider_data,
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(cache_file, json.dumps(payload, separators=(",", ":")))
    except OSError as exc:
        _warn(f"Could not write provider cache at {cache_file}: {exc}")


def _collect_days_with_fallback(
    provider: ProviderConfig,
    since: str | None,
    until: str | None,
    *,
    mode: str = "collection",
    timezone_name: str = "UTC",
    gemini_log_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Collect provider day rows with a shared failure-to-empty normalization path."""
    collector = getattr(sys.modules[__name__], provider.collect_fn)
    try:
        if provider.name == "gemini":
            if gemini_log_path is None:
                raw_data = collector(since, until, timezone_name)
            else:
                raw_data = collector(since, until, timezone_name, log_path=gemini_log_path)
        else:
            raw_data = collector(since, until)
    except Exception as exc:
        _warn(f"{provider.label} {mode} failed: {exc}")
        return {}
    if not isinstance(raw_data, dict):
        _warn(f"{provider.label} returned invalid data for {mode}")
        return {}
    return raw_data


def _collect_provider_data(
    since: str | None = None,
    until: str | None = None,
    timezone_name: str = "UTC",
    gemini_log_path: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect provider data for an explicit date window."""
    provider_data: dict[str, dict[str, dict[str, Any]]] = {}
    for p in PROVIDERS:
        print(f"  {p.label}...", file=sys.stderr)
        data = _collect_days_with_fallback(
            p,
            since,
            until,
            mode="collection",
            timezone_name=timezone_name,
            gemini_log_path=gemini_log_path,
        )
        print(f"    {len(data)} days", file=sys.stderr)
        provider_data[p.name] = data
    return provider_data


def _collect_provider_data_incremental(
    cache_path: Path | None = None,
    timezone_name: str = "UTC",
    gemini_log_path: str | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect provider data incrementally from cache, per provider."""
    today_compact = datetime.now().strftime("%Y%m%d")
    cached_provider_data = _load_provider_cache(cache_path=cache_path)
    provider_data: dict[str, dict[str, dict[str, Any]]] = {}

    for p in PROVIDERS:
        print(f"  {p.label}...", file=sys.stderr)
        cached_days = cached_provider_data.get(p.name, {})
        if not isinstance(cached_days, dict):
            _warn(f"{p.label} cache payload was invalid; rebuilding full history")
            cached_days = {}
        fresh_days: dict[str, dict[str, Any]] = {}

        if not cached_days:
            # No prior cache for this provider — fetch full history.
            fresh_days = _collect_days_with_fallback(
                p,
                None,
                None,
                mode="full collection",
                timezone_name=timezone_name,
                gemini_log_path=gemini_log_path,
            )
            combined_days = dict(fresh_days)
        else:
            latest_cached = max(cached_days.keys())
            since = _next_day_compact(latest_cached)
            if not since:
                fresh_days = _collect_days_with_fallback(
                    p,
                    None,
                    None,
                    mode="full collection",
                    timezone_name=timezone_name,
                    gemini_log_path=gemini_log_path,
                )
            elif since > today_compact:
                print("    up to date (cache)", file=sys.stderr)
            else:
                print(f"    incremental since {since}", file=sys.stderr)
                fresh_days = _collect_days_with_fallback(
                    p,
                    since,
                    today_compact,
                    mode="incremental collection",
                    timezone_name=timezone_name,
                    gemini_log_path=gemini_log_path,
                )

            combined_days = dict(cached_days)
            if fresh_days:
                combined_days.update(fresh_days)

        print(f"    {len(combined_days)} days", file=sys.stderr)
        provider_data[p.name] = combined_days

    return provider_data


def calculate_energy(tokens_input: int, tokens_output: int, tokens_cached: int) -> float:
    """Calculate energy in Wh with PUE and grid losses."""
    base = (
        tokens_output * ENERGY_PER_OUTPUT_TOKEN_WH
        + tokens_input * ENERGY_PER_INPUT_TOKEN_WH
        + tokens_cached * ENERGY_PER_CACHED_TOKEN_WH
    )
    return base * PUE * GRID_LOSS_FACTOR


def calculate_carbon(energy_wh: float) -> float:
    """Calculate CO2 in grams from energy in Wh, including embodied carbon."""
    energy_kwh = energy_wh / 1000
    return energy_kwh * CARBON_INTENSITY * EMBODIED_CARBON_FACTOR


def calculate_water(energy_wh: float) -> float:
    """Calculate water usage in mL from energy in Wh."""
    energy_kwh = energy_wh / 1000
    return energy_kwh * WATER_USE_EFFICIENCY * 1000  # Convert L to mL


def merge_data(
    provider_data: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge all provider data into a unified daily dataset."""
    all_dates = sorted({d for pdata in provider_data.values() for d in pdata})
    names = provider_names()

    merged = []
    for date in all_dates:
        row: dict[str, Any] = {"date": date}
        for name in names:
            pdata = provider_data.get(name, {})
            if date in pdata:
                raw_row = pdata[date]
                if not isinstance(raw_row, dict):
                    raw_row = {}

                input_tokens = _safe_int(raw_row.get("input_tokens", 0))
                output_tokens = _safe_int(raw_row.get("output_tokens", 0))
                cache_read_tokens = _safe_int(raw_row.get("cache_read_tokens", 0))
                cache_write_tokens = _safe_int(raw_row.get("cache_write_tokens", 0))
                cached_tokens = cache_read_tokens + cache_write_tokens
                cost = _safe_float(raw_row.get("cost", 0))

                energy = calculate_energy(input_tokens, output_tokens, cached_tokens)
                row[name] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens,
                    "cost": round(cost, 4),
                    "energy_wh": round(energy, 4),
                    "carbon_g": round(calculate_carbon(energy), 4),
                    "water_ml": round(calculate_water(energy), 4),
                }
            else:
                row[name] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0,
                    "energy_wh": 0,
                    "carbon_g": 0,
                    "water_ml": 0,
                }
        merged.append(row)
    return merged


def _json_dumps_html_safe(value: Any) -> str:
    """json.dumps with HTML-safe escaping for embedding in <script> blocks."""
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def compute_dashboard_data(
    data: list[dict[str, Any]],
    github_username: str | None = None,
    live_mode: bool = False,
    refresh_endpoint: str | None = None,
    refresh_token: str = "",
) -> dict[str, Any]:
    """Compute the config dict that the HTML template needs."""
    if github_username is None:
        github_username = detect_github_username()

    # Provider data presence (for default toggle state)
    provider_has_data = {
        name: any(
            r[name]["input_tokens"]
            + r[name]["output_tokens"]
            + r[name]["cache_read_tokens"]
            + r[name]["cache_write_tokens"]
            > 0
            for r in data
        )
        for name in provider_names()
    }

    # Date range
    min_date = data[0]["date"] if data else ""
    max_date = data[-1]["date"] if data else ""

    # Raw data for client-side filtering and rendering
    raw_data = []
    for r in data:
        row: dict[str, Any] = {"d": r["date"]}
        for p in PROVIDERS:
            cached_tokens = r[p.name]["cache_read_tokens"] + r[p.name]["cache_write_tokens"]
            row[p.key] = [
                r[p.name]["input_tokens"],
                r[p.name]["output_tokens"],
                cached_tokens,
                round(r[p.name]["cost"], 4),
            ]
        raw_data.append(row)

    return {
        "rawData": raw_data,
        "githubUser": github_username,
        "providerHasData": provider_has_data,
        "providers": [
            {
                "name": p.name,
                "displayName": p.display_name,
                "key": p.key,
                "color": p.color,
                "rates": {"input": p.rates[0], "output": p.rates[1], "cached": p.rates[2]},
            }
            for p in PROVIDERS
        ],
        "minDate": min_date,
        "maxDate": max_date,
        "electricityCostKwh": ELECTRICITY_COST_KWH,
        "generatedAt": datetime.now().strftime("%m/%d/%Y, %I:%M %p"),
        "liveMode": live_mode,
        "refreshEndpoint": refresh_endpoint or "",
        "refreshToken": refresh_token,
        "energyModel": {
            "outputWhPerToken": ENERGY_PER_OUTPUT_TOKEN_WH,
            "inputWhPerToken": ENERGY_PER_INPUT_TOKEN_WH,
            "cachedWhPerToken": ENERGY_PER_CACHED_TOKEN_WH,
            "pue": PUE,
            "gridLossFactor": GRID_LOSS_FACTOR,
            "carbonIntensity": CARBON_INTENSITY,
            "embodiedCarbonFactor": EMBODIED_CARBON_FACTOR,
            "waterUseEfficiency": WATER_USE_EFFICIENCY,
        },
    }


def _render_html_from_config(config: dict[str, Any]) -> str:
    """Render self-contained HTML from config using the template."""
    config_json = _json_dumps_html_safe(config)

    template_path = Path(__file__).resolve().parent / "template.html"
    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    return template.replace("TOKENPRINT_DATA_PLACEHOLDER", config_json)


def generate_html(data: list[dict[str, Any]], output_path: str) -> None:
    """Generate the self-contained HTML dashboard from template."""
    config = compute_dashboard_data(data)
    html = _render_html_from_config(config)
    _write_html_file(output_path, html)


def _default_output_path() -> str:
    """Return default dashboard output path."""
    return os.path.join(tempfile.gettempdir(), "tokenprint.html")


def _default_output_path_json() -> str:
    """Return default JSON output path."""
    return os.path.join(tempfile.gettempdir(), "tokenprint.json")


def _write_json_file(output_path: str, payload: dict[str, Any]) -> None:
    """Write JSON payload to output path."""
    _write_output_file(output_path, json.dumps(payload), "JSON output")


def _write_html_file(output_path: str, html: str) -> None:
    """Write rendered HTML to output path."""
    _write_output_file(output_path, html, "dashboard file")


def _write_output_file(output_path: str, content: str, label: str) -> None:
    """Write arbitrary output text with a user-friendly error path."""
    path = Path(output_path)
    if path.exists() and path.is_dir():
        raise RuntimeError(f"unable to write {label} at {output_path}: target is a directory")
    try:
        _write_text_atomic(path, content)
    except OSError as exc:
        raise RuntimeError(f"unable to write {label} at {output_path}: {exc}") from exc


def _write_text_atomic(path: Path, content: str) -> None:
    """Write text to disk using an atomic replace after completing a full write."""
    temp_file_path: Path | None = None
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file_path = Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_file_path, path)
    except OSError:
        if temp_file_path is not None and temp_file_path.exists():
            with suppress(OSError):
                temp_file_path.unlink()
        raise


def _collect_merged_usage_data(
    since: str | None,
    until: str | None,
    no_cache: bool,
    cache_path: Path | None = None,
    timezone_name: str = "UTC",
    gemini_log_path: str | None = None,
) -> list[dict[str, Any]]:
    """Collect provider data and return merged daily rows."""
    is_default_range = not since and not until
    if is_default_range and not no_cache:
        print("Collecting AI usage data (incremental)...")
        if gemini_log_path is None:
            provider_data = _collect_provider_data_incremental(cache_path=cache_path, timezone_name=timezone_name)
        else:
            provider_data = _collect_provider_data_incremental(
                cache_path=cache_path,
                timezone_name=timezone_name,
                gemini_log_path=gemini_log_path,
            )
    else:
        print("Collecting AI usage data...")
        if gemini_log_path is None:
            provider_data = _collect_provider_data(since, until, timezone_name)
        else:
            provider_data = _collect_provider_data(
                since,
                until,
                timezone_name,
                gemini_log_path=gemini_log_path,
            )

    if not any(provider_data.values()):
        raise RuntimeError("No usage data found from any source.")

    if is_default_range:
        _save_provider_cache(provider_data, cache_path=cache_path)

    merged = merge_data(provider_data)
    print(f"\nMerged: {len(merged)} days of data", file=sys.stderr)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI usage & impact dashboard")
    parser.add_argument("--since", help="Start date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--until", help="End date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true", help="Force full refresh (ignore incremental cache)")
    parser.add_argument("--live-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--refresh-endpoint", default="/api/refresh", help=argparse.SUPPRESS)
    parser.add_argument("--refresh-token", default="", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version=f"tokenprint {_tokenprint_version()}")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--cache-path", default="", help="Optional provider cache path (default: temp directory)")
    parser.add_argument(
        "--gemini-log-path",
        help="Path to Gemini telemetry log or directory (overrides TOKENPRINT_GEMINI_TELEMETRY_LOG_PATH)",
    )
    parser.add_argument("--timezone", default="UTC", help="Timezone for date filtering (IANA timezone)")
    parser.add_argument("--output-format", choices=["html", "json"], default="html", help="Output format: html or json")
    parser.add_argument("--check", action="store_true", help="Run preflight checks and exit")
    args = parser.parse_args()
    if args.output is not None:
        args.output = args.output.strip() or None
    if args.cache_path is not None:
        args.cache_path = args.cache_path.strip() or None
    if args.refresh_endpoint is not None:
        args.refresh_endpoint = args.refresh_endpoint.strip() or "/api/refresh"
    if args.refresh_token is not None:
        args.refresh_token = args.refresh_token.strip()
    if args.gemini_log_path is not None:
        args.gemini_log_path = args.gemini_log_path.strip() or None
    try:
        timezone_name = _normalize_timezone_name(args.timezone)
    except ValueError as exc:
        parser.error(str(exc))

    if args.check:
        if not _run_cli_check():
            sys.exit(1)
        sys.exit(0)

    # Validate date arguments (syntax + calendar validity), and ensure ranges are ordered.
    parsed_since = None
    parsed_until = None
    for name, val in [("since", args.since), ("until", args.until)]:
        parsed = _normalize_cli_date_arg(val)
        if val and not parsed:
            parser.error(f"--{name} must be YYYYMMDD or YYYY-MM-DD (got: {val})")
        if name == "since" and parsed:
            parsed_since = parsed
        elif name == "until" and parsed:
            parsed_until = parsed

    if parsed_since and parsed_until:
        parsed_since_date = datetime.strptime(parsed_since, "%Y%m%d").date()
        parsed_until_date = datetime.strptime(parsed_until, "%Y%m%d").date()
        if parsed_since_date > parsed_until_date:
            parser.error("--since must be before or equal to --until")

    try:
        cache_path = _resolve_cache_path(args.cache_path)
        merged = _collect_merged_usage_data(
            parsed_since,
            parsed_until,
            args.no_cache,
            cache_path=cache_path,
            timezone_name=timezone_name,
            gemini_log_path=args.gemini_log_path,
        )
    except RuntimeError:
        print("\nNo usage data found from any source.", file=sys.stderr)
        print("Make sure ccusage is installed: npm i -g ccusage", file=sys.stderr)
        sys.exit(1)

    config = compute_dashboard_data(
        merged,
        live_mode=args.live_mode,
        refresh_endpoint=args.refresh_endpoint if args.live_mode else "",
        refresh_token=args.refresh_token if args.live_mode else "",
    )
    output_path = args.output or (
        _default_output_path() if args.output_format == "html" else _default_output_path_json()
    )
    try:
        if args.output_format == "json":
            _write_json_file(output_path, config)
        else:
            html = _render_html_from_config(config)
            _write_html_file(output_path, html)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Dashboard written to: {output_path}")

    if args.output_format == "html" and not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
        print("Opened in browser.")


if __name__ == "__main__":
    main()
