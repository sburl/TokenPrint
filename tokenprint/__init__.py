#!/usr/bin/env python3
"""
TokenPrint Dashboard Generator

Collects token usage data from Claude Code (ccusage), Codex CLI (@ccusage/codex),
and Gemini CLI (OpenTelemetry logs), then generates an interactive HTML dashboard
showing usage trends, costs, and estimated environmental impact.

Usage:
    tokenprint [--since YYYYMMDD] [--until YYYYMMDD] [--no-cache] [--serve] [--no-open] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


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
    ProviderConfig("claude", "Claude Code", "c", "#6366f1",
                   "collect_claude_data", "Claude Code (ccusage)",
                   (3e-6, 15e-6, 0.30e-6)),
    ProviderConfig("codex", "Codex CLI", "x", "#22c55e",
                   "collect_codex_data", "Codex CLI (@ccusage/codex)",
                   (0.69e-6, 2.76e-6, 0.17e-6)),
    ProviderConfig("gemini", "Gemini CLI", "g", "#f59e0b",
                   "collect_gemini_data", "Gemini CLI (telemetry)",
                   (1.25e-6, 10.0e-6, 0.125e-6)),
)

# --- Energy / Carbon Model ---
ENERGY_PER_OUTPUT_TOKEN_WH = 0.001
ENERGY_PER_INPUT_TOKEN_WH = 0.0002
ENERGY_PER_CACHED_TOKEN_WH = 0.00005
PUE = 1.2                    # Power Usage Effectiveness (data center overhead)
EMBODIED_CARBON_FACTOR = 1.2  # +20% for hardware manufacturing
GRID_LOSS_FACTOR = 1.05       # 5% transmission losses (EIA)
CARBON_INTENSITY = 390        # gCO2e per kWh (US average)
WATER_USE_EFFICIENCY = 0.5    # liters per kWh
ELECTRICITY_COST_KWH = 0.13  # USD per kWh (EIA commercial average)
PROVIDER_CACHE_SCHEMA_VERSION = 1
PROVIDER_CACHE_FILENAME = "tokenprint-provider-cache-v1.json"

# Claude pricing fallback for models that may appear unpriced in ccusage output.
# Source: https://platform.claude.com/docs/en/about-claude/pricing (checked 2026-02-20).
CLAUDE_RATE_BY_MODEL_PREFIX: tuple[tuple[str, tuple[float, float]], ...] = (
    ("claude-opus-4-6", (5e-6, 25e-6)),
    ("claude-opus-4-5", (5e-6, 25e-6)),
    ("claude-opus-4-1", (15e-6, 75e-6)),
    ("claude-opus-4", (15e-6, 75e-6)),
    ("claude-sonnet-4-6", (3e-6, 15e-6)),
    ("claude-sonnet-4-5", (3e-6, 15e-6)),
    ("claude-sonnet-4", (3e-6, 15e-6)),
    ("claude-sonnet-3-7", (3e-6, 15e-6)),
    ("claude-haiku-4-5", (1e-6, 5e-6)),
    ("claude-haiku-3-5", (0.8e-6, 4e-6)),
    ("claude-haiku-3", (0.25e-6, 1.25e-6)),
)


def run_command(cmd: list[str], timeout: int = 60) -> str | None:
    """Run a command (list of args) and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _safe_float(val: Any) -> float:
    """Convert a value to float safely."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


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
        pass
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
    cache_write_rate = input_rate * 1.25
    cache_read_rate = input_rate * 0.10

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


def collect_claude_data(since: str | None = None, until: str | None = None) -> dict[str, dict[str, Any]]:
    """Collect Claude Code usage via ccusage."""
    cmd = ["ccusage", "daily", "--json"]
    if since:
        cmd.extend(["--since", since])
    if until:
        cmd.extend(["--until", until])

    output = run_command(cmd, timeout=120)
    if not output:
        print("  [skip] ccusage not available or returned no data", file=sys.stderr)
        return {}

    try:
        raw = json.loads(output)
    except json.JSONDecodeError:
        print("  [skip] ccusage returned invalid JSON", file=sys.stderr)
        return {}

    # ccusage wraps data in {"daily": [...]}
    data = raw.get("daily", raw) if isinstance(raw, dict) else raw

    daily: dict[str, dict[str, Any]] = {}
    estimated_missing_cost_total = 0.0
    estimated_models: set[str] = set()
    for entry in data:
        date = entry.get("date", "")
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
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0,
            }
        daily[date]["input_tokens"] += entry.get("inputTokens", 0) or 0
        daily[date]["output_tokens"] += entry.get("outputTokens", 0) or 0
        daily[date]["cache_read_tokens"] += entry.get("cacheReadTokens", 0) or 0
        daily[date]["cache_write_tokens"] += entry.get("cacheCreationTokens", 0) or 0
        daily[date]["cost"] += day_cost

    if estimated_missing_cost_total > 0:
        models_str = ", ".join(sorted(estimated_models)) if estimated_models else "unknown models"
        print(
            f"  [info] Added ${estimated_missing_cost_total:.2f} estimated Claude cost for unpriced models: {models_str}",
            file=sys.stderr,
        )
    return daily


def _parse_date_flexible(date_str: str | None) -> str | None:
    """Parse dates in ISO (2026-01-07) or human (Jan 7, 2026) format to YYYY-MM-DD."""
    if not date_str:
        return None
    # Try ISO format first (validate calendar correctness)
    if len(date_str) >= 10 and date_str[4] == '-':
        candidate = date_str[:10]
        try:
            datetime.strptime(candidate, "%Y-%m-%d")
            return candidate
        except ValueError:
            return None
    # Try human-readable formats
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


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
        print("  [skip] @ccusage/codex unavailable (missing install or invalid JSON)", file=sys.stderr)
        print("         Install once: npm i -g @ccusage/codex@18", file=sys.stderr)
        return {}

    # @ccusage/codex wraps data in {"daily": [...]}
    data = raw.get("daily", raw) if isinstance(raw, dict) else raw

    # First pass: collect all entries
    entries: list[tuple[str, int, int, int, float]] = []
    for entry in data:
        raw_date = entry.get("date", "")
        if not raw_date:
            continue
        date = _parse_date_flexible(raw_date)
        if not date:
            continue
        raw_input = entry.get("inputTokens", 0) or 0
        output_tok = entry.get("outputTokens", 0) or 0
        cached_tok = entry.get("cachedInputTokens", 0) or 0
        # Codex inputTokens includes cached — subtract to get non-cached input
        input_tok = max(0, raw_input - cached_tok)
        cost = entry.get("costUSD", 0) or 0
        entries.append((date, input_tok, output_tok, cached_tok, cost))

    # Use gpt-5-codex pricing for all unpriced days (including gpt-5.3-codex)
    rate_input = 0.69e-6    # $0.69/M input tokens
    rate_output = 2.76e-6   # $2.76/M output tokens
    rate_cached = 0.17e-6   # $0.17/M cached tokens

    daily: dict[str, dict[str, Any]] = {}
    for date, input_tok, output_tok, cached_tok, cost in entries:
        if not cost and (input_tok or output_tok):
            cost = input_tok * rate_input + output_tok * rate_output + cached_tok * rate_cached
        if date not in daily:
            daily[date] = {
                "provider": "codex",
                "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0,
            }
        daily[date]["input_tokens"] += input_tok
        daily[date]["output_tokens"] += output_tok
        daily[date]["cache_read_tokens"] += cached_tok
        daily[date]["cost"] += cost
    return daily


def collect_gemini_data(since: str | None = None, until: str | None = None) -> dict[str, dict[str, Any]]:
    """Collect Gemini CLI usage from OpenTelemetry log."""
    log_path = Path.home() / ".gemini" / "telemetry.log"
    if not log_path.exists():
        print("  [skip] Gemini telemetry log not found (~/.gemini/telemetry.log)", file=sys.stderr)
        print("         Run: bash install.sh (or bash setup-gemini-telemetry.sh)", file=sys.stderr)
        return {}

    daily: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "provider": "gemini",
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "cost": 0,
    })

    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract timestamp for date grouping
                ts = record.get("timestamp") or record.get("time") or record.get("Timestamp") or ""
                if not ts:
                    continue
                try:
                    date = ts[:10]  # YYYY-MM-DD
                    datetime.strptime(date, "%Y-%m-%d")
                except (ValueError, IndexError):
                    continue

                # Apply date filters
                date_compact = date.replace("-", "")
                if since and date_compact < since:
                    continue
                if until and date_compact > until:
                    continue

                # Look for token usage in attributes or body
                attrs = record.get("attributes", record.get("Attributes", {}))
                if isinstance(attrs, list):
                    attrs_dict: dict[str, Any] = {}
                    for a in attrs:
                        if not isinstance(a, dict):
                            continue
                        key = a.get("Key", a.get("key", ""))
                        val = a.get("Value", a.get("value", {}))
                        if isinstance(val, dict):
                            val = val.get("intValue", val.get("Int64Value", val.get("stringValue", 0)))
                        attrs_dict[key] = val
                    attrs = attrs_dict
                if not isinstance(attrs, dict):
                    continue

                raw_input = _safe_int(attrs.get("input_token_count", attrs.get("gen_ai.usage.input_tokens", 0)))
                output_tok = _safe_int(attrs.get("output_token_count", attrs.get("gen_ai.usage.output_tokens", 0)))
                cached_tok = _safe_int(attrs.get("cached_content_token_count", attrs.get("gen_ai.usage.cached_tokens", 0)))
                # Gemini input_token_count includes cached — subtract to get non-cached input
                input_tok = max(0, raw_input - cached_tok)

                if input_tok or output_tok:
                    daily[date]["input_tokens"] += input_tok
                    daily[date]["output_tokens"] += output_tok
                    daily[date]["cache_read_tokens"] += cached_tok
    except (OSError, PermissionError):
        print("  [skip] Could not read Gemini telemetry log (permission or I/O error)", file=sys.stderr)
        return {}

    # Estimate Gemini costs (Gemini 2.5 Pro pricing, cached = 10% of input rate)
    for d in daily.values():
        input_cost = d["input_tokens"] * 1.25 / 1_000_000
        output_cost = d["output_tokens"] * 10.00 / 1_000_000
        cached_cost = d["cache_read_tokens"] * 0.125 / 1_000_000
        d["cost"] = input_cost + output_cost + cached_cost

    return dict(daily)


def _safe_int(val: Any) -> int:
    """Convert a value to int safely."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _provider_cache_path() -> Path:
    """Return the provider cache location."""
    return Path(tempfile.gettempdir()) / PROVIDER_CACHE_FILENAME


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
    try:
        cost = float(raw.get("cost", 0) or 0)
    except (TypeError, ValueError):
        cost = 0.0
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
    empty: dict[str, dict[str, dict[str, Any]]] = {p.name: {} for p in PROVIDERS}
    if not cache_file.exists():
        return empty

    try:
        with open(cache_file) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty

    if not isinstance(raw, dict):
        return empty
    if raw.get("version") != PROVIDER_CACHE_SCHEMA_VERSION:
        return empty

    providers_blob = raw.get("providers")
    if not isinstance(providers_blob, dict):
        return empty

    normalized: dict[str, dict[str, dict[str, Any]]] = {p.name: {} for p in PROVIDERS}
    for p in PROVIDERS:
        raw_provider = providers_blob.get(p.name, {})
        if not isinstance(raw_provider, dict):
            continue
        provider_days: dict[str, dict[str, Any]] = {}
        for date_key, entry in raw_provider.items():
            if not isinstance(date_key, str):
                continue
            iso_date = _parse_date_flexible(date_key)
            if not iso_date:
                continue
            normalized_entry = _normalize_provider_day(p.name, entry)
            if normalized_entry:
                provider_days[iso_date] = normalized_entry
        normalized[p.name] = provider_days

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
        with open(cache_file, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
    except OSError:
        print("  [warn] Could not write provider cache", file=sys.stderr)


def _collect_provider_data(since: str | None = None, until: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect provider data for an explicit date window."""
    provider_data: dict[str, dict[str, dict[str, Any]]] = {}
    for p in PROVIDERS:
        print(f"  {p.label}...", file=sys.stderr)
        collector = getattr(sys.modules[__name__], p.collect_fn)
        data = collector(since, until)
        print(f"    {len(data)} days", file=sys.stderr)
        provider_data[p.name] = data
    return provider_data


def _collect_provider_data_incremental() -> dict[str, dict[str, dict[str, Any]]]:
    """Collect provider data incrementally from cache, per provider."""
    today_compact = datetime.now().strftime("%Y%m%d")
    cached_provider_data = _load_provider_cache()
    provider_data: dict[str, dict[str, dict[str, Any]]] = {}
    all_cached_dates = [d for pdata in cached_provider_data.values() for d in pdata.keys()]
    global_next_day = _next_day_compact(max(all_cached_dates)) if all_cached_dates else None

    for p in PROVIDERS:
        print(f"  {p.label}...", file=sys.stderr)
        collector = getattr(sys.modules[__name__], p.collect_fn)
        cached_days = cached_provider_data.get(p.name, {})
        fresh_days: dict[str, dict[str, Any]] = {}

        if not cached_days:
            if global_next_day and global_next_day <= today_compact:
                print(f"    incremental since {global_next_day}", file=sys.stderr)
                fresh_days = collector(global_next_day, today_compact)
            elif global_next_day and global_next_day > today_compact:
                print("    up to date (cache)", file=sys.stderr)
            else:
                fresh_days = collector(None, None)
            combined_days = dict(fresh_days)
        else:
            latest_cached = max(cached_days.keys())
            since = _next_day_compact(latest_cached)
            if not since:
                fresh_days = collector(None, None)
            elif since > today_compact:
                print("    up to date (cache)", file=sys.stderr)
            else:
                print(f"    incremental since {since}", file=sys.stderr)
                fresh_days = collector(since, today_compact)

            combined_days = dict(cached_days)
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
    provider_names = [p.name for p in PROVIDERS]

    merged = []
    for date in all_dates:
        row: dict[str, Any] = {"date": date}
        for name in provider_names:
            pdata = provider_data.get(name, {})
            if date in pdata:
                d = pdata[date]
                energy = calculate_energy(d["input_tokens"], d["output_tokens"], d["cache_read_tokens"])
                row[name] = {
                    "input_tokens": d["input_tokens"],
                    "output_tokens": d["output_tokens"],
                    "cache_read_tokens": d["cache_read_tokens"],
                    "cost": round(d["cost"], 4),
                    "energy_wh": round(energy, 4),
                    "carbon_g": round(calculate_carbon(energy), 4),
                    "water_ml": round(calculate_water(energy), 4),
                }
            else:
                row[name] = {
                    "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                    "cost": 0, "energy_wh": 0, "carbon_g": 0, "water_ml": 0,
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
) -> dict[str, Any]:
    """Compute the config dict that the HTML template needs."""
    if github_username is None:
        github_username = detect_github_username()

    # Provider data presence (for default toggle state)
    provider_has_data = {
        p.name: any(
            r[p.name]["input_tokens"] + r[p.name]["output_tokens"] + r[p.name]["cache_read_tokens"] > 0
            for r in data
        )
        for p in PROVIDERS
    }

    # Date range
    min_date = data[0]["date"] if data else ""
    max_date = data[-1]["date"] if data else ""

    # Raw data for client-side filtering and rendering
    raw_data = []
    for r in data:
        row: dict[str, Any] = {"d": r["date"]}
        for p in PROVIDERS:
            row[p.key] = [
                r[p.name]["input_tokens"], r[p.name]["output_tokens"],
                r[p.name]["cache_read_tokens"], round(r[p.name]["cost"], 4),
            ]
        raw_data.append(row)

    return {
        "rawData": raw_data,
        "githubUser": github_username,
        "providerHasData": provider_has_data,
        "providers": [
            {"name": p.name, "displayName": p.display_name, "key": p.key,
             "color": p.color, "rates": {"input": p.rates[0], "output": p.rates[1], "cached": p.rates[2]}}
            for p in PROVIDERS
        ],
        "minDate": min_date,
        "maxDate": max_date,
        "electricityCostKwh": ELECTRICITY_COST_KWH,
        "generatedAt": datetime.now().strftime("%m/%d/%Y, %I:%M %p"),
        "liveMode": live_mode,
        "refreshEndpoint": refresh_endpoint or "",
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
    with open(template_path) as f:
        template = f.read()

    return template.replace("TOKENPRINT_DATA_PLACEHOLDER", config_json)


def generate_html(data: list[dict[str, Any]], output_path: str) -> None:
    """Generate the self-contained HTML dashboard from template."""
    config = compute_dashboard_data(data)
    html = _render_html_from_config(config)

    with open(output_path, "w") as f:
        f.write(html)


def _default_output_path() -> str:
    """Return default dashboard output path."""
    return os.path.join(tempfile.gettempdir(), "tokenprint.html")


def _write_html_file(output_path: str, html: str) -> None:
    """Write rendered HTML to output path."""
    with open(output_path, "w") as f:
        f.write(html)


def _collect_merged_usage_data(since: str | None, until: str | None, no_cache: bool) -> list[dict[str, Any]]:
    """Collect provider data and return merged daily rows."""
    is_default_range = not since and not until
    if is_default_range and not no_cache:
        print("Collecting AI usage data (incremental)...")
        provider_data = _collect_provider_data_incremental()
    else:
        print("Collecting AI usage data...")
        provider_data = _collect_provider_data(since, until)

    if is_default_range:
        _save_provider_cache(provider_data)

    if not any(provider_data.values()):
        raise RuntimeError("No usage data found from any source.")

    merged = merge_data(provider_data)
    print(f"\nMerged: {len(merged)} days of data", file=sys.stderr)
    return merged


def _serve_dashboard(args: argparse.Namespace) -> None:
    """Run local live dashboard server with UI-triggered refresh."""
    refresh_path = "/api/refresh"
    open_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    base_url = f"http://{open_host}:{args.port}"
    output_path = args.output or _default_output_path()
    state_lock = threading.Lock()
    github_username = detect_github_username()
    state: dict[str, str] = {"html": ""}

    def rebuild_dashboard() -> dict[str, Any]:
        merged = _collect_merged_usage_data(args.since, args.until, args.no_cache)
        config = compute_dashboard_data(
            merged,
            github_username=github_username,
            live_mode=True,
            refresh_endpoint=refresh_path,
        )
        html = _render_html_from_config(config)
        state["html"] = html
        _write_html_file(output_path, html)
        return config

    # Initial build before serving.
    rebuild_dashboard()

    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args_log: Any) -> None:
            # Keep HTTP noise out of terminal.
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_html(self, status: int, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                with state_lock:
                    html = state["html"]
                self._send_html(200, html)
                return
            if path == "/api/status":
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != refresh_path:
                self._send_json(404, {"ok": False, "error": "not_found"})
                return

            acquired = state_lock.acquire(blocking=False)
            if not acquired:
                self._send_json(409, {"ok": False, "error": "refresh_in_progress"})
                return

            try:
                try:
                    config = rebuild_dashboard()
                    self._send_json(200, {"ok": True, "generatedAt": config.get("generatedAt", "")})
                except RuntimeError as err:
                    self._send_json(500, {"ok": False, "error": str(err)})
            finally:
                state_lock.release()

    try:
        server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    except OSError as err:
        print(f"Could not start server on {args.host}:{args.port}: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Live dashboard at {base_url}")
    print(f"Snapshot file: {output_path}")
    print("Press Ctrl-C to stop.")
    if not args.no_open:
        webbrowser.open(base_url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped live dashboard.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI usage & impact dashboard")
    parser.add_argument("--since", help="Start date (YYYYMMDD)")
    parser.add_argument("--until", help="End date (YYYYMMDD)")
    parser.add_argument("--no-cache", action="store_true", help="Force full refresh (ignore incremental cache)")
    parser.add_argument("--live-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--refresh-endpoint", default="/api/refresh", help=argparse.SUPPRESS)
    parser.add_argument("--serve", action="store_true", help="Run live local server with UI-triggered refresh")
    parser.add_argument("--host", default="127.0.0.1", help="Host for --serve (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port for --serve (default: 8765)")
    parser.add_argument("--no-open", action="store_true", help="Don't open in browser")
    parser.add_argument("--output", help="Output HTML path")
    args = parser.parse_args()

    # Validate date arguments (syntax + calendar validity)
    for name, val in [("since", args.since), ("until", args.until)]:
        if val:
            if not re.match(r"^\d{8}$", val):
                parser.error(f"--{name} must be YYYYMMDD format (got: {val})")
            try:
                datetime.strptime(val, "%Y%m%d")
            except ValueError:
                parser.error(f"--{name} is not a valid date (got: {val})")

    if args.serve:
        _serve_dashboard(args)
        return

    try:
        merged = _collect_merged_usage_data(args.since, args.until, args.no_cache)
    except RuntimeError:
        print("\nNo usage data found from any source.", file=sys.stderr)
        print("Make sure ccusage is installed: npm i -g ccusage", file=sys.stderr)
        sys.exit(1)

    config = compute_dashboard_data(
        merged,
        live_mode=args.live_mode,
        refresh_endpoint=args.refresh_endpoint if args.live_mode else "",
    )
    html = _render_html_from_config(config)
    output_path = args.output or _default_output_path()
    _write_html_file(output_path, html)
    print(f"Dashboard written to: {output_path}")

    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
        print("Opened in browser.")


if __name__ == "__main__":
    main()
