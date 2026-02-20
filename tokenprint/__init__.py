#!/usr/bin/env python3
"""
TokenPrint Dashboard Generator

Collects token usage data from Claude Code (ccusage), Codex CLI (@ccusage/codex),
and Gemini CLI (OpenTelemetry logs), then generates an interactive HTML dashboard
showing usage trends, costs, and estimated environmental impact.

Usage:
    tokenprint [--since YYYYMMDD] [--until YYYYMMDD] [--no-open] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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
    for entry in data:
        date = entry.get("date", "")
        if not date:
            continue
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
        daily[date]["cost"] += entry.get("totalCost", 0) or 0
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


def collect_codex_data(since: str | None = None, until: str | None = None) -> dict[str, dict[str, Any]]:
    """Collect Codex CLI usage via @ccusage/codex."""
    cmd = ["npx", "@ccusage/codex@18", "daily", "--json"]
    if since:
        cmd.extend(["--since", since])
    if until:
        cmd.extend(["--until", until])

    output = run_command(cmd, timeout=60)
    if not output:
        print("  [skip] @ccusage/codex not available or returned no data", file=sys.stderr)
        return {}

    try:
        raw = json.loads(output)
    except json.JSONDecodeError:
        print("  [skip] @ccusage/codex returned invalid JSON", file=sys.stderr)
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


def compute_dashboard_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the config dict that the HTML template needs."""
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


def generate_html(data: list[dict[str, Any]], output_path: str) -> None:
    """Generate the self-contained HTML dashboard from template."""
    config = compute_dashboard_data(data)
    config_json = _json_dumps_html_safe(config)

    template_path = Path(__file__).resolve().parent / "template.html"
    with open(template_path) as f:
        template = f.read()

    html = template.replace("TOKENPRINT_DATA_PLACEHOLDER", config_json)

    with open(output_path, "w") as f:
        f.write(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI usage & impact dashboard")
    parser.add_argument("--since", help="Start date (YYYYMMDD)")
    parser.add_argument("--until", help="End date (YYYYMMDD)")
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

    print("Collecting AI usage data...")

    provider_data: dict[str, dict[str, dict[str, Any]]] = {}
    for p in PROVIDERS:
        print(f"  {p.label}...", file=sys.stderr)
        collector = getattr(sys.modules[__name__], p.collect_fn)
        data = collector(args.since, args.until)
        print(f"    {len(data)} days", file=sys.stderr)
        provider_data[p.name] = data

    if not any(provider_data.values()):
        print("\nNo usage data found from any source.", file=sys.stderr)
        print("Make sure ccusage is installed: npm i -g ccusage", file=sys.stderr)
        sys.exit(1)

    merged = merge_data(provider_data)
    print(f"\nMerged: {len(merged)} days of data", file=sys.stderr)

    output_path = args.output
    if not output_path:
        output_path = os.path.join(tempfile.gettempdir(), "tokenprint.html")

    generate_html(merged, output_path)
    print(f"Dashboard written to: {output_path}")

    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
        print("Opened in browser.")


if __name__ == "__main__":
    main()
