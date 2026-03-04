#!/usr/bin/env python3
"""TokenPrint constants.

Extracted from core module to keep parsing/collection logic focused and reduce
future churn when model/rate assumptions change.
"""

from __future__ import annotations

# --- Claude ---
CLAUDE_RATE_INPUT_PER_TOKEN = 3e-6
CLAUDE_RATE_OUTPUT_PER_TOKEN = 15e-6
CLAUDE_RATE_CACHED_PER_TOKEN = 0.30e-6

# Claude prompt caching published multipliers:
# cache-write tokens cost 1.25x input
# cache-read tokens cost 0.10x input
CLAUDE_RATE_CACHE_WRITE_MULTIPLIER = 1.25
CLAUDE_RATE_CACHE_READ_MULTIPLIER = 0.10

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

GEMINI_RATE_INPUT_PER_TOKEN = 1.25e-6
GEMINI_RATE_OUTPUT_PER_TOKEN = 10.0e-6
GEMINI_RATE_CACHED_PER_TOKEN = 0.125e-6

CODEX_RATE_INPUT_PER_TOKEN = 0.69e-6
CODEX_RATE_OUTPUT_PER_TOKEN = 2.76e-6
CODEX_RATE_CACHED_PER_TOKEN = 0.17e-6

# --- Energy / Carbon Model ---
ENERGY_PER_OUTPUT_TOKEN_WH = 0.001
ENERGY_PER_INPUT_TOKEN_WH = 0.0002
ENERGY_PER_CACHED_TOKEN_WH = 0.00005
PUE = 1.2  # Power Usage Effectiveness (data center overhead)
EMBODIED_CARBON_FACTOR = 1.2  # +20% for hardware manufacturing
GRID_LOSS_FACTOR = 1.05  # 5% transmission losses (EIA)
CARBON_INTENSITY = 390  # gCO2e per kWh (US average)
WATER_USE_EFFICIENCY = 0.5  # liters per kWh
ELECTRICITY_COST_KWH = 0.13  # USD per kWh (EIA commercial average)

PROVIDER_CACHE_SCHEMA_VERSION = 2
PROVIDER_CACHE_FILENAME = "tokenprint-provider-cache-v1.json"
TOKENPRINT_CACHE_PATH_ENV_VAR = "TOKENPRINT_CACHE_PATH"
GEMINI_TELEMETRY_LOG_PATH_ENV_VAR = "TOKENPRINT_GEMINI_TELEMETRY_LOG_PATH"
