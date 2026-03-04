"""Tests for tokenprint.py pure functions."""

import json
import os
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
import tokenprint as tp
from unittest.mock import patch

import pytest

from tokenprint import (
    CLAUDE_RATE_BY_MODEL_PREFIX,
    CLAUDE_RATE_CACHED_PER_TOKEN,
    CLAUDE_RATE_CACHE_READ_MULTIPLIER,
    CLAUDE_RATE_CACHE_WRITE_MULTIPLIER,
    CLAUDE_RATE_INPUT_PER_TOKEN,
    CLAUDE_RATE_OUTPUT_PER_TOKEN,
    CODEX_RATE_CACHED_PER_TOKEN,
    CODEX_RATE_INPUT_PER_TOKEN,
    CODEX_RATE_OUTPUT_PER_TOKEN,
    GEMINI_RATE_CACHED_PER_TOKEN,
    GEMINI_RATE_INPUT_PER_TOKEN,
    GEMINI_RATE_OUTPUT_PER_TOKEN,
    CARBON_INTENSITY,
    EMBODIED_CARBON_FACTOR,
    ENERGY_PER_CACHED_TOKEN_WH,
    ENERGY_PER_INPUT_TOKEN_WH,
    ENERGY_PER_OUTPUT_TOKEN_WH,
    GRID_LOSS_FACTOR,
    PROVIDERS,
    PROVIDER_CACHE_SCHEMA_VERSION,
    PROVIDER_CACHE_FILENAME,
    PUE,
    GEMINI_TELEMETRY_LOG_PATH_ENV_VAR,
    WATER_USE_EFFICIENCY,
    _collect_provider_data_incremental,
    _collect_provider_data,
    _json_dumps_html_safe,
    _load_provider_cache,
    _collect_days_with_fallback,
    _normalize_gemini_attributes,
    _next_day_compact,
    _normalize_cli_date_arg,
    _normalize_timezone_name,
    _parse_date_flexible,
    _parse_gemini_timestamp,
    _resolve_gemini_log_path,
    _resolve_cache_path,
    _safe_int,
    _safe_float,
    _save_provider_cache,
    _write_json_file,
    calculate_carbon,
    calculate_energy,
    calculate_water,
    _write_html_file,
    _write_text_atomic,
    _run_cli_check,
    _empty_provider_cache,
    collect_claude_data,
    collect_codex_data,
    collect_gemini_data,
    compute_dashboard_data,
    detect_github_username,
    generate_html,
    main,
    merge_data,
    run_command,
)
from tokenprint.providers import PROVIDERS as MODULE_PROVIDERS, ProviderConfig, provider_by_key, provider_by_name, provider_name_set, provider_names, resolve_provider


@pytest.fixture(autouse=True)
def isolated_provider_cache(monkeypatch, tmp_path):
    """Keep provider cache isolated per test to avoid cross-test state leakage."""
    cache_file = tmp_path / PROVIDER_CACHE_FILENAME
    monkeypatch.setattr("tokenprint._provider_cache_path", lambda: cache_file)
    return cache_file


# --- _safe_int ---

class TestSafeInt:
    def test_true_is_rejected(self):
        assert _safe_int(True) == 0

    def test_false_is_rejected(self):
        assert _safe_int(False) == 0

    def test_normal_int(self):
        assert _safe_int(42) == 42

    def test_string_int(self):
        assert _safe_int("123") == 123

    def test_none(self):
        assert _safe_int(None) == 0

    def test_empty_string(self):
        assert _safe_int("") == 0

    def test_non_numeric(self):
        assert _safe_int("abc") == 0

    def test_float(self):
        assert _safe_int(3.9) == 3

    def test_infinite_is_rejected(self):
        assert _safe_int(float("inf")) == 0

    def test_negative_infinite_is_rejected(self):
        assert _safe_int(float("-inf")) == 0

    def test_nan_is_rejected(self):
        assert _safe_int(float("nan")) == 0


class TestSafeFloat:
    def test_true_is_rejected(self):
        assert _safe_float(True) == 0.0

    def test_false_is_rejected(self):
        assert _safe_float(False) == 0.0

    def test_normal_float(self):
        assert _safe_float(3.9) == 3.9

    def test_string_float(self):
        assert _safe_float("1.23") == 1.23

    def test_infinite_is_rejected(self):
        assert _safe_float(float("inf")) == 0.0

    def test_negative_infinite_is_rejected(self):
        assert _safe_float(float("-inf")) == 0.0

    def test_nan_is_rejected(self):
        assert _safe_float(float("nan")) == 0.0


# --- _parse_date_flexible ---

class TestParseDateFlexible:
    def test_compact_yyyymmdd_format(self):
        assert _parse_date_flexible("20260220") == "2026-02-20"

    def test_compact_yyyymmdd_with_whitespace(self):
        assert _parse_date_flexible(" 20260220 ") == "2026-02-20"

    def test_iso_format(self):
        assert _parse_date_flexible("2026-01-15") == "2026-01-15"

    def test_iso_with_time(self):
        assert _parse_date_flexible("2026-01-15T10:30:00") == "2026-01-15"

    def test_human_abbrev(self):
        assert _parse_date_flexible("Jan 15, 2026") == "2026-01-15"

    def test_human_full(self):
        assert _parse_date_flexible("January 15, 2026") == "2026-01-15"

    def test_human_no_comma(self):
        assert _parse_date_flexible("Jan 15 2026") == "2026-01-15"

    def test_invalid(self):
        assert _parse_date_flexible("not-a-date") is None

    def test_empty(self):
        assert _parse_date_flexible("") is None

    def test_invalid_iso_month(self):
        assert _parse_date_flexible("2026-13-01") is None

    def test_invalid_iso_day(self):
        assert _parse_date_flexible("2026-02-30") is None

    def test_compact_invalid(self):
        assert _parse_date_flexible("20261315") is None


class TestParseCliDateArg:
    def test_compact_date(self):
        assert _normalize_cli_date_arg("20260115") == "20260115"

    def test_iso_date(self):
        assert _normalize_cli_date_arg("2026-01-15") == "20260115"

    def test_iso_date_with_time(self):
        assert _normalize_cli_date_arg("2026-01-15T09:00:00") == "20260115"

    def test_whitespace(self):
        assert _normalize_cli_date_arg(" 2026-01-15 ") == "20260115"

    def test_invalid_date(self):
        assert _normalize_cli_date_arg("2026-13-01") is None

    def test_human_date_is_rejected(self):
        assert _normalize_cli_date_arg("Jan 15, 2026") is None

    def test_trailing_garbage_after_iso_date_is_rejected(self):
        assert _normalize_cli_date_arg("2026-01-15abc") is None


class TestNormalizeTimezoneName:
    def test_blank_timezone_defaults_to_utc(self):
        assert _normalize_timezone_name("") == "UTC"

    def test_whitespace_timezone_defaults_to_utc(self):
        assert _normalize_timezone_name("   ") == "UTC"

    def test_unknown_timezone_is_rejected(self):
        with pytest.raises(ValueError, match="unknown timezone"):
            _normalize_timezone_name("not-a-timezone")


class TestParseGeminiTimestamp:
    def test_rejects_boolean_timestamps(self):
        assert _parse_gemini_timestamp(True) is None
        assert _parse_gemini_timestamp(False) is None

    def test_rejects_non_numeric_string(self):
        assert _parse_gemini_timestamp("n/a") is None

    def test_supports_iso_timestamp(self):
        assert _parse_gemini_timestamp("2026-01-15T10:00:00Z") == "2026-01-15"

    def test_supports_iso_timestamp_with_offset(self):
        assert _parse_gemini_timestamp("2026-01-15T10:00:00+01:00") == "2026-01-15"

    def test_supports_scientific_notation(self):
        ts_seconds = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp())
        assert _parse_gemini_timestamp(f"{float(ts_seconds):.1e}") == "2026-01-15"

    def test_supports_large_ms_timestamps(self):
        ts_ms = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp()) * 1000 + 250
        assert _parse_gemini_timestamp(ts_ms) == "2026-01-15"

    def test_supports_timezone_conversion(self):
        # 2026-01-15T00:30:00Z -> 2026-01-14 in America/Los_Angeles (-08:00).
        assert _parse_gemini_timestamp("2026-01-15T00:30:00Z", "America/Los_Angeles") == "2026-01-14"

    def test_supports_timezone_conversion_for_numeric_timestamps(self):
        # 2026-01-15T01:00:00Z -> 2026-01-14 in America/New_York (-05:00) during winter.
        ts_seconds = int(datetime(2026, 1, 15, 1, 0, tzinfo=timezone.utc).timestamp())
        assert _parse_gemini_timestamp(ts_seconds, "America/New_York") == "2026-01-14"

    def test_rejects_non_finite_timestamps(self):
        assert _parse_gemini_timestamp(float("nan")) is None
        assert _parse_gemini_timestamp(float("inf")) is None


class TestGeminiRateConstants:
    def test_provider_rates_match_gemini_constants(self):
        gemini = next(p for p in PROVIDERS if p.name == "gemini")
        assert gemini.rates == (
            GEMINI_RATE_INPUT_PER_TOKEN,
            GEMINI_RATE_OUTPUT_PER_TOKEN,
            GEMINI_RATE_CACHED_PER_TOKEN,
        )


class TestClaudeRateConstants:
    def test_provider_rates_match_claude_constants(self):
        claude = next(p for p in PROVIDERS if p.name == "claude")
        assert claude.rates == (
            CLAUDE_RATE_INPUT_PER_TOKEN,
            CLAUDE_RATE_OUTPUT_PER_TOKEN,
            CLAUDE_RATE_CACHED_PER_TOKEN,
        )

    def test_model_prefix_reflects_legacy_claude_rates(self):
        sample = next(
            (rates for prefix, rates in CLAUDE_RATE_BY_MODEL_PREFIX if prefix.startswith("claude-sonnet")),
            None,
        )
        assert sample is not None
        input_rate, _ = sample
        assert input_rate == CLAUDE_RATE_INPUT_PER_TOKEN


class TestCodexRateConstants:
    def test_provider_rates_match_codex_constants(self):
        codex = next(p for p in PROVIDERS if p.name == "codex")
        assert codex.rates == (
            CODEX_RATE_INPUT_PER_TOKEN,
            CODEX_RATE_OUTPUT_PER_TOKEN,
            CODEX_RATE_CACHED_PER_TOKEN,
        )


class TestNormalizeGeminiAttributes:
    def test_normalizes_attribute_list(self):
        raw = [
            {"Key": "input_token_count", "Value": {"intValue": 10}},
            {"key": "output_token_count", "Value": {"stringValue": "20"}},
            {"Key": "invalid_key", "Value": {"stringValue": "x"}},
            {},
            "bad",
        ]
        assert _normalize_gemini_attributes(raw) == {
            "input_token_count": 10,
            "output_token_count": "20",
            "invalid_key": "x",
        }

    def test_normalizes_attribute_dict_with_string_keys_only(self):
        raw = {
            "input_token_count": "100",
            2: "ignored",
            "output_token_count": 40,
        }
        assert _normalize_gemini_attributes(raw) == {
            "input_token_count": "100",
            "output_token_count": 40,
        }

    def test_normalizes_attribute_dict_with_nested_gemini_values(self):
        raw = {
            "input_token_count": {"intValue": 10},
            "output_token_count": {"stringValue": "20"},
            "cached_content_token_count": {"Int64Value": 5},
            "missing": {"foo": "bar"},
            1: "ignored",
            "invalid": None,
        }
        assert _normalize_gemini_attributes(raw) == {
            "input_token_count": 10,
            "output_token_count": "20",
            "cached_content_token_count": 5,
            "missing": 0,
            "invalid": None,
        }

    def test_returns_none_for_unknown_shape(self):
        assert _normalize_gemini_attributes(123) is None


class TestResolveCachePath:
    def test_returns_none_when_not_set(self, monkeypatch):
        monkeypatch.delenv("TOKENPRINT_CACHE_PATH", raising=False)
        assert _resolve_cache_path("") is None

    def test_resolves_tilde_expanded_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        expected = tmp_path / "my-cache.json"
        assert _resolve_cache_path("~/my-cache.json") == expected

    def test_trims_whitespace(self, tmp_path, monkeypatch):
        path = tmp_path / "my-cache.json"
        monkeypatch.setenv("TOKENPRINT_CACHE_PATH", f" {path} ")
        assert _resolve_cache_path("") == path

    def test_resolves_dir_to_default_cache_file(self, tmp_path):
        cache_dir = tmp_path / "cache-dir"
        cache_dir.mkdir()
        assert _resolve_cache_path(str(cache_dir)) == cache_dir / PROVIDER_CACHE_FILENAME


class TestResolveGeminiLogPath:
    def test_default_uses_home_dir(self, monkeypatch):
        monkeypatch.setenv("HOME", "/tmp/tokenprint-home")
        assert _resolve_gemini_log_path() == Path("/tmp/tokenprint-home/.gemini/telemetry.log")

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        override = tmp_path / "custom" / "telemetry.log"
        monkeypatch.setenv(GEMINI_TELEMETRY_LOG_PATH_ENV_VAR, str(override))
        assert _resolve_gemini_log_path() == override

    def test_env_var_with_tilde_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv(GEMINI_TELEMETRY_LOG_PATH_ENV_VAR, "~/gemini-custom/telemetry.log")
        assert _resolve_gemini_log_path() == tmp_path / "gemini-custom" / "telemetry.log"

    def test_env_var_directory_is_file_appended(self, tmp_path, monkeypatch):
        target_dir = tmp_path / "telemetry-dir"
        target_dir.mkdir()
        monkeypatch.setenv(GEMINI_TELEMETRY_LOG_PATH_ENV_VAR, str(target_dir))
        assert _resolve_gemini_log_path() == target_dir / "telemetry.log"

    def test_env_var_trims_whitespace(self, tmp_path, monkeypatch):
        target_file = tmp_path / "telemetry.log"
        monkeypatch.setenv(GEMINI_TELEMETRY_LOG_PATH_ENV_VAR, f" {target_file} ")
        assert _resolve_gemini_log_path() == target_file


# --- incremental cache helpers ---

class TestProviderCacheHelpers:
    def test_next_day_compact(self):
        assert _next_day_compact("2026-02-19") == "20260220"

    def test_next_day_compact_invalid(self):
        assert _next_day_compact("not-a-date") is None

    def test_cache_round_trip(self, isolated_provider_cache):
        provider_data = {
            "claude": {
                "2026-02-19": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 10,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }
        _save_provider_cache(provider_data, cache_path=isolated_provider_cache)
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded["claude"]["2026-02-19"]["input_tokens"] == 100
        assert loaded["claude"]["2026-02-19"]["provider"] == "claude"
        assert loaded["codex"] == {}

    def test_empty_provider_cache_matches_registry(self):
        assert list(_empty_provider_cache().keys()) == list(provider_names())

    def test_legacy_cache_with_compact_dates(self, isolated_provider_cache):
        payload = {
            "version": PROVIDER_CACHE_SCHEMA_VERSION,
            "providers": {
                "claude": {
                    "20260220": {
                        "provider": "claude",
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_read_tokens": 10,
                        "cache_write_tokens": 0,
                        "cost": 0.01,
                    }
                },
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded["claude"]["2026-02-20"]["input_tokens"] == 100

    def test_legacy_cache_without_schema_wrapper(self, isolated_provider_cache):
        payload = {
            "claude": {
                "2026-02-21": {
                    "provider": "claude",
                    "input_tokens": 200,
                    "output_tokens": 25,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.02,
                }
            },
            "codex": {},
            "gemini": {},
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded["claude"]["2026-02-21"]["input_tokens"] == 200

    def test_legacy_cache_without_schema_wrapper_accepts_string_version(self, isolated_provider_cache):
        payload = {
            "version": str(PROVIDER_CACHE_SCHEMA_VERSION),
            "claude": {
                "2026-02-21": {
                    "provider": "claude",
                    "input_tokens": 200,
                    "output_tokens": 25,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.02,
                }
            },
            "codex": {},
            "gemini": {},
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded["claude"]["2026-02-21"]["input_tokens"] == 200

    def test_legacy_cache_without_schema_wrapper_accepts_float_version(self, isolated_provider_cache):
        payload = {
            "version": float(PROVIDER_CACHE_SCHEMA_VERSION),
            "claude": {
                "2026-02-22": {
                    "provider": "claude",
                    "input_tokens": 300,
                    "output_tokens": 35,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.03,
                }
            },
            "codex": {},
            "gemini": {},
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded["claude"]["2026-02-22"]["output_tokens"] == 35

    def test_unsupported_cache_version_is_ignored(self, isolated_provider_cache):
        payload = {
            "version": PROVIDER_CACHE_SCHEMA_VERSION + 1,
            "providers": {
                "claude": {
                    "2026-02-22": {
                        "provider": "claude",
                        "input_tokens": 100,
                    }
                }
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_cache_version_as_string_is_accepted(self, isolated_provider_cache):
        payload = {
            "version": str(PROVIDER_CACHE_SCHEMA_VERSION),
            "providers": {
                "claude": {},
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_cache_version_string_float_is_accepted(self, isolated_provider_cache):
        payload = {
            "version": f"{float(PROVIDER_CACHE_SCHEMA_VERSION):.1f}",
            "providers": {
                "claude": {},
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_cache_version_float_is_accepted(self, isolated_provider_cache):
        payload = {
            "version": float(PROVIDER_CACHE_SCHEMA_VERSION),
            "providers": {
                "claude": {},
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_cache_version_nan_is_rejected(self, isolated_provider_cache):
        payload = {
            "version": float("nan"),
            "providers": {
                "claude": {},
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_cache_version_scientific_notation_is_accepted(self, isolated_provider_cache):
        payload = {
            "version": f"{float(PROVIDER_CACHE_SCHEMA_VERSION):.2e}",
            "providers": {
                "claude": {},
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_cache_version_null_is_rejected(self, isolated_provider_cache):
        payload = {
            "version": None,
            "providers": {
                "claude": {},
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_cache_version_bool_is_rejected(self, isolated_provider_cache):
        payload = {
            "version": True,
            "providers": {
                "claude": {},
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_cache_version_non_integer_float_is_rejected(self, isolated_provider_cache):
        payload = {
            "version": 2.5,
            "providers": {
                "claude": {
                    "2026-02-22": {
                        "provider": "claude",
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cost": 1.0,
                    }
                },
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded == {provider: {} for provider in [p.name for p in PROVIDERS]}

    def test_non_dict_provider_payload_is_ignored(self, isolated_provider_cache):
        payload = {
            "version": PROVIDER_CACHE_SCHEMA_VERSION,
            "providers": {
                "claude": ["not", "a", "mapping"],
                "codex": {},
                "gemini": {},
            }
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded["claude"] == {}
        assert loaded["codex"] == {}
        assert loaded["gemini"] == {}

    def test_bool_cost_is_normalized_to_zero(self, isolated_provider_cache):
        payload = {
            "version": PROVIDER_CACHE_SCHEMA_VERSION,
            "providers": {
                "claude": {
                    "2026-02-19": {
                        "provider": "claude",
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_read_tokens": 10,
                        "cache_write_tokens": 0,
                        "cost": True,
                    }
                },
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded["claude"]["2026-02-19"]["cost"] == 0.0

    def test_nonfinite_cost_is_normalized_to_zero(self, isolated_provider_cache):
        payload = {
            "version": PROVIDER_CACHE_SCHEMA_VERSION,
            "providers": {
                "claude": {
                    "2026-02-20": {
                        "provider": "claude",
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cache_read_tokens": 10,
                        "cache_write_tokens": 0,
                        "cost": float("inf"),
                    }
                },
                "codex": {},
                "gemini": {},
            },
        }
        isolated_provider_cache.write_text(json.dumps(payload))
        loaded = _load_provider_cache(cache_path=isolated_provider_cache)
        assert loaded["claude"]["2026-02-20"]["cost"] == 0.0

    def test_save_provider_cache_cleans_temp_file_on_replace_failure(
        self, isolated_provider_cache, monkeypatch
    ):
        provider_data = {
            "claude": {
                "2026-02-20": {
                    "provider": "claude",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_tokens": 2,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        def fail_replace(_src: os.PathLike[str], _dst: os.PathLike[str]) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr(tp.os, "replace", fail_replace)
        _save_provider_cache(provider_data, cache_path=isolated_provider_cache)
        temp_prefix = f".{isolated_provider_cache.name}."
        assert not any(p.name.startswith(temp_prefix) for p in isolated_provider_cache.parent.iterdir())
        assert not isolated_provider_cache.exists()

    def test_save_provider_cache_reports_write_failures(self, isolated_provider_cache, monkeypatch, capsys):
        provider_data = {
            "claude": {
                "2026-02-20": {
                    "provider": "claude",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_tokens": 2,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        def fail_write(*_args, **_kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(tp, "_write_text_atomic", fail_write)
        _save_provider_cache(provider_data, cache_path=isolated_provider_cache)
        _, err = capsys.readouterr()
        assert "Could not write provider cache at" in err
        assert str(isolated_provider_cache) in err

    def test_save_provider_cache_creates_parent_path(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "nested" / "path" / "provider-cache.json"
        provider_data = {
            "claude": {
                "2026-02-23": {
                    "provider": "claude",
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        monkeypatch.setattr("tokenprint._provider_cache_path", lambda: cache_path)
        _save_provider_cache(provider_data)
        assert cache_path.exists()
        loaded = _load_provider_cache(cache_path=cache_path)
        assert loaded["claude"]["2026-02-23"]["cost"] == 0.01


# --- _json_dumps_html_safe ---

class TestJsonDumpsHtmlSafe:
    def test_escapes_angle_brackets(self):
        result = _json_dumps_html_safe({"name": "<script>alert(1)</script>"})
        assert "\\u003c" in result
        assert "\\u003e" in result
        assert "<script>" not in result

    def test_escapes_ampersand(self):
        result = _json_dumps_html_safe({"q": "a&b"})
        assert "\\u0026" in result
        assert "&" not in result

    def test_normal_data(self):
        result = _json_dumps_html_safe({"key": "value", "num": 42})
        parsed = json.loads(result)
        assert parsed == {"key": "value", "num": 42}


# --- calculate_energy ---

class TestCalculateEnergy:
    def test_zero(self):
        assert calculate_energy(0, 0, 0) == 0

    def test_output_only(self):
        expected = 1000 * ENERGY_PER_OUTPUT_TOKEN_WH * PUE * GRID_LOSS_FACTOR
        assert calculate_energy(0, 1000, 0) == pytest.approx(expected)

    def test_input_only(self):
        expected = 1000 * ENERGY_PER_INPUT_TOKEN_WH * PUE * GRID_LOSS_FACTOR
        assert calculate_energy(1000, 0, 0) == pytest.approx(expected)

    def test_cached_only(self):
        expected = 1000 * ENERGY_PER_CACHED_TOKEN_WH * PUE * GRID_LOSS_FACTOR
        assert calculate_energy(0, 0, 1000) == pytest.approx(expected)

    def test_all_types(self):
        base = (
            500 * ENERGY_PER_OUTPUT_TOKEN_WH
            + 1000 * ENERGY_PER_INPUT_TOKEN_WH
            + 2000 * ENERGY_PER_CACHED_TOKEN_WH
        )
        expected = base * PUE * GRID_LOSS_FACTOR
        assert calculate_energy(1000, 500, 2000) == pytest.approx(expected)


# --- calculate_carbon ---

class TestCalculateCarbon:
    def test_zero(self):
        assert calculate_carbon(0) == 0

    def test_one_kwh(self):
        expected = CARBON_INTENSITY * EMBODIED_CARBON_FACTOR
        assert calculate_carbon(1000) == pytest.approx(expected)

    def test_proportional(self):
        assert calculate_carbon(2000) == pytest.approx(2 * calculate_carbon(1000))


# --- calculate_water ---

class TestCalculateWater:
    def test_zero(self):
        assert calculate_water(0) == 0

    def test_one_kwh(self):
        expected = WATER_USE_EFFICIENCY * 1000  # L/kWh * 1000 mL/L
        assert calculate_water(1000) == pytest.approx(expected)


# --- run_command ---

class TestRunCommand:
    def test_successful_command(self):
        result = run_command(["echo", "hello"])
        assert result == "hello"

    def test_failing_command(self):
        result = run_command(["false"])
        assert result is None

    def test_nonexistent_binary(self):
        result = run_command(["__nonexistent_binary_xyz__"])
        assert result is None

    @patch("tokenprint.subprocess.run", side_effect=OSError("permission denied"))
    def test_os_error_is_treated_as_unavailable(self, mock_run):
        result = run_command(["tokenprint"])
        assert result is None

    def test_timeout(self):
        result = run_command(["sleep", "10"], timeout=1)
        assert result is None

    def test_strips_output(self):
        result = run_command(["echo", "  spaces  "])
        assert result == "spaces"


class TestRunCliCheck:
    def _set_tokenprint_file(self, monkeypatch, tmp_path, include_template: bool = True):
        fake_module = tmp_path / "tokenprint.py"
        fake_module.write_text("")
        if include_template:
            (tmp_path / "template.html").write_text("<!doctype html></html>")
        monkeypatch.setattr(tp, "__file__", str(fake_module))
        return fake_module

    def test_check_fails_without_template(self, monkeypatch, tmp_path):
        self._set_tokenprint_file(monkeypatch, tmp_path, include_template=False)
        with patch.object(tp, "_command_exists", return_value=True):
            assert _run_cli_check() is False

    def test_check_passes_with_missing_codex_as_optional(self, monkeypatch, tmp_path):
        self._set_tokenprint_file(monkeypatch, tmp_path, include_template=True)

        def command_exists(name: str) -> bool:
            return name == "ccusage"

        with patch.object(tp, "_command_exists", side_effect=command_exists):
            assert _run_cli_check() is True

    def test_check_requires_ccusage(self, monkeypatch, tmp_path):
        self._set_tokenprint_file(monkeypatch, tmp_path, include_template=True)

        def command_exists(name: str) -> bool:
            if name == "ccusage":
                return False
            return True

        with patch.object(tp, "_command_exists", side_effect=command_exists):
            assert _run_cli_check() is False


# --- detect_github_username ---

class TestDetectGithubUsername:
    @patch("tokenprint.run_command", side_effect=[None, None])
    def test_fallback_to_dev(self, mock_run):
        # When both gh and git fail, and no TTY for input
        with patch("builtins.input", side_effect=EOFError):
            assert detect_github_username() == "dev"

    @patch("tokenprint.run_command", side_effect=["ghuser", None])
    def test_gh_cli(self, mock_run):
        assert detect_github_username() == "ghuser"

    @patch("tokenprint.run_command", side_effect=[None, "gituser"])
    def test_git_config_fallback(self, mock_run):
        assert detect_github_username() == "gituser"

    @patch("tokenprint.run_command", side_effect=[None, None])
    def test_interactive_input(self, mock_run):
        with patch("builtins.input", return_value="manual"):
            assert detect_github_username() == "manual"

    @patch("tokenprint.run_command", side_effect=[None, None])
    def test_keyboard_interrupt(self, mock_run):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert detect_github_username() == "dev"


# --- merge_data ---

class TestMergeData:
    def test_empty(self):
        assert merge_data({"claude": {}, "codex": {}, "gemini": {}}) == []

    def test_single_provider(self):
        claude = {
            "2026-01-15": {
                "provider": "claude",
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_tokens": 30,
                "cache_write_tokens": 0,
                "cost": 0.05,
            }
        }
        result = merge_data({"claude": claude, "codex": {}, "gemini": {}})
        assert len(result) == 1
        row = result[0]
        assert row["date"] == "2026-01-15"
        assert row["claude"]["input_tokens"] == 100
        assert row["claude"]["output_tokens"] == 50
        assert row["claude"]["energy_wh"] > 0
        assert row["claude"]["carbon_g"] > 0
        assert row["codex"]["input_tokens"] == 0
        assert row["gemini"]["input_tokens"] == 0

    def test_multiple_providers_same_date(self):
        claude = {"2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.05}}
        codex = {"2026-01-15": {"provider": "codex", "input_tokens": 200, "output_tokens": 100, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.03}}
        result = merge_data({"claude": claude, "codex": codex, "gemini": {}})
        assert len(result) == 1
        assert result[0]["claude"]["input_tokens"] == 100
        assert result[0]["codex"]["input_tokens"] == 200

    def test_merge_data_uses_provider_name_order(self):
        claude = {"2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}}
        result = merge_data({"gemini": {}, "claude": claude, "codex": {}})
        row_keys = list(result[0].keys())
        assert row_keys == ["date", *provider_names()]

    def test_sorted_dates(self):
        claude = {"2026-01-20": {"provider": "claude", "input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}}
        codex = {"2026-01-10": {"provider": "codex", "input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}}
        result = merge_data({"claude": claude, "codex": codex, "gemini": {}})
        assert result[0]["date"] == "2026-01-10"
        assert result[1]["date"] == "2026-01-20"

    def test_energy_carbon_water_values(self):
        """Verify computed energy/carbon/water values match expectations."""
        claude = {"2026-01-15": {"provider": "claude", "input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 200, "cache_write_tokens": 0, "cost": 0.10}}
        result = merge_data({"claude": claude, "codex": {}, "gemini": {}})
        row = result[0]["claude"]
        expected_energy = calculate_energy(1000, 500, 200)
        assert row["energy_wh"] == pytest.approx(round(expected_energy, 4))
        assert row["carbon_g"] == pytest.approx(round(calculate_carbon(expected_energy), 4))
        assert row["water_ml"] == pytest.approx(round(calculate_water(expected_energy), 4))

    def test_cache_write_tokens_contribute_to_energy(self):
        """Energy should include both cache_read_tokens and cache_write_tokens."""
        claude = {"2026-01-15": {
            "provider": "claude",
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_tokens": 200,
            "cache_write_tokens": 300,
            "cost": 0.10,
        }}
        result = merge_data({"claude": claude, "codex": {}, "gemini": {}})
        row = result[0]["claude"]
        expected_energy = calculate_energy(1000, 500, 200 + 300)
        assert row["energy_wh"] == pytest.approx(round(expected_energy, 4))

    def test_handles_corrupted_provider_row(self):
        data = {
            "claude": {"2026-01-15": "bad-row"},
            "codex": {"2026-01-15": {"provider": "codex", "input_tokens": 50, "output_tokens": 25, "cache_read_tokens": 10, "cache_write_tokens": 0, "cost": 0.02}},
            "gemini": {},
        }
        result = merge_data(data)
        assert result[0]["claude"]["input_tokens"] == 0
        assert result[0]["codex"]["input_tokens"] == 50

    def test_handles_nonnumeric_fields(self):
        data = {
            "claude": {"2026-01-15": {"provider": "claude", "input_tokens": "abc", "output_tokens": None, "cache_read_tokens": "7", "cache_write_tokens": "x", "cost": "bad"}},
            "codex": {},
            "gemini": {},
        }
        result = merge_data(data)
        row = result[0]["claude"]
        assert row["input_tokens"] == 0
        assert row["output_tokens"] == 0
        assert row["cache_read_tokens"] == 7
        assert row["cost"] == 0

    def test_handles_nonfinite_cost_values(self):
        data = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 10,
                    "cache_write_tokens": 0,
                    "cost": float("inf"),
                }
            },
            "codex": {},
            "gemini": {},
        }
        result = merge_data(data)
        assert result[0]["claude"]["cost"] == 0

    def test_handles_boolean_numeric_fields(self):
        data = {
            "claude": {"2026-01-15": {"provider": "claude", "input_tokens": True, "output_tokens": False, "cache_read_tokens": True, "cache_write_tokens": False, "cost": True}},
            "codex": {},
            "gemini": {},
        }
        result = merge_data(data)
        row = result[0]["claude"]
        assert row["input_tokens"] == 0
        assert row["output_tokens"] == 0
        assert row["cache_read_tokens"] == 0
        assert row["cache_write_tokens"] == 0
        assert row["cost"] == 0


# --- compute_dashboard_data ---

class TestComputeDashboardData:
    def _make_data(self):
        return merge_data({
            "claude": {"2026-01-15": {"provider": "claude", "input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 200, "cache_write_tokens": 0, "cost": 0.10}},
            "codex": {"2026-01-16": {"provider": "codex", "input_tokens": 800, "output_tokens": 400, "cache_read_tokens": 100, "cache_write_tokens": 0, "cost": 0.05}},
            "gemini": {},
        })

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_returns_required_keys(self, mock_user):
        data = self._make_data()
        config = compute_dashboard_data(data)
        assert "rawData" in config
        assert "githubUser" in config
        assert "providerHasData" in config
        assert "providers" in config
        assert "minDate" in config
        assert "maxDate" in config
        assert "electricityCostKwh" in config
        assert "generatedAt" in config
        assert "liveMode" in config
        assert "refreshEndpoint" in config
        assert "energyModel" in config

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_energy_model_keys(self, mock_user):
        config = compute_dashboard_data(self._make_data())
        em = config["energyModel"]
        assert em["outputWhPerToken"] == ENERGY_PER_OUTPUT_TOKEN_WH
        assert em["inputWhPerToken"] == ENERGY_PER_INPUT_TOKEN_WH
        assert em["cachedWhPerToken"] == ENERGY_PER_CACHED_TOKEN_WH
        assert em["pue"] == PUE
        assert em["gridLossFactor"] == GRID_LOSS_FACTOR
        assert em["carbonIntensity"] == CARBON_INTENSITY
        assert em["embodiedCarbonFactor"] == EMBODIED_CARBON_FACTOR
        assert em["waterUseEfficiency"] == WATER_USE_EFFICIENCY

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_github_user(self, mock_user):
        config = compute_dashboard_data(self._make_data())
        assert config["githubUser"] == "testuser"

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_date_range(self, mock_user):
        config = compute_dashboard_data(self._make_data())
        assert config["minDate"] == "2026-01-15"
        assert config["maxDate"] == "2026-01-16"

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_provider_has_data(self, mock_user):
        config = compute_dashboard_data(self._make_data())
        assert config["providerHasData"]["claude"] is True
        assert config["providerHasData"]["codex"] is True
        assert config["providerHasData"]["gemini"] is False

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_raw_data_structure(self, mock_user):
        config = compute_dashboard_data(self._make_data())
        raw = config["rawData"]
        assert len(raw) == 2
        assert "d" in raw[0]
        # Verify keys come from registry
        for p in PROVIDERS:
            assert p.key in raw[0]
        assert len(raw[0][PROVIDERS[0].key]) == 4  # [input, output, cached, cost]

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_empty_data(self, mock_user):
        config = compute_dashboard_data([])
        assert config["rawData"] == []
        assert config["minDate"] == ""
        assert config["maxDate"] == ""

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_live_mode_fields(self, mock_user):
        config = compute_dashboard_data(self._make_data(), live_mode=True, refresh_endpoint="/api/refresh")
        assert config["liveMode"] is True
        assert config["refreshEndpoint"] == "/api/refresh"

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_raw_data_field_ordering(self, mock_user):
        """Verify raw data arrays are [input, output, cached, cost] in correct order."""
        data = merge_data({
            "claude": {"2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 200, "cache_read_tokens": 300, "cache_write_tokens": 0, "cost": 0.50}},
            "codex": {}, "gemini": {},
        })
        config = compute_dashboard_data(data)
        c = config["rawData"][0]["c"]
        assert c[0] == 100   # input
        assert c[1] == 200   # output
        assert c[2] == 300   # cached
        assert c[3] == 0.50  # cost


# --- collect_claude_data ---

class TestCollectClaudeData:
    @patch("tokenprint.run_command", return_value=None)
    def test_no_ccusage(self, mock_run):
        assert collect_claude_data() == {}

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": 1000, "outputTokens": 500, "cacheReadTokens": 200, "cacheCreationTokens": 50, "totalCost": 0.10}]}')
    def test_valid_data(self, mock_run):
        result = collect_claude_data()
        assert "2026-01-15" in result
        d = result["2026-01-15"]
        assert d["input_tokens"] == 1000
        assert d["output_tokens"] == 500
        assert d["cache_read_tokens"] == 200
        assert d["cost"] == 0.10

    @patch(
        "tokenprint.run_command",
        return_value='{"daily": ["bad-entry", 123, {"date": "2026-01-15", "inputTokens": 5, "outputTokens": 1, "cacheReadTokens": 2, "cacheCreationTokens": 0, "totalCost": 0.001}]}',
    )
    def test_skip_non_dict_entries(self, mock_run):
        result = collect_claude_data()
        assert result == {
            "2026-01-15": {
                "provider": "claude",
                "input_tokens": 5,
                "output_tokens": 1,
                "cache_read_tokens": 2,
                "cache_write_tokens": 0,
                "cost": 0.001,
            }
        }

    @patch("tokenprint.run_command", return_value="not json")
    def test_invalid_json(self, mock_run):
        assert collect_claude_data() == {}

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": 100, "outputTokens": 50, "cacheReadTokens": 10, "cacheCreationTokens": 0, "totalCost": 0.01}, {"date": "2026-01-15", "inputTokens": 200, "outputTokens": 100, "cacheReadTokens": 20, "cacheCreationTokens": 0, "totalCost": 0.02}]}')
    def test_same_date_aggregation(self, mock_run):
        result = collect_claude_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 300
        assert d["output_tokens"] == 150
        assert d["cache_read_tokens"] == 30
        assert d["cost"] == pytest.approx(0.03)

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": "bad", "outputTokens": "75", "cacheReadTokens": "3", "cacheCreationTokens": "bad", "totalCost": "bad"}]}')
    def test_malformed_numeric_fields_are_safely_coerced(self, mock_run):
        result = collect_claude_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 0
        assert d["output_tokens"] == 75
        assert d["cache_read_tokens"] == 3
        assert d["cache_write_tokens"] == 0
        assert d["cost"] == 0

    @patch(
        "tokenprint.run_command",
        return_value=json.dumps(
            {
                "daily": [
                    {
                        "date": "2026-01-15",
                        "inputTokens": float("inf"),
                        "outputTokens": float("-inf"),
                        "cacheReadTokens": float("nan"),
                        "cacheCreationTokens": 0,
                        "totalCost": float("inf"),
                    }
                ]
            }
        ),
    )
    def test_nonfinite_numeric_fields_are_safely_coerced(self, mock_run):
        result = collect_claude_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 0
        assert d["output_tokens"] == 0
        assert d["cache_read_tokens"] == 0
        assert d["cache_write_tokens"] == 0
        assert d["cost"] == 0

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": true, "outputTokens": false, "cacheReadTokens": true, "cacheCreationTokens": false, "totalCost": true}]}')
    def test_boolean_numeric_fields_are_rejected(self, mock_run):
        result = collect_claude_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 0
        assert d["output_tokens"] == 0
        assert d["cache_read_tokens"] == 0
        assert d["cache_write_tokens"] == 0
        assert d["cost"] == 0

    @patch("tokenprint.run_command", return_value='{"daily": {"not": "a-list"}}')
    def test_unexpected_json_shape_returns_empty(self, mock_run):
        assert collect_claude_data() == {}

    @patch(
        "tokenprint.run_command",
        return_value='{"daily": [{"date": "Jan 15, 2026", "inputTokens": 10, "outputTokens": 5, "cacheReadTokens": 1, "cacheCreationTokens": 0, "totalCost": 0.001}]}',
    )
    def test_human_date_format(self, mock_run):
        result = collect_claude_data()
        assert "2026-01-15" in result

    @patch(
        "tokenprint.run_command",
        return_value='{"daily": [{"date": "20260115", "inputTokens": 123, "outputTokens": 45, "cacheReadTokens": 10, "cacheCreationTokens": 0, "totalCost": 0.12}]}',
    )
    def test_compact_date_format_supported(self, mock_run):
        result = collect_claude_data()
        assert "2026-01-15" in result

    @patch(
        "tokenprint.run_command",
        return_value=json.dumps(
            {
                "daily": [
                    {
                        "date": "2026-02-20",
                        "inputTokens": 6300,
                        "outputTokens": 56000,
                        "cacheCreationTokens": 5400000,
                        "cacheReadTokens": 140000000,
                        "totalCost": 0.75756585,
                        "modelBreakdowns": [
                            {
                                "modelName": "claude-opus-4-6",
                                "inputTokens": 5961,
                                "outputTokens": 56165,
                                "cacheCreationTokens": 5450259,
                                "cacheReadTokens": 140124110,
                                "cost": 0,
                            },
                            {
                                "modelName": "claude-haiku-4-5-20251001",
                                "inputTokens": 339,
                                "outputTokens": 210,
                                "cacheCreationTokens": 271871,
                                "cacheReadTokens": 4163381,
                                "cost": 0.75756585,
                            },
                        ],
                    }
                ]
            }
        ),
    )
    def test_estimates_unpriced_claude_models_from_breakdown(self, mock_run):
        result = collect_claude_data()
        d = result["2026-02-20"]
        opus_rates = next(
            rates for prefix, rates in CLAUDE_RATE_BY_MODEL_PREFIX if prefix == "claude-opus-4-6"
        )
        opus_input, opus_output = opus_rates
        cache_write = opus_input * CLAUDE_RATE_CACHE_WRITE_MULTIPLIER
        cache_read = opus_input * CLAUDE_RATE_CACHE_READ_MULTIPLIER
        opus_est = (
            5961 * opus_input
            + 56165 * opus_output
            + 5450259 * cache_write
            + 140124110 * cache_read
        )
        assert d["cost"] == pytest.approx(0.75756585 + opus_est, rel=1e-9)

    @patch(
        "tokenprint.run_command",
        return_value=json.dumps(
            {
                "daily": [
                    {
                        "date": "2026-02-20",
                        "inputTokens": 1000,
                        "outputTokens": 500,
                        "cacheCreationTokens": 10000,
                        "cacheReadTokens": 20000,
                        "totalCost": 10.0,  # already above known breakdown sum
                        "modelBreakdowns": [
                            {
                                "modelName": "claude-opus-4-6",
                                "inputTokens": 1000,
                                "outputTokens": 500,
                                "cacheCreationTokens": 10000,
                                "cacheReadTokens": 20000,
                                "cost": 0,
                            },
                            {
                                "modelName": "claude-haiku-4-5-20251001",
                                "inputTokens": 1,
                                "outputTokens": 1,
                                "cacheCreationTokens": 1,
                                "cacheReadTokens": 1,
                                "cost": 1.0,
                            },
                        ],
                    }
                ]
            }
        ),
    )
    def test_does_not_double_count_when_total_cost_already_includes_unknowns(self, mock_run):
        result = collect_claude_data()
        assert result["2026-02-20"]["cost"] == pytest.approx(10.0)


# --- collect_codex_data ---

class TestCollectCodexData:
    @patch("tokenprint.run_command", return_value=None)
    def test_no_codex(self, mock_run):
        assert collect_codex_data() == {}

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": 1000, "outputTokens": 500, "cachedInputTokens": 300, "costUSD": 0.05}]}')
    def test_cached_subtraction(self, mock_run):
        result = collect_codex_data()
        d = result["2026-01-15"]
        # inputTokens (1000) includes cached (300), so non-cached = 700
        assert d["input_tokens"] == 700
        assert d["cache_read_tokens"] == 300

    @patch(
        "tokenprint.run_command",
        return_value='{"daily": ["bad-entry", 999, {"date": "2026-01-15", "inputTokens": 100, "outputTokens": 50, "cachedInputTokens": 10, "costUSD": 0.01}]}',
    )
    def test_skip_non_dict_entries(self, mock_run):
        result = collect_codex_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 90
        assert d["output_tokens"] == 50
        assert d["cache_read_tokens"] == 10

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "Jan 15, 2026", "inputTokens": 500, "outputTokens": 200, "cachedInputTokens": 0, "costUSD": 0}]}')
    def test_human_date_format(self, mock_run):
        result = collect_codex_data()
        assert "2026-01-15" in result

    @patch(
        "tokenprint.run_command",
        return_value='{"daily": [{"date": "20260115", "inputTokens": 300, "outputTokens": 100, "cachedInputTokens": 20, "costUSD": 0.01}]}',
    )
    def test_compact_date_format_supported(self, mock_run):
        result = collect_codex_data()
        assert "2026-01-15" in result

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": "12", "outputTokens": "bad", "cachedInputTokens": "3", "costUSD": "bad"}]}')
    def test_malformed_numeric_fields_are_safely_coerced(self, mock_run):
        result = collect_codex_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 9
        assert d["output_tokens"] == 0
        assert d["cache_read_tokens"] == 3
        assert d["cost"] == pytest.approx(
            9 * CODEX_RATE_INPUT_PER_TOKEN
            + 0 * CODEX_RATE_OUTPUT_PER_TOKEN
            + 3 * CODEX_RATE_CACHED_PER_TOKEN
        )

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": true, "outputTokens": false, "cachedInputTokens": true, "costUSD": false}]}')
    def test_boolean_numeric_fields_are_rejected(self, mock_run):
        result = collect_codex_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 0
        assert d["output_tokens"] == 0
        assert d["cache_read_tokens"] == 0
        assert d["cost"] == 0

    @patch(
        "tokenprint.run_command",
        return_value=json.dumps(
            {
                "daily": [
                    {
                        "date": "2026-01-15",
                        "inputTokens": float("inf"),
                        "outputTokens": float("-inf"),
                        "cachedInputTokens": float("nan"),
                        "costUSD": float("inf"),
                    }
                ]
            }
        ),
    )
    def test_nonfinite_numeric_fields_are_safely_coerced(self, mock_run):
        result = collect_codex_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 0
        assert d["output_tokens"] == 0
        assert d["cache_read_tokens"] == 0
        assert d["cost"] == 0

    @patch("tokenprint.run_command", return_value='{"daily": {"not": "a-list"}}')
    def test_unexpected_json_shape_returns_empty(self, mock_run):
        assert collect_codex_data() == {}

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": 1000, "outputTokens": 500, "cachedInputTokens": 0, "costUSD": 0}]}')
    def test_cost_estimation_when_zero(self, mock_run):
        result = collect_codex_data()
        d = result["2026-01-15"]
        # When costUSD is 0 but tokens exist, cost should be estimated
        assert d["cost"] > 0

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": 100, "outputTokens": 50, "cachedInputTokens": 200, "costUSD": 0.01}]}')
    def test_negative_cached_subtraction_clamped(self, mock_run):
        """When cachedInputTokens > inputTokens, input_tokens should be clamped to 0."""
        result = collect_codex_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 0
        assert d["cache_read_tokens"] == 200

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "2026-01-15", "inputTokens": 100, "outputTokens": 50, "cachedInputTokens": 10, "costUSD": 0.01}, {"date": "2026-01-15", "inputTokens": 200, "outputTokens": 100, "cachedInputTokens": 20, "costUSD": 0.02}]}')
    def test_same_date_aggregation(self, mock_run):
        result = collect_codex_data()
        d = result["2026-01-15"]
        assert d["input_tokens"] == 90 + 180  # (100-10) + (200-20)
        assert d["output_tokens"] == 150
        assert d["cache_read_tokens"] == 30


# --- collect_gemini_data ---

class TestCollectGeminiData:
    @patch("tokenprint._resolve_gemini_log_path")
    def test_uses_configured_telemetry_path(self, mock_resolve_path, tmp_path):
        # ensure custom path helper is consulted by collection.
        custom_path = tmp_path / "custom.telemetry.log"
        custom_path.write_text("")
        mock_resolve_path.return_value = custom_path
        result = collect_gemini_data()
        assert result == {}
        mock_resolve_path.assert_called_once_with()

    @patch("tokenprint._resolve_gemini_log_path")
    def test_uses_explicit_telemetry_path(self, mock_resolve_path, tmp_path):
        custom_path = tmp_path / "explicit.telemetry.log"
        mock_resolve_path.return_value = custom_path
        result = collect_gemini_data(gemini_log_path=str(custom_path))
        assert result == {}
        mock_resolve_path.assert_called_once_with(str(custom_path))

    @patch("tokenprint.Path")
    def test_no_log_file(self, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = False
        assert collect_gemini_data() == {}

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_valid_log(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        log_line = json.dumps({
            "timestamp": "2026-01-15T10:00:00Z",
            "attributes": {
                "input_token_count": 1000,
                "output_token_count": 500,
                "cached_content_token_count": 300,
            }
        })
        mock_file.return_value.__enter__ = lambda s: iter([log_line + "\n"])
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data()
        assert "2026-01-15" in result
        d = result["2026-01-15"]
        # input_token_count (1000) includes cached (300), so non-cached = 700
        assert d["input_tokens"] == 700
        assert d["cache_read_tokens"] == 300
        assert d["cost"] > 0

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_malformed_log_lines_skipped(self, mock_file, mock_path):
        """Malformed lines should be skipped without crashing."""
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        good_line = json.dumps({
            "timestamp": "2026-01-15T10:00:00Z",
            "attributes": {"input_token_count": 500, "output_token_count": 200, "cached_content_token_count": 0}
        })
        lines = [
            "not json at all\n",
            '{"timestamp": "2026-01-15T10:00:00Z", "attributes": "not-a-dict"}\n',
            '{"no_timestamp": true}\n',
            good_line + "\n",
        ]
        mock_file.return_value.__enter__ = lambda s: iter(lines)
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data()
        assert "2026-01-15" in result
        assert result["2026-01-15"]["input_tokens"] == 500

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_non_dict_record_is_skipped(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        lines = [
            "[1, 2, 3]\n",
            json.dumps({
                "timestamp": "2026-01-15T10:00:00Z",
                "attributes": {"input_token_count": 10, "output_token_count": 5, "cached_content_token_count": 2},
            }) + "\n",
        ]
        mock_file.return_value.__enter__ = lambda s: iter(lines)
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data()
        assert "2026-01-15" in result
        assert result["2026-01-15"]["input_tokens"] == 8

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_numeric_timestamp_is_parsed(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        ts_seconds = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp())
        ts_millis = ts_seconds * 1000
        ts_nanos = ts_seconds * 1_000_000_000
        ts_micros = ts_seconds * 1_000_000
        lines = [
            json.dumps({
                "timestamp": ts_seconds,
                "attributes": {"input_token_count": 100, "output_token_count": 50, "cached_content_token_count": 25},
            }) + "\n",
            json.dumps({
                "timestamp": ts_millis,
                "attributes": {"input_token_count": 100, "output_token_count": 50, "cached_content_token_count": 25},
            }) + "\n",
            json.dumps({
                "timestamp": ts_nanos,
                "attributes": {"input_token_count": 100, "output_token_count": 50, "cached_content_token_count": 25},
            }) + "\n",
            json.dumps({
                "timestamp": ts_micros,
                "attributes": {"input_token_count": 100, "output_token_count": 50, "cached_content_token_count": 25},
            }) + "\n",
            json.dumps({
                "timestamp": f"{float(ts_seconds)}",
                "attributes": {"input_token_count": 100, "output_token_count": 50, "cached_content_token_count": 25},
            }) + "\n",
        ]
        mock_file.return_value.__enter__ = lambda s: iter(lines)
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data()
        assert "2026-01-15" in result
        assert result["2026-01-15"]["input_tokens"] == 500

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_scientific_notation_timestamp_is_parsed(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        ts_seconds = int(datetime(2026, 1, 16, tzinfo=timezone.utc).timestamp())
        lines = [
            json.dumps({
                "time": f"{float(ts_seconds):.2e}",
                "attributes": {"input_token_count": 70, "output_token_count": 30, "cached_content_token_count": 10},
            }) + "\n",
        ]
        mock_file.return_value.__enter__ = lambda s: iter(lines)
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data()
        assert result["2026-01-16"]["input_tokens"] == 60

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_attribute_list_values_are_normalized(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        lines = [
            json.dumps({
                "Timestamp": "2026-01-17T08:00:00Z",
                "attributes": [
                    {"Key": "input_token_count", "Value": {"intValue": 90}},
                    {"key": "output_token_count", "Value": {"Int64Value": "45"}},
                    {"key": "cached_content_token_count", "Value": {"stringValue": "20"}},
                ],
            }) + "\n",
        ]
        mock_file.return_value.__enter__ = lambda s: iter(lines)
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data()
        assert result["2026-01-17"]["input_tokens"] == 70

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_boolean_numeric_fields_are_rejected(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        lines = [
            json.dumps({
                "timestamp": "2026-01-18T08:00:00Z",
                "attributes": {
                    "input_token_count": true,
                    "output_token_count": false,
                    "cached_content_token_count": true,
                },
            }) + "\n",
        ]
        mock_file.return_value.__enter__ = lambda s: iter(lines)
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data()
        assert "2026-01-18" in result
        assert result["2026-01-18"]["input_tokens"] == 0
        assert result["2026-01-18"]["output_tokens"] == 0
        assert result["2026-01-18"]["cache_read_tokens"] == 0

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_nonfinite_numeric_fields_are_safely_coerced(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        lines = [
            json.dumps({
                "timestamp": "2026-01-20T08:00:00Z",
                "attributes": {
                    "input_token_count": float("inf"),
                    "output_token_count": float("-inf"),
                    "cached_content_token_count": float("nan"),
                },
            }) + "\n",
        ]
        mock_file.return_value.__enter__ = lambda s: iter(lines)
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data()
        assert "2026-01-20" in result
        assert result["2026-01-20"]["input_tokens"] == 0
        assert result["2026-01-20"]["output_tokens"] == 0
        assert result["2026-01-20"]["cache_read_tokens"] == 0

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_timezone_filtering_changes_day_bucket(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        # 01:30Z converts to the prior date in America/Los_Angeles.
        line = json.dumps({
            "timestamp": "2026-01-16T01:30:00Z",
            "attributes": {"input_token_count": 100, "output_token_count": 0, "cached_content_token_count": 0},
        }) + "\n"
        mock_file.return_value.__enter__ = lambda s: iter([line])
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data(since="20260115", until="20260115", timezone_name="America/Los_Angeles")
        assert result["2026-01-15"]["input_tokens"] == 100

    @patch("tokenprint.Path")
    @patch("builtins.open")
    def test_timezone_name_is_forwarded_to_timestamp_parser(self, mock_file, mock_path):
        mock_path.home.return_value.__truediv__ = lambda s, x: mock_path
        mock_path.__truediv__ = lambda s, x: mock_path
        mock_path.exists.return_value = True

        lines = [
            json.dumps({
                "timestamp": "2026-01-14T20:30:00-05:00",
                "attributes": {"input_token_count": 50, "output_token_count": 20, "cached_content_token_count": 0},
            }) + "\n",
            json.dumps({
                "timestamp": "2026-01-15T12:00:00Z",
                "attributes": {"input_token_count": 75, "output_token_count": 25, "cached_content_token_count": 0},
            }) + "\n",
        ]
        mock_file.return_value.__enter__ = lambda s: iter(lines)
        mock_file.return_value.__exit__ = lambda s, *a: None

        result = collect_gemini_data(
            since="20260114",
            until="20260115",
            timezone_name="America/New_York",
        )
        assert result["2026-01-14"]["input_tokens"] == 50
        assert result["2026-01-15"]["input_tokens"] == 75


class TestCollectProviderData:
    @patch("tokenprint._warn")
    @patch("tokenprint.collect_claude_data", side_effect=RuntimeError("temporary collector failure"))
    def test_collect_days_with_fallback_warning_for_full_collection(self, mock_collect, mock_warn):
        claude = next(p for p in PROVIDERS if p.name == "claude")
        _collect_days_with_fallback(claude, "20260101", "20260131", mode="collection")

        mock_collect.assert_called_once_with("20260101", "20260131")
        assert any("collection failed" in str(call.args[0]) for call in mock_warn.call_args_list)

    @patch("tokenprint.collect_claude_data", side_effect=RuntimeError("collection failed"))
    def test_collect_days_with_fallback_handles_exception(self, mock_collect):
        claude = next(p for p in PROVIDERS if p.name == "claude")
        out = _collect_days_with_fallback(claude, None, None)
        mock_collect.assert_called_once_with(None, None)
        assert out == {}

    @patch("tokenprint.collect_claude_data", return_value=[])
    def test_collect_days_with_fallback_normalizes_non_dict_payload(self, mock_collect):
        claude = next(p for p in PROVIDERS if p.name == "claude")
        out = _collect_days_with_fallback(claude, "20260101", "20260131")
        mock_collect.assert_called_once_with("20260101", "20260131")
        assert out == {}

    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value=["not a dict"])
    def test_non_dict_provider_payload_becomes_empty(self, mock_claude, mock_codex, mock_gemini):
        out = _collect_provider_data("20260101", "20260131")
        assert out["claude"] == {}

    @patch("tokenprint.collect_codex_data", return_value={"2026-01-15": {"provider": "codex", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 10, "cache_write_tokens": 0, "cost": 0.02}})
    @patch("tokenprint.collect_claude_data", side_effect=RuntimeError("temporary collector failure"))
    @patch("tokenprint.collect_gemini_data", return_value={})
    def test_provider_errors_do_not_break_full_collection(self, mock_gemini, mock_claude, mock_codex):
        out = _collect_provider_data("20260101", "20260131")
        assert out["claude"] == {}
        assert out["codex"] == {"2026-01-15": {"provider": "codex", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 10, "cache_write_tokens": 0, "cost": 0.02}}
        assert out["gemini"] == {}

    @patch("tokenprint.collect_gemini_data")
    @patch("tokenprint.collect_codex_data")
    @patch("tokenprint.collect_claude_data")
    def test_timezone_is_passed_to_full_collection_collectors(self, mock_claude, mock_codex, mock_gemini):
        _collect_provider_data("20260101", "20260131", timezone_name="America/Los_Angeles")
        mock_claude.assert_called_once_with("20260101", "20260131")
        mock_codex.assert_called_once_with("20260101", "20260131")
        mock_gemini.assert_called_once_with("20260101", "20260131", "America/Los_Angeles")


# --- _collect_provider_data_incremental ---

class TestCollectProviderDataIncremental:
    @staticmethod
    def _day(provider):
        return {
            "provider": provider,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 10,
            "cache_write_tokens": 0,
            "cost": 0.01,
        }

    @patch("tokenprint.collect_gemini_data")
    @patch("tokenprint.collect_codex_data")
    @patch("tokenprint.collect_claude_data")
    @patch("tokenprint._load_provider_cache")
    def test_skips_collection_when_cache_is_current(self, mock_cache, mock_claude, mock_codex, mock_gemini):
        today_iso = datetime.now().strftime("%Y-%m-%d")
        mock_cache.return_value = {
            "claude": {today_iso: self._day("claude")},
            "codex": {today_iso: self._day("codex")},
            "gemini": {today_iso: self._day("gemini")},
        }

        out = _collect_provider_data_incremental()

        mock_claude.assert_not_called()
        mock_codex.assert_not_called()
        mock_gemini.assert_not_called()
        assert out["claude"][today_iso]["provider"] == "claude"
        assert out["codex"][today_iso]["provider"] == "codex"
        assert out["gemini"][today_iso]["provider"] == "gemini"

    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={})
    @patch("tokenprint._load_provider_cache")
    def test_timezone_is_passed_to_incremental_collectors(self, mock_cache, mock_claude, mock_codex, mock_gemini):
        mock_cache.return_value = {"claude": {}, "codex": {}, "gemini": {}}
        out = _collect_provider_data_incremental(timezone_name="America/Denver")
        assert out == {"claude": {}, "codex": {}, "gemini": {}}
        mock_claude.assert_called_once_with(None, None)
        mock_codex.assert_called_once_with(None, None)
        mock_gemini.assert_called_once_with(None, None, "America/Denver")

    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data")
    @patch("tokenprint._load_provider_cache")
    def test_fetches_only_new_days_per_provider(self, mock_cache, mock_claude, mock_codex, mock_gemini):
        today = datetime.now()
        today_iso = today.strftime("%Y-%m-%d")
        today_compact = today.strftime("%Y%m%d")
        yesterday_iso = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        mock_cache.return_value = {
            "claude": {yesterday_iso: self._day("claude")},
            "codex": {},
            "gemini": {},
        }
        mock_claude.return_value = {today_iso: self._day("claude")}

        out = _collect_provider_data_incremental()

        # Claude has cache through yesterday → incremental fetch from today.
        mock_claude.assert_called_once_with(today_compact, today_compact)
        # Codex and Gemini have no cache → full history fetch, not the global-max shortcut.
        mock_codex.assert_called_once_with(None, None)
        mock_gemini.assert_called_once_with(None, None, "UTC")
        assert yesterday_iso in out["claude"]
        assert today_iso in out["claude"]

    @patch("tokenprint.collect_gemini_data")
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data")
    @patch("tokenprint._load_provider_cache")
    def test_invalid_incremental_payload_is_ignored(self, mock_cache, mock_claude, mock_codex, mock_gemini):
        today = datetime.now()
        today_iso = today.strftime("%Y-%m-%d")
        today_compact = today.strftime("%Y%m%d")
        yesterday_iso = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        mock_cache.return_value = {
            "claude": {yesterday_iso: self._day("claude")},
            "codex": {},
            "gemini": {},
        }
        mock_claude.return_value = [("not", "a", "dict")]
        mock_codex.return_value = {}
        mock_gemini.return_value = {}

        out = _collect_provider_data_incremental()

        mock_claude.assert_called_once_with(today_compact, today_compact)
        assert out["claude"] == {yesterday_iso: self._day("claude")}
        assert out["codex"] == {}
        assert out["gemini"] == {}

    @patch("tokenprint.collect_gemini_data")
    @patch("tokenprint.collect_codex_data")
    @patch("tokenprint.collect_claude_data", side_effect=RuntimeError("collection failed"))
    @patch("tokenprint._load_provider_cache")
    def test_invalid_incremental_exception_is_ignored(self, mock_cache, mock_claude, mock_codex, mock_gemini):
        today = datetime.now()
        yesterday_iso = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        today_compact = today.strftime("%Y%m%d")
        mock_cache.return_value = {
            "claude": {yesterday_iso: self._day("claude")},
            "codex": {},
            "gemini": {},
        }

        out = _collect_provider_data_incremental()

        mock_claude.assert_called_once_with(today_compact, today_compact)
        assert out["claude"] == {yesterday_iso: self._day("claude")}

    @patch("tokenprint.collect_gemini_data")
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={})
    @patch("tokenprint._load_provider_cache")
    def test_invalid_cached_provider_payload_triggers_full_fetch(
        self, mock_cache, mock_claude, mock_codex, mock_gemini
    ):
        mock_cache.return_value = {
            "claude": ["not", "a", "dict"],
            "codex": {},
            "gemini": {},
        }

        out = _collect_provider_data_incremental()

        mock_claude.assert_called_once_with(None, None)
        mock_codex.assert_called_once_with(None, None)
        mock_gemini.assert_called_once_with(None, None, "UTC")
        assert out["claude"] == {}
        assert out["codex"] == {}
        assert out["gemini"] == {}

    @patch("tokenprint.collect_gemini_data")
    @patch("tokenprint.collect_codex_data")
    @patch("tokenprint.collect_claude_data")
    @patch("tokenprint._load_provider_cache")
    def test_mixed_cached_payload_paths_per_provider(self, mock_cache, mock_claude, mock_codex, mock_gemini):
        today = datetime.now()
        today_iso = today.strftime("%Y-%m-%d")
        today_compact = today.strftime("%Y%m%d")
        yesterday_iso = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        mock_cache.return_value = {
            "claude": ["not", "a", "dict"],
            "codex": {yesterday_iso: self._day("codex")},
            "gemini": {},
        }
        mock_claude.return_value = {today_iso: self._day("claude")}
        mock_codex.return_value = {today_iso: self._day("codex")}

        out = _collect_provider_data_incremental()

        mock_claude.assert_called_once_with(None, None)
        mock_codex.assert_called_once_with(today_compact, today_compact)
        mock_gemini.assert_called_once_with(None, None, "UTC")
        assert yesterday_iso in out["codex"]
        assert today_iso in out["codex"]
        assert out["claude"] == {today_iso: self._day("claude")}

    @patch("tokenprint._warn")
    @patch("tokenprint.collect_gemini_data")
    @patch("tokenprint.collect_codex_data")
    @patch("tokenprint.collect_claude_data", side_effect=RuntimeError("temporary collector failure"))
    @patch("tokenprint._load_provider_cache")
    def test_incremental_fallback_warning_for_failed_incremental_fetch(
        self, mock_cache, mock_claude, mock_codex, mock_gemini, mock_warn
    ):
        today = datetime.now()
        yesterday_iso = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        mock_cache.return_value = {
            "claude": {yesterday_iso: self._day("claude")},
            "codex": {},
            "gemini": {},
        }

        out = _collect_provider_data_incremental()

        assert out["claude"] == {yesterday_iso: self._day("claude")}
        assert any("incremental collection failed" in str(call.args[0]) for call in mock_warn.call_args_list)


# --- generate_html ---

class TestGenerateHtml:
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_smoke(self, mock_user):
        data = merge_data({
            "claude": {"2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}},
            "codex": {}, "gemini": {},
        })
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name
        try:
            generate_html(data, output_path)
            with open(output_path) as f:
                html = f.read()
            assert "const TP" in html
            assert "TOKENPRINT_DATA_PLACEHOLDER" not in html
            assert "updateDashboard" in html
            assert "chart.js" in html
        finally:
            os.unlink(output_path)

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_energy_model_in_output(self, mock_user):
        """Verify the generated HTML contains energy model constants from Python."""
        data = merge_data({
            "claude": {"2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}},
            "codex": {}, "gemini": {},
        })
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name
        try:
            generate_html(data, output_path)
            with open(output_path) as f:
                html = f.read()
            assert "energyModel" in html
            assert "outputWhPerToken" in html
        finally:
            os.unlink(output_path)

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_missing_template(self, mock_user):
        """generate_html should raise FileNotFoundError if template.html is missing."""
        import tokenprint as tp
        data = merge_data({
            "claude": {"2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}},
            "codex": {}, "gemini": {},
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            original_file = tp.__file__
            try:
                tp.__file__ = os.path.join(tmpdir, "tokenprint.py")
                with pytest.raises(FileNotFoundError):
                    generate_html(data, os.path.join(tmpdir, "output.html"))
            finally:
                tp.__file__ = original_file

    def test_write_html_file_creates_parent_path(self, tmp_path):
        nested = tmp_path / "a" / "b" / "tokenprint.html"
        _write_html_file(str(nested), "<html><body>ok</body></html>")
        assert nested.read_text() == "<html><body>ok</body></html>"

    def test_write_html_file_preserves_unicode(self, tmp_path):
        nested = tmp_path / "a" / "b" / "tokenprint.html"
        html = "<html><body>TokenPrint ✅</body></html>"
        _write_html_file(str(nested), html)
        assert nested.read_text(encoding="utf-8") == html

    def test_write_html_file_raises_runtime_error_for_directory_target(self, tmp_path):
        target = tmp_path
        with pytest.raises(RuntimeError, match="target is a directory"):
            _write_html_file(str(target), "<html></html>")

    def test_write_html_file_cleans_temp_file_on_replace_failure(self, tmp_path, monkeypatch):
        target = tmp_path / "a" / "b" / "tokenprint.html"

        def fail_replace(_src: os.PathLike[str], _dst: os.PathLike[str]) -> None:
            raise OSError("simulated replace failure")

        monkeypatch.setattr(tp.os, "replace", fail_replace)
        with pytest.raises(RuntimeError, match="unable to write dashboard file"):
            _write_html_file(str(target), "<html></html>")
        temp_prefix = f".{target.name}."
        assert not any(p.name.startswith(temp_prefix) for p in target.parent.iterdir())

    def test_write_json_file_creates_parent_path(self, tmp_path):
        nested = tmp_path / "a" / "b" / "tokenprint.json"
        payload = {"ok": True}
        _write_json_file(str(nested), payload)
        assert nested.read_text() == json.dumps(payload)

    def test_write_json_file_uses_atomic_replace(self, tmp_path):
        nested = tmp_path / "a" / "b" / "tokenprint.json"
        payload = {"ok": True}
        _write_json_file(str(nested), payload)
        temp_prefix = f".{nested.name}."
        assert not any(p.name.startswith(temp_prefix) for p in nested.parent.iterdir())

    def test_write_json_file_calls_fsync(self, tmp_path, monkeypatch):
        nested = tmp_path / "tokenprint.json"
        payload = {"ok": True}
        calls = []

        def fake_fsync(fd: int) -> None:
            calls.append(fd)

        monkeypatch.setattr(tp.os, "fsync", fake_fsync)
        _write_json_file(str(nested), payload)
        assert calls
        assert all(isinstance(fd, int) for fd in calls)

    def test_write_json_file_preserves_unicode(self, tmp_path):
        nested = tmp_path / "a" / "b" / "tokenprint.json"
        payload = {"label": "café"}
        _write_json_file(str(nested), payload)
        assert json.loads(nested.read_text(encoding="utf-8")) == payload

    def test_write_json_file_raises_runtime_error_for_directory_target(self, tmp_path):
        target = tmp_path
        with pytest.raises(RuntimeError, match="target is a directory"):
            _write_json_file(str(target), {"ok": True})


class TestWriteTextAtomic:
    def test_writes_and_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "data.txt"
        _write_text_atomic(nested, "ok")
        assert nested.read_text() == "ok"

    def test_calls_fsync_before_replace(self, tmp_path, monkeypatch):
        nested = tmp_path / "data.txt"
        calls = []

        def fake_fsync(fd: int) -> None:
            calls.append(fd)

        monkeypatch.setattr(tp.os, "fsync", fake_fsync)
        _write_text_atomic(nested, "ok")
        assert calls
        assert all(isinstance(fd, int) for fd in calls)

    def test_fsync_failure_cleans_temp_file(self, tmp_path, monkeypatch):
        nested = tmp_path / "a" / "b" / "data.txt"

        def fail_fsync(_: int) -> None:
            raise OSError("simulated fsync failure")

        monkeypatch.setattr(tp.os, "fsync", fail_fsync)
        with pytest.raises(OSError, match="simulated fsync failure"):
            _write_text_atomic(nested, "ok")
        temp_prefix = f".{nested.name}."
        assert not any(p.name.startswith(temp_prefix) for p in nested.parent.iterdir())


# --- main() ---

class TestMain:
    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_default_run_uses_incremental(self, mock_user, mock_incremental, mock_full, mock_save):
        mock_incremental.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }
        with patch("sys.argv", ["tokenprint", "--no-open"]):
            main()
        mock_incremental.assert_called_once_with(cache_path=None, timezone_name="UTC")
        mock_full.assert_not_called()
        mock_save.assert_called_once()

    @patch("tokenprint._run_cli_check", return_value=True)
    def test_check_flag_exits_zero(self, mock_check):
        with patch("sys.argv", ["tokenprint", "--check"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        mock_check.assert_called_once()

    @patch("tokenprint._run_cli_check", return_value=False)
    def test_check_flag_exits_nonzero(self, mock_check):
        with patch("sys.argv", ["tokenprint", "--check"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        mock_check.assert_called_once()

    def test_version_flag_exits(self):
        with patch("sys.argv", ["tokenprint", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    @patch("tokenprint._tokenprint_version", return_value="9.9.9")
    def test_version_flag_uses_runtime_lookup(self, mock_version, capsys):
        with patch("sys.argv", ["tokenprint", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        out, _ = capsys.readouterr()
        assert "tokenprint 9.9.9" in out
        mock_version.assert_called_once()

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_custom_cache_path_is_used(self, mock_user, mock_incremental, mock_full, mock_save, tmp_path):
        custom_cache = tmp_path / "custom-cache.json"
        mock_incremental.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }
        with patch("sys.argv", ["tokenprint", "--no-open", "--cache-path", str(custom_cache)]):
            main()
        mock_incremental.assert_called_once_with(cache_path=custom_cache, timezone_name="UTC")
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=custom_cache)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_custom_gemini_log_path_is_passed_to_incremental(
        self, mock_user, mock_incremental, mock_full, mock_save, tmp_path
    ):
        mock_incremental.return_value = {
            "claude": {},
            "codex": {},
            "gemini": {
                "2026-01-15": {
                    "provider": "gemini",
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cache_read_tokens": 1,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
        }
        with patch("sys.argv", ["tokenprint", "--no-open", "--gemini-log-path", str(tmp_path / "gemini.log")]):
            main()
        mock_incremental.assert_called_once_with(
            cache_path=None,
            timezone_name="UTC",
            gemini_log_path=str(tmp_path / "gemini.log"),
        )
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=None)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_whitespace_gemini_log_path_is_ignored_and_not_passed_to_collectors(
        self, mock_user, mock_incremental, mock_full, mock_save
    ):
        mock_incremental.return_value = {"claude": {}, "codex": {}, "gemini": {}}
        with patch("sys.argv", ["tokenprint", "--no-open", "--gemini-log-path", "   "]):
            main()
        mock_incremental.assert_called_once_with(cache_path=None, timezone_name="UTC")
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=None)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_cli_gemini_log_path_wins_over_env(
        self, mock_user, mock_incremental, mock_full, mock_save, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("TOKENPRINT_GEMINI_TELEMETRY_LOG_PATH", str(tmp_path / "env.gemini.log"))
        mock_incremental.return_value = {"claude": {}, "codex": {}, "gemini": {}}
        with patch(
            "sys.argv",
            [
                "tokenprint",
                "--no-open",
                "--gemini-log-path",
                str(tmp_path / "cli.gemini.log"),
            ],
        ):
            main()
        mock_incremental.assert_called_once_with(
            cache_path=None,
            timezone_name="UTC",
            gemini_log_path=str(tmp_path / "cli.gemini.log"),
        )

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_cache_directory_path_is_expanded_and_resolved(self, mock_user, mock_incremental, mock_full, mock_save, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache-dir"
        cache_dir.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))
        home_relative = "~/cache-dir"

        mock_incremental.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        expected_path = cache_dir / "tokenprint-provider-cache-v1.json"
        with patch("sys.argv", ["tokenprint", "--no-open", "--cache-path", home_relative]):
            main()
        mock_incremental.assert_called_once_with(cache_path=expected_path, timezone_name="UTC")
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=expected_path)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_env_cache_path_is_used_when_no_flag(self, mock_user, mock_incremental, mock_full, mock_save, tmp_path, monkeypatch):
        cache_file = tmp_path / "tokenprint-env-cache.json"
        monkeypatch.setenv("TOKENPRINT_CACHE_PATH", str(cache_file))
        mock_incremental.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        with patch("sys.argv", ["tokenprint", "--no-open"]):
            main()
        mock_incremental.assert_called_once_with(cache_path=cache_file, timezone_name="UTC")
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=cache_file)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_env_cache_path_with_whitespace_is_trimmed(self, mock_user, mock_incremental, mock_full, mock_save, tmp_path, monkeypatch):
        cache_file = tmp_path / "tokenprint-env-cache.json"
        monkeypatch.setenv("TOKENPRINT_CACHE_PATH", f" {cache_file} ")
        mock_incremental.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        with patch("sys.argv", ["tokenprint", "--no-open"]):
            main()
        mock_incremental.assert_called_once_with(cache_path=cache_file, timezone_name="UTC")
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=cache_file)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_empty_cache_path_flag_defers_to_env(
        self, mock_user, mock_incremental, mock_full, mock_save, tmp_path, monkeypatch
    ):
        env_cache = tmp_path / "env-cache.json"
        cli_cache = "   "
        monkeypatch.setenv("TOKENPRINT_CACHE_PATH", str(env_cache))
        mock_incremental.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        with patch("sys.argv", ["tokenprint", "--cache-path", cli_cache, "--no-open"]):
            main()
        mock_incremental.assert_called_once_with(cache_path=env_cache, timezone_name="UTC")
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=env_cache)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_cli_cache_path_wins_over_env_cache_path(
        self, mock_user, mock_incremental, mock_full, mock_save, tmp_path, monkeypatch
    ):
        env_cache = tmp_path / "env-cache.json"
        cli_cache = tmp_path / "cli-cache.json"
        monkeypatch.setenv("TOKENPRINT_CACHE_PATH", str(env_cache))
        mock_incremental.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        with patch("sys.argv", ["tokenprint", "--no-open", "--cache-path", str(cli_cache)]):
            main()
        mock_incremental.assert_called_once_with(cache_path=cli_cache, timezone_name="UTC")
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=cli_cache)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_cache_path_with_whitespace_is_trimmed(
        self, mock_user, mock_incremental, mock_full, mock_save, tmp_path, monkeypatch
    ):
        cache_file = tmp_path / "cli-cache.json"
        mock_incremental.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }

        with patch("sys.argv", ["tokenprint", "--no-open", "--cache-path", f" {cache_file} "]):
            main()
        mock_incremental.assert_called_once_with(cache_path=cache_file, timezone_name="UTC")
        mock_save.assert_called_once_with(mock_incremental.return_value, cache_path=cache_file)

    @patch("tokenprint._save_provider_cache")
    @patch("tokenprint._collect_provider_data")
    @patch("tokenprint._collect_provider_data_incremental")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_no_cache_flag_forces_full_collection(self, mock_user, mock_incremental, mock_full, mock_save):
        mock_full.return_value = {
            "claude": {
                "2026-01-15": {
                    "provider": "claude",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost": 0.01,
                }
            },
            "codex": {},
            "gemini": {},
        }
        with patch("sys.argv", ["tokenprint", "--no-cache", "--no-open"]):
            main()
        mock_incremental.assert_not_called()
        mock_full.assert_called_once_with(None, None, "UTC")
        mock_save.assert_called_once()

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_default_run(self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser):
        """main() should generate HTML and open browser by default."""
        with patch("sys.argv", ["tokenprint"]):
            main()
        mock_browser.assert_called_once()
        # Verify the file was created
        call_arg = mock_browser.call_args[0][0]
        assert "tokenprint.html" in call_arg

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_no_open_flag(self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser):
        """--no-open should skip browser."""
        with patch("sys.argv", ["tokenprint", "--no-open"]):
            main()
        mock_browser.assert_not_called()

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.generate_html")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_json_output_format(self, mock_claude, mock_codex, mock_gemini, mock_user, mock_generate_html, mock_browser):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name
        try:
            with patch("sys.argv", ["tokenprint", "--no-open", "--output-format", "json", "--output", output_path]):
                main()
            with open(output_path) as f:
                data = json.load(f)
            assert "rawData" in data
        finally:
            os.unlink(output_path)
        mock_generate_html.assert_not_called()
        mock_browser.assert_not_called()

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.compute_dashboard_data")
    @patch("tokenprint._collect_merged_usage_data")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint._write_json_file")
    def test_json_output_format_uses_default_json_path(
        self, mock_write_json, mock_user, mock_collect, mock_compute, mock_browser, tmp_path
    ):
        mock_collect.return_value = []
        mock_compute.return_value = {}
        with patch.object(tp.tempfile, "gettempdir", return_value=str(tmp_path)):
            with patch("sys.argv", ["tokenprint", "--no-open", "--output-format", "json"]):
                main()
        expected_path = str(tmp_path / "tokenprint.json")
        assert mock_write_json.call_args.args[0] == expected_path
        mock_browser.assert_not_called()

    @patch("tokenprint._write_json_file")
    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_whitespace_json_output_path_uses_default_json_path(
        self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser, mock_write_json, tmp_path
    ):
        with patch.object(tp.tempfile, "gettempdir", return_value=str(tmp_path)):
            with patch("sys.argv", ["tokenprint", "--no-open", "--output-format", "json", "--output", "   "]):
                main()
        expected_path = str(tmp_path / "tokenprint.json")
        assert mock_write_json.call_args.args[0] == expected_path
        mock_browser.assert_not_called()

    @patch("tokenprint._write_html_file")
    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_default_html_output_path(
        self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser, mock_write_html, tmp_path
    ):
        with patch.object(tp.tempfile, "gettempdir", return_value=str(tmp_path)):
            with patch("sys.argv", ["tokenprint", "--no-open"]):
                main()
        expected_path = str(tmp_path / "tokenprint.html")
        assert mock_write_html.call_args.args[0] == expected_path
        mock_browser.assert_not_called()

    @patch("tokenprint._write_html_file")
    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_whitespace_output_path_uses_default(
        self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser, mock_write_html, tmp_path
    ):
        with patch.object(tp.tempfile, "gettempdir", return_value=str(tmp_path)):
            with patch("sys.argv", ["tokenprint", "--output", "   ", "--no-open"]):
                main()
        expected_path = str(tmp_path / "tokenprint.html")
        assert mock_write_html.call_args.args[0] == expected_path
        mock_browser.assert_not_called()

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_custom_output(self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser):
        """--output should write to custom path."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name
        try:
            with patch("sys.argv", ["tokenprint", "--output", output_path, "--no-open"]):
                main()
            with open(output_path) as f:
                html = f.read()
            assert "TokenPrint" in html
        finally:
            os.unlink(output_path)

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_live_mode_flags_rendered(self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser):
        """--live-mode should embed live config for external daemon refresh endpoint."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = f.name
        try:
            with patch(
                "sys.argv",
                ["tokenprint", "--no-open", "--output", output_path, "--live-mode", "--refresh-endpoint", "/api/refresh"],
            ):
                main()
            with open(output_path) as f:
                html = f.read()
            assert '"liveMode": true' in html
            assert '"refreshEndpoint": "/api/refresh"' in html
        finally:
            os.unlink(output_path)

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.compute_dashboard_data")
    @patch("tokenprint._write_html_file")
    def test_live_mode_refresh_args_are_trimmed(
        self,
        mock_write_html,
        mock_compute,
        mock_user,
        mock_claude,
        mock_codex,
        mock_gemini,
        mock_browser,
    ):
        mock_compute.return_value = {"rawData": []}
        with patch("sys.argv", [
            "tokenprint",
            "--no-open",
            "--live-mode",
            "--refresh-endpoint",
            "  /api/refresh  ",
            "--refresh-token",
            "  secret-token  ",
        ]):
            main()

        assert mock_compute.call_args.kwargs["live_mode"] is True
        assert mock_compute.call_args.kwargs["refresh_endpoint"] == "/api/refresh"
        assert mock_compute.call_args.kwargs["refresh_token"] == "secret-token"
        mock_write_html.assert_called_once()
        mock_browser.assert_not_called()

    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={})
    def test_no_data_exits(self, mock_claude, mock_codex, mock_gemini):
        """main() should exit with code 1 when no data is found."""
        with patch("sys.argv", ["tokenprint", "--no-open"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_invalid_since_date(self):
        """--since with bad format should error."""
        with patch("sys.argv", ["tokenprint", "--since", "not-a-date"]), pytest.raises(SystemExit):
            main()

    @patch("tokenprint._collect_provider_data_incremental")
    def test_whitespace_only_since_date_is_invalid(self, mock_incremental):
        with patch("sys.argv", ["tokenprint", "--since", "   ", "--no-open"]):
            with pytest.raises(SystemExit):
                main()
        mock_incremental.assert_not_called()

    @patch("tokenprint._collect_provider_data_incremental")
    def test_whitespace_only_until_date_is_invalid(self, mock_incremental):
        with patch("sys.argv", ["tokenprint", "--until", "   ", "--no-open"]):
            with pytest.raises(SystemExit):
                main()
        mock_incremental.assert_not_called()

    def test_invalid_calendar_date(self):
        """--since with impossible calendar date should error."""
        with patch("sys.argv", ["tokenprint", "--since", "20261345"]), pytest.raises(SystemExit):
            main()

    def test_invalid_output_format(self):
        with patch("sys.argv", ["tokenprint", "--output-format", "yaml", "--no-open"]), pytest.raises(SystemExit):
            main()

    def test_since_after_until_is_rejected(self):
        """--since after --until should fail fast."""
        with patch("sys.argv", ["tokenprint", "--since", "20260131", "--until", "20260101", "--no-open"]):
            with pytest.raises(SystemExit):
                main()

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_since_until_passed_to_collectors(self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser):
        """Date args should be forwarded to collectors."""
        with patch("sys.argv", ["tokenprint", "--since", "20260101", "--until", "20260131", "--no-open"]):
            main()
        mock_claude.assert_called_once_with("20260101", "20260131")
        mock_codex.assert_called_once_with("20260101", "20260131")
        mock_gemini.assert_called_once_with("20260101", "20260131", "UTC")

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_iso_date_range_is_normalized_for_collectors(
        self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser
    ):
        """ISO date args should be normalized to compact format for collectors."""
        with patch("sys.argv", ["tokenprint", "--since", "2026-01-01", "--until", "2026-01-31", "--no-open"]):
            main()
        mock_claude.assert_called_once_with("20260101", "20260131")
        mock_codex.assert_called_once_with("20260101", "20260131")
        mock_gemini.assert_called_once_with("20260101", "20260131", "UTC")

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_whitespace_surrounded_dates_are_accepted_for_collectors(
        self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser
    ):
        with patch("sys.argv", ["tokenprint", "--since", " 2026-01-01 ", "--until", " 2026-01-31 ", "--no-open"]):
            main()
        mock_claude.assert_called_once_with("20260101", "20260131")
        mock_codex.assert_called_once_with("20260101", "20260131")
        mock_gemini.assert_called_once_with("20260101", "20260131", "UTC")

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_timezone_is_forwarded_to_collectors(
        self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser
    ):
        with patch("sys.argv", [
            "tokenprint",
            "--since",
            "20260101",
            "--until",
            "20260131",
            "--timezone",
            "America/Denver",
            "--no-open",
        ]):
            main()
        mock_claude.assert_called_once_with("20260101", "20260131")
        mock_codex.assert_called_once_with("20260101", "20260131")
        mock_gemini.assert_called_once_with("20260101", "20260131", "America/Denver")

    @patch("tokenprint.webbrowser.open")
    @patch("tokenprint.detect_github_username", return_value="testuser")
    @patch("tokenprint.collect_gemini_data", return_value={})
    @patch("tokenprint.collect_codex_data", return_value={})
    @patch("tokenprint.collect_claude_data", return_value={
        "2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50,
                       "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}
    })
    def test_gemini_log_path_is_forwarded_to_collectors(
        self, mock_claude, mock_codex, mock_gemini, mock_user, mock_browser
    ):
        with patch(
            "sys.argv",
            [
                "tokenprint",
                "--since",
                "20260101",
                "--until",
                "20260131",
                "--gemini-log-path",
                "/tmp/gemini.log",
                "--no-open",
            ],
        ):
            main()
        mock_claude.assert_called_once_with("20260101", "20260131")
        mock_codex.assert_called_once_with("20260101", "20260131")
        mock_gemini.assert_called_once_with("20260101", "20260131", "UTC", log_path="/tmp/gemini.log")

    def test_invalid_timezone_is_rejected(self):
        with patch("sys.argv", ["tokenprint", "--timezone", "not-a-timezone", "--no-open"]):
            with pytest.raises(SystemExit):
                main()


class TestModuleEntrypoint:
    def test_python_module_entrypoint_supports_version(self):
        repo_root = Path(__file__).resolve().parent
        env = os.environ.copy()
        python_path = str(repo_root)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = python_path + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

        result = subprocess.run(
            [sys.executable, "-m", "tokenprint", "--version"],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert result.stdout.startswith("tokenprint ")


# --- Date validation edge cases ---

class TestDateValidation:
    def test_valid_yyyymmdd(self):
        from datetime import datetime
        # Should not raise
        datetime.strptime("20260115", "%Y%m%d")

    def test_invalid_month(self):
        from datetime import datetime
        with pytest.raises(ValueError):
            datetime.strptime("20261315", "%Y%m%d")

    def test_invalid_day(self):
        from datetime import datetime
        with pytest.raises(ValueError):
            datetime.strptime("20260230", "%Y%m%d")


# --- Provider Registry ---

class TestProviderRegistry:
    def test_provider_registry_module_exports_provider_config(self):
        assert ProviderConfig is not None
        assert all(isinstance(p, ProviderConfig) for p in MODULE_PROVIDERS)

    def test_top_level_and_module_providers_match(self):
        top_level_names = [p.name for p in PROVIDERS]
        module_names = [p.name for p in MODULE_PROVIDERS]
        assert top_level_names == module_names
        assert len(MODULE_PROVIDERS) == len(PROVIDERS)

    def test_provider_lookup_works(self):
        p = provider_by_name("claude")
        assert p is not None
        assert p.name == "claude"
        assert p.display_name == "Claude Code"
        assert p is MODULE_PROVIDERS[0]
        assert provider_by_name("missing-provider") is None

    def test_provider_config_is_frozen(self):
        p = MODULE_PROVIDERS[0]
        with pytest.raises(AttributeError):
            p.name = "hijack"

    def test_unknown_provider_rates_fall_back_to_zero(self):
        assert tp._rates_for_provider("nope") == (0.0, 0.0, 0.0)

    def test_provider_key_rates_resolution_uses_resolver(self):
        assert tp._rates_for_provider("c") == tp._rates_for_provider("claude")

    def test_provider_lookup_by_key_works(self):
        p = provider_by_key("c")
        assert p is not None
        assert p.name == "claude"
        assert provider_by_key("z") is None

    def test_resolve_provider_with_name(self):
        p = resolve_provider("claude")
        assert p is not None
        assert p.name == "claude"

    def test_resolve_provider_with_key(self):
        p = resolve_provider("x")
        assert p is not None
        assert p.name == "codex"

    def test_resolve_provider_empty_is_none(self):
        assert resolve_provider("") is None
    def test_provider_names_helper_matches_registry_order(self):
        assert provider_names() == ("claude", "codex", "gemini")

    def test_provider_name_set_matches_registry(self):
        assert provider_name_set() == {"claude", "codex", "gemini"}

    def test_provider_count(self):
        assert len(PROVIDERS) == 3

    def test_unique_names(self):
        names = [p.name for p in PROVIDERS]
        assert len(names) == len(set(names))

    def test_unique_keys(self):
        keys = [p.key for p in PROVIDERS]
        assert len(keys) == len(set(keys))

    @patch("tokenprint.detect_github_username", return_value="testuser")
    def test_config_includes_providers(self, mock_user):
        """Verify compute_dashboard_data output includes providers list with correct structure."""
        data = merge_data({
            "claude": {"2026-01-15": {"provider": "claude", "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0, "cache_write_tokens": 0, "cost": 0.01}},
            "codex": {}, "gemini": {},
        })
        config = compute_dashboard_data(data)
        providers = config["providers"]
        assert len(providers) == len(PROVIDERS)
        for i, p in enumerate(PROVIDERS):
            assert providers[i]["name"] == p.name
            assert providers[i]["displayName"] == p.display_name
            assert providers[i]["key"] == p.key
            assert providers[i]["color"] == p.color
            assert providers[i]["rates"]["input"] == p.rates[0]
            assert providers[i]["rates"]["output"] == p.rates[1]
            assert providers[i]["rates"]["cached"] == p.rates[2]
