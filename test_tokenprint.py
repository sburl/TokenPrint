"""Tests for tokenprint.py pure functions."""

import json
import os
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from tokenprint import (
    _safe_int,
    _parse_date_flexible,
    _json_dumps_html_safe,
    calculate_energy,
    calculate_carbon,
    calculate_water,
    merge_data,
    compute_dashboard_data,
    generate_html,
    collect_claude_data,
    collect_codex_data,
    collect_gemini_data,
    run_command,
    detect_github_username,
    main,
    ProviderConfig,
    PROVIDERS,
    ENERGY_PER_OUTPUT_TOKEN_WH,
    ENERGY_PER_INPUT_TOKEN_WH,
    ENERGY_PER_CACHED_TOKEN_WH,
    PUE,
    GRID_LOSS_FACTOR,
    CARBON_INTENSITY,
    EMBODIED_CARBON_FACTOR,
    WATER_USE_EFFICIENCY,
)


# --- _safe_int ---

class TestSafeInt:
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


# --- _parse_date_flexible ---

class TestParseDateFlexible:
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

    def test_timeout(self):
        result = run_command(["sleep", "10"], timeout=1)
        assert result is None

    def test_strips_output(self):
        result = run_command(["echo", "  spaces  "])
        assert result == "spaces"


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

    @patch("tokenprint.run_command", return_value='{"daily": [{"date": "Jan 15, 2026", "inputTokens": 500, "outputTokens": 200, "cachedInputTokens": 0, "costUSD": 0}]}')
    def test_human_date_format(self, mock_run):
        result = collect_codex_data()
        assert "2026-01-15" in result

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


# --- main() ---

class TestMain:
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
        with patch("sys.argv", ["tokenprint", "--since", "not-a-date"]):
            with pytest.raises(SystemExit):
                main()

    def test_invalid_calendar_date(self):
        """--since with impossible calendar date should error."""
        with patch("sys.argv", ["tokenprint", "--since", "20261345"]):
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
        mock_gemini.assert_called_once_with("20260101", "20260131")


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
