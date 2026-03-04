# TokenPrint

**Created:** 2026-02-23-17-42
**Last Updated:** 2026-03-03-16-43

Track the true cost of AI coding tools — tokens, dollars, energy, carbon, and water.

TokenPrint collects usage data from **Claude Code**, **Codex CLI**, and **Gemini CLI**, then generates an interactive HTML dashboard with cost breakdowns, environmental impact estimates, and real-world equivalents.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Quick Start

```bash
# One-command setup (installs ccusage, codex, gemini telemetry)
bash install.sh

# Install CLI
pipx install -e .

# Run (one-shot)
tokenprint

# Or run directly as a module
python -m tokenprint

# Run (live dashboard with auto-refresh)
tokenprintd
```

Or install manually: `npm i -g ccusage @ccusage/codex@18` for Claude/Codex data. Gemini is optional.

## What You Get

An interactive dark-mode dashboard with:

- **Provider toggles** — Click Claude/Codex/Gemini to show/hide any provider across all charts, cards, and tables
- **Date range picker** — Filter to any date range
- **Daily + cumulative views** — Toggle each chart between daily bars and cumulative lines
- **Cost matrix** — Monthly cost breakdown by provider with token tooltips
- **Environmental impact cards** — Energy (Wh/kWh/MWh), carbon (g/kg/tonnes), water (mL/L), electricity cost
- **Real-world equivalents** — Burritos, stacked Bibles, Tesla miles, showers, flights, gas car miles
- **Incremental collection** — Default reruns fetch only new days per provider (fast daily refreshes)

### Charts

| Chart | Daily | Cumulative |
|-------|-------|------------|
| Cost | Stacked bars by provider | Line with total |
| Tokens | Stacked bars by provider | Line with total |
| Energy | Stacked bars by provider | Line with total |
| Carbon | Color-coded bars (green/amber/red) | Line with total |

## Usage

```bash
tokenprint                                      # Full history, opens in browser
tokenprint --since 20260201                     # From date
tokenprint --until 20260215                     # To date
tokenprint --since 20260201 --until 20260215    # Date range
tokenprint --since 2026-01-01                   # ISO date input
tokenprint --no-open                            # Generate without opening
tokenprint --output ~/report.html               # Custom output path
tokenprint --no-cache                           # Force full refresh (ignore incremental cache)
tokenprint --timezone America/Los_Angeles        # Bucket Gemini dates in a local timezone
tokenprint --gemini-log-path ~/gemini/telemetry.log  # Read Gemini usage from an alternate telemetry source
```

The default output is `/tmp/tokenprint.html`. TokenPrint keeps a provider cache in your temp directory (`/tmp/tokenprint-provider-cache-v1.json`) and by default only fetches days after each provider's last collected date. Use `--no-cache` for a full rebuild.

### Live Dashboard (Go Daemon)

For a persistent live dashboard where the **Refresh Data** button reruns collection without restarting:

```bash
tokenprintd                          # http://127.0.0.1:8765
tokenprintd --port 8877              # custom port
tokenprintd --host ::1               # IPv6 loopback
tokenprintd --refresh-token mysecret # required for non-loopback hosts
tokenprintd --host 0.0.0.0 --refresh-token mysecret # public/wildcard host
tokenprintd --cache-path /tmp/tokenprint-cache-v1.json # explicit cache location for daemon
tokenprintd --gemini-log-path ~/gemini/telemetry.log --no-open # custom Gemini telemetry location
```

Binding TokenPrint on any non-loopback host (for example `--host 0.0.0.0`) requires
`--refresh-token` and the `X-Tokenprint-Token` header for refresh endpoint calls.

`install.sh` builds and installs `tokenprintd` to `~/.local/bin`. The daemon runs `tokenprint` on startup, serves the dashboard, and handles refresh requests. See [`daemon/go/README.md`](daemon/go/README.md) for all flags.

## Data Sources

| Provider | Source | Install |
|----------|--------|---------|
| Claude Code | `ccusage daily --json` | `npm i -g ccusage` |
| Codex CLI | `ccusage-codex daily --json` | `npm i -g @ccusage/codex@18` |
| Gemini CLI | `~/.gemini/telemetry.log` (override with `TOKENPRINT_GEMINI_TELEMETRY_LOG_PATH` or `--gemini-log-path`) | One-time setup (see below) |

If `ccusage` returns `cost: 0` for newer Claude models, TokenPrint applies a model-rate fallback (from Anthropic published pricing) so daily totals are not undercounted.

### Gemini CLI Setup (Optional)

Gemini CLI doesn't expose usage data by default. Run the setup script to enable OpenTelemetry local logging:

```bash
bash setup-gemini-telemetry.sh
```

This adds telemetry config to `~/.gemini/settings.json`. Future Gemini CLI sessions will log token usage to `~/.gemini/telemetry.log`. Historical data cannot be backfilled — tracking starts from the point you enable it.

You can point TokenPrint at a different log location by setting:

```bash
export TOKENPRINT_GEMINI_TELEMETRY_LOG_PATH=/absolute/path/to/telemetry.log
```

If set to a directory, TokenPrint will read from `<directory>/telemetry.log`.

## Energy & Carbon Model

TokenPrint estimates environmental impact using industry averages:

| Parameter | Value | Source |
|-----------|-------|--------|
| Energy per output token | 0.001 Wh | IEA (2024), Luccioni et al. (2023) |
| Energy per input token | 0.0002 Wh | de Vries (2023), Patterson et al. (2022) |
| Energy per cached token | 0.00005 Wh | KV-cache architecture analysis |
| PUE (data center overhead) | 1.2x | Uptime Institute (2023), Google (2023) |
| Embodied carbon | +20% | Gupta et al. (2022) "ACT" |
| Grid transmission loss | 5% | EIA (2024), LBNL |
| Carbon intensity | 390 gCO2e/kWh | EPA eGRID (2022), IEA (2023) |
| Water usage efficiency | 0.5 L/kWh | Google (2023), Li et al. (2023) |
| Electricity price | $0.13/kWh | EIA (2024) US commercial average |

These are rough estimates. Actual impact varies by model, hardware, data center location, time of day, and renewable energy mix. The dashboard includes a full methodology section with all assumptions.

## CrossCheck Integration

If you use [CrossCheck](https://github.com/sburl/CrossCheck), TokenPrint is available as the `/ai-impact` skill:

```bash
/ai-impact                    # Same as running tokenprint
/ai-impact --since 20260201   # With date filter
/ai-impact --since 2026-01-01  # ISO date support
/ai-impact --output-format json --output /tmp/tokenprint.json
/ai-impact --timezone America/Los_Angeles --output-format json
``` 

## Architecture

For a deeper walkthrough of how collectors, cache persistence, merge/render flow,
and daemon refresh interact, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Command reference

- `--version` prints the installed package version from metadata.
- `--check` validates CLI prerequisites (template file and required collectors) before running.
- `--since` and `--until` accept `YYYYMMDD` or `YYYY-MM-DD` date formats.
- `--timezone` groups Gemini timestamps by an IANA timezone (defaults to `UTC`).
- `--output-format` chooses output renderer (`html` or `json`; default is `html`).
- `--output` sets the output path; defaults to:
  - HTML: `{tempdir}/tokenprint.html`
  - JSON: `{tempdir}/tokenprint.json`
- `--no-open` suppresses browser launch for HTML output.
- `--cache-path` overrides the provider cache file path (`tokenprint-provider-cache-v1.json` in temp by default, or set via `TOKENPRINT_CACHE_PATH`).
- `--gemini-log-path` can point token collection to a custom Gemini telemetry log file or directory.
- `TOKENPRINT_GEMINI_TELEMETRY_LOG_PATH` can point token collection to a custom log path without CLI flags.
- `python -m tokenprint` is supported as an alternate invocation for scripting environments.

## License

MIT
