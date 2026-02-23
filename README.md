# TokenPrint

Track the true cost of AI coding tools — tokens, dollars, energy, carbon, and water.

TokenPrint collects usage data from **Claude Code**, **Codex CLI**, and **Gemini CLI**, then generates an interactive HTML dashboard with cost breakdowns, environmental impact estimates, and real-world equivalents.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Quick Start

```bash
# One-command setup (installs ccusage, codex, gemini telemetry)
bash install.sh

# Install CLI
pipx install -e .

# Run
tokenprint                # One-shot static report
tokenprint --serve        # Live dashboard (UI refresh reruns data collection)
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
tokenprint --no-open                            # Generate without opening
tokenprint --output ~/report.html               # Custom output path
tokenprint --no-cache                           # Force full refresh (ignore incremental cache)
tokenprint --serve                              # Start local live server on http://127.0.0.1:8765
tokenprint --serve --port 8877                  # Live server on custom port
```

The default static output is `/tmp/tokenprint.html` — a fixed path so you can re-run and reload the file.
In `--serve` mode, the UI **Refresh Data** button triggers a real backend recollection run, shows a spinner, then auto-reloads with new data.
TokenPrint also keeps a local provider cache in your temp directory (for example `/tmp/tokenprint-provider-cache-v1.json`) and, by default, only fetches days after each provider's last collected date. Use `--no-cache` for a full rebuild.

## Data Sources

| Provider | Source | Install |
|----------|--------|---------|
| Claude Code | `ccusage daily --json` | `npm i -g ccusage` |
| Codex CLI | `ccusage-codex daily --json` | `npm i -g @ccusage/codex@18` |
| Gemini CLI | `~/.gemini/telemetry.log` | One-time setup (see below) |

If `ccusage` returns `cost: 0` for newer Claude models, TokenPrint applies a model-rate fallback (from Anthropic published pricing) so daily totals are not undercounted.

### Gemini CLI Setup (Optional)

Gemini CLI doesn't expose usage data by default. Run the setup script to enable OpenTelemetry local logging:

```bash
bash setup-gemini-telemetry.sh
```

This adds telemetry config to `~/.gemini/settings.json`. Future Gemini CLI sessions will log token usage to `~/.gemini/telemetry.log`. Historical data cannot be backfilled — tracking starts from the point you enable it.

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
```

## License

MIT
