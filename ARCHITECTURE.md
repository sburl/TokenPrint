# TokenPrint Architecture

**Created:** 2026-03-03-16-43
**Last Updated:** 2026-03-03-16-43

This document describes the current architecture and contracts between the core
Python CLI, local cache, and optional Go daemon.

## High-level architecture

1. **Collector phase (Python)** in `tokenprint/__init__.py`
   - Collects daily usage from each provider:
     - Claude via `ccusage`
     - Codex via `@ccusage/codex`
     - Gemini via local OpenTelemetry log parsing (`~/.gemini/telemetry.log`)
   - Each provider returns `Dict[str, Dict[str, Any]]` keyed by `YYYY-MM-DD`,
     where each row contains token counts and USD cost.
2. **Persistence phase (Python)**
   - Cache file is `tokenprint-provider-cache-v1.json` by default or
     `TOKENPRINT_CACHE_PATH`.
   - Cache schema version is normalized with migration hooks via
     `_extract_provider_payload`, `_coerce_cache_schema_version`,
     and `_migrate_provider_payload`.
3. **Merge phase (Python)**
   - `_collect_merged_usage_data` builds provider payloads and calls `merge_data`.
   - `merge_data` emits one row per date with provider-level tokens, costs, energy,
     carbon, and water summaries.
4. **Render phase (Python)**
   - `compute_dashboard_data` builds dashboard config consumed by template rendering.
   - `_render_html_from_config` injects serialized config into `template.html`.
5. **Daemon phase (Go, optional)**
   - `tokenprintd` hosts the HTML output and exposes `/api/refresh` (or custom path).
   - POST refresh triggers the Python CLI in configured mode and updates the same file.

## In-memory contracts

### Provider config contract

- Defined by `ProviderConfig` with:
  - `name`: internal key (`claude`, `codex`, `gemini`)
  - `collect_fn`: callable name in `tokenprint/__init__.py`
  - `label`: CLI output label
  - `rates`: `(input_rate, output_rate, cache_read_rate)` in USD/token
- All providers are expected to return daily maps keyed by `YYYY-MM-DD`.

### Daily usage row contract

Each provider row should be normalized to:

```python
{
  "provider": "claude|codex|gemini",
  "input_tokens": int,
  "output_tokens": int,
  "cache_read_tokens": int,
  "cache_write_tokens": int,
  "cost": float,
}
```

Missing/invalid values are coerced safely using `_safe_int` and `_safe_float`.

## Operational contracts

- CLI always collects with explicit date strings when provided:
  `--since` and `--until` are normalized to `YYYYMMDD`.
- Default run path is incremental cache mode (unless dates or `--no-cache` are set).
- Output supports HTML and JSON:
  - HTML (default): `--output` or `{tempdir}/tokenprint.html`
  - JSON: `--output` or `{tempdir}/tokenprint.json`
- Output writes are atomic via temp-file replace and path validation.

## Security posture

- Daemon host parsing is strict about loopback vs non-loopback hosts.
- Non-loopback hosts require `--refresh-token` and token checks on `/api/refresh`.
- Refresh endpoint requires same-origin `Origin` header or header-based API clients.
- Header token values are trimmed before comparison to reduce formatting false-negatives.

## Extension points

- Add a new provider by:
  1. Adding a `ProviderConfig` entry.
  2. Implementing a collector function returning normalized daily rows.
  3. Adding tests around malformed data normalization and collection failure handling.
