# TokenPrint Refactor Plan & PR Tracker

**Created:** 2026-03-03-16-43
**Last Updated:** 2026-03-03-16-43

Date created: 2026-03-03

This file tracks the full sequence of planned and executed PRs, and includes the staged roadmap requested in Step 4b.

## Split PR queue from monolith (current run)

The previous all-in-one PR was split into these tightly scoped PRs in this order:

- [chore: sync implemented PRs and repo hardening](https://github.com/sburl/TokenPrint/pull/7) — superseded and closed
- [docs: add architecture and planning documentation](https://github.com/sburl/TokenPrint/pull/8)
- [ci: add GitHub Actions workflow and guardrails](https://github.com/sburl/TokenPrint/pull/9)
- [refactor(go): simplify daemon command handling and tests](https://github.com/sburl/TokenPrint/pull/10)
- [refactor(python): modularize tokenprint providers and CLI entrypoint](https://github.com/sburl/TokenPrint/pull/11)

Current intended merge order: 8 -> 9 -> 10 -> 11.

## Operating Principles
- Small, tightly scoped PRs.
- No giant refactors in one PR.
- Every PR is internally reviewed, then externally reviewed with Gemini.
- The external review process (Gemini) must pass before merge.
- PRs should remain in `PR Queue` order unless explicitly reordered for blockers.

## Step 0 — Baseline
- Repository on latest remote main: ✅ `Already up to date`.
- Existing open PRs: 8→11 (`gh pr list --state open`).
- Existing CI: ✅ `.github/workflows/ci.yml` and helper scripts are present.

## PR Tracker (intended order)

| PR | Proposed Scope | Why this PR | Status | Dependencies | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | Add baseline planning/assessment docs | Creates the planning and feedback structure first | ✅ complete | none | Creates `user-questions-and-answers.md` + `plan-and-pr-tracking.md` |
| 1 | CI foundation + quality gates | Add `.github/workflows/ci.yml` with unit/integration split + nightly job shell | ✅ ready | none | Added `.github/workflows/ci.yml` with Python/Go PR checks and nightly extended checks |
| 2 | Accuracy/robustness hardening | Expand unit tests for parsing/date normalization and metric correctness edges | ✅ ready | 1 | Includes cache-write token energy math and Claude date-format parsing |
| 3 | Provider collection audit | Improve provider parsing robustness + explicit invariants + tests | ✅ ready | 2 | Added guardrails for non-dict collector rows + malformed-entry test coverage |
| 4 | Cache strategy cleanup | Make cache read/write versioned/migrated with invariant checks | ✅ ready | 3 | Added cache-version extraction, migration guardrails, and schema warning/repair paths |
| 5 | Security hardening (daemon) | Refresh endpoint auth/CORS policy and hardened defaults | ✅ ready | 3,4 | Enforced refresh-token requirement for non-loopback host bindings + loopback host helper tests |
| 6 | Config/refactor isolation | Pull constants and behavior into modules for maintainability | ✅ ready | 4 | Added environment-based cache-path resolution and path helper precedence for CLI/env fallback |
| 7 | Error-handling + messaging pass | Improve observability with consistent exceptions and user messages | ✅ ready | 6 | Added collection exception isolation, cache write/write-path messaging, and safer HTML writing |
| 8 | CLI quality pass | Add `--version` and `--check` preflight behavior for confidence | ✅ ready | 7 | Added explicit version output, preflight checks, and safer startup with deterministic exit behavior |
| 9 | Data output format | Add `--output-format json|html` and JSON export path | ✅ ready | 8 | Added JSON export mode with default JSON path and JSON file write helper |
| 10 | Version source-of-truth | Derive `--version` from installed package metadata | ✅ ready | 9 | Replaced hardcoded version constant with `importlib.metadata` lookup + safe fallback |
| 11 | Version lookup testability | Make version lookup runtime-callable and test `--version` path deterministically | ✅ ready | 10 | Wrapped version resolution in helper and added regression test for argparse output |
| 12 | Defensive merge path | Guard merge pipeline against malformed provider rows | ✅ ready | 11 | Make `merge_data` tolerate non-dict/non-numeric row values without crashing |
| 13 | Collector numeric coercion | Coerce provider collector token/cost fields safely to prevent malformed provider JSON crashes | ✅ ready | 12 | Add `_safe_int`/`_safe_float` usage in collectors and tests for malformed daily rows |
| 14 | Gemini collector hardening | Guard malformed Gemini telemetry records/timestamps before parser access | ✅ ready | 13 | Skip non-dict log records, coerce timestamp values defensively, and add regressions |
| 15 | Gemini timestamp parsing | Accept epoch-second and epoch-millisecond timestamps in telemetry parser | ✅ ready | 14 | Added `_parse_gemini_timestamp` and coverage for both second and millis formats |
| 16 | Gemini epoch resolution | Support nanosecond timestamps and keep timestamp parsing robust for numeric epochs | ✅ ready | 15 | Expand parser to normalize seconds/millis/nanos and add regression coverage |
| 17 | Gemini timestamp normalization precision | Add microsecond timestamp support and robust numeric string handling | ✅ ready | 16 | Prevent microsecond/nanosecond misclassification and cover edge numeric timestamp formats |
| 18 | Gemini timestamp guardrails | Reject boolean-like timestamps and support scientific-notation numeric timestamps | ✅ ready | 17 | Harden timestamp parser against invalid edge-cases and add targeted tests |
| 19 | Documentation consistency for output and cache options | Align docs with actual defaults and cache override behavior | ✅ ready | 18 | Remove mismatches in CLI docs for output paths and cache flags |
| 20 | Simplify Gemini timestamp path in collector | Remove duplicate manual parsing and add collector-level scientific-notation coverage for `time` key | ✅ ready | 19 | Keep timestamp normalization in one place and validate end-to-end behavior |
| 21 | Normalize Gemini attribute payload handling | Share parsing for dict/list attribute encodings and add regression coverage | ✅ ready | 20 | Parse `intValue`/`Int64Value`/`stringValue` safely for list-shaped attributes |
| 22 | Centralize Gemini pricing constants | Extract repeated Gemini rate literals into named constants for readability and reviewability | ✅ ready | 21 | Avoid magic numbers in token cost calculation |
| 23 | Gemini provider config rate consistency | Keep `ProviderConfig` and collector math using the same constants | ✅ ready | 22 | Prevent drift between pricing constants and provider metadata |
| 24 | Codex pricing constants consistency | Keep Codex rates in one source-of-truth and consume via provider config | ✅ ready | 23 | Prevent Codex rate drift between config metadata and collection math |
| 25 | Daemon hardening for public host bindings | Force auth token requirement for non-loopback hosts on refresh endpoint | ✅ ready | 5 | Hardens remote refresh attacks while preserving local loopback convenience |
| 26 | Gemini dict attribute normalization | Parse nested `intValue`/`Int64Value`/`stringValue` payloads in dict-style Gemini attrs | ✅ ready | 25 | Handles additional Gemini telemetry shapes without dropping counters |
| 27 | Daemon security docs and flag clarity | Clarify non-loopback token requirement in CLI/docs and remove ambiguity for users | ✅ ready | 25 | Keeps security model discoverable and auditable |
| 28 | Cache-path resolution tests | Add focused tests for `_resolve_cache_path` directory/file and tilde behavior | ✅ ready | 6 | Locks in precedence and normalization for cache path handling |
| 29 | Loopback host matching hardening | Trim/normalize loopback host input for consistent security checks | ✅ ready | 25 | Reduces accidental non-loopback false positives from formatting variants |
| 30 | Daemon IPv6 listen and public URL hardening | Build valid IPv6 bind URLs via net.JoinHostPort for daemon listener and browser URL | ✅ ready | 29 | Prevents invalid address binding/launch URL formatting for `::1` and bracketed IPv6 hosts |
| 31 | Daemon host normalization in config defaults | Trim configured host in `normalizeConfig` before binding/validation | ✅ ready | 30 | Prevents accidental misbinding and keeps IPv6/whitespace hosts stable |
| 32 | Daemon open-browser wildcard host fallback | Open daemon UI on loopback when bound to wildcard hosts (`0.0.0.0`, `::`) | ✅ ready | 31 | Keeps browser launch usable without guessing host syntax |
| 33 | Daemon port normalization to safe range | Clamp invalid or out-of-range `--port` values to a safe default | ✅ ready | 32 | Prevents invalid listen addresses and confusing startup failures |
| 34 | Daemon host normalization edge-case tests | Add explicit tests for zero port, bracketed host trimming, and public URL host edge cases | ✅ ready | 33 | Locks in config normalization invariants for loopback/boundary inputs |
| 35 | CLI helper decomposition and preflight test coverage | De-duplicate output writers and harden `--check` by adding explicit preflight tests | ✅ ready | 34 | Small maintainability + reliability pass in Python CLI checks/writers |
| 36 | Compact date parsing fallback | Support `YYYYMMDD` provider date inputs in `_parse_date_flexible` with normalization + tests | ✅ ready | 35 | Expands date compatibility while keeping strict validation |
| 37 | CLI date arg normalization | Normalize `--since`/`--until` using shared parser for compact and ISO input | ✅ ready | 36 | Improves CLI ergonomics while keeping collector semantics stable |
| 38 | CLI date format docs | Update README usage/examples and flag reference to document new date inputs | ✅ ready | 37 | Keeps documentation aligned with current CLI behavior |
| 39 | CrossCheck usage docs for JSON and ISO dates | Update CrossCheck examples to show new format support and JSON output mode | ✅ ready | 38 | Ensures integration docs match command behavior |
| 40 | Strict CLI date format parsing | Restrict CLI date args to compact/ISO only to match help text and avoid ambiguity | ✅ ready | 39 | Prevents accidental acceptance of unsupported date formats at entrypoint |
| 41 | Whitespace-tolerant CLI date handling tests | Add explicit test coverage that surrounding whitespace is accepted for `--since`/`--until` | ✅ ready | 40 | Locks in user-facing resilience for manual date entry |
| 42 | Normalize cached path whitespace | Trim and normalize cache-path env/flag values before resolving file path | ✅ ready | 41 | Prevents hard-to-debug failures from accidental whitespace |
| 43 | Refresh command documentation header | Keep file/module usage examples aligned with documented date formats | ✅ ready | 42 | Prevents stale embedded CLI guidance from diverging from runtime behavior |
| 44 | Cache path trimming test for CLI flag | Add regression test asserting whitespace in `--cache-path` is ignored | ✅ ready | 42 | Confirms CLI whitespace normalization mirrors env behavior |
| 45 | Whitespace-only CLI date rejection | Add regression test asserting blank/space-only `--since`/`--until` fails fast | ✅ ready | 40 | Prevents ambiguous input from reaching collector layer |
| 46 | Blank cache path flag defers to env var | Ensure whitespace-only `--cache-path` falls back to `TOKENPRINT_CACHE_PATH` | ✅ ready | 42 | Prevents accidental overrides from malformed CLI input |
| 47 | Clarify invalid date error messaging | Surface accepted date format directly in `argparse` validation error | ✅ ready | 40 | Improves CLI diagnostics for user typo recovery |
| 48 | Harden numeric coercion against booleans | Treat boolean fields as invalid numeric values when parsing token/cost inputs | ✅ ready | 47 | Prevents silent boolean -> 1/0 coercion artifacts in malformed payloads |
| 49 | Add boolean payload regression tests | Add provider/merge tests for bool numeric fields to lock in coercion behavior | ✅ ready | 48 | Ensures boolean field handling is permanently covered by tests |
| 50 | Normalize non-finite numeric values | Treat inf/nan cost inputs as invalid numeric payload values | ✅ ready | 49 | Prevents silent propagation of invalid floats into aggregates |
| 51 | Tighten Gemini timestamp edge tests | Add tests for ms-scale timestamps and non-finite values | ✅ ready | 50 | Improves resilience for malformed telemetry timestamp payloads |
| 52 | Harden _safe_int against non-finite values | Normalize int coercion for malformed numeric payloads that can crash collectors | ✅ ready | 51 | Prevents malformed payloads (inf/nan) from raising OverflowError during token parsing |
| 53 | Extract constants module | Move shared pricing and model constants to `tokenprint/constants.py` | ✅ ready | 52 | Keeps core logic focused and avoids constant sprawl |
| 54 | Centralize Claude pricing constants | Move Claude rate constants/model prefixes/multipliers into `constants.py` and reuse in estimates | ✅ ready | 53 | Avoids further duplication in `_estimate_claude_model_cost` and provider config |
| 55 | Extract daemon defaults into named constants | Consolidate duplicated daemon configuration literals in `daemon/go/main.go` and update tests to assert against shared defaults | ✅ complete | 54 | Keeps CLI/runtime defaults consistent across code and tests |
| 56 | Trim and normalize daemon config inputs | Make `normalizeConfig` trim whitespace for configurable string fields and add regression coverage | ✅ complete | 55 | Prevents malformed CLI values (spaces) from leaking into runner and endpoint behavior |
| 57 | Normalize daemon date/token fields in config | Trim and normalize `since`, `until`, and `refresh-token` so daemon starts predictably with whitespace input | ✅ complete | 56 | Keeps runner arguments stable and prevents accidental auth/header mismatches |
| 58 | Centralize daemon loopback host list | Reuse a single loopback host list for host checks and CORS-like origin allowlist checks | ✅ complete | 57 | Reduces host policy divergence between token checks and origin policy |
| 59 | Document daemon config input normalization | Update daemon README to describe whitespace trimming and token-host requirements | ✅ complete | 58 | Makes config behavior explicit for operators and reduces onboarding confusion |
| 60 | Harden cache cost normalization from disk | Normalize provider cache `cost` through shared safe coercion to reject bool/non-finite values | ✅ complete | 59 | Prevents malformed cache payloads from silently converting to non-zero costs |
| 61 | Make command execution robust to OSError in subprocess calls | Treat command-launch OSErrors as unavailable command and keep collection paths resilient | ✅ complete | 60 | Prevents collectors from failing unexpectedly on runtime process launch issues |
| 62 | Accept numeric cache schema versions expressed as strings | Parse numeric string `version` fields when loading cache payloads | ✅ complete | 61 | Preserves compatibility with manually-edited caches and avoids unnecessary invalidation |
| 63 | Accept whole-number cache schema versions expressed as floats | Parse float `version` values when they represent whole numbers (e.g., `2.0`) | ✅ complete | 62 | Improves tolerance for manually generated JSON cache files |
| 64 | Normalize cache schema version for legacy payloads | Apply version coercion consistently to legacy/no-wrapper cache payload formats | ✅ complete | 63 | Keeps provider cache format compatibility across versions and payload shapes |
| 65 | Accept integral version strings in string form (e.g., `2.0`) | Handle cache payload `version` strings that parse as whole-number floats | ✅ complete | 64 | Reduces accidental cache invalidation for JSON float-string encodings |
| 66 | Harden cache schema version coercion with helper and edge-case tests | Centralize version normalization and reject null/bool/NaN/Inf versions | ✅ complete | 65 | Makes cache-version handling explicit and easier to maintain |
| 67 | Use explicit UTF-8 file I/O for cache/template/output reads and writes | Ensure cache/template and output artifacts are written/read consistently across platforms | ✅ complete | 66 | Keeps persisted artifacts reliable with predictable encoding |
| 68 | Use atomic file writes for cache and output artifacts | Reduce partial-write risk during interrupted runs by writing via temporary then replace | ✅ complete | 67 | Improves reliability and data integrity for generated files |
| 69 | Validate atomic write cleanup on replace failure | Add regression coverage that temp file cleanup occurs when atomic replacement fails | ✅ complete | 68 | Keeps temporary write paths from lingering on write errors |
| 70 | Add cache-write failure cleanup regression | Ensure cache persistence has the same temp-file cleanup guarantees as output writes | ✅ complete | 69 | Keeps partial cache writes from leaving temp artifacts |
| 71 | Default JSON output path regression test | Assert `--output-format json` defaults to tokenprint.json in temp dir | ✅ complete | 70 | Prevents regressions in output path selection for JSON mode |
| 72 | Harden atomic writer with fsync and coverage | Ensure atomic writes flush before replace and are regression-tested | ✅ complete | 71 | Improves durability against sudden process termination during write |
| 73 | Verify cache writes create nested parent directories | Confirm `_save_provider_cache` can persist cache into new nested directories | ✅ complete | 72 | Guards deployment setups where cache path points into deep path trees |
| 74 | Add direct atomic writer helper coverage | Test `_write_text_atomic` parent creation and fsync behavior directly | ✅ complete | 73 | Ensures atomic durability semantics are protected independent of wrappers |
| 75 | Simplify exception fallbacks in username/timestamp parsers | Remove implicit fall-through `pass` blocks with explicit return paths | ✅ complete | 74 | Makes control flow clearer for `detect_github_username` and timestamp parsing |
| 76 | Add default HTML output path regression | Verify default HTML output path uses `tokenprint.html` under temp dir | ✅ complete | 75 | Prevents future regressions in default CLI output behavior |
| 77 | Remove silent exception pass in atomic cleanup | Replace cleanup `pass` with explicit `suppress(OSError)` for readability | ✅ complete | 76 | Tightens atomic-write control-flow and avoids empty `except` blocks |
| 78 | Add parser edge-case test for non-numeric Gemini timestamp strings | Ensure `_parse_gemini_timestamp` returns `None` for non-numeric string input | ✅ complete | 77 | Closes a parser edge-case gap for malformed telemetry |
| 79 | Improve cache write failure diagnostics | Include path and error details when provider cache persistence fails | ✅ complete | 78 | Makes on-disk failures observable without debugging cache write code paths |
| 80 | Add invalid output-format CLI regression | Ensure argparse rejects unsupported `--output-format` values | ✅ complete | 79 | Prevents silent/ambiguous behavior for malformed format flags |
| 81 | Add explicit output format runtime guard | Fail fast on unsupported `--output-format` values in `main` | ✅ complete | 80 | Prevents malformed output formats from slipping through parser changes |
| 82 | Normalize CLI output path whitespace | Treat whitespace-only `--output` as unset and fall back to default path | ✅ complete | 81 | Improves CLI resilience to accidental whitespace input |
| 83 | Parse ISO strings in Gemini timestamp parser | Support ISO-8601 timestamp strings (including Z and offsets) before numeric parsing | ✅ complete | 82 | Fixes real telemetry ingestion path and aligns with existing Gemini log fixture formats |
| 84 | Remove redundant output-format guard | Rely on `argparse` choices and drop duplicate runtime validation | ✅ complete | 83 | Small control-flow cleanup reducing duplicated validation |
| 85 | Normalize JSON output path whitespace | Treat whitespace-only `--output` as default for JSON output mode | ✅ complete | 84 | Keeps CLI output defaults consistent across formats |
| 86 | Add CLI whitespace guard for `--until` | Ensure whitespace-only `--until` is rejected like whitespace-only `--since` | ✅ complete | 85 | Closes asymmetric date argument validation edge case |
| 87 | Validate atomic writer cleanup on fsync error | Ensure temp file is removed when fsync raises during write | ✅ complete | 86 | Strengthens durability guarantees for interrupted writes |
| 88 | Reject directory output targets early | Provide clearer failure for `--output`/JSON path that points to an existing directory | ✅ complete | 87 | Keeps CLI writes from entering replace-stage errors and gives actionable message |
| 89 | Normalize daemon-style refresh config in CLI main args | Trim whitespace from `--refresh-endpoint` and `--refresh-token` before rendering live-mode config | ✅ complete | 88 | Improves reliability of live-mode refresh configuration and prevents whitespace-only token leakage |
| 90 | Trim request token header for daemon refresh endpoint | Trim whitespace around `X-Tokenprint-Token` before comparison | ✅ complete | 89 | Improves robustness against header formatting noise without changing token semantics |
| 91 | Consolidate provider collection fallback handling | Centralize provider collector try/except/type-shape handling for full and incremental collection paths | ✅ ready | 90 | Removes duplication and keeps collector-failure behavior consistent |
| 92 | Add `ARCHITECTURE.md` with component contracts | Document current architecture and data flow across collectors, merge, template, and daemon | ✅ ready | 91 | Improves onboarding and supports maintainability planning |
| 93 | Link architecture docs from user-facing docs | Add `ARCHITECTURE.md` reference to README architecture section | ✅ ready | 92 | Helps users and contributors discover component contracts quickly |
| 94 | Add regression tests for provider collection fallback edge cases | Add explicit coverage for non-dict and exception-based collector payload failures in full and incremental flows | ✅ ready | 93 | Prevents bad collector returns from causing merge-time surprises |
| 95 | Harden provider collection fallback tests (invalid payloads + warning paths) | Extend unit coverage for `_collect_days_with_fallback` warnings and mixed incremental failure modes | ✅ ready | 94 | Confirms bad collector/cache payloads stay isolated and observable |
| 96 | Add nightly dependency integrity checks to CI | Add `pip check` and `go mod verify` under scheduled CI to catch environment drift | ✅ ready | 95 | Improves release reliability by detecting dependency/pin breakage early |
| 97 | Add nightly vulnerability/secret scan steps in CI | Add `pip-audit`, `govulncheck`, and lightweight secret-pattern scan to scheduled checks | ✅ ready | 96 | Strengthens repository security posture with proactive early-warning checks |
| 98 | Encapsulate secret scanning in CI helper script | Add reusable `.github/scripts/ci-secret-scan.sh` and wire it into nightly workflow | ✅ ready | 97 | Keeps security tooling centralized and easier to adjust |
| 99 | Add shell script syntax checks to CI | Add reusable `.github/scripts/ci-shell-syntax-check.sh` and run in both PR and nightly workflows | ✅ ready | 98 | Expands CI guardrails with zero-dependency script linting |
| 100 | Extend nightly Go checks with static analysis | Add `staticcheck` install+run in nightly `go` workflow step | ✅ ready | 99 | Catches static lint issues beyond `go vet` before release windows |
| 101 | Add timezone-aware Gemini bucketing | Add `--timezone`, pass it through collection paths, and add parser/CLI/date-range coverage | ✅ ready | 100 | Makes Gemini daily grouping explicit and configurable by local day boundary |
| 102 | Make Gemini telemetry log path configurable via environment | Add `TOKENPRINT_GEMINI_TELEMETRY_LOG_PATH` path resolution (trim, expanduser, directory support) and test coverage | ✅ ready | 101 | Keeps telemetry ingestion adaptable across install paths and CI/test fixtures without code changes |
| 103 | Add CLI override for Gemini telemetry log path | Add `--gemini-log-path` and thread it through collector invocation | ✅ ready | 102 | Supports explicit operational overrides without requiring env vars |
| 104 | Propagate Gemini log path from daemon to tokenprint | Add `tokenprintd --gemini-log-path` and tests to pass through to tokenprint command args | ✅ ready | 103 | Keeps daemon/live dashboard and collector overrides aligned |
| 105 | Pass cache path through daemon to tokenprint | Add `tokenprintd --cache-path` and build helper coverage for propagated tokenprint args | ✅ ready | 104 | Makes daemon cache behavior explicit and consistent across direct and daemon-run collection |
| 106 | Tighten daemon public-host token enforcement | Reject empty/whitespace token headers for required public-host auth paths | ✅ ready | 105 | Prevents ambiguous auth bypass when public host has non-empty refresh requirement intent |
| 107 | Add module invocation path `python -m tokenprint` | Add `tokenprint/__main__.py` and tests for module execution | ✅ ready | 106 | Supports canonical module execution in scripting and build environments |
| 108 | Extract provider registry to dedicated module | Move `ProviderConfig` and `PROVIDERS` out of `tokenprint/__init__.py` into `tokenprint/providers.py` | ✅ ready | 107 | Reduces monolith size and clarifies provider extensibility boundary |
| 109 | Add provider lookup helper in dedicated registry | Add `provider_by_name()` in `tokenprint/providers.py` and use it in rate resolution | ✅ ready | 108 | Removes repetitive provider lookup logic in core loops |
| 110 | Centralize provider-name set creation in registry | Add `provider_name_set()` to `tokenprint/providers.py` and reuse it in cache payload extraction | ✅ ready | 109 | Keeps provider-name derivations in one place for future registry changes |
| 111 | Optimize provider lookup with indexed mapping | Add internal `_PROVIDER_BY_NAME` mapping for O(1) provider lookups | ✅ ready | 110 | Prepares registry access for future plugin lookups and avoids repeated linear scans |
| 112 | Make provider-name set cached and immutable | Add cached `_PROVIDER_NAME_SET` to avoid recomputation in repeated lookups | ✅ ready | 111 | Improves provider-name derivation performance and predictability |
| 113 | Add provider lookup by compact key | Add `provider_by_key()` and internal key-index map for future key-aware rendering paths | ✅ ready | 112 | Encapsulates compact key resolution logic in provider registry |
| 114 | Add ordered provider name helper | Add `provider_names()` for explicit registry order access from callers/tests | ✅ ready | 113 | Keeps provider ordering semantics in one location and avoids repeated list comprehensions |
| 115 | Add provider resolution helper | Add `resolve_provider()` to look up by name or compact key from a single call site | ✅ ready | 114 | Reduces call-site branching when a caller accepts both provider identifier forms |
| 116 | Use unified provider resolver in core rate lookup | Switch `_rates_for_provider` to call `resolve_provider` and remove name-only assumptions | ✅ ready | 115 | Keeps future caller usage consistent with centralized provider lookup semantics |
| 117 | Use provider name helper in provider-only loops | Replace provider-name list comprehensions in cache, merge, and dashboard helpers with `provider_names()` | ✅ ready | 116 | Centralizes iteration order and keeps cache/dashboard behavior aligned with registry metadata |

## 3-pass review protocol (for each PR)
1. Internal review of changed files and invariants.
2. Gemini review loop:
   - Run the CrossCheck-compatible Gemini review process.
   - If feedback indicates issues, patch and repeat until explicit “ready to merge” signal.
3. Re-open the PR tracker entry and only mark as `ready` after both internal + Gemini checks are satisfied.

## Stage planning (4b-style plan creation)
No existing do-work queue was found in-repo, so I am creating one now.

### A) Improvements (10)
1. Split `tokenprint/__init__.py` into modules (`cli.py`, `collectors.py`, `models.py`, `render.py`) while preserving API.
2. Add explicit typed model/DTOs for provider rows.
3. Replace inline magic constants with named config groups.
4. Add helper for date parsing with strict validation and telemetry warnings.
5. Add cache schema version migrations and migration tests.
6. Add graceful handling for partial/corrupt logs and partial provider data.
7. Add CLI `--version` and `--check` modes.
8. Add `--output-format json|html` for machine-readable exports.
9. Add `--timezone` support for rendering and filters.
10. Normalize cache read/write path via env var or `--cache-path` flag.

### B) Big Vision (10)
1. Add provider plugin interface to add future providers without touching core loop.
2. Add provider registry JSON schema and provider capability metadata.
3. Add interactive CLI wizard for first-run setup and onboarding checks.
4. Add trend and anomaly summary report generation (text and HTML summary cards).
5. Add historical smoothing / rolling average charts.
6. Add export formats for CSV and JSON snapshots.
7. Add timezone-aware date-range semantics and locale-safe rendering.
8. Add user profile persistence for display name and default filters.
9. Add optional storage backend abstraction (local + optional cloud sync stub).
10. Add offline mode to avoid blocking rendering when one provider is unavailable.

### C) Maintenance (10)
1. Add `.github/workflows` with fast + nightly matrix and artifacting.
2. Enforce formatting and lint gates for both Python and Go.
3. Add mutation/invariant tests for provider merge math.
4. Add integration test harness for `tokenprint` file generation path.
5. Add go test coverage for handler error/status transitions in daemon.
6. Add fixture-based tests for template rendering and script integration.
7. Add security scan command integration in CI (`scan-secrets`, dependency checks).
8. Add signed changelog and release notes format.
9. Add architecture notes in `ARCHITECTURE.md` with component contracts.
10. Add dead-code sweep report process and monthly cleanup PR requirement.

## Stage group constraints
- Minimum tasks per stage: 5
- Maximum tasks per stage: 15
- PRs remain single-topic and scoped.
- Every completed stage should re-run Step 3 (bug/quality/security/CI/redundancy cycles) before moving forward.

## Stage sequencing with your requested pass loops
For each numbered stage below, we complete:
1) first pass implementation,
2) second pass verification/review,
3) third pass cleanup/compaction.

### Stage 1: Baseline quality and safety hardening
- Execute PRs 1-3
- Then rerun Step 3 passes

### Stage 2: Data correctness and cache reliability
- Execute PRs 4-6
- Then rerun Step 3 passes

### Stage 3: Refactor and maintainability
- Execute PRs 6-7
- Then rerun Step 3 passes

### Stage 4: Feature expansion from queue
- Execute PRs from A/B/C as chosen order, grouped into ≤15 task stages
- Then rerun Step 3 passes

### Stage 5: Final reassessment
- Re-run Step 4b queue creation once all stages are complete
- Freeze with roadmap/maintenance commitments for next cycle

## Current stage status
- **Current:** Split PR queue now active
- **Next PR:** #8, then #9, then #10, then #11
