# tokenprintd (Go)

**Created:** 2026-02-23-17-42
**Last Updated:** 2026-03-03-16-43

Local daemon for TokenPrint with live browser refresh support.

## What it does

- Runs an initial `tokenprint` collection on startup
- Serves the generated dashboard HTML
- Exposes a refresh API that reruns collection without restarting the daemon
- Uses `--live-mode` so the dashboard's refresh button calls the daemon API

## Endpoints

- `GET /` and `GET /index.html`: serve dashboard HTML
- `GET /api/status`: daemon status JSON
- `POST /api/refresh` (or custom path): trigger recollection

## Run

```bash
go run ./daemon/go --port 8765
```

The daemon normalizes most string inputs by trimming surrounding whitespace before startup, including:

- `--host`
- `--output`
- `--tokenprint-bin`
- `--refresh-path`
- `--since` / `--until`
- `--cache-path`
- `--refresh-token`
- `--gemini-log-path`

## Key flags

- `--host` bind host (default `127.0.0.1`; accepts loopback IPv6 like `::1` or `[::1]`)
- `--port` bind port (default `8765`)
- `--output` dashboard output path (default temp `tokenprint.html`)
- `--tokenprint-bin` tokenprint binary path (default `tokenprint`)
- `--since`, `--until` optional date bounds (`YYYYMMDD`)
- `--no-cache` force full recollection
- `--cache-path` pass through custom cache path to tokenprint (default cache location or TOKENPRINT_CACHE_PATH)
- `--refresh-path` custom refresh endpoint path (default `/api/refresh`)
- `--gemini-log-path` path to pass through to tokenprint as `--gemini-log-path`
- `--timeout` refresh timeout (default `10m`)
- `--refresh-token` shared token for refresh auth (`X-Tokenprint-Token`)  
  Required when binding to non-loopback hosts (e.g. `0.0.0.0`).
- `--no-open` do not auto-open browser

## Example with refresh token

```bash
go run ./daemon/go --refresh-token local-dev-secret
curl -X POST http://127.0.0.1:8765/api/refresh \
  -H 'X-Tokenprint-Token: local-dev-secret'
```

## Example with cache path

```bash
go run ./daemon/go --cache-path /tmp/tokenprint-cache-v1.json --refresh-token local-dev-secret
```

## Public host example

```bash
go run ./daemon/go --host 0.0.0.0 --port 8765 --refresh-token local-dev-secret
curl -X POST http://127.0.0.1:8765/api/refresh \
  -H 'X-Tokenprint-Token: local-dev-secret'
```

## IPv6 loopback example

```bash
go run ./daemon/go --host ::1 --port 8765
```


## Test

```bash
go test ./daemon/go
```
