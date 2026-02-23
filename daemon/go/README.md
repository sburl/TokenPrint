# tokenprintd (Go)

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

## Key flags

- `--host` bind host (default `127.0.0.1`)
- `--port` bind port (default `8765`)
- `--output` dashboard output path (default temp `tokenprint.html`)
- `--tokenprint-bin` tokenprint binary path (default `tokenprint`)
- `--since`, `--until` optional date bounds (`YYYYMMDD`)
- `--no-cache` force full recollection
- `--refresh-path` custom refresh endpoint path (default `/api/refresh`)
- `--timeout` refresh timeout (default `10m`)
- `--refresh-token` optional shared token for refresh auth (`X-Tokenprint-Token`)
- `--no-open` do not auto-open browser

## Example with refresh token

```bash
go run ./daemon/go --refresh-token local-dev-secret
curl -X POST http://127.0.0.1:8765/api/refresh \
  -H 'X-Tokenprint-Token: local-dev-secret'
```

## Test

```bash
go test ./daemon/go
```
