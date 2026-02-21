# tokenprintd (Go prototype)

This is a prototype local daemon for TokenPrint.

## Endpoints
- `GET /` -> serves generated dashboard HTML
- `GET /api/status` -> health check
- `POST /api/refresh` -> reruns `tokenprint` and returns JSON

## Run (once Go is installed)

```bash
go run ./daemon/go/main.go --port 8765
```

Options:
- `--tokenprint`: path to tokenprint binary
- `--output`: output HTML path (default `/tmp/tokenprint.html` on macOS)
- `--no-cache`: pass `--no-cache` to tokenprint
- `--since`, `--until`: optional date filters

## Notes
- This is a spike, not yet integrated into the Python CLI.
- Next steps: add tests, auth token for refresh endpoint, and graceful shutdown.
