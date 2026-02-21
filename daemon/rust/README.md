# tokenprintd (Rust prototype)

This is a prototype local daemon for TokenPrint.

## Endpoints
- `GET /` -> serves generated dashboard HTML
- `GET /api/status` -> health check
- `POST /api/refresh` -> reruns `tokenprint` and returns JSON

## Run (once Rust is installed)

```bash
cd daemon/rust
cargo run
```

Configuration via env vars:
- `TOKENPRINTD_HOST` (default `127.0.0.1`)
- `TOKENPRINTD_PORT` (default `8765`)
- `TOKENPRINTD_OUTPUT` (default temp `tokenprint.html`)
- `TOKENPRINTD_TOKENPRINT_BIN` (default `tokenprint`)
- `TOKENPRINTD_NO_CACHE=1`
- `TOKENPRINTD_SINCE`, `TOKENPRINTD_UNTIL`

## Notes
- This is a spike, not yet integrated into the Python CLI.
- Prototype keeps dependencies minimal and skips polished time formatting.
