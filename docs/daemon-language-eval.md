# Daemon Language Evaluation: Go vs Rust

This compares a local TokenPrint daemon that serves:
- `GET /` dashboard HTML
- `GET /api/status`
- `POST /api/refresh` (runs `tokenprint --no-open --output ...`)

## Recommendation

Start with **Go** for the first production daemon.

Why:
- very fast to build and iterate for small HTTP services
- lower learning curve for first-time systems language users
- single static binary output and easy local distribution
- built-in stdlib HTTP server is enough for this use case

Use **Rust** if you later want stronger compile-time guarantees and are willing to pay a steeper learning curve.

## Practical Tradeoffs

### Go
- Pros:
  - simplest path from prototype to working daemon
  - great stdlib for HTTP/process management
  - fewer concepts to learn initially
  - quick compile times
- Cons:
  - weaker type-level safety than Rust
  - some runtime errors that Rust could catch at compile time

### Rust
- Pros:
  - strongest memory/thread safety guarantees
  - excellent reliability once code compiles
  - great performance profile
- Cons:
  - harder for first-time users (ownership/borrowing/lifetimes)
  - slower initial development
  - more setup/boilerplate for async HTTP stacks

## Suggested Path for You

1. Build and ship v1 daemon in Go.
2. Keep API contract stable (`/api/status`, `/api/refresh`).
3. If needed, re-implement in Rust later behind the same contract.

## Learning Plan (first week)

### Go
- Day 1: variables, structs, interfaces, errors
- Day 2: `net/http`, handlers, JSON encode/decode
- Day 3: `os/exec`, command timeouts, logs
- Day 4: config flags + graceful shutdown
- Day 5: tests with `httptest`

### Rust
- Day 1: ownership/borrowing + structs/enums
- Day 2: `Result`/`Option` + error handling
- Day 3: async basics (`tokio`) + simple endpoint
- Day 4: process spawning + state synchronization
- Day 5: integration tests

## What is Scaffolded in This Repo

- `daemon/go/main.go`: minimal Go daemon prototype
- `daemon/rust/Cargo.toml` + `daemon/rust/src/main.rs`: minimal Rust daemon prototype

Both expose the same HTTP contract and call `tokenprint` as a child process.
