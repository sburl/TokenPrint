use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::{net::SocketAddr, path::PathBuf, process::Command, sync::Arc, time::{SystemTime, UNIX_EPOCH}};
use tokio::sync::Mutex;

#[derive(Clone)]
struct AppState {
    tokenprint_cmd: String,
    output_path: PathBuf,
    no_cache: bool,
    since: Option<String>,
    until: Option<String>,
    refresh_lock: Arc<Mutex<()>>,
}

impl AppState {
    fn run_tokenprint(&self) -> Result<(), String> {
        let mut args = vec![
            "--no-open".to_string(),
            "--output".to_string(),
            self.output_path.to_string_lossy().to_string(),
        ];

        if self.no_cache {
            args.push("--no-cache".to_string());
        }
        if let Some(s) = &self.since {
            args.push("--since".to_string());
            args.push(s.clone());
        }
        if let Some(u) = &self.until {
            args.push("--until".to_string());
            args.push(u.clone());
        }

        let status = Command::new(&self.tokenprint_cmd)
            .args(args)
            .status()
            .map_err(|e| format!("failed to spawn tokenprint: {e}"))?;

        if status.success() {
            Ok(())
        } else {
            Err(format!("tokenprint exited with status: {status}"))
        }
    }
}

async fn index(State(state): State<AppState>) -> impl IntoResponse {
    match tokio::fs::read_to_string(&state.output_path).await {
        Ok(html) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/html; charset=utf-8")],
            html,
        )
            .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({"ok": false, "error": "dashboard_not_found"})),
        )
            .into_response(),
    }
}

async fn status() -> impl IntoResponse {
    Json(json!({"ok": true}))
}

async fn refresh(State(state): State<AppState>) -> impl IntoResponse {
    let _guard = state.refresh_lock.lock().await;

    match state.run_tokenprint() {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({"ok": true, "generatedAt": chrono_like_now()})),
        ),
        Err(err) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"ok": false, "error": err}))),
    }
}

fn chrono_like_now() -> String {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{epoch}")
}

#[tokio::main]
async fn main() {
    let host = std::env::var("TOKENPRINTD_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("TOKENPRINTD_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(8765);

    let output_path = std::env::var("TOKENPRINTD_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("tokenprint.html"));

    let state = AppState {
        tokenprint_cmd: std::env::var("TOKENPRINTD_TOKENPRINT_BIN").unwrap_or_else(|_| "tokenprint".to_string()),
        output_path,
        no_cache: std::env::var("TOKENPRINTD_NO_CACHE").ok().as_deref() == Some("1"),
        since: std::env::var("TOKENPRINTD_SINCE").ok(),
        until: std::env::var("TOKENPRINTD_UNTIL").ok(),
        refresh_lock: Arc::new(Mutex::new(())),
    };

    if let Err(err) = state.run_tokenprint() {
        eprintln!("initial tokenprint run failed: {err}");
        std::process::exit(1);
    }

    let app = Router::new()
        .route("/", get(index))
        .route("/index.html", get(index))
        .route("/api/status", get(status))
        .route("/api/refresh", post(refresh))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}").parse().expect("invalid bind address");
    println!("tokenprintd-rust listening at http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}
