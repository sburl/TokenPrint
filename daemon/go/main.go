package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Config controls daemon behavior.
type Config struct {
	Host           string
	Port           int
	OutputPath     string
	TokenprintBin  string
	NoCache        bool
	Since          string
	Until          string
	RefreshPath    string
	RefreshTimeout time.Duration
	RefreshToken   string
	NoOpen         bool
}

// App serves dashboard content and refresh endpoints.
type App struct {
	cfg          Config
	runCollector func(context.Context) error

	mu            sync.Mutex
	refreshing    bool
	lastGenerated string
	lastError     string
}

func defaultConfig() Config {
	return Config{
		Host:           "127.0.0.1",
		Port:           8765,
		OutputPath:     filepath.Join(os.TempDir(), "tokenprint.html"),
		TokenprintBin:  "tokenprint",
		RefreshPath:    "/api/refresh",
		RefreshTimeout: 10 * time.Minute,
	}
}

func normalizeConfig(cfg Config) Config {
	if cfg.Host == "" {
		cfg.Host = "127.0.0.1"
	}
	if cfg.Port <= 0 {
		cfg.Port = 8765
	}
	if cfg.OutputPath == "" {
		cfg.OutputPath = filepath.Join(os.TempDir(), "tokenprint.html")
	}
	if cfg.TokenprintBin == "" {
		cfg.TokenprintBin = "tokenprint"
	}
	if cfg.RefreshPath == "" {
		cfg.RefreshPath = "/api/refresh"
	}
	if !strings.HasPrefix(cfg.RefreshPath, "/") {
		cfg.RefreshPath = "/" + cfg.RefreshPath
	}
	if cfg.RefreshTimeout <= 0 {
		cfg.RefreshTimeout = 10 * time.Minute
	}
	return cfg
}

func buildRunner(cfg Config) func(context.Context) error {
	cfg = normalizeConfig(cfg)
	return func(ctx context.Context) error {
		args := []string{"--no-open", "--output", cfg.OutputPath, "--live-mode", "--refresh-endpoint", cfg.RefreshPath}
		if cfg.NoCache {
			args = append(args, "--no-cache")
		}
		if cfg.Since != "" {
			args = append(args, "--since", cfg.Since)
		}
		if cfg.Until != "" {
			args = append(args, "--until", cfg.Until)
		}

		cmd := exec.CommandContext(ctx, cfg.TokenprintBin, args...)
		cmd.Stdout = os.Stdout
		var stderr bytes.Buffer
		cmd.Stderr = io.MultiWriter(os.Stderr, &stderr)

		err := cmd.Run()
		if err == nil {
			return nil
		}
		msg := strings.TrimSpace(stderr.String())
		if msg != "" {
			return fmt.Errorf("tokenprint failed: %w (%s)", err, msg)
		}
		return fmt.Errorf("tokenprint failed: %w", err)
	}
}

func newApp(cfg Config, runner func(context.Context) error) *App {
	cfg = normalizeConfig(cfg)
	if runner == nil {
		runner = buildRunner(cfg)
	}
	return &App{cfg: cfg, runCollector: runner}
}

func (a *App) refreshStart() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.refreshing {
		return false
	}
	a.refreshing = true
	return true
}

func (a *App) refreshFinish(err error, generatedAt string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.refreshing = false
	if err != nil {
		a.lastError = err.Error()
		return
	}
	a.lastError = ""
	a.lastGenerated = generatedAt
}

func (a *App) statusSnapshot() (bool, string, string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.refreshing, a.lastGenerated, a.lastError
}

func (a *App) warmup(ctx context.Context) error {
	if err := a.runCollector(ctx); err != nil {
		a.refreshFinish(err, "")
		return err
	}
	a.refreshFinish(nil, time.Now().Format("01/02/2006, 03:04 PM"))
	return nil
}

func writeJSON(w http.ResponseWriter, status int, payload map[string]any) {
	raw, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, "json_marshal_error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_, _ = w.Write(raw)
}

func (a *App) statusHandler(w http.ResponseWriter, _ *http.Request) {
	refreshing, generatedAt, lastError := a.statusSnapshot()
	resp := map[string]any{
		"ok":         true,
		"refreshing": refreshing,
	}
	if generatedAt != "" {
		resp["generatedAt"] = generatedAt
	}
	if lastError != "" {
		resp["lastError"] = lastError
	}
	writeJSON(w, http.StatusOK, resp)
}

func (a *App) indexHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method_not_allowed"})
		return
	}
	f, err := os.Open(a.cfg.OutputPath)
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]any{"ok": false, "error": "dashboard_not_found"})
		return
	}
	defer f.Close()

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method == http.MethodHead {
		w.WriteHeader(http.StatusOK)
		return
	}
	_, _ = io.Copy(w, f)
}

func (a *App) refreshHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method_not_allowed"})
		return
	}
	if a.cfg.RefreshToken != "" {
		tok := r.Header.Get("X-Tokenprint-Token")
		if tok != a.cfg.RefreshToken {
			writeJSON(w, http.StatusUnauthorized, map[string]any{"ok": false, "error": "unauthorized"})
			return
		}
	}

	if !a.refreshStart() {
		writeJSON(w, http.StatusConflict, map[string]any{"ok": false, "error": "refresh_in_progress"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), a.cfg.RefreshTimeout)
	defer cancel()

	err := a.runCollector(ctx)
	if err != nil {
		a.refreshFinish(err, "")
		status := http.StatusInternalServerError
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			status = http.StatusGatewayTimeout
		}
		writeJSON(w, status, map[string]any{"ok": false, "error": err.Error()})
		return
	}

	generatedAt := time.Now().Format("01/02/2006, 03:04 PM")
	a.refreshFinish(nil, generatedAt)
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "generatedAt": generatedAt})
}

func (a *App) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", a.indexHandler)
	mux.HandleFunc("/index.html", a.indexHandler)
	mux.HandleFunc("/api/status", a.statusHandler)
	mux.HandleFunc(a.cfg.RefreshPath, a.refreshHandler)
	return mux
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "linux":
		cmd = exec.Command("xdg-open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		return
	}
	_ = cmd.Start()
}

func main() {
	cfg := defaultConfig()
	flag.StringVar(&cfg.Host, "host", cfg.Host, "Bind host")
	flag.IntVar(&cfg.Port, "port", cfg.Port, "Bind port")
	flag.StringVar(&cfg.OutputPath, "output", cfg.OutputPath, "Dashboard output path")
	flag.StringVar(&cfg.TokenprintBin, "tokenprint-bin", cfg.TokenprintBin, "tokenprint binary path")
	flag.BoolVar(&cfg.NoCache, "no-cache", false, "Pass --no-cache to tokenprint")
	flag.StringVar(&cfg.Since, "since", "", "Optional since (YYYYMMDD)")
	flag.StringVar(&cfg.Until, "until", "", "Optional until (YYYYMMDD)")
	flag.StringVar(&cfg.RefreshPath, "refresh-path", cfg.RefreshPath, "Refresh API path")
	flag.DurationVar(&cfg.RefreshTimeout, "timeout", cfg.RefreshTimeout, "Refresh timeout")
	flag.StringVar(&cfg.RefreshToken, "refresh-token", "", "Optional shared token required for POST refresh")
	flag.BoolVar(&cfg.NoOpen, "no-open", false, "Do not auto-open browser")
	flag.Parse()

	cfg = normalizeConfig(cfg)
	app := newApp(cfg, nil)

	warmCtx, warmCancel := context.WithTimeout(context.Background(), cfg.RefreshTimeout)
	defer warmCancel()
	if err := app.warmup(warmCtx); err != nil {
		log.Fatalf("initial tokenprint run failed: %v", err)
	}

	openHost := cfg.Host
	if openHost == "0.0.0.0" {
		openHost = "127.0.0.1"
	}
	url := fmt.Sprintf("http://%s:%d", openHost, cfg.Port)
	addr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)

	httpServer := &http.Server{
		Addr:              addr,
		Handler:           app.handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	stopCh := make(chan os.Signal, 1)
	signal.Notify(stopCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stopCh
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(ctx)
	}()

	log.Printf("tokenprintd-go listening at %s", url)
	log.Printf("dashboard output: %s", cfg.OutputPath)
	if !cfg.NoOpen {
		openBrowser(url)
	}

	err := httpServer.ListenAndServe()
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
