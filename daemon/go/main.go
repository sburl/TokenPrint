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
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	defaultHost             = "127.0.0.1"
	defaultPort             = 8765
	defaultOutputFilename   = "tokenprint.html"
	defaultTokenprintBinary = "tokenprint"
	defaultRefreshPath      = "/api/refresh"
	defaultRefreshTimeout   = 10 * time.Minute
	defaultReadHeaderTO     = 5 * time.Second
	defaultShutdownTO       = 5 * time.Second
	maxPort                 = 65535
	generatedAtFormat       = "01/02/2006, 03:04 PM"
	loopbackHostFallback    = defaultHost
)

var loopbackHosts = []string{
	defaultHost,
	"::1",
	"localhost",
	"::ffff:7f00:1",
}

// Config controls daemon behavior.
type Config struct {
	Host           string
	Port           int
	OutputPath     string
	TokenprintBin  string
	NoCache        bool
	Since          string
	Until          string
	CachePath      string
	RefreshPath    string
	RefreshTimeout time.Duration
	RefreshToken   string
	GeminiLogPath  string
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
		Host:           defaultHost,
		Port:           defaultPort,
		OutputPath:     filepath.Join(os.TempDir(), defaultOutputFilename),
		TokenprintBin:  defaultTokenprintBinary,
		RefreshPath:    defaultRefreshPath,
		RefreshTimeout: defaultRefreshTimeout,
	}
}

func normalizeConfig(cfg Config) Config {
	cfg.Host = strings.TrimSpace(cfg.Host)
	cfg.OutputPath = strings.TrimSpace(cfg.OutputPath)
	cfg.TokenprintBin = strings.TrimSpace(cfg.TokenprintBin)
	cfg.RefreshPath = strings.TrimSpace(cfg.RefreshPath)
	cfg.Since = strings.TrimSpace(cfg.Since)
	cfg.Until = strings.TrimSpace(cfg.Until)
	cfg.CachePath = strings.TrimSpace(cfg.CachePath)
	cfg.RefreshToken = strings.TrimSpace(cfg.RefreshToken)
	cfg.GeminiLogPath = strings.TrimSpace(cfg.GeminiLogPath)

	if cfg.Host == "" {
		cfg.Host = defaultHost
	}
	if cfg.Port <= 0 {
		cfg.Port = defaultPort
	}
	if cfg.Port > maxPort {
		cfg.Port = defaultPort
	}
	if cfg.OutputPath == "" {
		cfg.OutputPath = filepath.Join(os.TempDir(), defaultOutputFilename)
	}
	if cfg.TokenprintBin == "" {
		cfg.TokenprintBin = defaultTokenprintBinary
	}
	if cfg.RefreshPath == "" {
		cfg.RefreshPath = defaultRefreshPath
	}
	if !strings.HasPrefix(cfg.RefreshPath, "/") {
		cfg.RefreshPath = "/" + cfg.RefreshPath
	}
	if cfg.RefreshTimeout <= 0 {
		cfg.RefreshTimeout = defaultRefreshTimeout
	}
	return cfg
}

func isLoopbackHost(host string) bool {
	host = strings.ToLower(strings.TrimSpace(host))
	host = normalizedLoopbackHost(host)
	for _, allowed := range loopbackHosts {
		if host == allowed {
			return true
		}
	}
	return false
}

func normalizedLoopbackHost(host string) string {
	host = strings.TrimSpace(host)
	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		return strings.TrimPrefix(strings.TrimSuffix(host, "]"), "[")
	}
	return host
}

func formatListenAddress(host string, port int) string {
	return net.JoinHostPort(normalizedLoopbackHost(host), strconv.Itoa(port))
}

func formatPublicURL(host string, port int) string {
	host = strings.ToLower(strings.TrimSpace(host))
	if host == "0.0.0.0" || host == "::" {
		host = loopbackHostFallback
	}
	return "http://" + formatListenAddress(host, port)
}

func buildTokenprintArgs(cfg Config) []string {
	args := []string{
		"--no-open", "--output", cfg.OutputPath, "--live-mode", "--refresh-endpoint", cfg.RefreshPath,
	}
	if cfg.RefreshToken != "" {
		args = append(args, "--refresh-token", cfg.RefreshToken)
	}
	if cfg.NoCache {
		args = append(args, "--no-cache")
	}
	if cfg.Since != "" {
		args = append(args, "--since", cfg.Since)
	}
	if cfg.Until != "" {
		args = append(args, "--until", cfg.Until)
	}
	if cfg.CachePath != "" {
		args = append(args, "--cache-path", cfg.CachePath)
	}
	if cfg.GeminiLogPath != "" {
		args = append(args, "--gemini-log-path", cfg.GeminiLogPath)
	}
	return args
}

func buildRunner(cfg Config) func(context.Context) error {
	cfg = normalizeConfig(cfg)
	return func(ctx context.Context) error {
		args := buildTokenprintArgs(cfg)
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
	a.refreshFinish(nil, time.Now().Format(generatedAtFormat))
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

// originAllowed returns true when the Origin header is absent (non-browser client)
// or matches a loopback origin on our port (same-origin browser request).
func (a *App) originAllowed(origin string) bool {
	if origin == "" {
		return true
	}
	allowed := make([]string, 0, len(loopbackHosts))
	for _, host := range loopbackHosts {
		allowed = append(allowed, "http://"+formatListenAddress(host, a.cfg.Port))
	}
	for _, o := range allowed {
		if origin == o {
			return true
		}
	}
	return false
}

func (a *App) refreshHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"ok": false, "error": "method_not_allowed"})
		return
	}
	if !a.originAllowed(r.Header.Get("Origin")) {
		writeJSON(w, http.StatusForbidden, map[string]any{"ok": false, "error": "forbidden"})
		return
	}
	requiresToken := a.cfg.RefreshToken != "" || !isLoopbackHost(a.cfg.Host)
	if requiresToken {
		tok := strings.TrimSpace(r.Header.Get("X-Tokenprint-Token"))
		if tok == "" || tok != a.cfg.RefreshToken {
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

	generatedAt := time.Now().Format(generatedAtFormat)
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
	flag.StringVar(&cfg.CachePath, "cache-path", cfg.CachePath, "Path to tokenprint cache file or directory")
	flag.StringVar(&cfg.RefreshPath, "refresh-path", cfg.RefreshPath, "Refresh API path")
	flag.DurationVar(&cfg.RefreshTimeout, "timeout", cfg.RefreshTimeout, "Refresh timeout")
	flag.StringVar(
		&cfg.RefreshToken,
		"refresh-token",
		"",
		"Shared token required for POST refresh when bound to non-loopback hosts",
	)
	flag.StringVar(&cfg.GeminiLogPath, "gemini-log-path", cfg.GeminiLogPath, "Path to Gemini telemetry log or directory (passed to tokenprint)")
	flag.BoolVar(&cfg.NoOpen, "no-open", false, "Do not auto-open browser")
	flag.Parse()

	cfg = normalizeConfig(cfg)
	app := newApp(cfg, nil)

	if !isLoopbackHost(cfg.Host) && cfg.RefreshToken == "" {
		log.Fatal("refusing to run refresh endpoint without --refresh-token when host is non-loopback")
	}

	warmCtx, warmCancel := context.WithTimeout(context.Background(), cfg.RefreshTimeout)
	defer warmCancel()
	if err := app.warmup(warmCtx); err != nil {
		log.Fatalf("initial tokenprint run failed: %v", err)
	}

	url := formatPublicURL(cfg.Host, cfg.Port)
	addr := formatListenAddress(cfg.Host, cfg.Port)

	httpServer := &http.Server{
		Addr:              addr,
		Handler:           app.handler(),
		ReadHeaderTimeout: defaultReadHeaderTO,
	}

	stopCh := make(chan os.Signal, 1)
	signal.Notify(stopCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stopCh
		ctx, cancel := context.WithTimeout(context.Background(), defaultShutdownTO)
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
