package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestNormalizeConfigDefaults(t *testing.T) {
	cfg := normalizeConfig(Config{RefreshPath: "api/refresh"})
	if cfg.Host != defaultHost {
		t.Fatalf("expected default host, got %q", cfg.Host)
	}
	if cfg.Port != defaultPort {
		t.Fatalf("expected default port, got %d", cfg.Port)
	}
	if cfg.TokenprintBin != defaultTokenprintBinary {
		t.Fatalf("expected default tokenprint binary, got %q", cfg.TokenprintBin)
	}
	if cfg.RefreshPath != "/api/refresh" {
		t.Fatalf("expected normalized refresh path, got %q", cfg.RefreshPath)
	}
	if cfg.RefreshTimeout != defaultRefreshTimeout {
		t.Fatalf("expected default timeout, got %s", cfg.RefreshTimeout)
	}
}

func TestNormalizeConfigTrimsStringInputs(t *testing.T) {
	cfg := normalizeConfig(Config{
		Host:          "  127.0.0.1  ",
		OutputPath:    "  /tmp/tokenprint.html  ",
		TokenprintBin: "  tokenprint  ",
		RefreshPath:   "  api/refresh  ",
		Since:         "  20260101  ",
		Until:         " 20260131 ",
		CachePath:     "  /tmp/daemon-cache  ",
		RefreshToken:  "  secret  ",
		GeminiLogPath: "  /tmp/gemini.log  ",
	})
	if cfg.Host != "127.0.0.1" {
		t.Fatalf("expected host trimmed, got %q", cfg.Host)
	}
	if cfg.OutputPath != "/tmp/tokenprint.html" {
		t.Fatalf("expected output path trimmed, got %q", cfg.OutputPath)
	}
	if cfg.TokenprintBin != defaultTokenprintBinary {
		t.Fatalf("expected tokenprint binary preserved from trimmed value, got %q", cfg.TokenprintBin)
	}
	if cfg.RefreshPath != "/api/refresh" {
		t.Fatalf("expected refresh path trimmed and normalized, got %q", cfg.RefreshPath)
	}
	if cfg.Since != "20260101" {
		t.Fatalf("expected since date trimmed, got %q", cfg.Since)
	}
	if cfg.Until != "20260131" {
		t.Fatalf("expected until date trimmed, got %q", cfg.Until)
	}
	if cfg.CachePath != "/tmp/daemon-cache" {
		t.Fatalf("expected cache path trimmed, got %q", cfg.CachePath)
	}
	if cfg.RefreshToken != "secret" {
		t.Fatalf("expected refresh token trimmed, got %q", cfg.RefreshToken)
	}
	if cfg.GeminiLogPath != "/tmp/gemini.log" {
		t.Fatalf("expected gemini log path trimmed, got %q", cfg.GeminiLogPath)
	}
}

func TestNormalizeConfigDefaultsOnWhitespace(t *testing.T) {
	cfg := normalizeConfig(Config{
		OutputPath:    "   ",
		TokenprintBin: "   ",
		RefreshPath:   "   ",
	})
	if cfg.OutputPath != filepath.Join(os.TempDir(), defaultOutputFilename) {
		t.Fatalf("expected whitespace output path to fall back, got %q", cfg.OutputPath)
	}
	if cfg.TokenprintBin != defaultTokenprintBinary {
		t.Fatalf("expected whitespace tokenprint binary to fall back, got %q", cfg.TokenprintBin)
	}
	if cfg.RefreshPath != defaultRefreshPath {
		t.Fatalf("expected whitespace refresh path to fall back, got %q", cfg.RefreshPath)
	}
}

func TestNormalizeConfigFallsBackInvalidPort(t *testing.T) {
	cfgZero := normalizeConfig(Config{Port: 0})
	if cfgZero.Port != defaultPort {
		t.Fatalf("expected zero port to fall back to default, got %d", cfgZero.Port)
	}

	cfg := normalizeConfig(Config{Port: 70000})
	if cfg.Port != defaultPort {
		t.Fatalf("expected invalid port to fall back to default, got %d", cfg.Port)
	}
}

func TestNormalizeConfigTrimsHost(t *testing.T) {
	cfg := normalizeConfig(Config{Host: "  ::1  "})
	if cfg.Host != "::1" {
		t.Fatalf("expected host trimmed to ::1, got %q", cfg.Host)
	}
}

func TestNormalizeConfigPreservesBracketedIPv6Host(t *testing.T) {
	cfg := normalizeConfig(Config{Host: "  [::1]  "})
	if cfg.Host != "[::1]" {
		t.Fatalf("expected host to preserve bracketed IPv6 host, got %q", cfg.Host)
	}
}

func TestIsLoopbackHost(t *testing.T) {
	if !isLoopbackHost(defaultHost) {
		t.Fatalf("expected loopback for %q", defaultHost)
	}
	if !isLoopbackHost("::1") {
		t.Fatalf("expected loopback for ::1")
	}
	if !isLoopbackHost(" [::1] ") {
		t.Fatalf("expected loopback for bracketed IPv6 host with whitespace")
	}
	if !isLoopbackHost(" [::ffff:7f00:1] ") {
		t.Fatalf("expected loopback for mapped IPv4 loopback IPv6 host with whitespace")
	}
	if !isLoopbackHost("localhost") {
		t.Fatalf("expected loopback for localhost")
	}
	if isLoopbackHost("192.168.1.10") {
		t.Fatalf("expected non-loopback for private IP")
	}
	if isLoopbackHost("0.0.0.0") {
		t.Fatalf("expected non-loopback for 0.0.0.0")
	}
}

func TestFormatListenAndPublicURL(t *testing.T) {
	if got := formatListenAddress("::1", defaultPort); got != "[::1]:"+strconv.Itoa(defaultPort) {
		t.Fatalf("expected IPv6 listen address [::1]:%d, got %q", defaultPort, got)
	}
	if got := formatListenAddress(" [::1] ", defaultPort); got != "[::1]:"+strconv.Itoa(defaultPort) {
		t.Fatalf("expected bracket-trimmed IPv6 listen address [::1]:%d, got %q", defaultPort, got)
	}
	if got := formatListenAddress(defaultHost, defaultPort); got != defaultHost+":"+strconv.Itoa(defaultPort) {
		t.Fatalf("expected IPv4 listen address, got %q", got)
	}
	if got := formatPublicURL("::1", defaultPort); got != "http://[::1]:"+strconv.Itoa(defaultPort) {
		t.Fatalf("expected IPv6 URL, got %q", got)
	}
	if got := formatPublicURL("::", defaultPort); got != "http://"+loopbackHostFallback+":"+strconv.Itoa(defaultPort) {
		t.Fatalf("expected unspecified IPv6 host fallback URL, got %q", got)
	}
	if got := formatPublicURL(" LOCALHOST ", defaultPort); got != "http://localhost:"+strconv.Itoa(defaultPort) {
		t.Fatalf("expected normalized localhost URL, got %q", got)
	}
	if got := formatPublicURL("0.0.0.0", defaultPort); got != "http://"+loopbackHostFallback+":"+strconv.Itoa(defaultPort) {
		t.Fatalf("expected non-loopback host fallback URL, got %q", got)
	}
}

func TestBuildTokenprintArgsIncludesGeminiLogPath(t *testing.T) {
	cfg := normalizeConfig(Config{
		OutputPath:    "/tmp/tokenprint.html",
		RefreshPath:   "/api/refresh",
		GeminiLogPath: "/tmp/custom-gemini.log",
		CachePath:     "/tmp/daemon-cache.json",
		RefreshToken:  "secret",
		NoCache:       true,
		Since:         "20260101",
		Until:         "20260131",
	})
	args := buildTokenprintArgs(cfg)

	expect := []string{
		"--no-open",
		"--output", "/tmp/tokenprint.html",
		"--live-mode",
		"--refresh-endpoint", "/api/refresh",
		"--refresh-token", "secret",
		"--no-cache",
		"--since", "20260101",
		"--until", "20260131",
		"--cache-path", "/tmp/daemon-cache.json",
		"--gemini-log-path", "/tmp/custom-gemini.log",
	}
	if len(args) != len(expect) {
		t.Fatalf("expected %d args, got %d: %#v", len(expect), len(args), args)
	}
	for i, expected := range expect {
		if args[i] != expected {
			t.Fatalf("expected args[%d]=%q, got %q", i, expected, args[i])
		}
	}
}

func TestBuildTokenprintArgsOmitOptionalPathsWhenEmpty(t *testing.T) {
	cfg := normalizeConfig(Config{OutputPath: "/tmp/tokenprint.html", RefreshPath: "/api/refresh"})
	args := buildTokenprintArgs(cfg)
	for i := 0; i < len(args)-1; i++ {
		if args[i] == "--cache-path" {
			t.Fatalf("did not expect --cache-path when cache path is empty: %#v", args)
		}
		if args[i] == "--gemini-log-path" {
			t.Fatalf("did not expect --gemini-log-path when gemini log path is empty: %#v", args)
		}
	}
}

func TestRefreshHandlerTreatsEmptyTokenAsUnauthorizedWhenRequired(t *testing.T) {
	app := newApp(Config{Host: "0.0.0.0", RefreshToken: ""}, func(context.Context) error { return nil })
	handler := app.handler()

	noTokenStatus, noTokenBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if noTokenStatus != http.StatusUnauthorized {
		t.Fatalf("expected 401 when public host requires token and no token is provided, got %d body=%s", noTokenStatus, noTokenBody)
	}

	wsTokenStatus, wsTokenBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", " ")
	if wsTokenStatus != http.StatusUnauthorized {
		t.Fatalf("expected 401 when public host token is whitespace-only, got %d body=%s", wsTokenStatus, wsTokenBody)
	}
}

func TestIndexHandlerServesHTML(t *testing.T) {
	outputPath := filepath.Join(t.TempDir(), defaultOutputFilename)
	if err := os.WriteFile(outputPath, []byte("<html>ok</html>"), 0o644); err != nil {
		t.Fatalf("write output file: %v", err)
	}

	app := newApp(Config{OutputPath: outputPath}, func(context.Context) error { return nil })
	handler := app.handler()

	status, body, headers := performRequest(t, handler, http.MethodGet, "/", "")
	if status != http.StatusOK {
		t.Fatalf("expected 200 from /, got %d", status)
	}
	if !strings.Contains(headers.Get("Content-Type"), "text/html") {
		t.Fatalf("expected html content-type, got %q", headers.Get("Content-Type"))
	}
	if !strings.Contains(body, "<html>ok</html>") {
		t.Fatalf("unexpected body: %q", body)
	}

	headStatus, _, _ := performRequest(t, handler, http.MethodHead, "/", "")
	if headStatus != http.StatusOK {
		t.Fatalf("expected 200 from HEAD /, got %d", headStatus)
	}
}

func TestRefreshHandlerSuccessAndStatus(t *testing.T) {
	var runs atomic.Int32
	app := newApp(Config{}, func(context.Context) error {
		runs.Add(1)
		return nil
	})
	handler := app.handler()

	status, body, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if status != http.StatusOK {
		t.Fatalf("expected 200 refresh, got %d body=%s", status, body)
	}
	if runs.Load() != 1 {
		t.Fatalf("expected 1 refresh run, got %d", runs.Load())
	}

	statusCode, statusBody, _ := performRequest(t, handler, http.MethodGet, "/api/status", "")
	if statusCode != http.StatusOK {
		t.Fatalf("expected 200 status, got %d", statusCode)
	}

	var payload map[string]any
	if err := json.Unmarshal([]byte(statusBody), &payload); err != nil {
		t.Fatalf("decode status payload: %v", err)
	}
	if payload["ok"] != true {
		t.Fatalf("expected ok=true payload, got %+v", payload)
	}
	if _, ok := payload["generatedAt"].(string); !ok {
		t.Fatalf("expected generatedAt in payload, got %+v", payload)
	}
}

func TestRefreshHandlerErrorPublishesStatusAndPreservesGeneratedAt(t *testing.T) {
	attempts := 0
	app := newApp(Config{}, func(context.Context) error {
		attempts++
		if attempts == 1 {
			return nil
		}
		return errors.New("collector failed")
	})
	handler := app.handler()

	firstStatus, firstBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if firstStatus != http.StatusOK {
		t.Fatalf("expected first refresh to succeed, got %d body=%s", firstStatus, firstBody)
	}
	var firstPayload map[string]any
	if err := json.Unmarshal([]byte(firstBody), &firstPayload); err != nil {
		t.Fatalf("decode first refresh payload: %v", err)
	}
	firstGeneratedAt, ok := firstPayload["generatedAt"].(string)
	if !ok || firstGeneratedAt == "" {
		t.Fatalf("expected generatedAt in first refresh payload, got %+v", firstPayload)
	}

	errorStatus, errorBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if errorStatus != http.StatusInternalServerError {
		t.Fatalf("expected second refresh to fail with 500, got %d body=%s", errorStatus, errorBody)
	}

	statusCode, statusBody, _ := performRequest(t, handler, http.MethodGet, "/api/status", "")
	if statusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d body=%s", statusCode, statusBody)
	}
	var statusPayload map[string]any
	if err := json.Unmarshal([]byte(statusBody), &statusPayload); err != nil {
		t.Fatalf("decode status payload: %v", err)
	}
	if statusPayload["lastError"] != "collector failed" {
		t.Fatalf("expected lastError in status, got %+v", statusPayload["lastError"])
	}
	if got, ok := statusPayload["generatedAt"].(string); !ok || got != firstGeneratedAt {
		t.Fatalf("expected generatedAt to remain %q after failed refresh, got %+v", firstGeneratedAt, statusPayload["generatedAt"])
	}
}

func TestRefreshHandlerConflictWhileInProgress(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})

	app := newApp(Config{}, func(ctx context.Context) error {
		close(started)
		select {
		case <-release:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	})
	handler := app.handler()

	type refreshResult struct {
		status int
		err    error
	}
	firstResult := make(chan refreshResult, 1)
	go func() {
		req := httptest.NewRequest(http.MethodPost, "/api/refresh", nil)
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		firstResult <- refreshResult{status: rr.Code}
	}()

	<-started
	secondStatus, secondBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if secondStatus != http.StatusConflict {
		t.Fatalf("expected 409 conflict, got %d body=%s", secondStatus, secondBody)
	}

	close(release)
	select {
	case got := <-firstResult:
		if got.err != nil {
			t.Fatalf("first refresh request failed: %v", got.err)
		}
		if got.status != http.StatusOK {
			t.Fatalf("expected first refresh 200, got %d", got.status)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for first refresh response")
	}
}

func TestRefreshHandlerRequiresToken(t *testing.T) {
	app := newApp(Config{RefreshToken: "secret"}, func(context.Context) error { return nil })
	handler := app.handler()

	unauthStatus, unauthBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if unauthStatus != http.StatusUnauthorized {
		t.Fatalf("expected 401 without token, got %d body=%s", unauthStatus, unauthBody)
	}

	authStatus, authBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "secret")
	if authStatus != http.StatusOK {
		t.Fatalf("expected 200 with token, got %d body=%s", authStatus, authBody)
	}
}

func TestRefreshHandlerTrimsWhitespaceToken(t *testing.T) {
	app := newApp(Config{RefreshToken: "secret"}, func(context.Context) error { return nil })
	handler := app.handler()

	authStatus, authBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "  secret  ")
	if authStatus != http.StatusOK {
		t.Fatalf("expected 200 with trimmed token, got %d body=%s", authStatus, authBody)
	}

	authStatus, authBody, _ = performRequest(t, handler, http.MethodPost, "/api/refresh", "secret  ")
	if authStatus != http.StatusOK {
		t.Fatalf("expected 200 with trailing whitespace token, got %d body=%s", authStatus, authBody)
	}
}

func TestRefreshHandlerRequiresTokenForPublicHost(t *testing.T) {
	app := newApp(Config{Host: "0.0.0.0"}, func(context.Context) error { return nil })
	handler := app.handler()

	noTokenStatus, noTokenBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if noTokenStatus != http.StatusUnauthorized {
		t.Fatalf("expected 401 for public host without token, got %d body=%s", noTokenStatus, noTokenBody)
	}

	app = newApp(Config{Host: "0.0.0.0", RefreshToken: "secret"}, func(context.Context) error { return nil })
	handler = app.handler()
	authedStatus, authedBody, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "secret")
	if authedStatus != http.StatusOK {
		t.Fatalf("expected 200 for public host with token, got %d body=%s", authedStatus, authedBody)
	}
}

func TestRefreshHandlerTimeoutMapsToGatewayTimeout(t *testing.T) {
	app := newApp(Config{RefreshTimeout: 20 * time.Millisecond}, func(ctx context.Context) error {
		<-ctx.Done()
		return ctx.Err()
	})
	handler := app.handler()

	status, body, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if status != http.StatusGatewayTimeout {
		t.Fatalf("expected 504 timeout, got %d body=%s", status, body)
	}
}

func TestRefreshHandlerBlocksCrossOrigin(t *testing.T) {
	app := newApp(Config{Port: defaultPort}, func(context.Context) error { return nil })
	handler := app.handler()

	// Cross-origin browser request → 403.
	req := httptest.NewRequest(http.MethodPost, "/api/refresh", nil)
	req.Header.Set("Origin", "https://evil.example.com")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for cross-origin, got %d", rr.Code)
	}

	// Same-origin browser request → 200.
	req2 := httptest.NewRequest(http.MethodPost, "/api/refresh", nil)
	req2.Header.Set("Origin", "http://"+defaultHost+":"+strconv.Itoa(defaultPort))
	rr2 := httptest.NewRecorder()
	handler.ServeHTTP(rr2, req2)
	if rr2.Code != http.StatusOK {
		t.Fatalf("expected 200 for same-origin, got %d", rr2.Code)
	}

	// No Origin header (curl / programmatic) → 200.
	status, body, _ := performRequest(t, handler, http.MethodPost, "/api/refresh", "")
	if status != http.StatusOK {
		t.Fatalf("expected 200 for no-origin client, got %d body=%s", status, body)
	}

	app = newApp(Config{Host: "::1", Port: defaultPort}, func(context.Context) error { return nil })
	handler = app.handler()
	req3 := httptest.NewRequest(http.MethodPost, "/api/refresh", nil)
	req3.Header.Set("Origin", "http://[::1]:"+strconv.Itoa(defaultPort))
	rr3 := httptest.NewRecorder()
	handler.ServeHTTP(rr3, req3)
	if rr3.Code != http.StatusOK {
		t.Fatalf("expected 200 for IPv6 same-origin, got %d", rr3.Code)
	}
}

func performRequest(t *testing.T, handler http.Handler, method string, path string, token string) (int, string, http.Header) {
	t.Helper()
	req := httptest.NewRequest(method, path, nil)
	if token != "" {
		req.Header.Set("X-Tokenprint-Token", token)
	}
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	res := rr.Result()
	defer res.Body.Close()

	raw, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("read response body: %v", err)
	}
	return res.StatusCode, string(raw), res.Header
}
