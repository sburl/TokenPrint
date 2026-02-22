package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestNormalizeConfigDefaults(t *testing.T) {
	cfg := normalizeConfig(Config{RefreshPath: "api/refresh"})
	if cfg.Host != "127.0.0.1" {
		t.Fatalf("expected default host, got %q", cfg.Host)
	}
	if cfg.Port != 8765 {
		t.Fatalf("expected default port, got %d", cfg.Port)
	}
	if cfg.TokenprintBin != "tokenprint" {
		t.Fatalf("expected default tokenprint binary, got %q", cfg.TokenprintBin)
	}
	if cfg.RefreshPath != "/api/refresh" {
		t.Fatalf("expected normalized refresh path, got %q", cfg.RefreshPath)
	}
	if cfg.RefreshTimeout != 10*time.Minute {
		t.Fatalf("expected default timeout, got %s", cfg.RefreshTimeout)
	}
}

func TestIndexHandlerServesHTML(t *testing.T) {
	outputPath := filepath.Join(t.TempDir(), "tokenprint.html")
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
