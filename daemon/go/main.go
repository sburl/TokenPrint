package main

import (
    "encoding/json"
    "flag"
    "fmt"
    "io"
    "log"
    "net/http"
    "os"
    "os/exec"
    "path/filepath"
    "sync"
    "time"
)

type server struct {
    tokenprintCmd string
    outputPath    string
    noCache       bool
    since         string
    until         string

    mu         sync.Mutex
    refreshing bool
}

type jsonResp map[string]any

func (s *server) runTokenprint() error {
    args := []string{"--no-open", "--output", s.outputPath}
    if s.noCache {
        args = append(args, "--no-cache")
    }
    if s.since != "" {
        args = append(args, "--since", s.since)
    }
    if s.until != "" {
        args = append(args, "--until", s.until)
    }

    cmd := exec.Command(s.tokenprintCmd, args...)
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    return cmd.Run()
}

func writeJSON(w http.ResponseWriter, status int, payload jsonResp) {
    w.Header().Set("Content-Type", "application/json")
    w.WriteHeader(status)
    _ = json.NewEncoder(w).Encode(payload)
}

func (s *server) statusHandler(w http.ResponseWriter, _ *http.Request) {
    writeJSON(w, http.StatusOK, jsonResp{"ok": true})
}

func (s *server) refreshHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        writeJSON(w, http.StatusMethodNotAllowed, jsonResp{"ok": false, "error": "method_not_allowed"})
        return
    }

    s.mu.Lock()
    if s.refreshing {
        s.mu.Unlock()
        writeJSON(w, http.StatusConflict, jsonResp{"ok": false, "error": "refresh_in_progress"})
        return
    }
    s.refreshing = true
    s.mu.Unlock()

    defer func() {
        s.mu.Lock()
        s.refreshing = false
        s.mu.Unlock()
    }()

    if err := s.runTokenprint(); err != nil {
        writeJSON(w, http.StatusInternalServerError, jsonResp{"ok": false, "error": err.Error()})
        return
    }

    writeJSON(w, http.StatusOK, jsonResp{
        "ok":          true,
        "generatedAt": time.Now().Format("01/02/2006, 03:04 PM"),
    })
}

func (s *server) indexHandler(w http.ResponseWriter, _ *http.Request) {
    f, err := os.Open(s.outputPath)
    if err != nil {
        writeJSON(w, http.StatusNotFound, jsonResp{"ok": false, "error": "dashboard_not_found"})
        return
    }
    defer f.Close()

    w.Header().Set("Content-Type", "text/html; charset=utf-8")
    _, _ = io.Copy(w, f)
}

func main() {
    host := flag.String("host", "127.0.0.1", "Bind host")
    port := flag.Int("port", 8765, "Bind port")
    output := flag.String("output", filepath.Join(os.TempDir(), "tokenprint.html"), "Dashboard output path")
    tokenprintCmd := flag.String("tokenprint", "tokenprint", "tokenprint binary path")
    noCache := flag.Bool("no-cache", false, "Pass --no-cache to tokenprint")
    since := flag.String("since", "", "Optional since (YYYYMMDD)")
    until := flag.String("until", "", "Optional until (YYYYMMDD)")
    flag.Parse()

    srv := &server{
        tokenprintCmd: *tokenprintCmd,
        outputPath:    *output,
        noCache:       *noCache,
        since:         *since,
        until:         *until,
    }

    if err := srv.runTokenprint(); err != nil {
        log.Fatalf("initial tokenprint run failed: %v", err)
    }

    mux := http.NewServeMux()
    mux.HandleFunc("/", srv.indexHandler)
    mux.HandleFunc("/index.html", srv.indexHandler)
    mux.HandleFunc("/api/status", srv.statusHandler)
    mux.HandleFunc("/api/refresh", srv.refreshHandler)

    addr := fmt.Sprintf("%s:%d", *host, *port)
    log.Printf("tokenprintd-go listening at http://%s", addr)
    if err := http.ListenAndServe(addr, mux); err != nil {
        log.Fatal(err)
    }
}
