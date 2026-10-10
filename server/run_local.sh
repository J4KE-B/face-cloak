#!/usr/bin/env bash
# Laptop fallback: run the demo backend on CPU and expose it through a free Cloudflare quick tunnel.
#   ./server/run_local.sh                  -> generates a random passcode and prints it
#   DEMO_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')" ./server/run_local.sh
# Env: DEMO_SITE (default https://face-cloak.vercel.app), VERCEL_TEAM_SLUG (enables CORS for your Vercel previews).
# DEMO_KEY must be >= 12 characters.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DEMO_SITE="${DEMO_SITE:-https://face-cloak.vercel.app}"
export DEMO_SITE
# cloudflared is pinned to one release and verified against its published sha256 (GitHub release notes of
# cloudflare/cloudflared 2026.10.0, "SHA256 Checksums": cloudflared-linux-amd64; also equals the asset digest).
CF_VERSION="2026.10.0"
CF_SHA256="d33ff2d14475178d2012c2c56beba87389ac5ded27649519f198a7d3134a99db"
PORT="${PORT:-8000}"
BIN="$HOME/.local/bin"
LOGDIR="$(mktemp -d)"   # backend / tunnel logs only (no images are ever logged)
PIDS=()

if [[ -z "${DEMO_KEY:-}" ]]; then
  DEMO_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(12))')"
  echo "DEMO_KEY not set -> generated passcode: $DEMO_KEY"
fi
if [[ ${#DEMO_KEY} -lt 12 ]]; then
  echo "DEMO_KEY must be at least 12 characters (unset it to auto-generate one)." >&2
  exit 1
fi
export DEMO_KEY

killtree() {  # kill a process and all its descendants
  local p="$1" c
  for c in $(pgrep -P "$p" 2>/dev/null || true); do killtree "$c"; done
  kill "$p" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  echo; echo "Stopping tunnel and backend..."
  for pid in "${PIDS[@]:-}"; do [[ -n "$pid" ]] && killtree "$pid"; done
  wait 2>/dev/null || true
  rm -rf "$LOGDIR"
}
trap cleanup EXIT INT TERM

# cloudflared: pinned official release binary, sha256-verified (never "latest")
CF="$BIN/cloudflared-$CF_VERSION"
if [[ ! -x "$CF" ]] || ! echo "$CF_SHA256  $CF" | sha256sum -c --status -; then
  echo "Downloading cloudflared $CF_VERSION to $CF ..."
  mkdir -p "$BIN"
  curl -fsSL -o "$CF.part" "https://github.com/cloudflare/cloudflared/releases/download/$CF_VERSION/cloudflared-linux-amd64"
  if ! echo "$CF_SHA256  $CF.part" | sha256sum -c --status -; then
    rm -f "$CF.part"; echo "cloudflared checksum mismatch - refusing to run it" >&2; exit 1
  fi
  chmod +x "$CF.part"; mv "$CF.part" "$CF"
fi
"$CF" --version >/dev/null || { echo "cloudflared does not run" >&2; exit 1; }

# Backend: CPU, 8 threads, memory/CPU capped (laptop heavy-job rule), low priority.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 FACECLOAK_THREADS=8
PY="$REPO/.venv/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3)"
echo "Starting backend on 127.0.0.1:$PORT (loading models, ~30-60 s) ..."
systemd-run --user --scope -q -p MemoryMax=9G -p MemoryHigh=8G -p MemorySwapMax=0 -p CPUQuota=800% -p IOWeight=50 \
  nice -n 10 "$PY" -m server --host 127.0.0.1 --port "$PORT" >"$LOGDIR/backend.log" 2>&1 &
PIDS+=($!)

for i in $(seq 1 180); do
  if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "${PIDS[0]}" 2>/dev/null; then echo "Backend exited:" >&2; tail -20 "$LOGDIR/backend.log" >&2; exit 1; fi
  sleep 2
  [[ $i -eq 180 ]] && { echo "Backend did not become healthy in 6 min" >&2; tail -20 "$LOGDIR/backend.log" >&2; exit 1; }
done
echo "Backend healthy."

"$CF" tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate >"$LOGDIR/tunnel.log" 2>&1 &
PIDS+=($!)

URL=""
for i in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGDIR/tunnel.log" | head -1 || true)"
  [[ -n "$URL" ]] && break
  kill -0 "${PIDS[1]}" 2>/dev/null || { echo "cloudflared exited:" >&2; tail -20 "$LOGDIR/tunnel.log" >&2; exit 1; }
  sleep 1
done
[[ -z "$URL" ]] && { echo "Could not read the tunnel URL" >&2; tail -20 "$LOGDIR/tunnel.log" >&2; exit 1; }

ENC="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$URL")"
echo
echo "==================================================================="
echo " Tunnel URL : $URL"
echo " Passcode   : $DEMO_KEY"
echo " Auto-connect link (open this):"
echo "   $DEMO_SITE/demo?backend=$ENC"
echo "==================================================================="
echo "Ctrl-C to stop. Backend log: $LOGDIR/backend.log"
wait "${PIDS[1]}"
