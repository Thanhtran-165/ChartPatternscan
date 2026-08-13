#!/usr/bin/env bash
# refresh_pattern_pipeline.sh — M4 (13/08/2026): 1 lệnh end-to-end pipeline mẫu hình V3.
#
# Các bước:
#   1. rescan events V3  (A/scripts/rescan_pattern_events_v3.py — cwd = repo A)
#   2. build profile V3   (B/market_stats/scripts/build_stock_pattern_profiles.py → staging web_v3)
#   3. deploy (TÙY CHỌN --deploy): copy staging → B/web/ + C/market_stats/web/ (server 8766 đọc C)
#   4. verify server 8766 đang serve bản mới nhất (curl + so generated_at)
#   5. mail preview       (A/scanner/send_realtime_scan_email.py — KHÔNG gửi; --send-mail để gửi thật)
#
# Thiết kế bám hiện trạng THẬT (khảo sát 13/08/2026 — Explore agent):
#   - KHÔNG có bước "split" profiles per-mã trong hệ thống: cả 2 bản build (V2/V3) đều
#     ghi MỘT file JSON tổng có profiles dict per mã → bỏ bước split.
#   - Server 8766 = SimpleHTTPRequestHandler (C/market_stats/local_server.py) đọc file
#     tươi từ disk mỗi request + Cache-Control no-cache → KHÔNG cần restart khi chỉ đổi
#     stock_pattern_profiles.json (giữ --restart-server opt-in cho trường hợp đổi code server).
#   - KHÔNG tự copy web_v3/ → web/ (quy tắc nghiệm thu: chưa ký thì không đụng production)
#     → bước deploy bị gate sau cờ --deploy có chủ đích.
#
# Dùng: scripts/refresh_pattern_pipeline.sh            # rescan + build staging + verify + mail preview
#       scripts/refresh_pattern_pipeline.sh --check    # chỉ báo trạng thái, không làm gì
#       scripts/refresh_pattern_pipeline.sh --skip-rescan --deploy
#
# Cờ: --check | --skip-rescan | --skip-build | --skip-mail | --skip-verify
#      --deploy | --send-mail | --restart-server | --rescan-background | --out <path>
set -euo pipefail

# ---------------------------------------------------------------- đường dẫn
A="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"                     # repo nghiên cứu (rescan + mail)
B="/Users/bobo/Library/Mobile Documents/com~apple~CloudDocs/main sonet"  # iCloud gốc (build)
C="/Users/bobo/dev/market_stats_v2"                                      # dev chạy server 8766
STAGING_OUT="${B}/market_stats/web_v3/stock_pattern_profiles.json"
PROD_B="${B}/market_stats/web/stock_pattern_profiles.json"
PROD_C="${C}/market_stats/web/stock_pattern_profiles.json"
DB="${C}/market_cache/stock_ohlcv/latest.sqlite"
SERVER_URL="http://127.0.0.1:8766"
PY="/opt/homebrew/bin/python3.14"
RESCAN_SCRIPT="${A}/scripts/rescan_pattern_events_v3.py"
BUILD_SCRIPT="${B}/market_stats/scripts/build_stock_pattern_profiles.py"
EVENTS_GLOB="${A}/artifacts/scanner_v2_v3/*/db_active/events.csv"
STALE_DAYS=7
LOG_DIR="${A}/logs/refresh_pipeline"
TS="$(date +%Y%m%d_%H%M%S)"
STEP_LOG="${LOG_DIR}/pipeline_${TS}.log"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | grep -v '^#!'
}

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${STEP_LOG}"; }

# ---------------------------------------------------------------- cờ
DO_CHECK=0; DO_RESCAN=1; DO_BUILD=1; DO_MAIL=1; DO_VERIFY=1
DO_DEPLOY=0; SEND_MAIL=0; RESTART_SERVER=0; RESCAN_BG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) DO_CHECK=1; shift ;;
    --skip-rescan) DO_RESCAN=0; shift ;;
    --skip-build) DO_BUILD=0; shift ;;
    --skip-mail) DO_MAIL=0; shift ;;
    --skip-verify) DO_VERIFY=0; shift ;;
    --deploy) DO_DEPLOY=1; shift ;;
    --send-mail) SEND_MAIL=1; shift ;;
    --restart-server) RESTART_SERVER=1; shift ;;
    --rescan-background) RESCAN_BG=1; shift ;;
    --out) STAGING_OUT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Không biết cờ: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------- check (nhận diện khi cần chạy)
run_check() {
  cat > "${LOG_DIR}/check_${TS}.py" <<'PYEOF'
import glob, json, os, sys, time
from datetime import date, datetime
A, B, C = os.environ["A"], os.environ["B"], os.environ["C"]
STALE = int(os.environ["STALE_DAYS"])
evs = glob.glob(os.environ["EVENTS_GLOB"])
today = date.today()

print("== TRẠNG THÁI PIPELINE MẪU HÌNH V3 ==")
print(f"events V3: {len(evs)} file")
if evs:
    m = max(os.path.getmtime(e) for e in evs)
    d = datetime.fromtimestamp(m).date()
    print(f"  events mới nhất: {d} ({(today - d).days} ngày trước)"
          + ("  ⚠️ CẦN RESCAN" if (today - d).days > STALE else ""))
    n = 0
    for e in evs:
        try:
            n += sum(1 for _ in open(e, encoding="utf-8"))
        except Exception:
            pass
    print(f"  tổng dòng events ≈ {n}")
else:
    print("  KHÔNG CÓ events — chạy pipeline (rescan) trước khi build")

def meta(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("metadata", {})
    except Exception:
        return {}
for label, p in (("staging web_v3", os.environ["STAGING_OUT"]),
                 ("prod B web", os.environ["PROD_B"]),
                 ("prod C (server)", os.environ["PROD_C"])):
    m = meta(p)
    if not m:
        print(f"{label}: KHÔNG CÓ file {p}")
        continue
    gen = m.get("generated_at", "?")
    stale = ""
    try:
        gd = datetime.fromisoformat(gen).date()
        stale = "  ⚠️ STALE >7 ngày" if (today - gd).days > STALE else ""
    except Exception:
        pass
    print(f"{label}: v={m.get('pipeline_version')} generated={gen}{stale}")

db = os.environ["DB"]
if os.path.exists(db):
    import sqlite3
    try:
        c = sqlite3.connect(db)
        mx = c.execute("SELECT MAX(time) FROM stock_price_history").fetchone()[0]
        print(f"DB OHLCV max date: {mx}" + ("  ⚠️ dữ liệu thị trường cũ — chạy refresh market trước" if mx and mx < str(today) else ""))
        c.close()
    except Exception:
        print("DB OHLCV: không đọc được")
PYEOF
  A="$A" B="$B" C="$C" STAGING_OUT="$STAGING_OUT" PROD_B="$PROD_B" PROD_C="$PROD_C" \
  DB="$DB" EVENTS_GLOB="$EVENTS_GLOB" STALE_DAYS="$STALE_DAYS" "${PY}" "${LOG_DIR}/check_${TS}.py" | tee -a "${STEP_LOG}"
}

if [[ "${DO_CHECK}" -eq 1 ]]; then
  log "CHẾ ĐỘ --check: chỉ báo trạng thái, không làm gì."
  run_check
  exit 0
fi

log "=== refresh_pattern_pipeline bắt đầu (${TS}) ==="
run_check

# ---------------------------------------------------------------- 1. rescan
if [[ "${DO_RESCAN}" -eq 1 ]]; then
  log "BƯỚC 1/5: rescan events V3 (1599 mã — có thể mất nhiều phút)"
  cd "${A}"
  if [[ "${RESCAN_BG}" -eq 1 ]]; then
    nohup "${PY}" "${RESCAN_SCRIPT}" > "${LOG_DIR}/rescan_${TS}.log" 2>&1 &
    RESCAN_PID=$!
    log "  rescan chạy nền (pid ${RESCAN_PID}) — chờ kết thúc..."
    if ! wait "${RESCAN_PID}"; then
      log "LỖI: rescan fail — xem ${LOG_DIR}/rescan_${TS}.log"; exit 1
    fi
  else
    "${PY}" "${RESCAN_SCRIPT}" 2>&1 | tee -a "${LOG_DIR}/rescan_${TS}.log"
  fi
  log "✓ rescan xong"
else
  log "BƯỚC 1/5: BỎ QUA rescan (--skip-rescan) — dùng events hiện có"
fi

# ---------------------------------------------------------------- 2. build
if [[ "${DO_BUILD}" -eq 1 ]]; then
  log "BƯỚC 2/5: build profile V3 → ${STAGING_OUT}"
  cd "${B}/market_stats"
  "${PY}" "${BUILD_SCRIPT}" --out "${STAGING_OUT}" 2>&1 | tee -a "${LOG_DIR}/build_${TS}.log"
  log "✓ build xong"
else
  log "BƯỚC 2/5: BỎ QUA build (--skip-build)"
fi

# ---------------------------------------------------------------- 3. deploy (gate)
if [[ "${DO_DEPLOY}" -eq 1 ]]; then
  log "BƯỚC 3/5: deploy staging → production (chạy --deploy là chủ đích, đã cân nhắc)"
  if [[ ! -f "${STAGING_OUT}" ]]; then
    log "LỖI: staging chưa tồn tại — chạy build trước"; exit 1
  fi
  cp "${STAGING_OUT}" "${PROD_B}"
  cp "${STAGING_OUT}" "${PROD_C}"
  log "✓ đã copy → B/web/ + C/market_stats/web/ (server 8766 đọc C)"
else
  log "BƯỚC 3/5: deploy BỊ BỎ QUA — mặc định chỉ build staging (web_v3). Chạy --deploy để copy sang production."
fi

# ---------------------------------------------------------------- 4. restart + verify
if [[ "${RESTART_SERVER}" -eq 1 ]]; then
  log "BƯỚC 4/5: khởi động lại server 8766 (opt-in)"
  launchctl kickstart -k "gui/$(id -u)/com.bobo.marketstats.local" 2>/dev/null \
    || log "⚠️ kickstart fail — server có thể chưa chạy (bỏ qua)"
else
  log "BƯỚC 4/5: KHÔNG restart server — static server đọc file tươi từ disk, không cần restart cho data đổi"
fi

if [[ "${DO_VERIFY}" -eq 1 ]]; then
  log "  verify: server 8766 đang serve bản nào?"
  if curl -sf "${SERVER_URL}/stock_pattern_profiles.json" -o "${LOG_DIR}/served_${TS}.json"; then
    served_ver=$("${PY}" -c "import json,sys; d=json.load(open(sys.argv[1]))['metadata']; print(d.get('pipeline_version'), d.get('generated_at'))" "${LOG_DIR}/served_${TS}.json" 2>/dev/null || echo "?")
    staging_ver=$("${PY}" -c "import json,sys; d=json.load(open(sys.argv[1]))['metadata']; print(d.get('pipeline_version'), d.get('generated_at'))" "${STAGING_OUT}" 2>/dev/null || echo "?")
    log "  server serve: ${served_ver}"
    log "  staging     : ${staging_ver}"
    if [[ -n "${served_ver}" && "${served_ver}" == "${staging_ver}" && "${served_ver}" != "?" ]]; then
      log "✓ server đã serve bản mới nhất"
    else
      log "⚠️ server serve bản KHÁC staging (chưa --deploy hoặc deploy chưa tới) — không chặn pipeline"
    fi
  else
    log "⚠️ server 8766 không trả lời — bỏ qua verify (không chặn pipeline)"
  fi
fi

# ---------------------------------------------------------------- 5. mail
MAIL_PROFILE="${STAGING_OUT}"
[[ "${DO_DEPLOY}" -eq 1 ]] && MAIL_PROFILE="${PROD_B}"
if [[ "${DO_MAIL}" -eq 1 ]]; then
  log "BƯỚC 5/5: mail preview (KHÔNG gửi — thêm --send-mail để gửi thật)"
  cd "${A}"
  MAIL_ARGS=(-m scanner.send_realtime_scan_email --include-buy-setup --focus-buy-watchlist --attach-pdf)
  [[ "${SEND_MAIL}" -eq 1 ]] && MAIL_ARGS+=(--send)
  REALTIME_SCAN_V3_PROFILE="${MAIL_PROFILE}" "${PY}" "${MAIL_ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/mail_${TS}.log"
  log "✓ mail xong (preview; đã gửi nếu --send-mail)"
else
  log "BƯỚC 5/5: BỎ QUA mail (--skip-mail)"
fi

log "=== HOÀN TẤT pipeline (exit 0). Log đầy đủ: ${STEP_LOG} ==="
exit 0
