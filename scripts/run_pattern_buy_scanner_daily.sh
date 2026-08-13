#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# M4-1b (13/08/2026): KHÔNG dùng .venv trong iCloud — launchd chạy bị
# "OSError: Resource deadlock avoided" (venv 136MB nằm trong iCloud Drive) và
# .venv thiếu vnstock_data. Thứ tự ưu tiên:
#   1) venv sponsor golden (~/dev/main-sonet-runtime/.venv-vnstock-sponsor311):
#      có vnii → tier golden 500 req/phút (đã test refresh 100 mã RPM 300, 0 lỗi).
#   2) python3.14 homebrew: chạy được nhưng tier free 60 req/phút.
# Ghi đè bằng PATTERN_BUY_PYTHON nếu cần.
SPONSOR_PY="${HOME}/dev/main-sonet-runtime/.venv-vnstock-sponsor311/bin/python"
if [[ -n "${PATTERN_BUY_PYTHON:-}" ]]; then
  PY="${PATTERN_BUY_PYTHON}"
elif [[ -x "${SPONSOR_PY}" ]]; then
  PY="${SPONSOR_PY}"
elif [[ -x "/opt/homebrew/bin/python3.14" ]]; then
  PY="/opt/homebrew/bin/python3.14"
elif command -v python3.11 >/dev/null 2>&1; then
  PY="python3.11"
else
  PY="${PYTHON_BIN:-python3}"
fi

LOG_DIR="${REPO_ROOT}/logs/pattern_buy_scanner"
LOCK_DIR="${REPO_ROOT}/logs/pattern_buy_scanner.lock"
mkdir -p "${LOG_DIR}" "${REPO_ROOT}/artifacts/realtime_scan/latest"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Pattern BUY scanner is already running. Exit."
  exit 2
fi
trap 'rm -rf "${LOCK_DIR}"' EXIT

RUN_TS="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/pattern_buy_scanner_${RUN_TS}.log"
LATEST_LOG="${LOG_DIR}/latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
rm -f "${LATEST_LOG}" 2>/dev/null || true
ln -s "${LOG_FILE}" "${LATEST_LOG}" 2>/dev/null || {
  echo "Warning: cannot update latest log symlink: ${LATEST_LOG}"
}

echo "============================================================"
echo "Pattern BUY scanner daily run"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Repo: ${REPO_ROOT}"
echo "Python: ${PY}"
echo "Log: ${LOG_FILE}"
echo "============================================================"

export REALTIME_SCAN_EMAIL_TO="${REALTIME_SCAN_EMAIL_TO:-stevetransg@gmail.com}"

LOOKBACK_DAYS="${PATTERN_BUY_LOOKBACK_DAYS:-7}"
LIMIT_EACH="${PATTERN_BUY_LIMIT_EACH:-20}"
BUY_SETUP_LIMIT="${PATTERN_BUY_SETUP_LIMIT_PER_PATTERN:-8}"
HISTORY_HORIZON_DAYS="${PATTERN_BUY_HISTORY_HORIZON_DAYS:-120}"
REFRESH_STALENESS_DAYS="${PATTERN_BUY_REFRESH_STALENESS_DAYS:-0}"
REFRESH_MIN_LATEST_SYMBOLS="${PATTERN_BUY_REFRESH_MIN_LATEST_SYMBOLS:-80}"
REFRESH_SCOPE="${PATTERN_BUY_REFRESH_SCOPE:-vn100}"
REFRESH_RPM="${PATTERN_BUY_REFRESH_RPM:-180}"
REFRESH_MAX_ERRORS="${PATTERN_BUY_REFRESH_MAX_ERRORS:-80}"
REFRESH_TIMEOUT_SECONDS="${PATTERN_BUY_REFRESH_TIMEOUT_SECONDS:-10}"
REFRESH_COMMAND_TIMEOUT_SECONDS="${PATTERN_BUY_REFRESH_COMMAND_TIMEOUT_SECONDS:-3600}"

refresh_symbol_args=()
if [[ "${REFRESH_SCOPE}" == "vn100" ]]; then
  while IFS= read -r symbol; do
    [[ -n "${symbol}" ]] && refresh_symbol_args+=(--refresh-symbol "${symbol}")
  done < <("${PY}" - <<'PY'
import sqlite3
from pathlib import Path

db = Path("../market_stats/cache/membership_history.sqlite")
with sqlite3.connect(db) as conn:
    rows = conn.execute(
        """
        SELECT DISTINCT ticker
        FROM index_membership_history
        WHERE index_code IN ('VN30', 'VN100')
          AND effective_to IS NULL
        ORDER BY ticker
        """
    ).fetchall()
for (symbol,) in rows:
    print(str(symbol).upper())
PY
)
  echo "Refresh scope: VN100/VN30 (${#refresh_symbol_args[@]} args, $(( ${#refresh_symbol_args[@]} / 2 )) symbols)"
  # M4 (13/08/2026): nếu python chết hoặc DB membership lỗi → mảng rỗng.
  # Cảnh báo thay vì để mail quét âm thầm trên dữ liệu cũ không refresh.
  if [[ ${#refresh_symbol_args[@]} -eq 0 ]]; then
    echo "CẢNH BÁO: không lấy được danh sách VN100/VN30 (DB membership thiếu hoặc python lỗi) — sẽ quét trên dữ liệu cũ, không refresh."
  fi
elif [[ "${REFRESH_SCOPE}" != "all" ]]; then
  echo "Unsupported PATTERN_BUY_REFRESH_SCOPE=${REFRESH_SCOPE}; use vn100 or all."
  exit 2
fi

cmd=(
  "${PY}" -m scanner.send_realtime_scan_email
  --refresh-data
  --strict-refresh
  --include-buy-setup
  --focus-buy-watchlist
  --attach-pdf
  --lookback-days "${LOOKBACK_DAYS}"
  --limit-each "${LIMIT_EACH}"
  --buy-setup-limit-per-pattern "${BUY_SETUP_LIMIT}"
  --history-horizon-days "${HISTORY_HORIZON_DAYS}"
  --refresh-staleness-days "${REFRESH_STALENESS_DAYS}"
  --refresh-min-latest-symbols "${REFRESH_MIN_LATEST_SYMBOLS}"
  --refresh-rpm "${REFRESH_RPM}"
  --refresh-max-errors "${REFRESH_MAX_ERRORS}"
  --refresh-timeout-seconds "${REFRESH_TIMEOUT_SECONDS}"
  --refresh-command-timeout-seconds "${REFRESH_COMMAND_TIMEOUT_SECONDS}"
)
# M4 (13/08/2026): /bin/bash 3.2 + set -u — expand mảng rỗng trực tiếp gây
# "unbound variable" (lỗi job 17:00 11-12/08). Chỉ append khi có phần tử.
if [[ ${#refresh_symbol_args[@]} -gt 0 ]]; then
  cmd+=( "${refresh_symbol_args[@]}" )
fi

if [[ "${PATTERN_BUY_REFRESH_FORCE:-0}" =~ ^(1|true|yes|on)$ ]]; then
  cmd+=(--refresh-force)
fi

if [[ "${PATTERN_BUY_SEND:-1}" =~ ^(1|true|yes|on)$ ]]; then
  cmd+=(--send)
else
  echo "PATTERN_BUY_SEND=0: build preview artifacts only, do not send email."
fi

echo
echo "Running:"
printf ' %q' "${cmd[@]}"
echo
echo

"${cmd[@]}"

echo
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S %Z')"
