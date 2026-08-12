#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LABEL="${PATTERN_BUY_LAUNCHD_LABEL:-com.bobo.pattern-buy-scanner}"
RUN_HOUR="${PATTERN_BUY_RUN_HOUR:-17}"
RUN_MINUTE="${PATTERN_BUY_RUN_MINUTE:-00}"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
RUNNER="${REPO_ROOT}/scripts/run_pattern_buy_scanner_daily.sh"
LOG_DIR="${REPO_ROOT}/logs/pattern_buy_scanner"
LAUNCHD_SUPPORT_DIR="${HOME}/Library/Application Support/PatternBuyScanner"
LAUNCHD_WRAPPER="${LAUNCHD_SUPPORT_DIR}/run_pattern_buy_scanner_daily.sh"
LAUNCHD_LOG_DIR="${LAUNCHD_SUPPORT_DIR}/logs"

usage() {
  cat <<EOF
Usage: $0 install|uninstall|status|run|logs

Environment overrides:
  PATTERN_BUY_RUN_HOUR=17
  PATTERN_BUY_RUN_MINUTE=00
  PATTERN_BUY_SEND=1
  REALTIME_SCAN_EMAIL_TO=stevetransg@gmail.com
EOF
}

ensure_runner() {
  if [[ ! -x "${RUNNER}" ]]; then
    chmod +x "${RUNNER}"
  fi
  mkdir -p "${LAUNCHD_SUPPORT_DIR}" "${LAUNCHD_LOG_DIR}"
  cat > "${LAUNCHD_WRAPPER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
EOF
  tail -n +7 "${RUNNER}" >> "${LAUNCHD_WRAPPER}"
  chmod +x "${LAUNCHD_WRAPPER}"
}

install_agent() {
  ensure_runner
  mkdir -p "$(dirname "${PLIST}")" "${LOG_DIR}"
  cat > "${PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${LAUNCHD_WRAPPER}</string>
  </array>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>${RUN_HOUR}</integer>
    <key>Minute</key>
    <integer>${RUN_MINUTE}</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>${LAUNCHD_LOG_DIR}/launchd.out.log</string>

  <key>StandardErrorPath</key>
  <string>${LAUNCHD_LOG_DIR}/launchd.err.log</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>REALTIME_SCAN_EMAIL_TO</key>
    <string>${REALTIME_SCAN_EMAIL_TO:-stevetransg@gmail.com}</string>
    <key>PATTERN_BUY_SEND</key>
    <string>${PATTERN_BUY_SEND:-1}</string>
    <key>PATTERN_BUY_REFRESH_STALENESS_DAYS</key>
    <string>${PATTERN_BUY_REFRESH_STALENESS_DAYS:-0}</string>
    <key>PATTERN_BUY_REFRESH_MIN_LATEST_SYMBOLS</key>
    <string>${PATTERN_BUY_REFRESH_MIN_LATEST_SYMBOLS:-80}</string>
    <key>PATTERN_BUY_REFRESH_SCOPE</key>
    <string>${PATTERN_BUY_REFRESH_SCOPE:-vn100}</string>
    <key>PATTERN_BUY_REFRESH_RPM</key>
    <string>${PATTERN_BUY_REFRESH_RPM:-180}</string>
    <key>PATTERN_BUY_REFRESH_MAX_ERRORS</key>
    <string>${PATTERN_BUY_REFRESH_MAX_ERRORS:-80}</string>
    <key>PATTERN_BUY_REFRESH_TIMEOUT_SECONDS</key>
    <string>${PATTERN_BUY_REFRESH_TIMEOUT_SECONDS:-10}</string>
    <key>PATTERN_BUY_REFRESH_COMMAND_TIMEOUT_SECONDS</key>
    <string>${PATTERN_BUY_REFRESH_COMMAND_TIMEOUT_SECONDS:-3600}</string>
  </dict>
</dict>
</plist>
EOF

  launchctl unload "${PLIST}" >/dev/null 2>&1 || true
  launchctl load "${PLIST}"
  echo "Installed ${LABEL}"
  echo "Schedule: ${RUN_HOUR}:${RUN_MINUTE}"
  echo "Plist: ${PLIST}"
}

uninstall_agent() {
  launchctl unload "${PLIST}" >/dev/null 2>&1 || true
  rm -f "${PLIST}"
  echo "Uninstalled ${LABEL}"
}

case "${1:-}" in
  install)
    install_agent
    ;;
  uninstall)
    uninstall_agent
    ;;
  status)
    launchctl list | grep "${LABEL}" || true
    echo "Plist: ${PLIST}"
    ;;
  run)
    ensure_runner
    "${RUNNER}"
    ;;
  logs)
    mkdir -p "${LOG_DIR}"
    tail -n 200 "${LOG_DIR}/latest.log" 2>/dev/null || tail -n 200 "${LOG_DIR}/launchd.out.log" 2>/dev/null || true
    ;;
  *)
    usage
    exit 2
    ;;
esac
