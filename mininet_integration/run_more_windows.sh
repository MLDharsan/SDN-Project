#!/usr/bin/env bash
# Extra Mininet validation windows (run as root from the project root):
#   sudo bash mininet_integration/run_more_windows.sh
# Runs are ordered most-important first. Each run is ~65 min (4 policies x 150 steps). A failed run does not stop the rest.
cd "$(dirname "$0")/.." || exit 1
PY=venv/bin/python
LOG=mininet_results/batch_logs
mkdir -p "$LOG"

run() {  # profile start mode name
  echo "=== $(date '+%F %T') start $4 ===" | tee -a "$LOG/batch.log"
  $PY -m mininet_integration.cleanup >/dev/null 2>&1
  $PY -m mininet_integration.run_mininet_experiment \
      --profile "$1" --start-index "$2" --mode "$3" --steps 150 --run-name "$4" \
      2>&1 | tee "$LOG/$4.log" | grep -E 'done:|Results:|Error|Traceback'
  echo "=== $(date '+%F %T') end $4 (exit ${PIPESTATUS[0]}) ===" | tee -a "$LOG/batch.log"
}

# Crucial (run first)
run real_case_study  5250 current    real_w5250_current
run real_case_study  5250 predictive real_w5250_predictive
run synthetic        600  current    syn_w600_current
run synthetic        600  predictive syn_w600_predictive
# Optional extra windows (Ctrl+C after the crucial four if short of time)
run synthetic        1500 current    syn_w1500_current
run synthetic        1500 predictive syn_w1500_predictive
run real_case_study  7350 current    real_w7350_current
run real_case_study  7350 predictive real_w7350_predictive

$PY -m mininet_integration.cleanup >/dev/null 2>&1
chown -R "${SUDO_UID:-0}:${SUDO_GID:-0}" mininet_results
echo "ALL DONE $(date '+%F %T')" | tee -a "$LOG/batch.log"
