#!/bin/bash
# P-arm full run: generation -> judging -> aggregation; everything logged to logs/full_run.log
cd "$(dirname "$0")" || exit 1
LOG=logs/full_run.log
PY=${PY:-python}
echo "=== [$(date '+%F %T')] P-arm full run started ===" >> "$LOG"
$PY scripts/02_generate_answers.py --yes >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 02 FAILED" >> "$LOG"; exit 1; }
echo "=== [$(date '+%F %T')] generation done ===" >> "$LOG"
$PY scripts/03_judge_answers.py --yes >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 03 FAILED" >> "$LOG"; exit 1; }
echo "=== [$(date '+%F %T')] judging done ===" >> "$LOG"
$PY scripts/04_aggregate_placebo.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 04 FAILED" >> "$LOG"; exit 1; }
echo "=== [$(date '+%F %T')] P-arm full run finished (see reports/placebo_report.txt) ===" >> "$LOG"
