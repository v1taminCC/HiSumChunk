#!/bin/bash
# CR baseline full run: description generation (SGLang) -> stop server -> GPU embedding -> retrieval -> generation -> judging -> report
cd "$(dirname "$0")" || exit 1
LOG=logs/full_run.log
PY=${PY:-python}
echo "=== [$(date '+%F %T')] CR full run started ===" >> "$LOG"
$PY scripts/01_gen_contexts.py --yes >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 01 FAILED" >> "$LOG"; exit 1; }
echo "=== [$(date '+%F %T')] description generation done; stopping SGLang ===" >> "$LOG"
pkill -f sglang.launch_server
for i in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && break
  sleep 10
done
nvidia-smi --query-gpu=memory.used --format=csv >> "$LOG" 2>&1
$PY scripts/02_build_ctx_chunks.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 02 FAILED" >> "$LOG"; exit 1; }
$PY scripts/03_embed_ctx_chunks.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 03 FAILED" >> "$LOG"; exit 1; }
$PY scripts/04_retrieve_top10.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 04 FAILED" >> "$LOG"; exit 1; }
$PY scripts/05_build_top5_contexts.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 05 FAILED" >> "$LOG"; exit 1; }
$PY scripts/06_generate_answers.py --yes >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 06 FAILED" >> "$LOG"; exit 1; }
$PY scripts/07_judge_answers.py --yes >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 07 FAILED" >> "$LOG"; exit 1; }
$PY scripts/08_report.py >> "$LOG" 2>&1 || { echo "[$(date '+%F %T')] 08 FAILED" >> "$LOG"; exit 1; }
echo "=== [$(date '+%F %T')] CR full run finished (see reports/cr_report.txt) ===" >> "$LOG"
