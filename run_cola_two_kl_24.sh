#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
WORKER=""
DRY_RUN=0
while (( $# )); do
    case "$1" in
        --worker) WORKER="${2:?--worker requires 0 or 1}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Usage: $0 [--worker 0|1] [--dry-run]" >&2; exit 2 ;;
    esac
done
if [[ -n "$WORKER" && "$WORKER" != 0 && "$WORKER" != 1 ]]; then
    echo "Worker must be 0 or 1" >&2; exit 2
fi
GPU_ID="${GPU_ID:-${WORKER:-0}}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must identify one physical GPU" >&2; exit 2
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LOG_ROOT="$PROJECT_ROOT/logs/cola_two_kl_24"
mkdir -p "$LOG_ROOT"
# New serial queues exclude both workers; worker locks exclude duplicate GPUs.
exec 7>"$LOG_ROOT/schedule.lock"
ARGS=()
if [[ -n "$WORKER" ]]; then
    export COLA_QUEUE_WORKER="$WORKER"
    LOG_DIR="$LOG_ROOT/gpu_$WORKER"
    mkdir -p "$LOG_DIR"
    flock -n -s 7 || { echo "Another serial queue is running" >&2; exit 75; }
    exec 8>"$LOG_ROOT/physical_gpu_$GPU_ID.lock"
    flock -n 8 || { echo "Another worker uses GPU $GPU_ID" >&2; exit 75; }
    exec 9>"$LOG_DIR/queue.lock"
    flock -n 9 || { echo "Another worker $WORKER is running" >&2; exit 75; }
    # Only the registered legacy scheduler can overlap during a safe handoff.
    "$PYTHON_BIN" - "$WORKER" <<'PYCHECK'
import fcntl,json,sys
from pathlib import Path
root=Path('logs/cola_two_kl_24')
with (root/'queue.lock').open('a') as lock:
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  request=root/'handoff_request.json'
  if sys.argv[1]!='1' or not request.exists():
   raise SystemExit('Legacy queue is active; this worker must wait for handoff')
  old=json.loads(request.read_text())['legacy_pid']
  command=(Path('/proc')/str(old)/'cmdline').read_bytes().replace(b'\0',b' ')
  current=json.loads((root/'current.json').read_text())
  if b'cola_two_kl_queue.py' not in command or current['variant'] not in ('full','no_mff'):
   raise SystemExit('Legacy queue ownership cannot be verified')
PYCHECK
    ARGS=(--worker "$WORKER")
else
    LOG_DIR="$LOG_ROOT"
    flock -n 7 || { echo "Another queue or dual-GPU worker is running" >&2; exit 75; }
    exec 9>"$LOG_ROOT/queue.lock"
    flock -n 9 || { echo "Another cola_two_kl_24 queue is running" >&2; exit 75; }
fi
export GPU_ID CUDA_VISIBLE_DEVICES="$GPU_ID" TZ=Asia/Shanghai
if (( DRY_RUN )); then
    exec "$PYTHON_BIN" -u cola_two_kl_queue.py "${ARGS[@]}" --dry-run
fi
exec > >(tee -a "$LOG_DIR/queue.log") 2>&1
printf 'Queue start: PID=%s worker=%s GPU=%s time=%s Python=%s log=%s\n' "$$" "${WORKER:-serial}" "$GPU_ID" "$(date -Is)" "$PYTHON_BIN" "$LOG_DIR/queue.log"
exec "$PYTHON_BIN" -u cola_two_kl_queue.py "${ARGS[@]}"
