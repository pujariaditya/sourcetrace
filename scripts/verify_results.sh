#!/usr/bin/env bash
# One-command end-to-end verification of the MLAAD v5 + STOPA numbers.
#
# Stages (each echoed):
#   1. download the frontends (WavLM-Large + EnCodec-24kHz)
#   2. datasets: download STOPA; MLAAD is assumed present (huge) unless REPRO_DOWNLOAD_MLAAD=1
#   3. extract per-group features for each task
#   4. train (fit) a Method per task and checkpoint it
#   5. evaluate each checkpoint vs SOTA and write results JSON
#
# Everything is parameterized by the sourcetrace ST_* env vars plus the REPRO_*
# knobs below. Re-running is safe: downloads resume, feature extraction skips
# cached groups, and checkpoints are overwritten.
#
# Env (with defaults):
#   ST_DATA_ROOT            base for datasets/caches      (default: ./data)
#   ST_MLAAD_ROOT/…         see sourcetrace/config.py
#   PYTHON                  interpreter                   (default: python)
#   REPRO_TASKS             space list of tasks           (default: "mlaad_v5 stopa")
#   REPRO_CKPT_DIR          checkpoint dir                (default: <root>/checkpoints)
#   REPRO_MLAAD_SEED        MLAAD v5 fit seed             (default: 0 -> FPR95 0.50)
#   REPRO_RESULTS           results JSON path             (default: <root>/runs/verify.json)
#   REPRO_DOWNLOAD_MLAAD=1  also pull MLAAD (~242GB)      (default: skip; assume present)
#   REPRO_SKIP_MODELS=1     skip the model download stage
#   REPRO_SKIP_STOPA_DL=1   skip the STOPA download stage
# --help: the header comment above IS the documentation, so print it rather than
# maintaining a second copy that can drift out of date. Anything other than -h/--help
# is rejected: this script takes no positional arguments and silently ignoring one
# would look like it had been honoured.
usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") : ;;
  *) echo "$(basename "$0"): unexpected argument '$1'" >&2
     echo "This script takes no arguments; run '$(basename "$0") --help'." >&2
     exit 2 ;;
esac

set -euo pipefail

# Keep ~/.local/lib/pythonX.Y/site-packages out of the interpreter. A newer
# huggingface_hub installed there shadows the env's copy, and transformers (pinned
# >=4.38,<4.40 for EnCodec) then refuses to import -- suggesting an upgrade that
# would break the pin. See sourcetrace/features/_deps.py.
export PYTHONNOUSERSITE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# `python -m sourcetrace.*` and the shipped assets/ and configs/ both resolve
# from the repository root, so the package entry points run from there -- and so
# the defaults below are anchored there rather than at the caller's cwd.
ROOT="$(cd "$HERE/.." && pwd)"

PYTHON="${PYTHON:-python}"
TASKS="${REPRO_TASKS:-mlaad_v5 stopa}"
CKPT_DIR="${REPRO_CKPT_DIR:-$ROOT/checkpoints}"
MLAAD_SEED="${REPRO_MLAAD_SEED:-0}"   # seed 0 + split seed 42 -> MLAAD v5 FPR95 0.50
RESULTS="${REPRO_RESULTS:-$ROOT/runs/verify.json}"

echo "======================================================================"
echo " sourcetrace verify  |  tasks: $TASKS  |  python: $PYTHON"
echo "======================================================================"
mkdir -p "$CKPT_DIR" "$(dirname "$RESULTS")"

# --- stage 1: models ------------------------------------------------------ #
if [[ "${REPRO_SKIP_MODELS:-0}" == "1" ]]; then
  echo "[verify] (1/5) models: SKIPPED (REPRO_SKIP_MODELS=1)"
else
  echo "[verify] (1/5) downloading frontends (WavLM + EnCodec) ..."
  "$PYTHON" "$HERE/download_models.py"
fi

# --- stage 2: datasets ---------------------------------------------------- #
echo "[verify] (2/5) datasets ..."
if [[ " $TASKS " == *" stopa "* && "${REPRO_SKIP_STOPA_DL:-0}" != "1" ]]; then
  echo "[verify]   STOPA: download + extract"
  bash "$HERE/fetch_stopa.sh"
else
  echo "[verify]   STOPA download: skipped"
fi
if [[ " $TASKS " == *" mlaad_v5 "* ]]; then
  if [[ "${REPRO_DOWNLOAD_MLAAD:-0}" == "1" ]]; then
    echo "[verify]   MLAAD: downloading (~242GB) ..."
    "$PYTHON" "$HERE/fetch_mlaad.py"
  else
    echo "[verify]   MLAAD: assumed already present at ST_MLAAD_ROOT "
    echo "               (set REPRO_DOWNLOAD_MLAAD=1 to fetch)."
  fi
fi

# --- stages 3-5: per task ------------------------------------------------- #
for task in $TASKS; do
  echo "----------------------------------------------------------------------"
  echo "[verify] TASK: $task"
  echo "[verify] (3/5) extracting features ..."
  (cd "$ROOT" && "$PYTHON" -m sourcetrace.extract --dataset "$task")

  echo "[verify] (4/5) training + checkpoint ..."
  if [[ "$task" == "mlaad_v5" ]]; then
    # Determinism lever, and it must precede CUDA init -- see method.py._seed_all.
    (cd "$ROOT" && CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
      "$PYTHON" -m sourcetrace.train --task "$task" \
      --seed "$MLAAD_SEED" --checkpoint "$CKPT_DIR/$task.pt")
  else
    (cd "$ROOT" && "$PYTHON" -m sourcetrace.train --task "$task" \
      --checkpoint "$CKPT_DIR/$task.pt")
  fi
done

echo "----------------------------------------------------------------------"
echo "[verify] (5/5) evaluating vs SOTA ..."
EVAL_ARGS=()
if [[ " $TASKS " == *" mlaad_v5 "* && " $TASKS " == *" stopa "* ]]; then
  EVAL_ARGS=(--task both
             --checkpoint-mlaad "$CKPT_DIR/mlaad_v5.pt"
             --checkpoint-stopa "$CKPT_DIR/stopa.pt")
else
  for task in $TASKS; do
    EVAL_ARGS=(--task "$task" --checkpoint "$CKPT_DIR/$task.pt")
  done
fi
(cd "$ROOT" && "$PYTHON" -m sourcetrace.evaluate "${EVAL_ARGS[@]}" --json "$RESULTS")

echo "======================================================================"
echo "[verify] DONE. checkpoints in $CKPT_DIR ; results in $RESULTS"
echo "======================================================================"
