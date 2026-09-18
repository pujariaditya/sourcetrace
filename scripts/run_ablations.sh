#!/usr/bin/env bash
# MLAAD v5 ablation sweep for the paper's Table 2 + Table 1 ID-accuracy.
#
# Runs the full model and five leave-one-out variants, each fit-and-evaluated in one
# shot (evaluate fits at seed 0 when no checkpoint is given). Features are extracted
# ONCE up front; the ablations live in the head/scoring (one configs/*.yaml each), so
# the same 2133-d cache serves every variant.
#
# Usage:  bash scripts/run_ablations.sh
# Output: results/ablation/<variant>.json  and a summary printed at the end.
# --help: the header comment above IS the documentation, so print it rather than
# maintaining a second copy that can drift out of date. Anything other than -h/--help
# is rejected: this script takes no positional arguments and silently ignoring one
# would look like it had been honoured.
usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  "") : ;;
  *) echo "$(basename "$0"): unexpected argument '$1'" >&2
     echo "This script takes no arguments; run '$(basename "$0") --help'." >&2
     exit 2 ;;
esac

set -uo pipefail
cd "$(dirname "$0")/.."

# Machine-local paths (ST_MLAAD_ROOT, ST_FEATURE_CACHE, ...). Gitignored, so it is
# absent on a fresh clone -- fall back to the ST_* defaults in config.py.
[ -f ./.st_env.sh ] && source ./.st_env.sh

# Keep ~/.local site-packages out: a newer huggingface_hub there shadows the env's
# copy and breaks the pinned transformers. See sourcetrace/features/_deps.py.
export PYTHONNOUSERSITE=1

# Determinism lever (must precede CUDA init) -- see method.py._seed_all.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# No hardcoded interpreter: this defaulted to a path under one developer's home
# for a while, which works everywhere that developer runs it and nowhere else.
PY="${PYTHON:-python}"
OUT=results/ablation
mkdir -p "$OUT"

echo "### 1. models (WavLM + EnCodec) ###"
$PY scripts/download_models.py || { echo "model download failed"; exit 1; }

echo "### 2. extract MLAAD v5 features (once) ###"
$PY -m sourcetrace.extract --dataset mlaad_v5 || { echo "extract failed"; exit 1; }

# variant name -> the config that defines it ("" = the frozen champion defaults,
# which configs/base.yaml also encodes and is checked against).
#
# These were ST_ABL_* environment strings living only in this array, so what a
# given results/ablation/*.json had been fit under was recorded nowhere a reader
# would look. Each arm is now a file that states its own override and names the
# result it targets. The env knobs still work and still win over a config file;
# see sourcetrace/config.py.
declare -A VARIANTS=(
  [full]=""
  [no_codec]="configs/ablation_no_codec.yaml"
  [no_signature]="configs/ablation_no_signature.yaml"
  [no_gating]="configs/ablation_no_gating.yaml"
  [conformal_only]="configs/ablation_conformal_only.yaml"
  [rmd_only]="configs/ablation_rmd_only.yaml"
  [fixed_proj]="configs/ablation_fixed_proj.yaml"
)
ORDER=(full no_codec no_signature no_gating conformal_only rmd_only fixed_proj)

echo "### 3. ablation sweep (7 x ~35-40 min) ###"
for v in "${ORDER[@]}"; do
  cfg="${VARIANTS[$v]}"
  echo "--- variant: $v   [${cfg:-champion defaults}] ---"
  "$PY" -m sourcetrace.evaluate \
      --task mlaad_v5 --fit-seed 0 --json "$OUT/$v.json" \
      ${cfg:+--config "$cfg"} \
    || echo "variant $v FAILED (continuing)"
done

echo "### summary ###"
$PY - <<'EOF'
import json, glob, os
rows=[]
for f in sorted(glob.glob("results/ablation/*.json")):
    d=json.load(open(f))
    m=d.get("results", {}).get("mlaad_v5", {})
    rows.append((os.path.basename(f)[:-5], m.get("fpr95"), m.get("ood_eer"), m.get("id_acc")))
print(f"{'variant':16s} {'FPR95':>8s} {'OOD-EER':>8s} {'ID-acc':>8s}")
for n,a,b,c in rows:
    print(f"{n:16s} {a!s:>8} {b!s:>8} {c!s:>8}")
EOF
