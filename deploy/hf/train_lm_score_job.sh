#!/bin/bash
# One LM-score training run on a Hugging Face Jobs GPU, evaluated and uploaded.
#
#   hf jobs run --flavor a100-large --secrets HF_TOKEN --timeout 6h \
#     -e MODEL=qwen3-4b -e REF=generality -e RUN=qwen3-4b-lms-s0 \
#     -e BENCH_REPO=<public decision benchmark dataset> -e BENCH_REVISION=<sha> \
#     -e BENCH_RENAME="<dir>=incumbent <dir>=hosted-service" \
#     pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime \
#     bash -c "$(cat deploy/hf/train_lm_score_job.sh)"
#
# `train_job.sh`'s setup -- the repository at REF, the pinned corpora, the
# private teacher streams -- then `scripts/train_lm_score.py` with the same mix
# as the local 0.6B run, so the two differ in the backbone alone. Evaluated on
# the generality suite, web actions, and (when BENCH_REPO is given) the public
# decision benchmark beside the predictions it publishes. The benchmark's id
# and its systems' names come from the launch command, not this file.
set -euo pipefail
: "${MODEL:?}" "${REF:=generality}" "${RUN:?}" "${SEED:=0}" "${SCALE:=0.35}"
: "${DATA_REPO:=jacobbeckdev/trigon-private-data}" "${OUT_REPO:=jacobbeckdev/trigon-runs}"
: "${BENCH_REPO:=}" "${BENCH_REVISION:=main}" "${BENCH_RENAME:=}"
cd /tmp
python - <<PY
import io, tarfile, urllib.request
url = "https://codeload.github.com/howlerops/trigon/tar.gz/${REF}"
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())).extractall(".")
PY
cd trigon-*/
# The image's Python is Debian-managed (PEP 668); the container is thrown
# away, and installing over it keeps the image's own CUDA build of torch.
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q -e ".[train,server,convert]" "transformers>=4.51" "huggingface_hub[cli]>=1.0" httpx2
export TRIGON_CORPUS_CACHE=/tmp/corpora TRIGON_WEIGHTS_CACHE=/tmp/backbones PYTHONUNBUFFERED=1
mkdir -p "$TRIGON_CORPUS_CACHE"
hf download "$DATA_REPO" --type dataset --local-dir /tmp/private
for stream in teacher-workflows teacher-local; do
  mkdir -p "$TRIGON_CORPUS_CACHE/$stream"
  cp "/tmp/private/$stream/cases.jsonl.gz" "$TRIGON_CORPUS_CACHE/$stream/"
done
python scripts/convert_corpus.py measuring_hate_speech
python scripts/convert_corpus.py boolq
if [ "${DRY:-0}" = 1 ]; then
  # Setup only, on a CPU flavor: everything a GPU hour would fail on first.
  python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
  python -c "from trigon.backends.hub import BACKBONES; print(BACKBONES['$MODEL'])"
  [ -z "$BENCH_REPO" ] || hf download "$BENCH_REPO" --type dataset --revision "$BENCH_REVISION" --local-dir /tmp/bench
  python - <<'PY'
import sys
sys.path.insert(0, "scripts")
from train_mix import MIX
from trigon.evals.corpora import load
for name in [*MIX, "teacher-local", "mind2web-train", "wanli"]:
    print(name, "train", len(load(name, "train", purpose="train")))
for name in ("banking77", "clinc150", "boolq", "mind2web"):
    print(name, "test", len(load(name, "test", purpose="eval")))
PY
  echo "DRY OK"; exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=/tmp/run
python scripts/train_lm_score.py --model "$MODEL" --device cuda --seed "$SEED" --scale "$SCALE" \
  --max-prompt-tokens 1280 --extra teacher-local=3200 --extra mind2web-train=2864 \
  --extra wanli=4000 --name "$RUN" --out "$OUT"
# Upload after every stage: a failure late in the job must not take the
# trained adapter with it. (It once did: a scoring crash after 2.3 GPU-hours
# exited before the only upload, and the 4B adapter was lost.)
save() { hf upload "$OUT_REPO" "$OUT" "$RUN" --private --exclude "*/raw/*" --commit-message "$RUN: $1" || true; }
save "adapter and calibrators"
python scripts/generality.py --lm "$OUT/adapter.pt" --device cuda -n 1000 --out "$OUT/generality" || true
save "generality"
python scripts/webact.py --lm "$OUT/adapter.pt" --device cuda -n 600 --out "$OUT/webact" || true
save "web actions"
if [ -n "$BENCH_REPO" ]; then
  hf download "$BENCH_REPO" --type dataset --revision "$BENCH_REVISION" --local-dir /tmp/bench
  renames=()
  for pair in $BENCH_RENAME; do renames+=(--rename "$pair"); done
  python scripts/decision_bench.py --cases /tmp/bench/cases --lm "$OUT/adapter.pt" --device cuda \
    --reference /tmp/bench/predictions "${renames[@]}" --out "$OUT/decision-bench" \
    | tee "$OUT/decision-bench.md" || true
  save "decision benchmark"
fi
echo "done: https://huggingface.co/$OUT_REPO/tree/main/$RUN"
