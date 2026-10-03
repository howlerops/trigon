#!/bin/bash
# One broad training run on a Hugging Face Jobs GPU, evaluated and uploaded.
#
#   hf jobs run --flavor a100-large --secrets HF_TOKEN --timeout 8h \
#     -e BACKBONE=qwen3-4b -e REF=generality -e RUN=qwen3-4b-broad-s0 \
#     pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime \
#     bash -c "$(cat deploy/hf/train_job.sh)"
#
# Everything a run reads is fetched here: the repository at REF from GitHub,
# the public corpora from their pinned sources, and the two generated teacher
# streams from a private dataset repository. The backbone's forward is checked
# against `transformers` before a GPU-hour is spent on training it, and the
# bundle and both evaluations go to a private model repository.
set -euo pipefail
: "${BACKBONE:?}" "${REF:=generality}" "${RUN:?}" "${EPOCHS:=4}" "${SEED:=0}"
: "${DATA_REPO:=jacobbeckdev/trigon-private-data}" "${OUT_REPO:=jacobbeckdev/trigon-runs}"
cd /tmp
python - <<PY
import io, tarfile, urllib.request
url = "https://codeload.github.com/howlerops/trigon/tar.gz/${REF}"
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())).extractall(".")
PY
cd trigon-*/
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
python scripts/backbone_parity.py --backbone "$BACKBONE" | tee /tmp/parity.log
grep -q '^PARITY' /tmp/parity.log || { echo "backbone parity failed; not training"; exit 1; }
OUT=/tmp/run
python scripts/train_mix.py --backbone "$BACKBONE" --device cuda --epochs "$EPOCHS" --seed "$SEED" \
  --extra teacher-local=3200 --extra mind2web-train=2864 --extra wanli=6000 --out "$OUT"
python scripts/generality.py --bundle "$OUT" -n 1000 --out "$OUT/generality"
python scripts/webact.py --bundle "$OUT" -n 600 --out "$OUT/webact"
cp /tmp/parity.log "$OUT/"
rm -f "$OUT/resume.pt"; rm -rf "$OUT/generality/raw" "$OUT/webact/raw"
hf upload "$OUT_REPO" "$OUT" "$RUN" --commit-message "$RUN: $BACKBONE, $EPOCHS epochs, seed $SEED"
echo "done: https://huggingface.co/$OUT_REPO/tree/main/$RUN"
