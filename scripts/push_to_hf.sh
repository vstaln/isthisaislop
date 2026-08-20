#!/usr/bin/env bash
# Push train_all.parquet (1.46GB) to HuggingFace Dataset — no Drive needed.
# Usage:
#   echo hf_*** > /tmp/hf_token && bash scripts/push_to_hf.sh /tmp/hf_token
#   or: HF_TOKEN=hf_*** bash scripts/push_to_hf.sh
#   or: bash scripts/push_to_hf.sh  # will try huggingface-cli login cache

set -euo pipefail
cd "$(dirname "$0")/.."

TOKEN_FILE="${1:-}"
if [[ -n "$TOKEN_FILE" && -f "$TOKEN_FILE" ]]; then
  HF_TOKEN="$(cat "$TOKEN_FILE" | tr -d ' \n')"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "[hf] using token from $TOKEN_FILE (${#HF_TOKEN} chars)"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  echo "[hf] using HF_TOKEN env (${#HF_TOKEN} chars)"
else
  echo "[hf] no token file/env — trying cached login"
fi

# install hub if missing
if ! command -v huggingface-cli >/dev/null; then
  echo "[hf] installing huggingface_hub"
  pip install -q huggingface_hub
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | head -n 5
fi

REPO="vstaln/isthisaislop-data"
echo "[hf] creating dataset repo $REPO if needed"
huggingface-cli repo create "$REPO" --type dataset --exist-ok 2>&1 | head -n 10 || true

SRC="data/training/train_all.parquet"
if [[ ! -f "$SRC" ]]; then echo "missing $SRC"; exit 1; fi
ls -lh "$SRC"

echo "[hf] uploading $SRC → $REPO/train_all.parquet (1.46GB, resumable)"
huggingface-cli upload "$REPO" "$SRC" train_all.parquet --repo-type dataset

echo "[hf] also uploading spans for reproducibility (optional)"
for f in data/training/spans_artem9k_train.parquet data/training/spans_writingprompts_train.parquet data/training/spans_coai_train.parquet; do
  [[ -f "$f" ]] && huggingface-cli upload "$REPO" "$f" "$(basename "$f")" --repo-type dataset || true
done

echo "[hf] done — verify: https://huggingface.co/datasets/$REPO"
echo "[hf] Colab download snippet:"
cat <<'PY'
!pip install -q huggingface_hub
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download("vstaln/isthisaislop-data", "train_all.parquet", repo_type="dataset")
shutil.copy(path, "train_all.parquet")
print("ready", path)
PY
