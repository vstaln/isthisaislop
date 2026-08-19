#!/usr/bin/env bash
# Overnight pipeline: finish artem9k labeling → label writingprompts → rebuild
# train_all.parquet → write a DONE marker. Safe to rerun (each step is resumable).
#
# Usage: setsid nohup bash scripts/overnight.sh > /tmp/overnight.log 2>&1 &

cd /home/vstaln/slop-detector || exit 1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a /tmp/overnight.log; }

log "=== overnight pipeline start ==="

# Step 1: artem9k labeling (resumable, will skip finished parts)
log "STEP 1: artem9k labeling (resumes if partially done)"
uv run python scripts/label_artem9k.py --human-n 200000 --workers 4 --chunk 1000 >> /tmp/overnight.log 2>&1
log "STEP 1 done: $(ls /tmp/label_artem9k/part_*.parquet 2>/dev/null | wc -l) parts, out=$(ls -la data/training/spans_artem9k_train.parquet 2>/dev/null | awk '{print $5}')"

# Step 2: writingprompts labeling (100k human fiction)
log "STEP 2: writingprompts labeling (100k)"
uv run python scripts/label_writingprompts.py --n 100000 --workers 4 --chunk 1000 >> /tmp/overnight.log 2>&1
log "STEP 2 done: $(ls /tmp/label_writingprompts/part_*.parquet 2>/dev/null | wc -l) parts, out=$(ls -la data/training/spans_writingprompts_train.parquet 2>/dev/null | awk '{print $5}')"

# Step 3: rebuild train_all.parquet (now includes artem9k + writingprompts)
log "STEP 3: rebuild train_all.parquet"
uv run python scripts/build_training_parquet.py >> /tmp/overnight.log 2>&1
log "STEP 3 done"

# Step 4: verify
log "STEP 4: verify"
uv run python - <<'EOF' >> /tmp/overnight.log 2>&1
import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from fine_tune_lfm import load_rows
from pathlib import Path
from collections import Counter
rows = load_rows(None, Path("data/training/train_all.parquet"), False)
print(f"VERIFY: {len(rows)} rows")
print(f"  labels: {Counter(r['label'] for r in rows)}")
print(f"  registers: {Counter(r['register'] for r in rows)}")
EOF
log "VERIFY done"

# Done marker
touch data/training/OVERNIGHT_DONE
log "=== OVERNIGHT COMPLETE ==="
