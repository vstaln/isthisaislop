#!/usr/bin/env bash
# Wait for the artem9k labeling job to finish, then rebuild train_all.parquet
# so the user wakes up to a training-ready corpus.
#
# Usage: setsid nohup bash scripts/finalize_artem9k.sh > /tmp/finalize.log 2>&1 &

LOG=/tmp/label_artem9k.log
OUT=data/training/spans_artem9k_train.parquet
MAX_WAIT_S=$(( 6 * 3600 ))   # 6h cap; labeling ETA ~3h

echo "[finalize] $(date) waiting for labeling..."
start=$(date +%s)
while :; do
    # done when the final parquet exists
    if [ -f "$OUT" ]; then
        echo "[finalize] $(date) final parquet exists — labeling complete"
        break
    fi
    # or when the labeler process is gone AND no progress in the last 5 min
    if ! pgrep -f label_artem9k.py > /dev/null; then
        last=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
        now=$(date +%s)
        if [ $(( now - last )) -gt 300 ]; then
            echo "[finalize] $(date) labeler process gone, log quiet 5min — assuming done/crashed"
            break
        fi
    fi
    elapsed=$(( $(date +%s) - start ))
    if [ $elapsed -gt $MAX_WAIT_S ]; then
        echo "[finalize] $(date) TIMEOUT after ${elapsed}s — aborting"
        exit 1
    fi
    sleep 60
done

echo "[finalize] $(date) rebuilding train_all.parquet..."
cd /home/vstaln/slop-detector || exit 1
uv run python scripts/build_training_parquet.py 2>&1 | tail -5

echo "[finalize] $(date) done"
