#!/bin/bash
# Progress check for the artem9k labeling job (scripts/label_artem9k.py)
LOG=${1:-/tmp/label_artem9k.log}
echo "=== last progress ==="
grep "chunk" "$LOG" 2>/dev/null | tail -1
echo "=== parts (docs labeled) ==="
ls /tmp/label_artem9k/part_*.parquet 2>/dev/null | wc -l
echo "=== final parquet? ==="
ls -la data/training/spans_artem9k_train.parquet 2>/dev/null || echo "not yet"
echo "=== running? ==="
pgrep -f label_artem9k.py > /dev/null && echo "YES" || echo "NO (finished/died)"
