#!/usr/bin/env bash
# Run ESMFold embedding extraction across 4 GPUs in parallel.
#
# Usage:
#   bash scripts/run_parallel_extraction.sh
#
# Logs are written to logs/extract_rank{0..3}.log
# All 4 processes share the same output dir; skipping already-existing .pt files
# means it's safe to rerun after interruption.

set -euo pipefail

INPUT_DIR="${1:-$HOME/data/pdb}"
OUTPUT_DIR="${2:-$HOME/data/esm_fold_embeddings}"
WORLD_SIZE=4
MAX_SEQ_LEN=512
MIN_PLDDT=70.0
BATCH_SIZE=4   # fp16 via autocast; pTM patch handles NaN in batched inference

mkdir -p logs "$OUTPUT_DIR"

echo "Starting extraction: $INPUT_DIR -> $OUTPUT_DIR"
echo "world_size=$WORLD_SIZE  max_seq_len=$MAX_SEQ_LEN  min_plddt=$MIN_PLDDT  batch_size=$BATCH_SIZE"

PIDS=()
for RANK in 0 1 2 3; do
    LOG="logs/extract_rank${RANK}.log"
    echo "  Launching rank $RANK on cuda:$RANK  (log: $LOG)"
    uv run python scripts/extract_embeddings.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --device "cuda:$RANK" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --min_plddt "$MIN_PLDDT" \
        --batch_size "$BATCH_SIZE" \
        --parse_workers 8 \
        --sort_buffer 64 \
        --rank "$RANK" \
        --world_size "$WORLD_SIZE" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "All ranks launched (PIDs: ${PIDS[*]}). Waiting..."
FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  rank $i finished OK"
    else
        echo "  rank $i FAILED (exit $?)"
        FAIL=1
    fi
done

if [[ $FAIL -eq 0 ]]; then
    echo "Extraction complete. Output: $OUTPUT_DIR"
    echo "Next step — CHEAP encoding (dim=64, min_ptm=0.7):"
    echo ""
    echo "  torchrun --nproc_per_node=4 scripts/encode_cheap.py \\"
    echo "      --input_dir $OUTPUT_DIR \\"
    echo "      --output_dir \$HOME/data/cheap64 \\"
    echo "      --s_s_dim 64 \\"
    echo "      --min_ptm 0.7 \\"
    echo "      --max_len $MAX_SEQ_LEN \\"
    echo "      --batch_size 64"
else
    echo "Some ranks failed. Check logs/extract_rank*.log"
    exit 1
fi
