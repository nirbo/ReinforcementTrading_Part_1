#!/bin/bash
# Sequential training script for A/B comparison
# Runs baseline first, then box features

set -e

echo "=========================================="
echo "A/B Training Comparison"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Check if baseline is already running
if pgrep -f "train_baseline.py" > /dev/null; then
    echo "Baseline training already running, waiting for completion..."
    while pgrep -f "train_baseline.py" > /dev/null; do
        sleep 60
        echo "  $(date +%H:%M:%S) - Still running..."
    done
    echo "Baseline training completed!"
else
    echo "Starting baseline training..."
    python scripts/train_baseline.py
fi

echo ""
echo "=========================================="
echo "Starting box features training..."
echo "=========================================="
python scripts/train_box_features.py

echo ""
echo "=========================================="
echo "All training complete!"
echo "End time: $(date)"
echo "=========================================="
