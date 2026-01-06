#!/bin/bash
# Wait for baseline training to complete, then start box features training

echo "Waiting for baseline training to complete..."
echo "Started: $(date)"

while pgrep -f "train_baseline.py" > /dev/null; do
    PROGRESS=$(tail -1 logs/baseline_training.log 2>/dev/null | grep -oP '\d+/10,000,000' || echo "checking...")
    echo -ne "\r$(date +%H:%M:%S) - Progress: $PROGRESS"
    sleep 30
done

echo ""
echo "Baseline training completed at: $(date)"
echo ""

# Check baseline results
if [ -f models/baseline_no_box/*/final_stats.json ]; then
    echo "Baseline stats:"
    cat models/baseline_no_box/*/final_stats.json
    echo ""
fi

echo "Starting box features training..."
nohup python scripts/train_box_features.py > logs/box_features_training.log 2>&1 &
BOX_PID=$!
echo "Box features training started with PID: $BOX_PID"
echo "Monitor with: tail -f logs/box_features_training.log"
