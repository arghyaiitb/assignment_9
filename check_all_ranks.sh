#!/bin/bash
# Script to check the last debug logs from all ranks to identify where each is stuck

LOG_DIR="${1:-/data/logs}"

echo "==========================================="
echo "Checking DEBUG logs from all 8 ranks"
echo "==========================================="
echo ""

for rank in 0 1 2 3 4 5 6 7; do
    if [ $rank -eq 0 ]; then
        logfile="$LOG_DIR/train.log"
    else
        logfile="$LOG_DIR/train_rank${rank}.log"
    fi
    
    echo "========== RANK $rank =========="
    if [ -f "$logfile" ]; then
        echo "Last 10 DEBUG logs:"
        grep "\[DEBUG\]" "$logfile" | tail -10
        echo ""
    else
        echo "Log file not found: $logfile"
        echo ""
    fi
done

echo "==========================================="
echo "Summary: Where is each rank?"
echo "==========================================="
for rank in 0 1 2 3 4 5 6 7; do
    if [ $rank -eq 0 ]; then
        logfile="$LOG_DIR/train.log"
    else
        logfile="$LOG_DIR/train_rank${rank}.log"
    fi
    
    if [ -f "$logfile" ]; then
        last_debug=$(grep "\[DEBUG\]" "$logfile" | tail -1)
        echo "Rank $rank: $last_debug"
    fi
done

