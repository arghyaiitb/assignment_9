#!/bin/bash

echo "Killing stuck training processes..."

# Kill all python processes running main.py or train.py
pkill -f "python.*main\.py"
pkill -f "python.*train\.py"

# Also kill by PIDs if we know them (from nvidia-smi output)
for pid in 45255 45256 45257 45258 45259 45260 45261 45262; do
    if ps -p $pid > /dev/null 2>&1; then
        echo "Killing PID $pid"
        kill -9 $pid 2>/dev/null
    fi
done

# Wait a moment
sleep 2

# Check if any processes are still running
remaining=$(ps aux | grep -E "python.*(main|train)\.py" | grep -v grep | wc -l)
if [ $remaining -eq 0 ]; then
    echo "All training processes killed successfully"
else
    echo "Warning: $remaining training processes may still be running"
    ps aux | grep -E "python.*(main|train)\.py" | grep -v grep
fi

# Clear GPU memory
echo "Checking GPU status..."
nvidia-smi
