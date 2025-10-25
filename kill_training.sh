#!/bin/bash

echo "Killing stuck training processes..."

# Show current training processes
echo "Current training processes:"
ps aux | grep -E "python.*(main|train)\.py" | grep -v grep || echo "None found"

# Kill all python processes running main.py or train.py
pkill -9 -f "python.*main\.py"
pkill -9 -f "python.*train\.py"

# Also kill any PyTorch multiprocessing spawned processes
pkill -9 -f "torch.*multiprocessing"

# Kill any processes holding the distributed training port
PORT_PIDS=$(lsof -ti:12355 2>/dev/null)
if [ ! -z "$PORT_PIDS" ]; then
    echo "Killing processes holding port 12355: $PORT_PIDS"
    kill -9 $PORT_PIDS 2>/dev/null
fi

# Wait a moment
sleep 2

# Check if any processes are still running
remaining=$(ps aux | grep -E "python.*(main|train)\.py" | grep -v grep | wc -l)
if [ $remaining -eq 0 ]; then
    echo "✓ All training processes killed successfully"
else
    echo "⚠ Warning: $remaining training processes may still be running"
    ps aux | grep -E "python.*(main|train)\.py" | grep -v grep
fi

# Check if port is now free
if lsof -ti:12355 > /dev/null 2>&1; then
    echo "⚠ Warning: Port 12355 is still in use"
    lsof -ti:12355
else
    echo "✓ Port 12355 is now free"
fi

# Clear GPU memory
echo ""
echo "GPU status:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "No GPU processes found"
