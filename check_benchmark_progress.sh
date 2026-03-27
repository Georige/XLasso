#!/bin/bash
# Check benchmark progress

EXP_DIR="/home/lili/lyn/clear/NLasso/XLasso/experiments/results/benchmark"
LOG_FILE="/tmp/benchmark_progress.log"
OUTPUT_FILE="/tmp/claude-1004/-home-lili-lyn/d8aa9e88-68c7-47a4-9ada-d6e58c0cc00f/tasks/bd6m6yob8.output"

echo "=== Benchmark Progress Check: $(date) ===" >> $LOG_FILE

# Find latest experiment directory
LATEST_DIR=$(ls -t "$EXP_DIR" 2>/dev/null | head -1)
if [ -n "$LATEST_DIR" ]; then
    FULL_PATH="$EXP_DIR/$LATEST_DIR"
    echo "Latest experiment: $LATEST_DIR" >> $LOG_FILE
    echo "Contents:" >> $LOG_FILE
    ls -la "$FULL_PATH" >> $LOG_FILE 2>&1

    # Check for raw.csv to see how many trials completed
    if [ -f "$FULL_PATH/raw.csv" ]; then
        LINES=$(wc -l < "$FULL_PATH/raw.csv")
        echo "Raw results: $LINES lines (including header = $(($LINES - 1)) trials)" >> $LOG_FILE

        # Count by model
        echo "Trials per model:" >> $LOG_FILE
        cut -d',' -f3 "$FULL_PATH/raw.csv" | sort | uniq -c >> $LOG_FILE 2>&1
    else
        echo "No raw.csv yet" >> $LOG_FILE
    fi
fi

# Check if process is still running
if ps aux | grep -v grep | grep "bd6m6yob8" > /dev/null 2>&1; then
    echo "Status: RUNNING" >> $LOG_FILE
else
    echo "Status: COMPLETED or STOPPED" >> $LOG_FILE
fi

# Check output
if [ -f "$OUTPUT_FILE" ]; then
    echo "Last 15 lines of output:" >> $LOG_FILE
    tail -15 "$OUTPUT_FILE" >> $LOG_FILE 2>&1
fi

echo "---" >> $LOG_FILE