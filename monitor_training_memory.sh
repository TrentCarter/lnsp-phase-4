#!/bin/bash
# Monitor memory usage during training
# Usage: ./monitor_training_memory.sh <pid_of_training_process>

if [ -z "$1" ]; then
  echo "Usage: $0 <pid>"
  echo "Example: ./launch_v4_cpu.sh & ./monitor_training_memory.sh \$!"
  exit 1
fi

TRAIN_PID=$1
LOG_FILE="memory_profile_$(date +%Y%m%d_%H%M%S).log"

echo "Monitoring PID $TRAIN_PID..."
echo "Logging to: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

echo "Time,RSS_GB,VSZ_GB,%MEM" > "$LOG_FILE"

while kill -0 $TRAIN_PID 2>/dev/null; do
  TIMESTAMP=$(date +"%H:%M:%S")
  STATS=$(ps -p $TRAIN_PID -o rss=,vsz=,%mem= | awk '{printf "%.2f,%.2f,%.1f", $1/1024/1024, $2/1024/1024, $3}')
  echo "$TIMESTAMP,$STATS" >> "$LOG_FILE"

  # Also print to console
  RSS_GB=$(echo $STATS | cut -d',' -f1)
  echo "[$TIMESTAMP] RSS: ${RSS_GB} GB"

  sleep 5
done

echo ""
echo "Process $TRAIN_PID terminated"
echo "Peak memory usage:"
tail -n +2 "$LOG_FILE" | sort -t',' -k2 -rn | head -1
echo ""
echo "Full log: $LOG_FILE"
