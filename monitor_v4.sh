#!/bin/bash
while true; do
  clear
  echo "=== V4 Training Monitor ==="
  echo "Time: $(date)"
  echo ""
  ps aux | grep "19596" | grep -v grep | awk '{print "CPU: " $3 "% | Memory: " $4 "%"}'
  echo ""
  tail -3 logs/twotower_v4_cpu_20251021_122123.log | grep "Training:" | tail -1
  echo ""
  echo "Estimated completion: ~1:30-2:00 PM EDT"
  echo "Press Ctrl+C to stop monitoring"
  sleep 10
done
