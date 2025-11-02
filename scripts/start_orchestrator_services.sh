#!/bin/bash

# Start Orchestrator Encoder/Decoder Services (Ports 7001/7002)
# These services provide the CORRECT encoder/decoder combination

set -e

echo "Starting Orchestrator Encoder/Decoder Services..."
echo "=================================================="
echo ""

# Kill any existing processes on these ports
echo "Checking for existing processes..."
lsof -ti:7001 | xargs -r kill -9 2>/dev/null || true
lsof -ti:7002 | xargs -r kill -9 2>/dev/null || true
sleep 2

# Start encoder service on port 7001
echo "Starting Encoder Service (port 7001)..."
./.venv/bin/uvicorn app.api.orchestrator_encoder_server:app \
  --host 127.0.0.1 \
  --port 7001 \
  > /tmp/orchestrator_encoder.log 2>&1 &

ENCODER_PID=$!
echo "  → Encoder PID: $ENCODER_PID"

# Start decoder service on port 7002
echo "Starting Decoder Service (port 7002)..."
./.venv/bin/uvicorn app.api.orchestrator_decoder_server:app \
  --host 127.0.0.1 \
  --port 7002 \
  > /tmp/orchestrator_decoder.log 2>&1 &

DECODER_PID=$!
echo "  → Decoder PID: $DECODER_PID"

# Wait for services to initialize
echo ""
echo "Waiting for services to initialize..."
sleep 8

# Check health
echo ""
echo "Health Checks:"
echo "=================================================="

if curl -s http://localhost:7001/health > /dev/null 2>&1; then
    echo "✓ Encoder Service (7001): HEALTHY"
    curl -s http://localhost:7001/health | python3 -m json.tool | head -10
else
    echo "✗ Encoder Service (7001): NOT RESPONDING"
    echo "  Check logs: tail -f /tmp/orchestrator_encoder.log"
fi

echo ""

if curl -s http://localhost:7002/health > /dev/null 2>&1; then
    echo "✓ Decoder Service (7002): HEALTHY"
    curl -s http://localhost:7002/health | python3 -m json.tool | head -10
else
    echo "✗ Decoder Service (7002): NOT RESPONDING"
    echo "  Check logs: tail -f /tmp/orchestrator_decoder.log"
fi

echo ""
echo "=================================================="
echo "Services Started Successfully!"
echo ""
echo "Encoder: http://localhost:7001 (PID: $ENCODER_PID)"
echo "Decoder: http://localhost:7002 (PID: $DECODER_PID)"
echo ""
echo "Logs:"
echo "  tail -f /tmp/orchestrator_encoder.log"
echo "  tail -f /tmp/orchestrator_decoder.log"
echo ""
echo "To stop services:"
echo "  kill $ENCODER_PID $DECODER_PID"
echo "=================================================="
