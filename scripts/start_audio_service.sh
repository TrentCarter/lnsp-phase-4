#!/bin/bash
# Start Audio Service for PAS Agent Swarm HMI

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔊 Starting PAS Audio Service...${NC}"

# Check if port 6103 is already in use
if lsof -ti:6103 > /dev/null 2>&1; then
    echo -e "${RED}Port 6103 is already in use. Stopping existing process...${NC}"
    lsof -ti:6103 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}✗ Virtual environment not found at .venv${NC}"
    exit 1
fi

# Check for required dependencies
echo -e "${BLUE}Checking dependencies...${NC}"

# Check for f5_tts_mlx
if ! python3 -c "import f5_tts_mlx" 2>/dev/null; then
    echo -e "${RED}✗ f5_tts_mlx not found. Please install it first.${NC}"
    echo -e "${BLUE}Install with: pip install f5-tts-mlx${NC}"
    exit 1
fi

# Check for numpy
if ! python3 -c "import numpy" 2>/dev/null; then
    echo -e "${RED}✗ numpy not found. Installing...${NC}"
    pip install numpy
fi

echo -e "${GREEN}✓ All dependencies available${NC}"

# Check if reference audio exists
REF_AUDIO="/Users/trentcarter/Artificial_Intelligence/Voice_Clips/Sophia3.wav"
if [ -f "$REF_AUDIO" ]; then
    echo -e "${GREEN}✓ Reference audio found: $REF_AUDIO${NC}"
else
    echo -e "${RED}⚠ Reference audio not found: $REF_AUDIO${NC}"
    echo -e "${BLUE}TTS will fail unless you provide a valid ref_audio path${NC}"
fi

# Start the audio service
echo -e "${BLUE}Starting Audio Service on port 6103...${NC}"

# Run in foreground (use & to run in background)
PYTHONPATH=. uvicorn services.audio.audio_service:app \
    --host 127.0.0.1 \
    --port 6103 \
    --reload \
    --log-level info

# Alternative: Run in background
# PYTHONPATH=. uvicorn services.audio.audio_service:app \
#     --host 127.0.0.1 \
#     --port 6103 \
#     --reload \
#     --log-level info &
#
# AUDIO_PID=$!
# echo -e "${GREEN}✓ Audio Service started with PID: $AUDIO_PID${NC}"
# echo "To stop: kill $AUDIO_PID"
