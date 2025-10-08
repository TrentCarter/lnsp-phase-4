#!/bin/bash
# TMD-LS Lane Management Script
# Manages all 4 Ollama instances for lane specialist architecture

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tmp"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a port is running
check_port() {
    local port=$1
    curl -s http://localhost:$port/api/tags >/dev/null 2>&1
}

# Function to get model name on a port
get_model_name() {
    local port=$1
    local model=$(curl -s http://localhost:$port/api/tags 2>/dev/null | jq -r '.models[0].name' 2>/dev/null || echo "unknown")
    echo "$model"
}

# Function to start all instances
start_all() {
    echo "Starting all TMD-LS lane specialists..."

    # Check if default Ollama is running
    if ! check_port 11434; then
        echo -e "${YELLOW}Starting Llama 3.1 on port 11434 (default)...${NC}"
        echo "Please run 'ollama serve' in a separate terminal for the default instance"
        echo "Waiting 5 seconds..."
        sleep 5
    else
        echo -e "${GREEN}✅ Llama 3.1 already running on port 11434${NC}"
    fi

    # Start TinyLlama
    if ! check_port 11435; then
        echo -e "${YELLOW}Starting TinyLlama on port 11435...${NC}"
        OLLAMA_HOST=127.0.0.1:11435 ollama serve > ${LOG_DIR}/ollama_tinyllama.log 2>&1 &
        echo "PID: $!"
        sleep 3
    else
        echo -e "${GREEN}✅ TinyLlama already running on port 11435${NC}"
    fi

    # Start Phi3
    if ! check_port 11436; then
        echo -e "${YELLOW}Starting Phi3 on port 11436...${NC}"
        OLLAMA_HOST=127.0.0.1:11436 ollama serve > ${LOG_DIR}/ollama_phi3.log 2>&1 &
        echo "PID: $!"
        sleep 3
    else
        echo -e "${GREEN}✅ Phi3 already running on port 11436${NC}"
    fi

    # Start Granite3
    if ! check_port 11437; then
        echo -e "${YELLOW}Starting Granite3 on port 11437...${NC}"
        OLLAMA_HOST=127.0.0.1:11437 ollama serve > ${LOG_DIR}/ollama_granite.log 2>&1 &
        echo "PID: $!"
        sleep 3
    else
        echo -e "${GREEN}✅ Granite3 already running on port 11437${NC}"
    fi

    echo ""
    status
}

# Function to stop all instances
stop_all() {
    echo "Stopping all TMD-LS lane specialists..."
    pkill -f "OLLAMA_HOST=127.0.0.1:11435 ollama serve" || echo "TinyLlama (11435) not running"
    pkill -f "OLLAMA_HOST=127.0.0.1:11436 ollama serve" || echo "Phi3 (11436) not running"
    pkill -f "OLLAMA_HOST=127.0.0.1:11437 ollama serve" || echo "Granite3 (11437) not running"
    echo -e "${YELLOW}Note: Default Ollama (11434) not stopped. Use 'pkill ollama' to stop all.${NC}"
}

# Function to show status
status() {
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║              TMD-LS LANE SPECIALIST STATUS                           ║"
    echo "╠══════════════════════════════════════════════════════════════════════╣"

    # Port 11434
    if check_port 11434; then
        model=$(get_model_name 11434)
        echo -e "║ Port 11434 (Llama 3.1):   ${GREEN}✅ Running${NC} ($model)"
    else
        echo -e "║ Port 11434 (Llama 3.1):   ${RED}❌ Down${NC}"
    fi

    # Port 11435
    if check_port 11435; then
        model=$(get_model_name 11435)
        echo -e "║ Port 11435 (TinyLlama):   ${GREEN}✅ Running${NC} ($model)"
    else
        echo -e "║ Port 11435 (TinyLlama):   ${RED}❌ Down${NC}"
    fi

    # Port 11436
    if check_port 11436; then
        model=$(get_model_name 11436)
        echo -e "║ Port 11436 (Phi3):        ${GREEN}✅ Running${NC} ($model)"
    else
        echo -e "║ Port 11436 (Phi3):        ${RED}❌ Down${NC}"
    fi

    # Port 11437
    if check_port 11437; then
        model=$(get_model_name 11437)
        echo -e "║ Port 11437 (Granite3):    ${GREEN}✅ Running${NC} ($model)"
    else
        echo -e "║ Port 11437 (Granite3):    ${RED}❌ Down${NC}"
    fi

    echo "╚══════════════════════════════════════════════════════════════════════╝"
}

# Function to test all instances
test_all() {
    echo "Testing all TMD-LS lane specialists..."
    echo ""

    local test_prompt="What is AI? Answer in one sentence."

    # Test each port
    for port in 11434 11435 11436 11437; do
        if check_port $port; then
            model=$(get_model_name $port)
            echo -e "${GREEN}Testing port $port ($model)...${NC}"

            response=$(curl -s http://localhost:$port/api/generate \
                -d "{\"model\": \"$model\", \"prompt\": \"$test_prompt\", \"stream\": false}" \
                2>/dev/null | jq -r '.response' 2>/dev/null | head -c 100)

            if [ -n "$response" ]; then
                echo "  Response: ${response}..."
                echo -e "  ${GREEN}✅ PASS${NC}"
            else
                echo -e "  ${RED}❌ FAIL${NC}"
            fi
        else
            echo -e "${RED}Port $port not running${NC}"
        fi
        echo ""
    done
}

# Function to show logs
logs() {
    local service=$1

    case $service in
        tinyllama|11435)
            tail -f ${LOG_DIR}/ollama_tinyllama.log
            ;;
        phi3|11436)
            tail -f ${LOG_DIR}/ollama_phi3.log
            ;;
        granite|granite3|11437)
            tail -f ${LOG_DIR}/ollama_granite.log
            ;;
        *)
            echo "Usage: $0 logs [tinyllama|phi3|granite]"
            exit 1
            ;;
    esac
}

# Main command dispatcher
case "${1:-status}" in
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        sleep 2
        start_all
        ;;
    status)
        status
        ;;
    test)
        test_all
        ;;
    logs)
        logs $2
        ;;
    *)
        echo "TMD-LS Lane Management Script"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|test|logs}"
        echo ""
        echo "Commands:"
        echo "  start    - Start all lane specialist instances"
        echo "  stop     - Stop all lane specialist instances (except default)"
        echo "  restart  - Restart all instances"
        echo "  status   - Show status of all instances"
        echo "  test     - Test all instances with a sample query"
        echo "  logs     - View logs (logs tinyllama|phi3|granite)"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 status"
        echo "  $0 test"
        echo "  $0 logs tinyllama"
        exit 1
        ;;
esac
