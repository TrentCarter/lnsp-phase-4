#!/bin/bash

# Navigate to the project directory
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4/services/file_manager

# Run the FastAPI application
uvicorn app:app --host 0.0.0.0 --port 6102
