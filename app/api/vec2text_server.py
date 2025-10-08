#!/usr/bin/env python3
"""
Vec2Text FastAPI Server
Keeps vec2text models (JXE + IELab) warm in memory for fast decoding
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import numpy as np
import sys
import os
import subprocess
import json

app = FastAPI(
    title="Vec2Text Decoding Server",
    description="Always-on vec2text service for LNSP (JXE + IELab decoders)",
    version="1.0.0"
)

# For now, we'll use subprocess to call the isolated script
# TODO: Refactor vec2text_isolated.py to be importable as a library


class Vec2TextRequest(BaseModel):
    """Request for vec2text decoding"""
    vectors: List[List[float]] = Field(..., description="List of 768D vectors to decode")
    subscribers: str = Field(default="jxe,ielab", description="Comma-separated decoders: jxe,ielab")
    steps: int = Field(default=1, description="Number of decoding steps (1-20)")
    device: str = Field(default="cpu", description="Device: cpu, mps, cuda")


class Vec2TextResponse(BaseModel):
    """Response with decoded texts"""
    results: List[dict]
    count: int


class TextToVecRequest(BaseModel):
    """Request for text encoding (via GTR-T5) then decoding"""
    texts: List[str] = Field(..., description="Texts to encode then decode")
    subscribers: str = Field(default="jxe,ielab", description="Comma-separated decoders")
    steps: int = Field(default=1, description="Number of decoding steps")


@app.on_event("startup")
async def startup():
    """Check vec2text availability on startup"""
    print("Vec2Text server starting...")
    # Check if vec2text script exists
    script_path = os.path.join(
        os.path.dirname(__file__),
        "../../app/vect_text_vect/vec_text_vect_isolated.py"
    )
    if not os.path.exists(script_path):
        print(f"⚠️  Warning: vec2text script not found at {script_path}")
    else:
        print("✅ Vec2Text server ready")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "decoders": ["jxe", "ielab"],
        "dimensions": 768
    }


@app.post("/decode", response_model=Vec2TextResponse)
async def decode_vectors(request: Vec2TextRequest):
    """Decode vectors to text using vec2text"""
    if not request.vectors:
        raise HTTPException(status_code=400, detail="No vectors provided")

    # For now, we save vectors to temp file and call subprocess
    # TODO: Refactor to use imported library instead

    try:
        import tempfile
        import json

        results = []

        for i, vector in enumerate(request.vectors):
            # Create temp file with vector
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({"vector": vector}, f)
                temp_path = f.name

            try:
                # Call vec2text script
                cmd = [
                    "./venv/bin/python3",
                    "app/vect_text_vect/vec_text_vect_isolated.py",
                    "--input-vector-file", temp_path,
                    "--subscribers", request.subscribers,
                    "--vec2text-backend", "isolated",
                    "--output-format", "json",
                    "--steps", str(request.steps),
                    "--devices", request.device
                ]

                env = os.environ.copy()
                env["VEC2TEXT_FORCE_PROJECT_VENV"] = "1"
                env["VEC2TEXT_DEVICE"] = request.device
                env["TOKENIZERS_PARALLELISM"] = "false"

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=60
                )

                if result.returncode == 0:
                    # Parse JSON output
                    output = json.loads(result.stdout)
                    results.append(output)
                else:
                    results.append({
                        "error": result.stderr,
                        "index": i
                    })

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        return Vec2TextResponse(
            results=results,
            count=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")


@app.post("/encode-decode")
async def encode_then_decode(request: TextToVecRequest):
    """Encode texts to vectors (GTR-T5) then decode back (round-trip test)"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        results = []

        for text in request.texts:
            # Call vec2text script with text input
            cmd = [
                "./venv/bin/python3",
                "app/vect_text_vect/vec_text_vect_isolated.py",
                "--input-text", text,
                "--subscribers", request.subscribers,
                "--vec2text-backend", "isolated",
                "--output-format", "json",
                "--steps", str(request.steps)
            ]

            env = os.environ.copy()
            env["VEC2TEXT_FORCE_PROJECT_VENV"] = "1"
            env["VEC2TEXT_DEVICE"] = "cpu"
            env["TOKENIZERS_PARALLELISM"] = "false"

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=60
            )

            if result.returncode == 0:
                output = json.loads(result.stdout)
                results.append(output)
            else:
                results.append({
                    "error": result.stderr,
                    "original_text": text
                })

        return {
            "results": results,
            "count": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encode-decode failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8766,
        log_level="info"
    )
