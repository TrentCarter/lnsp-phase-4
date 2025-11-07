#!/usr/bin/env python3
"""
Audio Service for PAS Agent Swarm HMI

Provides unified audio endpoints for:
- Text-to-Speech (TTS) using f5_tts_mlx
- MIDI note playback for sequencer events
- Tone/beep generation
- Audio file playback
- Volume control and audio mixing

Usage:
    uvicorn services.audio.audio_service:app --host 127.0.0.1 --port 6103 --reload
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PAS Audio Service",
    description="Unified audio API for HMI sound effects, TTS, and music",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
AUDIO_DIR = Path("/tmp/pas_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Default reference voice for TTS
DEFAULT_REF_AUDIO = Path("/Users/trentcarter/Artificial_Intelligence/Voice_Clips/Sophia3.wav")
DEFAULT_REF_TEXT = "Hi babe. I just wanted to wish you good night with my voice because I miss you and I wanted to share that."

# Audio state
audio_state = {
    "master_volume": 0.7,
    "tts_enabled": True,
    "notes_enabled": True,
    "current_playback": None,
    "queue": []
}


# ============================================================================
# Pydantic Models
# ============================================================================

class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis"""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=1000)
    ref_audio: Optional[str] = Field(None, description="Path to reference audio file")
    ref_text: Optional[str] = Field(None, description="Reference text matching ref_audio")
    speed: float = Field(1.0, description="Speech speed multiplier", ge=0.5, le=2.0)
    method: str = Field("midpoint", description="Generation method (midpoint, euler, rk4)")
    auto_play: bool = Field(True, description="Automatically play generated audio")


class NoteRequest(BaseModel):
    """Request model for MIDI note playback"""
    note: int = Field(..., description="MIDI note number (21-108, middle C=60)", ge=21, le=108)
    duration: float = Field(0.3, description="Note duration in seconds", ge=0.05, le=5.0)
    velocity: int = Field(100, description="Note velocity/volume (0-127)", ge=0, le=127)
    instrument: str = Field("piano", description="Instrument type (piano, sine, square, sawtooth)")


class ToneRequest(BaseModel):
    """Request model for tone/beep generation"""
    frequency: float = Field(440.0, description="Frequency in Hz", ge=20.0, le=20000.0)
    duration: float = Field(0.2, description="Duration in seconds", ge=0.01, le=5.0)
    volume: float = Field(0.5, description="Volume (0.0-1.0)", ge=0.0, le=1.0)
    waveform: str = Field("sine", description="Waveform type (sine, square, sawtooth, triangle)")


class FilePlayRequest(BaseModel):
    """Request model for audio file playback"""
    file_path: str = Field(..., description="Path to audio file")
    volume: Optional[float] = Field(None, description="Playback volume (0.0-1.0)", ge=0.0, le=1.0)


class VolumeUpdate(BaseModel):
    """Request model for volume control"""
    master_volume: float = Field(..., description="Master volume (0.0-1.0)", ge=0.0, le=1.0)


class AudioResponse(BaseModel):
    """Standard response model for audio operations"""
    success: bool
    message: str
    audio_file: Optional[str] = None
    duration: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "audio_service",
        "port": 6103,
        "timestamp": datetime.utcnow().isoformat(),
        "master_volume": audio_state["master_volume"],
        "tts_enabled": audio_state["tts_enabled"],
        "notes_enabled": audio_state["notes_enabled"]
    }


@app.get("/status")
async def get_status():
    """Get current audio service status"""
    return {
        "master_volume": audio_state["master_volume"],
        "tts_enabled": audio_state["tts_enabled"],
        "notes_enabled": audio_state["notes_enabled"],
        "current_playback": audio_state["current_playback"],
        "queue_length": len(audio_state["queue"]),
        "temp_audio_dir": str(AUDIO_DIR),
        "ref_audio_exists": DEFAULT_REF_AUDIO.exists() if DEFAULT_REF_AUDIO else False
    }


# ============================================================================
# Text-to-Speech (TTS)
# ============================================================================

@app.post("/audio/tts", response_model=AudioResponse)
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate speech from text using f5_tts_mlx

    Example:
        curl -X POST http://localhost:6103/audio/tts \\
          -H "Content-Type: application/json" \\
          -d '{"text": "Agent Alpha has completed the task.", "speed": 1.0}'
    """
    if not audio_state["tts_enabled"]:
        return AudioResponse(
            success=False,
            message="TTS is disabled in settings"
        )

    try:
        logger.info(f"TTS request: '{request.text}' (speed={request.speed})")

        # Resolve reference audio
        ref_audio = request.ref_audio or str(DEFAULT_REF_AUDIO)
        ref_text = request.ref_text or DEFAULT_REF_TEXT

        if not Path(ref_audio).exists():
            raise HTTPException(status_code=400, detail=f"Reference audio not found: {ref_audio}")

        # Create output file
        timestamp = int(time.time() * 1000)
        output_file = AUDIO_DIR / f"tts_{timestamp}.wav"

        # Build f5_tts_mlx command
        cmd = [
            "python3", "-m", "f5_tts_mlx.generate",
            "--ref-audio", ref_audio,
            "--ref-text", ref_text,
            "--text", request.text,
            "--method", request.method,
            "--speed", str(request.speed),
            "--output", str(output_file)
        ]

        logger.info(f"Running TTS command: {' '.join(cmd)}")

        # Run TTS generation (blocking, but should be fast with MLX)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.error(f"TTS generation failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {result.stderr}")

        if not output_file.exists():
            raise HTTPException(status_code=500, detail="TTS output file not created")

        logger.info(f"TTS generated: {output_file} ({output_file.stat().st_size} bytes)")

        # Auto-play if requested
        if request.auto_play:
            background_tasks.add_task(play_audio_file, str(output_file))

        return AudioResponse(
            success=True,
            message=f"TTS generated successfully",
            audio_file=str(output_file)
        )

    except subprocess.TimeoutExpired:
        logger.error("TTS generation timed out")
        raise HTTPException(status_code=504, detail="TTS generation timed out")
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MIDI Note Playback
# ============================================================================

@app.post("/audio/note", response_model=AudioResponse)
async def play_note(request: NoteRequest, background_tasks: BackgroundTasks):
    """
    Play a MIDI note (for sequencer events)

    Example:
        curl -X POST http://localhost:6103/audio/note \\
          -H "Content-Type: application/json" \\
          -d '{"note": 60, "duration": 0.3, "velocity": 100}'
    """
    if not audio_state["notes_enabled"]:
        return AudioResponse(
            success=False,
            message="Note playback is disabled in settings"
        )

    try:
        logger.info(f"Playing note: {request.note} (velocity={request.velocity}, duration={request.duration})")

        # Generate tone for the MIDI note
        frequency = midi_to_frequency(request.note)
        volume = request.velocity / 127.0 * audio_state["master_volume"]

        # Generate audio
        audio_file = await generate_tone(
            frequency=frequency,
            duration=request.duration,
            volume=volume,
            waveform=request.instrument
        )

        # Play in background
        background_tasks.add_task(play_audio_file, audio_file)

        return AudioResponse(
            success=True,
            message=f"Playing note {request.note} ({frequency:.2f} Hz)",
            audio_file=audio_file,
            duration=request.duration
        )

    except Exception as e:
        logger.error(f"Note playback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Tone Generation
# ============================================================================

@app.post("/audio/tone", response_model=AudioResponse)
async def play_tone(request: ToneRequest, background_tasks: BackgroundTasks):
    """
    Generate and play a tone/beep

    Example:
        curl -X POST http://localhost:6103/audio/tone \\
          -H "Content-Type: application/json" \\
          -d '{"frequency": 440, "duration": 0.2, "volume": 0.5}'
    """
    try:
        logger.info(f"Generating tone: {request.frequency} Hz, {request.duration}s")

        volume = request.volume * audio_state["master_volume"]

        audio_file = await generate_tone(
            frequency=request.frequency,
            duration=request.duration,
            volume=volume,
            waveform=request.waveform
        )

        background_tasks.add_task(play_audio_file, audio_file)

        return AudioResponse(
            success=True,
            message=f"Playing {request.frequency} Hz tone",
            audio_file=audio_file,
            duration=request.duration
        )

    except Exception as e:
        logger.error(f"Tone generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Audio File Playback
# ============================================================================

@app.post("/audio/play", response_model=AudioResponse)
async def play_file(request: FilePlayRequest, background_tasks: BackgroundTasks):
    """
    Play an audio file

    Example:
        curl -X POST http://localhost:6103/audio/play \\
          -H "Content-Type: application/json" \\
          -d '{"file_path": "/path/to/audio.wav"}'
    """
    try:
        file_path = Path(request.file_path)

        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Audio file not found: {file_path}")

        logger.info(f"Playing audio file: {file_path}")

        volume = request.volume if request.volume is not None else audio_state["master_volume"]

        background_tasks.add_task(play_audio_file, str(file_path), volume)

        return AudioResponse(
            success=True,
            message=f"Playing {file_path.name}",
            audio_file=str(file_path)
        )

    except Exception as e:
        logger.error(f"File playback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Volume Control
# ============================================================================

@app.post("/audio/volume")
async def set_volume(request: VolumeUpdate):
    """
    Set master volume

    Example:
        curl -X POST http://localhost:6103/audio/volume \\
          -H "Content-Type: application/json" \\
          -d '{"master_volume": 0.8}'
    """
    audio_state["master_volume"] = request.master_volume
    logger.info(f"Master volume set to {request.master_volume}")

    return {
        "success": True,
        "message": f"Volume set to {request.master_volume}",
        "master_volume": audio_state["master_volume"]
    }


@app.post("/audio/enable")
async def enable_audio(tts: bool = True, notes: bool = True):
    """
    Enable/disable audio features

    Example:
        curl -X POST "http://localhost:6103/audio/enable?tts=true&notes=false"
    """
    audio_state["tts_enabled"] = tts
    audio_state["notes_enabled"] = notes

    logger.info(f"Audio features updated: TTS={tts}, Notes={notes}")

    return {
        "success": True,
        "tts_enabled": audio_state["tts_enabled"],
        "notes_enabled": audio_state["notes_enabled"]
    }


# ============================================================================
# Helper Functions
# ============================================================================

def midi_to_frequency(note: int) -> float:
    """
    Convert MIDI note number to frequency in Hz
    A4 (440 Hz) = MIDI note 69
    """
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


async def generate_tone(
    frequency: float,
    duration: float,
    volume: float,
    waveform: str = "sine"
) -> str:
    """
    Generate a tone and save to WAV file

    Returns:
        Path to generated WAV file
    """
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Generate waveform
    if waveform == "sine" or waveform == "piano":
        wave = np.sin(2 * np.pi * frequency * t)
    elif waveform == "square":
        wave = np.sign(np.sin(2 * np.pi * frequency * t))
    elif waveform == "sawtooth":
        wave = 2 * (t * frequency - np.floor(t * frequency + 0.5))
    elif waveform == "triangle":
        wave = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
    else:
        wave = np.sin(2 * np.pi * frequency * t)

    # Apply envelope (fade in/out to avoid clicks)
    fade_samples = int(sample_rate * 0.01)  # 10ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    wave[:fade_samples] *= fade_in
    wave[-fade_samples:] *= fade_out

    # Apply volume
    wave = wave * volume

    # Convert to 16-bit PCM
    audio = (wave * 32767).astype(np.int16)

    # Save to WAV file
    timestamp = int(time.time() * 1000)
    output_file = AUDIO_DIR / f"tone_{timestamp}.wav"

    # Write WAV file manually (to avoid dependency on scipy/wave)
    import wave
    with wave.open(str(output_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    logger.info(f"Generated tone: {output_file} ({frequency:.2f} Hz, {duration}s)")

    return str(output_file)


async def play_audio_file(file_path: str, volume: Optional[float] = None):
    """
    Play an audio file using macOS afplay
    """
    try:
        vol = volume if volume is not None else audio_state["master_volume"]

        audio_state["current_playback"] = {
            "file": file_path,
            "started": time.time()
        }

        logger.info(f"Playing: {file_path} (volume={vol})")

        # Use afplay on macOS (non-blocking)
        cmd = ["afplay", "-v", str(vol), file_path]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await process.wait()

        audio_state["current_playback"] = None

        logger.info(f"Playback complete: {file_path}")

    except Exception as e:
        logger.error(f"Playback error: {e}", exc_info=True)
        audio_state["current_playback"] = None


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize audio service"""
    logger.info("🔊 Audio Service starting up...")
    logger.info(f"Audio temp directory: {AUDIO_DIR}")
    logger.info(f"Reference audio: {DEFAULT_REF_AUDIO} (exists={DEFAULT_REF_AUDIO.exists()})")
    logger.info(f"Master volume: {audio_state['master_volume']}")
    logger.info("Audio Service ready on port 6103")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Audio Service shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6103)
