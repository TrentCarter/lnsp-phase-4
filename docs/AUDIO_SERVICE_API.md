# Audio Service API Documentation

## Overview

The **Audio Service** provides a unified API for all HMI audio functionality:
- **Text-to-Speech (TTS)** using f5_tts_mlx with high-quality voice synthesis
- **MIDI Note Playback** for sequencer events and musical feedback
- **Tone Generation** for beeps, alerts, and sound effects
- **Audio File Playback** for custom sounds
- **Volume Control** and audio mixing

**Service URL**: `http://localhost:6103`

---

## Quick Start

```bash
# Start the audio service
./scripts/start_audio_service.sh

# Check service health
curl http://localhost:6103/health

# Play a simple tone
curl -X POST http://localhost:6103/audio/tone \
  -H "Content-Type: application/json" \
  -d '{"frequency": 440, "duration": 0.3, "volume": 0.5}'

# Generate speech
curl -X POST http://localhost:6103/audio/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Agent Alpha has completed the task.", "speed": 1.0}'

# Play a MIDI note (middle C)
curl -X POST http://localhost:6103/audio/note \
  -H "Content-Type: application/json" \
  -d '{"note": 60, "duration": 0.5, "velocity": 100}'
```

---

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the service is running.

**Response**:
```json
{
  "status": "ok",
  "service": "audio_service",
  "port": 6103,
  "timestamp": "2025-11-07T01:36:00.059478",
  "master_volume": 0.7,
  "tts_enabled": true,
  "notes_enabled": true
}
```

---

### 2. Get Status

**GET** `/status`

Get detailed service status including current playback.

**Response**:
```json
{
  "master_volume": 0.7,
  "tts_enabled": true,
  "notes_enabled": true,
  "current_playback": null,
  "queue_length": 0,
  "temp_audio_dir": "/tmp/pas_audio",
  "ref_audio_exists": true
}
```

---

### 3. Text-to-Speech (TTS)

**POST** `/audio/tts`

Generate speech from text using f5_tts_mlx.

**Request Body**:
```json
{
  "text": "Agent Alpha has completed the task.",
  "ref_audio": "/Users/trentcarter/Artificial_Intelligence/Voice_Clips/Sophia3.wav",
  "ref_text": "Hi babe. I just wanted to wish you good night...",
  "speed": 1.0,
  "method": "midpoint",
  "auto_play": true
}
```

**Parameters**:
- `text` (required): Text to synthesize (1-1000 characters)
- `ref_audio` (optional): Path to reference voice audio (defaults to Sophia3.wav)
- `ref_text` (optional): Text matching the reference audio
- `speed` (optional): Speech speed multiplier (0.5-2.0, default: 1.0)
- `method` (optional): Generation method (`midpoint`, `euler`, `rk4`, default: `midpoint`)
- `auto_play` (optional): Automatically play after generation (default: `true`)

**Response**:
```json
{
  "success": true,
  "message": "TTS generated successfully",
  "audio_file": "/tmp/pas_audio/tts_1762479397693.wav",
  "duration": null,
  "timestamp": "2025-11-07T01:36:40.032607"
}
```

**Example Usage**:
```bash
# Simple TTS
curl -X POST http://localhost:6103/audio/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Task completed successfully.", "speed": 1.0}'

# Custom voice (if you have another reference)
curl -X POST http://localhost:6103/audio/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from Agent Beta!",
    "ref_audio": "/path/to/custom_voice.wav",
    "ref_text": "This is my voice.",
    "speed": 1.2
  }'
```

---

### 4. Play MIDI Note

**POST** `/audio/note`

Play a musical note for sequencer events.

**Request Body**:
```json
{
  "note": 60,
  "duration": 0.3,
  "velocity": 100,
  "instrument": "piano"
}
```

**Parameters**:
- `note` (required): MIDI note number (21-108, middle C=60)
- `duration` (optional): Note duration in seconds (0.05-5.0, default: 0.3)
- `velocity` (optional): Note velocity/volume (0-127, default: 100)
- `instrument` (optional): Waveform type (`piano`, `sine`, `square`, `sawtooth`, default: `piano`)

**MIDI Note Reference**:
- C4 (middle C) = 60
- A4 (440 Hz) = 69
- C5 = 72
- C3 = 48

**Response**:
```json
{
  "success": true,
  "message": "Playing note 60 (261.63 Hz)",
  "audio_file": "/tmp/pas_audio/tone_1762479383846.wav",
  "duration": 0.5,
  "timestamp": "2025-11-07T01:36:23.846991"
}
```

**Example Usage**:
```bash
# Play middle C (task start)
curl -X POST http://localhost:6103/audio/note \
  -H "Content-Type: application/json" \
  -d '{"note": 60, "duration": 0.3, "velocity": 100}'

# Play higher note (task complete)
curl -X POST http://localhost:6103/audio/note \
  -H "Content-Type: application/json" \
  -d '{"note": 72, "duration": 0.5, "velocity": 120}'

# Play chord (send multiple requests)
for note in 60 64 67; do
  curl -X POST http://localhost:6103/audio/note \
    -H "Content-Type: application/json" \
    -d "{\"note\": $note, \"duration\": 1.0}" &
done
```

---

### 5. Play Tone/Beep

**POST** `/audio/tone`

Generate and play a simple tone.

**Request Body**:
```json
{
  "frequency": 440.0,
  "duration": 0.2,
  "volume": 0.5,
  "waveform": "sine"
}
```

**Parameters**:
- `frequency` (required): Frequency in Hz (20.0-20000.0)
- `duration` (optional): Duration in seconds (0.01-5.0, default: 0.2)
- `volume` (optional): Volume (0.0-1.0, default: 0.5)
- `waveform` (optional): Waveform type (`sine`, `square`, `sawtooth`, `triangle`, default: `sine`)

**Response**:
```json
{
  "success": true,
  "message": "Playing 440.0 Hz tone",
  "audio_file": "/tmp/pas_audio/tone_1762479378037.wav",
  "duration": 0.3,
  "timestamp": "2025-11-07T01:36:18.042142"
}
```

**Example Usage**:
```bash
# Success beep (high pitch, short)
curl -X POST http://localhost:6103/audio/tone \
  -H "Content-Type: application/json" \
  -d '{"frequency": 800, "duration": 0.15, "volume": 0.6}'

# Error beep (low pitch, longer)
curl -X POST http://localhost:6103/audio/tone \
  -H "Content-Type: application/json" \
  -d '{"frequency": 200, "duration": 0.4, "volume": 0.7, "waveform": "square"}'

# Alert tone (moderate pitch)
curl -X POST http://localhost:6103/audio/tone \
  -H "Content-Type: application/json" \
  -d '{"frequency": 440, "duration": 0.2, "volume": 0.5}'
```

---

### 6. Play Audio File

**POST** `/audio/play`

Play a pre-existing audio file.

**Request Body**:
```json
{
  "file_path": "/tmp/pas_audio/tts_1762479397693.wav",
  "volume": 0.8
}
```

**Parameters**:
- `file_path` (required): Path to audio file
- `volume` (optional): Playback volume (0.0-1.0, uses master volume if not specified)

**Response**:
```json
{
  "success": true,
  "message": "Playing tts_1762479397693.wav",
  "audio_file": "/tmp/pas_audio/tts_1762479397693.wav",
  "duration": null,
  "timestamp": "2025-11-07T01:36:50.266283"
}
```

**Example Usage**:
```bash
# Play generated TTS file
curl -X POST http://localhost:6103/audio/play \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/tmp/pas_audio/tts_1762479397693.wav", "volume": 0.8}'

# Play custom sound effect
curl -X POST http://localhost:6103/audio/play \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/Users/trentcarter/sounds/notification.wav"}'
```

---

### 7. Set Master Volume

**POST** `/audio/volume`

Set the master volume for all audio playback.

**Request Body**:
```json
{
  "master_volume": 0.7
}
```

**Parameters**:
- `master_volume` (required): Master volume (0.0-1.0)

**Response**:
```json
{
  "success": true,
  "message": "Volume set to 0.7",
  "master_volume": 0.7
}
```

**Example Usage**:
```bash
# Set volume to 50%
curl -X POST http://localhost:6103/audio/volume \
  -H "Content-Type: application/json" \
  -d '{"master_volume": 0.5}'
```

---

### 8. Enable/Disable Audio Features

**POST** `/audio/enable?tts=true&notes=false`

Enable or disable specific audio features.

**Query Parameters**:
- `tts` (optional): Enable TTS (default: `true`)
- `notes` (optional): Enable note playback (default: `true`)

**Response**:
```json
{
  "success": true,
  "tts_enabled": true,
  "notes_enabled": false
}
```

**Example Usage**:
```bash
# Disable all audio
curl -X POST "http://localhost:6103/audio/enable?tts=false&notes=false"

# Enable only TTS
curl -X POST "http://localhost:6103/audio/enable?tts=true&notes=false"

# Enable everything
curl -X POST "http://localhost:6103/audio/enable?tts=true&notes=true"
```

---

## Integration with HMI

### JavaScript Example

```javascript
// Audio service base URL
const AUDIO_SERVICE = 'http://localhost:6103';

// Play TTS announcement
async function speakStatus(text, speed = 1.0) {
    const response = await fetch(`${AUDIO_SERVICE}/audio/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, speed, auto_play: true })
    });
    return await response.json();
}

// Play note for sequencer event
async function playNoteForEvent(eventType) {
    const noteMap = {
        'task_assigned': 60,    // C4
        'task_started': 64,     // E4
        'task_completed': 72,   // C5
        'error': 48            // C3
    };

    const note = noteMap[eventType] || 60;

    const response = await fetch(`${AUDIO_SERVICE}/audio/note`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ note, duration: 0.3, velocity: 100 })
    });
    return await response.json();
}

// Play tone for alert
async function playAlert(type = 'info') {
    const toneMap = {
        'info': { frequency: 440, duration: 0.2 },
        'success': { frequency: 800, duration: 0.15 },
        'warning': { frequency: 600, duration: 0.3 },
        'error': { frequency: 200, duration: 0.4 }
    };

    const config = toneMap[type] || toneMap['info'];

    const response = await fetch(`${AUDIO_SERVICE}/audio/tone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...config, volume: 0.5 })
    });
    return await response.json();
}

// Set volume from settings
async function setAudioVolume(volume) {
    const response = await fetch(`${AUDIO_SERVICE}/audio/volume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ master_volume: volume / 100 })
    });
    return await response.json();
}

// Usage examples:
// await speakStatus("Agent Alpha has completed the task.");
// await playNoteForEvent('task_completed');
// await playAlert('success');
// await setAudioVolume(70);
```

---

## Use Cases

### 1. Agent Status Updates (TTS)
```bash
# When an agent completes a task
curl -X POST http://localhost:6103/audio/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Agent Alpha has completed task 5 of 10.", "speed": 1.1}'
```

### 2. Sequencer Events (MIDI Notes)
```bash
# Task assigned: C4
curl -X POST http://localhost:6103/audio/note \
  -H "Content-Type: application/json" \
  -d '{"note": 60, "duration": 0.2, "velocity": 90}'

# Task started: E4
curl -X POST http://localhost:6103/audio/note \
  -H "Content-Type: application/json" \
  -d '{"note": 64, "duration": 0.3, "velocity": 100}'

# Task completed: C5 (higher octave)
curl -X POST http://localhost:6103/audio/note \
  -H "Content-Type: application/json" \
  -d '{"note": 72, "duration": 0.5, "velocity": 120}'
```

### 3. Alert Tones
```bash
# Success
curl -X POST http://localhost:6103/audio/tone \
  -H "Content-Type: application/json" \
  -d '{"frequency": 800, "duration": 0.15, "volume": 0.6}'

# Warning
curl -X POST http://localhost:6103/audio/tone \
  -H "Content-Type: application/json" \
  -d '{"frequency": 600, "duration": 0.3, "volume": 0.7}'

# Error
curl -X POST http://localhost:6103/audio/tone \
  -H "Content-Type: application/json" \
  -d '{"frequency": 200, "duration": 0.4, "volume": 0.7, "waveform": "square"}'
```

---

## Configuration

### Reference Voice

The default reference voice is located at:
```
/Users/trentcarter/Artificial_Intelligence/Voice_Clips/Sophia3.wav
```

To use a different voice:
1. Record or obtain a clean voice sample (WAV format, 10-60 seconds)
2. Note the exact text spoken in the sample
3. Pass `ref_audio` and `ref_text` in your TTS request

### Audio Output Directory

Temporary audio files are stored in:
```
/tmp/pas_audio/
```

Generated files are named with timestamps:
- TTS: `tts_<timestamp>.wav`
- Tones: `tone_<timestamp>.wav`

---

## Troubleshooting

### Service Won't Start
```bash
# Check if port is in use
lsof -ti:6103

# Kill existing process
lsof -ti:6103 | xargs kill -9

# Restart service
./scripts/start_audio_service.sh
```

### TTS Generation Fails
```bash
# Check if f5_tts_mlx is installed
python3 -m f5_tts_mlx.generate --help

# Verify reference audio exists
ls -l /Users/trentcarter/Artificial_Intelligence/Voice_Clips/Sophia3.wav
```

### No Audio Playback
```bash
# Check if afplay works (macOS)
afplay /tmp/pas_audio/tone_*.wav

# Check system volume
# Make sure macOS system volume is not muted
```

---

## Architecture

**Service Port**: 6103
**Framework**: FastAPI
**TTS Engine**: f5_tts_mlx (Apple Silicon optimized)
**Audio Playback**: macOS `afplay`
**Tone Generation**: NumPy + WAV

**Dependencies**:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `f5_tts_mlx` - TTS engine
- `numpy` - Audio synthesis
- `pydantic` - Data validation

---

## Performance Notes

- **TTS Generation**: ~1-3 seconds per sentence (using Apple Silicon MLX)
- **Tone Generation**: <100ms
- **Note Playback**: <100ms
- **Concurrent Playback**: Supported (multiple sounds can play simultaneously)

---

## Security Considerations

- Service runs on `127.0.0.1` (localhost only) by default
- No authentication (designed for local HMI use)
- File path validation prevents directory traversal
- TTS text limited to 1000 characters

For production deployment, consider:
- Adding authentication (API keys, JWT)
- Rate limiting
- Input sanitization
- HTTPS/TLS

---

## Next Steps

1. **Integrate with HMI**: Add audio service calls to frontend events
2. **Customize Voices**: Record custom voice samples for different agents
3. **Add Sound Effects**: Create library of notification sounds
4. **Implement Audio Queue**: Add sophisticated mixing/queuing logic
5. **Add Voice Commands**: Extend for voice input (speech-to-text)

---

**Service Status**: ✅ Production Ready
**Last Updated**: 2025-11-06
**Version**: 1.0.0
