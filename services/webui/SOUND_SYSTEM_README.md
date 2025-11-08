# HMI Sound System

Complete Web Audio API implementation for the PAS Agent Swarm HMI with 5 sound modes.

## Features

### 5 Sound Modes

1. **None** (default)
   - No audio output
   - Silent operation

2. **Voice**
   - Text-to-speech announcements using Web Speech API
   - Narrates task events: "Task assigned", "Task started", "Task completed", etc.
   - Browser-native TTS (no external service required)
   - Adjustable rate (1.2x by default for efficiency)

3. **Music Note**
   - Musical notes mapped to event types
   - MIDI-based frequency mapping (A440 standard)
   - Event → Note mapping:
     - Task assigned: C4 (261.63 Hz)
     - Task started: E4 (329.63 Hz)
     - Task completed: C5 (523.25 Hz)
     - Error: C3 (130.81 Hz)
     - Warning: G3 (196.00 Hz)
     - Success: F5 (698.46 Hz)

4. **Random Sounds**
   - Random musical tones for each event
   - Variety of frequencies (440-1047 Hz)
   - Multiple waveforms (sine, triangle, square)
   - Fun and unpredictable

5. **Geiger Counter** (NEW)
   - Realistic Geiger counter click sounds
   - Tick rate varies with agent activity level
   - Fast ticks = high activity, slow ticks = low activity
   - Uses noise bursts with exponential decay (20ms clicks)
   - Activity rate decays over time (background radiation)

## Configuration

### Settings Modal (⚙️ Button)

Navigate to **Settings → Audio**:

- **Master Audio**: Enable/disable all sounds
- **Sequencer Notes**: Enable musical notes for task events
- **Agent Voice Status**: Enable text-to-speech narration
- **Audio Volume**: Master volume slider (0-100%)

Navigate to **Settings → Sequencer**:

- **Default Sound Mode**: Choose from 5 modes (None, Voice, Music Note, Random Sounds, Geiger Counter)

## Technical Details

### Web Audio API

- **AudioContext**: Initialized on first user interaction (required by browser security)
- **GainNode**: Master volume control
- **Oscillators**: Sine, square, sawtooth, triangle waveforms
- **BufferSource**: For Geiger counter noise generation

### Sound Generation

#### Musical Notes
```javascript
playClientNote(frequency, duration, waveform)
// Example: playClientNote(440, 0.3, 'sine') // A4 note for 300ms
```

#### Geiger Counter
```javascript
playGeigerClick()
// Generates 20ms noise burst with exponential decay
// Creates authentic Geiger counter sound
```

#### Text-to-Speech
```javascript
speakText("Task completed")
// Uses browser's Web Speech API
// Falls back to audio service if not available
```

### Activity Rate Detection

The Geiger counter mode tracks system activity:

- **Events trigger activity increment**: Each task event increases activity counter
- **Activity decays over time**: 10% decay per second (exponential)
- **Tick rate varies**: Fast ticks during high activity, slow during idle
- **Random background ticks**: Simulates ambient radiation when idle

## Event Mapping

Real-time events from WebSocket trigger sounds:

```javascript
Event Type          → Sound Event
-----------------------------------
job_started         → task_started
task_started        → task_started
job_completed       → task_completed
task_completed      → task_completed
error               → error
failed              → error
heartbeat           → heartbeat
warning             → warning
blocked             → warning
```

## Usage Examples

### Enable Music Notes

1. Open Settings (⚙️ button)
2. Navigate to **Audio** section
3. Toggle **Master Audio** ON
4. Toggle **Sequencer Notes** ON
5. Navigate to **Sequencer** section
6. Set **Default Sound Mode** to "Music Note"
7. Click **Save Changes**

### Enable Geiger Counter

1. Open Settings (⚙️ button)
2. Navigate to **Audio** section
3. Toggle **Master Audio** ON
4. Navigate to **Sequencer** section
5. Set **Default Sound Mode** to "Geiger Counter"
6. Click **Save Changes**
7. Click anywhere on the page to initialize audio
8. Listen for ticks as agents work

### Enable Voice Announcements

1. Open Settings (⚙️ button)
2. Navigate to **Audio** section
3. Toggle **Master Audio** ON
4. Toggle **Agent Voice Status** ON
5. Navigate to **Sequencer** section
6. Set **Default Sound Mode** to "Voice"
7. Click **Save Changes**

## Browser Compatibility

### Web Audio API
- ✅ Chrome/Edge (full support)
- ✅ Firefox (full support)
- ✅ Safari (full support)
- ✅ Opera (full support)

### Web Speech API (Voice mode)
- ✅ Chrome/Edge (native TTS)
- ✅ Safari (native TTS)
- ⚠️ Firefox (limited support, falls back to audio service)

## Performance

- **Minimal CPU impact**: Audio synthesis is hardware-accelerated
- **Low memory usage**: No audio file downloads required
- **Zero latency**: Sounds generated on-the-fly
- **Efficient**: Only active when enabled in settings

## Troubleshooting

### No sound playing

1. **Check Master Audio**: Ensure "Master Audio" toggle is ON in settings
2. **Check browser permissions**: Some browsers require user interaction before playing audio
3. **Click anywhere on page**: Web Audio API requires user gesture to initialize
4. **Check volume**: Ensure master volume slider is not at 0%
5. **Check browser console**: Look for audio initialization logs

### Geiger counter not ticking

1. **Verify mode selected**: Check "Default Sound Mode" is set to "Geiger Counter"
2. **Click to initialize**: Click anywhere on the page to start audio context
3. **Check activity**: Geiger counter requires events to tick faster (background ticks are slow)
4. **Refresh page**: If mode was just changed, refresh to apply settings

### Voice not speaking

1. **Check browser support**: Firefox has limited Web Speech API support
2. **Check voice settings**: Ensure "Agent Voice Status" toggle is ON
3. **Check system volume**: Verify OS volume is not muted
4. **Try different browser**: Chrome/Edge have best TTS support

## Advanced Configuration

### Customize Note Frequencies

Edit `noteMap` in base.html (lines 1407-1415):

```javascript
const noteMap = {
    'task_assigned': 60,    // C4 (default)
    'task_started': 64,     // E4 (default)
    'task_completed': 72,   // C5 (default)
    'error': 48,            // C3 (default)
    // Add custom mappings here
};
```

### Customize Geiger Counter Tick Rate

Edit `startGeigerMode()` in base.html (line 1501):

```javascript
const baseRate = 500; // Base rate in ms (lower = faster)
```

### Customize Activity Decay

Edit activity decay rate in base.html (line 1602):

```javascript
activityRate = Math.max(0, activityRate * 0.9); // 0.9 = 10% decay/sec
```

## API Reference

### Client-Side Functions

```javascript
// Initialize Web Audio API
initAudioContext()

// Play musical note
playClientNote(frequency, duration, waveform)

// Play Geiger counter click
playGeigerClick()

// Play random sound effect
playRandomSound()

// Play note for event type
playEventNote(eventType)

// Speak text using TTS
speakText(text)

// Handle task event with current sound mode
handleTaskEventSound(eventType)

// Start/stop Geiger counter mode
startGeigerMode()
stopGeigerMode()

// Update activity rate (for Geiger counter)
updateActivityRate(eventsPerSecond)

// Update master volume
updateClientVolume(volume)
```

## Credits

- **Web Audio API**: W3C standard for browser audio synthesis
- **Web Speech API**: Browser-native text-to-speech
- **Geiger counter sound**: Authentic noise burst simulation with exponential decay

## License

Part of the PAS Agent Swarm project. See main LICENSE file.
