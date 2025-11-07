# HMI Sequencer View & Audio Settings Implementation

**Date:** 2025-11-06
**Status:** ✅ Completed

---

## 📋 Summary

Implemented the **Sequencer View** (MIDI-style task timeline) and comprehensive **Audio Settings** for the PAS Agent Swarm HMI, as specified in `PRD_Human_Machine_Interface_HMI.md`.

---

## 🎯 Features Implemented

### 1. **Sequencer View** — MIDI-Style Task Visualizer

**Location:** http://localhost:6101/sequencer

**Visual Design:**
- **Horizontal Timeline:** Time scrolls left-to-right (like DAW/MIDI sequencer)
- **Vertical Rows:** Each agent is a row, sorted by tier (VP → Directors → Managers → Workers)
- **Task Blocks:** Colored rectangles representing agent tasks
- **Playhead:** Red vertical line showing current time position

**Color Encoding:**
- 🟦 **Blue:** Running (0-25% complete)
- 🟨 **Yellow:** Running (25-75% complete)
- 🟩 **Green:** Running (75-100% complete) or Done
- 🟧 **Orange:** Blocked/Waiting
- 🟪 **Purple:** Awaiting Approval
- 🟥 **RED:** Stuck/Error (no progress for >2 heartbeat intervals)
- ⬜ **Gray:** Done/Idle

**Interactive Controls:**
- **Play/Pause:** Animate playhead through timeline
- **Stop:** Reset playhead to start
- **Zoom:** In/Out controls (10%-1000% zoom)
- **Time Range:** 5min, 15min, 30min, 1hr, 2hr, 4hr
- **Click Timeline:** Jump playhead to timestamp
- **Click Task Block:** Show detailed tooltip
- **Hover Task:** Display task info (agent, status, progress, duration)

**Technical Implementation:**
- Canvas-based rendering for smooth performance
- Auto-refresh support (respects settings)
- Fetches real-time data from `/api/sequencer`
- Falls back to demo data if event stream unavailable

---

### 2. **Audio Settings** — Sound Controls

**Location:** Settings panel (⚙️ button in navigation)

**Audio Settings Section:**
1. **Master Audio Toggle**
   - Enable/disable all sound output
   - Default: OFF

2. **Sequencer Notes Toggle**
   - Musical note sonification for task events (assign, start, complete)
   - Pitch mapping: VP (low) → Directors → Managers → Workers (high)
   - Default: OFF

3. **Agent Voice Status Toggle**
   - Text-to-speech narration of agent status updates
   - Voice depth mapping: VP (deep) → Workers (light)
   - Default: OFF

4. **Audio Volume Slider**
   - Master volume level (0-100%)
   - Real-time percentage display
   - Default: 70%

**Persistence:**
- All settings saved to browser localStorage
- Settings persist across page reloads
- Shared between Dashboard, Tree View, and Sequencer

---

### 3. **Time Zone Setting**

**Location:** Settings panel → Display section

**Options:**
- EST (US Eastern) — **Default**
- CST (US Central)
- MST (US Mountain)
- PST (US Pacific)
- UTC
- GMT (London)
- JST (Tokyo)

**Usage:**
- All timestamps displayed in selected timezone
- Affects Dashboard, Tree View, and Sequencer

---

## 📁 Files Modified/Created

### Created:
- `services/webui/templates/sequencer.html` — Sequencer view template
- `docs/HMI_SEQUENCER_AND_AUDIO_SETTINGS.md` — This file

### Modified:
- `services/webui/templates/base.html`:
  - Added Sequencer to navigation
  - Added Audio Settings section (4 controls)
  - Added Time Zone setting
  - Updated default settings object
  - Added toggle functions for audio controls
  - Volume slider event listener

- `services/webui/hmi_app.py`:
  - Added `/sequencer` route
  - Added `/api/sequencer` endpoint
  - Fetches agents from Registry (sorted by tier)
  - Fetches tasks from Event Stream (recent events)
  - Falls back to demo tasks if unavailable

- `docs/PRDs/PRD_Human_Machine_Interface_HMI.md`:
  - Updated **Section 4.3** with detailed Sequencer specs
  - Added **Section 10.1** with Audio Settings specs

---

## 🧪 Testing

### Test Sequencer View:

1. **Navigate to Sequencer:**
   ```
   http://localhost:6101/sequencer
   ```

2. **Verify Canvas Rendering:**
   - Timeline grid with 5-minute intervals
   - Agent rows (50 agents from Registry)
   - Task blocks (demo data if no real events)

3. **Test Controls:**
   - Click **Play** → Playhead animates
   - Click **Pause** → Playhead stops
   - Click **Stop** → Playhead resets to start
   - Click **Zoom +/-** → Canvas zooms in/out
   - Change **Time Range** → Updates timeline width
   - Click **Refresh** → Reloads data

4. **Test Interactions:**
   - Click timeline → Playhead jumps to position
   - Hover task block → Tooltip appears
   - Click task block → Console logs details

### Test Audio Settings:

1. **Open Settings:**
   - Click **⚙️ Settings** in navigation

2. **Verify Audio Controls:**
   - Master Audio toggle (default OFF)
   - Sequencer Notes toggle (default OFF)
   - Agent Voice Status toggle (default OFF)
   - Volume slider (default 70%)

3. **Test Persistence:**
   - Enable Master Audio → Save
   - Reload page → Setting persists
   - Change volume to 50% → Save
   - Reload → Volume still 50%

### Test Time Zone Setting:

1. **Open Settings:**
   - Go to Display section
   - Find Time Zone dropdown

2. **Change Timezone:**
   - Select "PST (US Pacific)"
   - Save changes
   - Verify timestamps update (Dashboard, Sequencer)

---

## 📊 API Endpoints

### `/api/sequencer` (GET)

**Response:**
```json
{
  "agents": [
    {
      "service_id": "agent-architect",
      "name": "Architect",
      "tier": "1",
      "status": "ok",
      "agent_role": "coord"
    },
    ...
  ],
  "tasks": [
    {
      "task_id": "demo_task_0",
      "agent_id": "agent-architect",
      "name": "Demo Task 1",
      "status": "running",
      "progress": 0.3,
      "start_time": 1730926725.0,
      "end_time": null
    },
    ...
  ],
  "timestamp": "2025-11-06T18:38:42.032424"
}
```

**Parameters:**
- None (future: `?from=timestamp&to=timestamp` for time range filtering)

---

## 🎨 Visual Design

### Sequencer Layout:

```
┌────────────────────────────────────────────────────────────────────┐
│ [▶ Play] [⏹ Stop]  [− 100% +]  [Last 1 hour ▼]  [🔄 Refresh]   │
├────────────────────────────────────────────────────────────────────┤
│ VP         │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│               │
│ Director-1 │      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                          │
│ Director-2 │            ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                     │
│ Manager-1  │                  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                   │
│ Manager-2  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓│                                     │
│ Worker-1   │              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                │
│ Worker-2   │                          ▓▓▓▓▓▓▓▓▓▓▓▓▓│             │
│            │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│            │          Playhead →                                │
└────────────────────────────────────────────────────────────────────┘
```

**Legend:**
- Each `▓` block = Task (color = status/progress)
- Playhead (red line) = Current time position
- Width = Task duration
- Opacity = Progress (0% = 40%, 100% = 100%)

---

## 🔊 Audio Implementation (Future)

**Note:** Audio settings UI is complete, but audio playback is **not yet implemented**. The settings are saved and ready for integration with:

1. **Web Audio API** — For musical note synthesis
2. **Web Speech API** — For text-to-speech narration
3. **Tone.js** or **Howler.js** — For richer audio capabilities

**When implementing audio:**
- Check `currentSettings.masterAudioEnabled` before playing sounds
- Check `currentSettings.sequencerNotesEnabled` for note events
- Check `currentSettings.agentVoiceEnabled` for TTS
- Use `currentSettings.audioVolume / 100` as gain multiplier

**Example:**
```javascript
const settings = getSettings();

if (settings.masterAudioEnabled && settings.sequencerNotesEnabled) {
    const gain = settings.audioVolume / 100;
    playNote(frequency, duration, gain);
}
```

---

## 🚀 Next Steps

**P1 — Audio Playback:**
1. Integrate Web Audio API for sequencer notes
2. Implement pitch mapping (VP = low, Workers = high)
3. Add rate limiting (≤8 notes/sec global, ≤2/sec per-agent)

**P2 — Voice Narration:**
1. Integrate Web Speech API for TTS
2. Implement voice depth mapping (VP = deep, Workers = light)
3. Add ducking (reduce music during speech)

**P3 — Advanced Sequencer:**
1. Task scrubbing (drag playhead, timeline zooms)
2. Solo/Mute per-agent (click agent row)
3. Export timeline as MP4/GIF (replay mode)
4. Vertical zoom (resize row height)

**P4 — Real-Time Integration:**
1. WebSocket connection for live task updates
2. Automatic playhead advance (follow "now")
3. Highlight active tasks with pulsing border
4. Task completion animations

---

## 📖 Documentation

**User Guide:**
1. **Access Sequencer:** Click "Sequencer" in navigation
2. **Playback Controls:** Use Play/Pause/Stop to animate timeline
3. **Zoom:** Adjust zoom level to see more/less time detail
4. **Time Range:** Select how far back to view (5min - 4hr)
5. **Task Details:** Hover or click task blocks for info

**Audio Settings:**
1. **Open Settings:** Click ⚙️ Settings button
2. **Enable Audio:** Toggle Master Audio ON
3. **Select Types:** Enable Sequencer Notes and/or Agent Voice
4. **Adjust Volume:** Drag slider to desired level (0-100%)
5. **Save:** Click "Save Changes"

**Time Zone:**
1. **Open Settings:** Click ⚙️ Settings
2. **Display Section:** Find Time Zone dropdown
3. **Select Zone:** Choose from EST/PST/UTC/etc.
4. **Save:** Click "Save Changes"
5. **Verify:** Timestamps update throughout HMI

---

## ✅ Acceptance Criteria

- [x] Sequencer view accessible via `/sequencer` route
- [x] Timeline displays agents as rows (sorted by tier)
- [x] Task blocks render with correct colors (status/progress)
- [x] Playhead animates on Play, stops on Pause, resets on Stop
- [x] Zoom controls adjust timeline scale (10%-1000%)
- [x] Time range selector changes visible window (5min-4hr)
- [x] Clicking timeline moves playhead to position
- [x] Hovering task shows tooltip with details
- [x] Audio settings UI complete (Master, Notes, Voice, Volume)
- [x] Time zone setting with 7 options (default EST)
- [x] All settings persist to localStorage
- [x] Settings apply across all views (Dashboard, Tree, Sequencer)
- [x] Auto-refresh respects settings (60s default)
- [x] API endpoint `/api/sequencer` returns agents + tasks
- [x] Falls back to demo data if event stream unavailable
- [x] PRD updated with Sequencer and Audio specs

---

## 🎯 Current Status

**✅ Complete:**
- Sequencer view (MIDI-style timeline)
- Audio settings UI (toggles + volume)
- Time zone setting
- Persistent settings storage
- API endpoints for sequencer data
- PRD documentation updates

**⚠️ Pending (Future Work):**
- Audio playback implementation (Web Audio API)
- Voice narration (Web Speech API)
- Real-time task updates via WebSocket
- Advanced sequencer features (solo/mute, export)

---

**End of Document**
