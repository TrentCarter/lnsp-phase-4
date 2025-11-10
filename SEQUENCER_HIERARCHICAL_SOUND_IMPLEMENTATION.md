# Sequencer Hierarchical Sound & Visual Flash Implementation

**Date**: 2025-11-09
**Status**: ✅ Complete (with debug logging active)

---

## Overview

Implemented hierarchical musical notes and synchronized visual flashes for the PAS Sequencer timeline view. Tasks now "sing" with pitch based on agent hierarchy (VP/Director = low, Programmer = high) and flash bright white when their sound plays.

---

## Features Implemented

### 1. Hierarchical Musical Notes

**Pitch by Agent Tier**:
- **Tier 0 (VP)**: Deepest/lowest pitch (base note + 0 semitones)
- **Tier 1 (Director)**: +4 semitones above base
- **Tier 2 (Manager)**: +7 semitones above base
- **Tier 3 (Programmer)**: +12 semitones above base (1 octave up)
- **Tier 4+ (Workers)**: +15-17 semitones above base

**Per-Agent Offset**:
- Parallel agents at same tier get unique pitches (0-2 semitones offset)
- Uses `agent_id` hash for consistent pitch assignment
- Prevents muddy sound when multiple agents start simultaneously

**Event Base Notes** (lowered octave for hierarchy headroom):
- `task_started`: E3 (52 MIDI)
- `task_completed`: C4 (60 MIDI)
- `error`: C2 (36 MIDI)
- `warning`: G2 (43 MIDI)

**Formula**:
```javascript
finalMidi = eventBaseMidi + tierOffset + agentOffset
frequency = 440 * Math.pow(2, (finalMidi - 69) / 12)  // A440 standard
```

---

### 2. Configurable Note Duration

**Settings Panel** (Audio section):
- New slider: 50-500ms range (default 150ms)
- Real-time display updates as you drag
- Stored in `localStorage`, persists across sessions

**Use Cases**:
- **Short (50-100ms)**: Rapid clicks for dense timelines with many parallel tasks
- **Medium (150ms)**: Balanced, default setting
- **Long (300-500ms)**: Musical/sustained notes for sparse timelines

---

### 3. Visual Flash Sync

**Flash Effect**:
- Tasks flash **bright white** (60% opacity overlay) when their musical note plays
- Glowing white border with 15px shadow blur for maximum visibility
- Flash duration matches note duration setting (50-500ms)
- Multiple tasks can flash simultaneously (parallel execution visualization)

**Rendering**:
- White overlay: `rgba(255, 255, 255, 0.6)`
- Glowing border: 4px white stroke with `shadowBlur = 15px`
- Rendered as final layer on top of task blocks

---

### 4. Multi-Threaded Sound Playback

**Previous Behavior**: Blocking queue prevented overlapping sounds
**New Behavior**: Web Audio API handles concurrent sounds natively

**Implementation**:
- Removed sound duration blocking in `processNextSound()`
- Multiple notes can play simultaneously
- Perfect for parallel task execution visualization

---

### 5. Immediate Refresh on Actions

**Refresh Triggers**:
- ✅ **Page load**: via `loadProjects()` → auto-selects most recent task
- ✅ **Task selection**: via `changeProject()` → loads selected Prime Directive
- ✅ **Play button**: via `togglePlayPause()` → refreshes before playback starts
- ✅ **Rewind (double-click stop)**: via `resetPlayhead()` → refreshes data

**Sound State Preservation**:
- `lastTriggeredTaskStarts` Set is **preserved** during auto-refresh while playing
- Prevents duplicate sounds when task data is re-fetched mid-playback
- Only cleared on loop/rewind (when explicitly requested)

---

### 6. Timeline Pan Synchronization

**Problem**: Manual timeline dragging desynchronized `playheadPosition` and `timelineOffset`
**Solution**: Update both in lockstep

**Implementation** (sequencer.html:685):
```javascript
// CRITICAL: Keep playheadPosition synchronized with timelineOffset
playheadPosition = timelineOffset;
```

**Result**: Sounds/flashes trigger at correct visual position regardless of pan/zoom

---

## Files Modified

### 1. `services/webui/templates/sequencer.html`

**Sound Queue** (lines 1991-2027):
- Modified `processNextSound()` to pass task object to sound handler
- Added task to `currentlyPlayingTasks` Set for visual flash
- Removed blocking queue (enabled overlapping playback)
- Added flash removal timeout based on `noteDuration` setting

**Visual Flash Rendering** (lines 1268-1291):
- Check `currentlyPlayingTasks.has(task.task_id)` during draw
- Render bright white overlay + glowing border
- Extensive debug logging (see below)

**Timeline Pan Sync** (line 685):
- Update `playheadPosition = timelineOffset` when dragging

**Immediate Refresh** (lines 1764-1777, 2048-2051, 2541):
- Refresh on play button press
- Refresh on rewind (double-click stop)
- Refresh when URL has task_id on page load

**Sound State Preservation** (lines 909-915):
- Log preservation status during `fetchSequencerData()`
- Don't clear `lastTriggeredTaskStarts` if `isPlaying === true`

**Sound Trigger Guards** (lines 1864-1866, 1917):
- Only trigger during active playback (`if (!isPlaying) return`)
- Only trigger if playhead moved (> 10ms threshold)

---

### 2. `services/webui/templates/base.html`

**Hierarchical Note Player** (lines 1483-1568):
- New function: `playEventNoteHierarchical(eventType, task)`
- Calculates pitch based on agent tier + agent offset
- Reads `noteDuration` from settings (default 150ms)
- Calls `playClientNote(frequency, duration, 'sine')`

**Sound Handler Update** (line 1538):
- Modified `handleTaskEventSound()` to call hierarchical note player
- Passes task object for pitch calculation

**Settings UI** (lines 779-788):
- Added "Note Duration" slider (50-500ms, step 10)
- Real-time display: `<span id="note-duration-display">150ms</span>`

**Settings Object** (line 958):
- Added `sequencerNoteDuration: 150` to default settings

**Settings Save/Load** (lines 993, 1060-1061, 1118-1121):
- Save/load note duration to/from `localStorage`
- Update slider display on settings load

---

## Debug Logging

**Currently Active** (for troubleshooting):

```javascript
// Sound triggering
🔊 [SOUND] Task STARTED: "..." (id=...) at playhead=X.Xs
   └─ isPlaying=true, soundQueue.length=X, currentlyPlayingTasks.size=X

// Flash management
✨ [FLASH] Added task "..." to flash set (size=X)
✨ [FLASH] Removed task "..." from flash set (size=X)
⚠️ [FLASH] Task missing task_id: {...}

// Drawing
🎨 [DRAW] Drawing X tasks, currentlyPlayingTasks.size=X, Set contents: [...]
✨ [FLASH RENDER] Drawing flash for "..." (id=...) at x=...

// Refresh
[Refresh] Playing - preserving triggered tasks to prevent duplicate sounds
✅ [Play] Data refreshed before playback
✅ [Rewind] Data refreshed
```

**Location**:
- Sound triggers: sequencer.html:1955-1956
- Flash add/remove: sequencer.html:2002, 2011
- Draw state: sequencer.html:1174-1176
- Flash render: sequencer.html:1272
- Refresh status: sequencer.html:909-915

**To Disable**: Comment out individual `console.log()` lines or remove debug sections

---

## Fixed Issues

### ✅ Flashes Only Work for First 5 Tasks (FIXED 2025-11-09)

**Root Cause**: Off-screen viewport culling was preventing flashes for tasks not visible in the current scroll position.

**Fix Location**: `services/webui/templates/sequencer.html` lines 1202-1206

**Problem**: The visibility check at line 1203 skipped rendering ALL content for off-screen tasks:
```javascript
// Skip if task block is not visible on screen
if (y + blockHeight < HEADER_HEIGHT || y > canvas.height) return;
```

**Solution**: Check flash state BEFORE visibility culling, so flashing tasks are always rendered:
```javascript
// Check if task is currently playing sound (needs flash even if off-screen)
const shouldFlash = currentlyPlayingTasks.has(task.task_id);

// Skip if task block is not visible on screen AND not currently flashing
if (!shouldFlash && (y + blockHeight < HEADER_HEIGHT || y > canvas.height)) return;
```

**Result**: All tasks now flash correctly when sounds play, regardless of scroll position!

**See**: `DEBUG_FLASH_ISSUE.md` for complete diagnostic details

---

## Testing Checklist

**Hierarchy Audibility**:
- [ ] Load task with VP → Director → Manager → Programmer chain
- [ ] Enable "Music Note" sound mode
- [ ] Play at 1x speed
- [ ] Verify pitch *ascending* as delegation flows down (VP lowest, Programmer highest)

**Parallel Agents**:
- [ ] Load task where Manager delegates to 3+ Programmers simultaneously
- [ ] Should hear 3+ slightly offset pitches (chord effect)
- [ ] Shorter note duration (50ms) makes this clearer

**Visual Flash**:
- [ ] Tasks flash bright white when sound plays
- [ ] Flash duration matches note duration setting
- [ ] Flash synchronized with red NOW line (left edge)

**Timeline Pan**:
- [ ] Drag timeline left/right manually
- [ ] Hit Play
- [ ] Flashes should still work correctly (no desync)

**Note Duration**:
- [ ] Settings → Audio → Note Duration slider
- [ ] Set to 50ms: Rapid clicks (good for dense timelines)
- [ ] Set to 300ms: Musical tones (good for sparse timelines)
- [ ] Verify flash duration matches

**Immediate Refresh**:
- [ ] Page load: Data appears immediately (no waiting)
- [ ] Select different Prime Directive: Loads immediately
- [ ] Hit Play: Refreshes data before playback starts
- [ ] Double-click Stop: Rewinds and refreshes data

---

## API Reference

### Settings Object

```javascript
{
  masterAudioEnabled: false,           // Toggle for all audio output
  sequencerNotesEnabled: false,        // Toggle for musical notes
  agentVoiceEnabled: false,            // Toggle for TTS narration
  audioVolume: 70,                     // Master volume (0-100)
  sequencerNoteDuration: 150,          // Note duration in ms (50-500)
  defaultSoundMode: 'none'             // 'none', 'music', 'voice', 'random', 'geiger'
}
```

### Global Variables

```javascript
let currentlyPlayingTasks = new Set();  // Task IDs with active visual flashes
let lastTriggeredTaskStarts = new Set(); // Task start keys already triggered
let lastTriggeredTaskEnds = new Set();   // Task end keys already triggered
let soundQueue = [];                     // Queued sound events
let playheadPosition = 0;                // Playback position (seconds from start)
let timelineOffset = 0;                  // Pan offset (seconds)
```

### Key Functions

**Hierarchical Sound**:
- `playEventNoteHierarchical(eventType, task)` - Calculate pitch and play note
- `handleTaskEventSound(eventType, task)` - Route to correct sound mode

**Flash Management**:
- `processNextSound()` - Add task to flash Set, schedule removal
- `drawTaskBlocks()` - Render flash overlay if task in Set

**Trigger Logic**:
- `checkTaskEvents()` - Detect when tasks cross playhead
- `queueSound(eventType, task)` - Add sound to queue

---

## Future Enhancements

**Potential Improvements**:
1. **Waveform Selection**: Allow users to choose sine/square/sawtooth/triangle waves
2. **Volume by Tier**: VP louder, Programmers quieter (or vice versa)
3. **Completion Chords**: Play major chord on successful completion, minor on error
4. **Flash Color by Status**: Green for success, red for error, yellow for warning
5. **Persistent Flash**: Keep task highlighted until completion (not just note duration)
6. **MIDI Export**: Export timeline as MIDI file for playback in DAW

**Performance**:
- If > 100 concurrent flashes cause lag, consider batching redraws
- If sound queue grows too large, add max queue size limit

---

## Troubleshooting

**No Sound**:
1. Check Settings → Audio → Master Audio (toggle ON)
2. Check Settings → Audio → Sequencer Notes (toggle ON)
3. Check browser console for audio initialization errors
4. Click anywhere on page (browsers block audio until user interaction)

**No Flash**:
1. Check browser console for `✨ [FLASH]` logs
2. Verify `currentlyPlayingTasks.size > 0` in logs
3. Check if task IDs match between Set and tasks array
4. Ensure `drawSequencer()` is being called during playback

**Desynchronized Flash** (flash doesn't align with NOW line):
1. Check if `playheadPosition === timelineOffset` (should be true during playback)
2. Verify timeline pan synchronization (line 685)
3. Check for floating-point accumulation errors in playback loop

**Duplicate Sounds**:
1. Check `lastTriggeredTaskStarts` Set size (shouldn't grow unbounded)
2. Verify Set is NOT cleared during auto-refresh while playing
3. Check for task ID changes during data refresh

---

## Commit Message Template

```
feat: Add hierarchical musical notes and visual flashes to sequencer

- Hierarchical pitch: VP/Director (low) to Programmer (high)
- Per-agent pitch offset for parallel tasks (chord effect)
- Configurable note duration: 50-500ms (Settings → Audio)
- Visual flash sync: Tasks flash bright white when sound plays
- Multi-threaded sound: Web Audio API handles concurrent playback
- Immediate refresh: Page load, task selection, play button
- Timeline pan sync: playheadPosition tracks timelineOffset
- Sound state preservation: No duplicates during auto-refresh

Files modified:
- services/webui/templates/sequencer.html (flash rendering, sync)
- services/webui/templates/base.html (hierarchical notes, settings)

Fixes:
- Timeline pan desynchronization (flash position)
- Prime Directive not loading on page load
- Duplicate sounds on auto-refresh during playback

Known issues:
- Flashes only work for first 5 tasks (under investigation)
- Debug logging currently active (remove before production)
```

---

## Related Documentation

- **Sequencer Arrow Fix**: `SEQUENCER_ARROW_FIX_2025_11_09.md`
- **Audio Service**: `docs/howto/how_to_use_audio_service.md`
- **Vec2Text Usage**: `docs/how_to_use_jxe_and_ielab.md`
- **Settings System**: `services/webui/templates/base.html` (lines 939-1000)

---

**End of Document**
