# Project Backlog

This document tracks future enhancements and features that are planned but not currently in active development.

---

## Task Intake System Enhancements

### Option 2: Enhanced Verdict CLI Interactive Mode
**Priority:** Medium
**Effort:** 2-3 days
**Status:** 🔵 Backlog

**Description:**
Add interactive mode to Verdict CLI that prompts for task information directly from the terminal, without requiring Claude Code.

**Motivation:**
- Self-contained in Verdict tool
- No Claude Code dependency
- Can be used outside Claude Code context
- Useful for command-line workflows

**Implementation:**
```bash
./bin/verdict interactive
# Prompts user with questions via CLI
# - Task title?
# - Goal description?
# - Entry point files?
# - Success criteria?
# - Constraints?
# Formats and submits when ready
```

**Dependencies:**
- None (standalone enhancement)

**References:**
- Related to `/pas-task` (Option 1 - Completed Nov 11, 2025)
- See: `.claude/commands/pas-task.md`

---

### Option 3: Hybrid Task Intake (Best of Both)
**Priority:** High
**Effort:** 3-4 days
**Status:** 🔵 Backlog

**Description:**
Combine `/pas-task` conversational interface (Option 1) with enhanced Verdict CLI (Option 2) for seamless task intake across contexts.

**Motivation:**
- `/pas-task` for conversational, in-context work (Claude Code)
- Verdict CLI for direct command-line usage
- Shared task formatting logic
- Consistent experience across interfaces

**Architecture:**
```
┌─────────────────┐         ┌──────────────────┐
│ /pas-task       │────────▶│ Task Formatter   │
│ (Claude Code)   │         │ (Shared Library) │
└─────────────────┘         └──────────────────┘
                                     │
┌─────────────────┐                 │
│ verdict         │────────▶────────┘
│ interactive     │                 │
│ (CLI)           │                 ▼
└─────────────────┘         ┌──────────────────┐
                            │ Verdict Submit   │
                            │ (Gateway API)    │
                            └──────────────────┘
```

**Implementation:**
1. **Shared Library** (`src/task_intake/formatter.py`):
   - Task validation logic
   - Prime Directive JSON formatting
   - Success criteria templates

2. **Claude Code Integration** (`.claude/commands/pas-task.md`):
   - Uses shared formatter
   - Adds conversational layer
   - Real-time status tracking

3. **CLI Integration** (`bin/verdict`):
   - Add `interactive` subcommand
   - Uses shared formatter
   - Terminal-based prompts (no LLM needed)

**Benefits:**
- DRY (Don't Repeat Yourself) - Single source of truth for task formatting
- Consistent validation across interfaces
- Easy to add new intake methods (web UI, API, etc.)
- Better testability

**Dependencies:**
- Option 1 (Completed - `/pas-task`)
- Option 2 (Backlog - Verdict interactive)

**References:**
- `/pas-task`: `.claude/commands/pas-task.md`
- P0 Stack: `docs/P0_END_TO_END_INTEGRATION.md`
- Verdict CLI: `bin/verdict`

---

## Future Enhancements

### Task Templates
**Priority:** Low
**Effort:** 1-2 days
**Status:** 🔵 Backlog

**Description:**
Pre-defined task templates for common operations (e.g., "Add Unit Tests", "Fix Bug", "Refactor Module").

**Benefits:**
- Faster task submission
- Consistent task structure
- Reduces user input burden

---

### Task History and Replay
**Priority:** Medium
**Effort:** 2-3 days
**Status:** 🔵 Backlog

**Description:**
Store task history and allow users to replay or modify previous tasks.

**Benefits:**
- Iterative refinement
- Learn from past tasks
- Quick re-runs with tweaks

---

## Legend

- 🔵 **Backlog**: Planned but not started
- 🟡 **In Progress**: Currently being worked on
- ✅ **Complete**: Shipped to production

---

**Last Updated:** 2025-11-11
