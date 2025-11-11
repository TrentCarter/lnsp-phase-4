# Restore Context from Last Session

This command loads the context from the previous work session to help you quickly resume work.

## What This Does

1. **Loads `docs/last_summary.md` into context** - This contains a concise summary of what was accomplished in the last session
2. **Shows you where you left off** - Key changes, current state, and next steps
3. **Provides quick start instructions** - Immediate actions to resume work

## Step 1: Load Last Summary

Read and present the contents of `docs/last_summary.md` to the user.

If the file doesn't exist:
- Inform the user that no previous session summary was found
- Suggest running `/wrap-up` at the end of their next session to create one
- Offer to search for the most recent session summary in `docs/session_summaries/`

## Step 2: Verify Current State

After loading the summary, verify the current state:

1. **Check git branch:**
   ```bash
   git branch --show-current
   ```
   Compare to the branch in `last_summary.md`

2. **Check git status:**
   ```bash
   git status --short
   ```
   Show any uncommitted changes

3. **Check running services:**
   ```bash
   lsof -ti:6101 > /dev/null && echo "HMI running" || echo "HMI not running"
   ```

## Step 3: Present Quick Start

Based on the "Quick Start Next Session" section in `last_summary.md`, present:
- Immediate next actions
- Any services that need to be started
- Files to review or edit
- Tests to run

## Step 4: Offer Next Steps

Ask the user:
- "Would you like to continue where you left off?"
- "Do you need me to explain any of the previous changes?"
- "Should I start a specific task from the 'What Needs Work' list?"

## Important Notes:

- **DO NOT** load `all_project_summary.md` - it's too large and is archival only
- **DO load** `last_summary.md` - it's designed for quick context loading
- Focus on helping the user resume work quickly
- If unclear, ask what they want to work on

## Example Output

After running `/restore`, you should present something like:

```
📋 Restored Context from Last Session

**Date:** 2025-11-11
**Branch:** feature/aider-lco-p0
**Duration:** 2 hours

## What Was Accomplished Last Session

[Summary from last_summary.md]

## Current State

**What's Working:**
- [List from last_summary.md]

**What Needs Work:**
- [ ] [Items from last_summary.md]

## Quick Start

Ready to resume! Here's what you were working on:
1. [First task]
2. [Second task]

**Current Git Status:**
[Show git status output]

**Services Status:**
[Show service status]

Would you like to continue where you left off?
```
