# Wrap-Up and Document Session

You are about to help the user wrap up the current work session. Follow these steps carefully:

## Step 0: Archive Previous Summary

**CRITICAL**: Before creating the new summary, archive the previous one:

1. **Check if `last_summary.md` exists:**
   ```bash
   ls docs/last_summary.md
   ```

2. **If it exists, append it to `all_project_summary.md`:**
   - Read the contents of `docs/last_summary.md`
   - Open (or create) `docs/all_project_summary.md`
   - Append to the file:
     ```
     ===
     [Previous Date from last_summary.md]

     [Full contents of last_summary.md]

     ```
   - This file is for archival only - DO NOT load it into context

3. **If `last_summary.md` doesn't exist:**
   - Skip archival step (this is the first session)

## Step 1: Review Recent Changes

Review what has been changed in this session:
1. Use `git status` to see modified files
2. Use `git diff` to see actual changes (focus on key files, not all diffs)
3. Identify the main features/fixes that were implemented

## Step 2: Update Documentation

Based on the changes you found:

1. **Check if CLAUDE.md needs updates:**
   - Read the relevant sections of CLAUDE.md
   - If new features were added, update the "Current Status" section
   - If new components were created, add them to the appropriate sections
   - If configurations changed, update the configuration sections

2. **Update any relevant PRD or design docs:**
   - Look for PRD files in `docs/PRDs/` related to the work
   - Update implementation status if features were completed
   - Add any new sections for features that were added

3. **Check README or other docs:**
   - If user-facing features were added, consider updating README
   - If API changes were made, update API documentation

## Step 3: Create Session Summaries

Create TWO summary files:

### A. `docs/last_summary.md` (FOR CONTEXT LOADING)

**This file is designed to be loaded into context with `/restore`**

Keep it concise but comprehensive:

```markdown
# Last Session Summary

**Date:** YYYY-MM-DD
**Duration:** [e.g., 2 hours]
**Branch:** [current git branch]

## What Was Accomplished

[2-3 sentence overview of main achievements]

## Key Changes

### 1. [Feature/Fix Name]
**Files:** `path/to/file.ext:line-range`
**Summary:** [1-2 sentences explaining the change and why]

### 2. [Next Feature/Fix]
**Files:** `path/to/file.ext:line-range`
**Summary:** [1-2 sentences explaining the change and why]

## Files Modified

- `file1.ext` - [Brief description]
- `file2.ext` - [Brief description]

## Current State

**What's Working:**
- [Key working features]

**What Needs Work:**
- [ ] [Known issues or next steps]

## Important Context for Next Session

- [Any critical context to remember]
- [Configuration changes]
- [Breaking changes or gotchas]

## Quick Start Next Session

1. [First thing to do when resuming work]
2. [Second thing]
3. [Third thing]
```

### B. `docs/session_summaries/YYYY-MM-DD_session_summary.md` (DETAILED ARCHIVE)

**This is the detailed archive - NOT for context loading**

Use the comprehensive structure:

```markdown
# Session Summary: [Date]

## Overview
Detailed paragraph explaining what was accomplished.

## Changes Made

### 1. [Feature/Fix Name]
**Files Changed:**
- `path/to/file.ext:line-range` - Description of change
- `path/to/file2.ext:line-range` - Description of change

**What Changed:**
- Detailed explanation of what was implemented
- Why it was implemented this way
- Any important technical decisions

**Testing:**
- How to test this feature
- Expected behavior

### 2. [Next Feature/Fix]
[Same structure as above]

## Files Modified

Complete list with line numbers:
- `file1.ext:123-456` - Brief description
- `file2.ext:789-1011` - Brief description

## Next Steps

If there are known issues or follow-up tasks:
- [ ] Task 1
- [ ] Task 2

## Notes

Any important context for future sessions:
- Configuration changes
- Breaking changes
- Dependencies added/removed
```

## Step 4: Git Status Check

Before committing:
1. Show `git status` one final time
2. List any untracked files that might need attention
3. Identify files that should be committed

## Step 5: Commit and Push Changes

**Automatically commit and push all session documentation:**

1. **Add all documentation files:**
   ```bash
   git add docs/last_summary.md docs/all_project_summary.md docs/session_summaries/ .claude/commands/ CLAUDE.md docs/readme.txt
   ```

2. **Create commit with descriptive message:**
   Use this format:
   ```bash
   git commit -m "$(cat <<'EOF'
   docs: wrap-up session YYYY-MM-DD - [brief summary]

   Session summary:
   - [Main accomplishment 1]
   - [Main accomplishment 2]
   - [Main accomplishment 3]

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

3. **Push to remote:**
   ```bash
   git push
   ```

4. **Verify:**
   ```bash
   git status
   ```

## Step 6: Final Checklist

Provide a final checklist:
- [ ] Previous summary archived to `all_project_summary.md`
- [ ] New `last_summary.md` created (concise, for `/restore`)
- [ ] Detailed summary created in `session_summaries/` (archival)
- [ ] All documentation updated (CLAUDE.md, PRDs, etc.)
- [ ] All changes committed and pushed
- [ ] Ready for `/clear`

## Important Notes:

- **DO** commit and push all documentation changes automatically
- **DO NOT** run `/clear` - just prepare for it
- **DO NOT** load `all_project_summary.md` into context (it's archival only)
- Focus on making `last_summary.md` concise and useful for `/restore`
- Include specific file paths and line numbers
- The detailed summary goes in `session_summaries/` for historical record

## File Locations Reference

- `docs/last_summary.md` - Current session (LOAD THIS with `/restore`)
- `docs/all_project_summary.md` - All previous sessions (DO NOT LOAD)
- `docs/session_summaries/YYYY-MM-DD_session_summary.md` - Today's detailed archive
