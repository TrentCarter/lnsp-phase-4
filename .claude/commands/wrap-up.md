# Wrap-Up Session

Help the user wrap up the current work session efficiently.

## Usage

```
/wrap-up                  # Create summary only (no git operations)
/wrap-up --git            # Create summary + git commit and push
/wrap-up --restart        # Create summary + restart HMI service
/wrap-up --git --restart  # All three: summary + git + restart HMI
```

## Step 1: Archive Previous Summary

If `docs/last_summary.md` exists, archive it without reading:

```bash
if [ -f docs/last_summary.md ]; then
  echo -e "\n===\n$(date '+%Y-%m-%d %H:%M:%S')\n" >> docs/all_project_summary.md
  cat docs/last_summary.md >> docs/all_project_summary.md
fi
```

## Step 2: Create New Summary

Create `docs/last_summary.md` based on conversation context:

```markdown
# Last Session Summary

**Date:** YYYY-MM-DD (Session N)
**Duration:** ~X minutes/hours
**Branch:** [current branch from git]

## What Was Accomplished

[2-3 sentence overview]

## Key Changes

### 1. [Feature/Fix Name]
**Files:** `path/to/file.ext:lines` or (NEW, size)
**Summary:** [1-2 sentences]

### 2. [Next Change]
**Files:** `path/to/file.ext:lines`
**Summary:** [1-2 sentences]

## Files Modified

- `file1.ext` - Brief description
- `file2.ext` - Brief description

## Current State

**What's Working:**
- ✅ [Key working features]

**What Needs Work:**
- [ ] [Next steps or known issues]

## Important Context for Next Session

1. **[Key Context Item]**: Brief explanation
2. **[Another Item]**: Brief explanation

## Quick Start Next Session

1. **Use `/restore`** to load this summary
2. [Next immediate action]
3. [Another action]
```

**Guidelines:**
- Base summary on conversation history (what user requested, what you did)
- Keep concise but informative
- Include file paths with line numbers or sizes
- Focus on "what" and "why", not detailed "how"

## Step 3: Git Operations (Optional)

**Only if `--git` flag is present:**

1. **Show current status:**
   ```bash
   git status --short
   ```

2. **Add documentation files:**
   ```bash
   git add docs/last_summary.md docs/all_project_summary.md CLAUDE.md
   ```

3. **Commit with summary:**
   ```bash
   git commit -m "$(cat <<'EOF'
   docs: wrap-up session YYYY-MM-DD - [brief summary]

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

4. **Push:**
   ```bash
   git push
   ```

## Step 4: Restart HMI (Optional)

**Only if `--restart` flag is present:**

1. **Kill existing HMI process:**
   ```bash
   lsof -ti:6101 | xargs -r kill -9
   ```

2. **Wait for clean shutdown:**
   ```bash
   sleep 2
   ```

3. **Restart HMI:**
   ```bash
   PYTHONPATH=. ./.venv/bin/python services/webui/hmi_app.py > /tmp/hmi.log 2>&1 &
   ```

4. **Verify it started:**
   ```bash
   sleep 3 && curl -s http://localhost:6101 > /dev/null && echo "✅ HMI restarted successfully on http://localhost:6101" || echo "❌ HMI failed to restart - check /tmp/hmi.log"
   ```

## Step 5: Audio Notification

Play "wrap up complete" voice notification using macOS TTS:

```bash
say "wrap up complete"
```

This alerts the user that the wrap-up process has finished and they can take action.

## Step 6: Completion

Confirm completion:
- ✅ Summary created in `docs/last_summary.md`
- ✅ Previous summary archived to `docs/all_project_summary.md`
- ✅ [If --git] Changes committed and pushed
- ✅ [If --restart] HMI service restarted on http://localhost:6101
- ✅ Audio notification played: "wrap up complete"

Ready for `/clear` when you're done.

## Notes

- DO NOT read files unless necessary for context
- DO NOT run `git diff` or review changes (waste of tokens)
- Base summary on conversation history, not git inspection
- Keep focused on deliverables, not process
- **DO NOT kill background services** unless explicitly requested by user
- **DO NOT clean up running services** - they are meant to persist between sessions
- Services that should stay running:
  - Model Pool Manager (port 8050)
  - Model services (ports 8051-8099)
  - Provider Router (port 6103)
  - P0 Stack (Gateway 6120, PAS 6100, Registry 6121, etc.)
  - Vec2Text services (ports 7001, 7002)
  - HMI (port 6101) - only restart if `--restart` flag is used
- Only kill processes if they are temporary test processes or explicitly mentioned in the session
