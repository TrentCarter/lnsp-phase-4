# Wrap-Up Session

Help the user wrap up the current work session efficiently.

## Usage

```
/wrap-up          # Create summary only (no git operations)
/wrap-up --git    # Create summary + git commit and push
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

## Step 4: Completion

Confirm completion:
- ✅ Summary created in `docs/last_summary.md`
- ✅ Previous summary archived to `docs/all_project_summary.md`
- ✅ [If --git] Changes committed and pushed

Ready for `/clear` when you're done.

## Notes

- DO NOT read files unless necessary for context
- DO NOT run `git diff` or review changes (waste of tokens)
- Base summary on conversation history, not git inspection
- Keep focused on deliverables, not process
