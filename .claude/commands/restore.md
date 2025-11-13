# Restore Context from Last Session

This command loads the context from the previous work session to help you quickly resume work.

## Usage

- `/restore` - Load last session summary (concise output)
- `/restore --git` - Load last session summary + detailed git status

## What This Does

1. **Loads `docs/last_summary.md` into context** - Contains summary of last session
2. **Shows where you left off** - Key changes, current state, next steps
3. **Checks running services** - Verify what's currently active
4. **Optional git details** - Only with `--git` flag

## Step 1: Load Last Summary

Read and present the contents of `docs/last_summary.md` to the user in **CONCISE** format.

If the file doesn't exist:
- Inform the user that no previous session summary was found
- Suggest running `/wrap-up` at the end of their next session to create one

## Step 2: Verify Current State

**Always check:**
1. **Check running services:**
   ```bash
   lsof -ti:6101 > /dev/null && echo "HMI running" || echo "HMI not running"
   lsof -ti:6120 > /dev/null && echo "Gateway running" || echo "Gateway not running"
   curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && echo "Ollama running" || echo "Ollama not running"
   ```

**Only if `--git` flag is present:**
1. **Check git branch:**
   ```bash
   git branch --show-current
   ```

2. **Check git status:**
   ```bash
   git status --short
   ```

3. **Show uncommitted file details**

## Step 3: Present Concise Summary

**Default output format (NO --git flag):**

```
📋 Restored Context

**Last Session (DATE):** [One-line summary of what was accomplished]

**Services:** [✓/✗ for HMI, Gateway, Ollama]

**What's Working:**
- [3-5 bullet points max]

**What Needs Work:**
- [ ] [Top 3 priorities]

Would you like to [continue/commit/start next task]?
```

## Step 4: Announce Completion with TTS

After presenting the summary, run:
```bash
say "Claude Ready"
```

**With --git flag:**

Add these sections:
```
**Current Branch:** [branch name]

**Uncommitted Changes:**
[git status --short output]

**Files Modified:**
- [list of modified files with brief description]
```

## Important Notes:

- **BE CONCISE** - Default output should be <15 lines
- **DO NOT** load `all_project_summary.md` - it's archival only
- **DO load** `last_summary.md` - designed for quick context loading
- **DO NOT** show verbose git details unless `--git` flag is used
- Focus on helping the user resume work quickly

## Example Outputs

### Example 1: `/restore` (default - concise)

```
📋 Restored Context

**Last Session (2025-11-13):** Fixed LLM chat interface bugs - multi-provider routing working

**Services:** ✓ HMI, Gateway, Ollama

**What's Working:**
- Ollama models streaming correctly
- Multi-provider routing infrastructure
- Input re-enabling fixed

**What Needs Work:**
- [ ] Browser cache (need hard refresh)
- [ ] OpenAI/Google SDK implementation
- [ ] Token cost tracking

Would you like to commit the current work or continue with next priorities?
```

### Example 2: `/restore --git` (verbose)

```
📋 Restored Context

**Last Session (2025-11-13):** Fixed LLM chat interface bugs - multi-provider routing working

**Current Branch:** feature/aider-lco-p0

**Services:** ✓ HMI, Gateway, Ollama

**Uncommitted Changes:**
 M services/gateway/gateway.py
 M services/webui/templates/llm.html
?? services/webui/templates/llm_multi_select.js

**What's Working:**
- Ollama models streaming correctly
- Multi-provider routing infrastructure
- Input re-enabling fixed

**What Needs Work:**
- [ ] Browser cache (need hard refresh)
- [ ] Commit current session work
- [ ] OpenAI/Google SDK implementation

Would you like to commit using `/wrap-up`?
```
