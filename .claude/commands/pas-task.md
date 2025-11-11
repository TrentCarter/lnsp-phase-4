# /pas-task - Submit Task to PAS/PLMS System

This command provides a conversational interface for submitting tasks to the P0 stack (Gateway → PAS Root → Aider-LCO). As your consultant, I'll help refine your task requirements before submitting to the Architect.

## What This Does

1. **Consultant Mode** - I act as your requirements analyst, asking clarifying questions
2. **Structured Intake** - Gather all necessary information for a well-formed Prime Directive
3. **Validation** - Ensure the task is clear, actionable, and has success criteria
4. **Submission** - Format and submit via Verdict CLI to the P0 stack
5. **Status Tracking** - Monitor progress and report back in real-time
6. **Results Review** - Show final output and validate completion

## Workflow

### Step 1: Activate Consultant Mode

I become your requirements analyst. My role:
- Understand what you want to accomplish
- Clarify ambiguities and assumptions
- Ensure the task is well-scoped
- Reference: `docs/contracts/DIRENG_SYSTEM_PROMPT.md` (Two-tier model)

### Step 2: Gather Task Information

Use the `AskUserQuestion` tool to collect structured information:

**Question Set 1: Core Task**
- "What would you like to accomplish?"
  - Options: "Add new feature", "Fix bug", "Refactor code", "Write tests"
- "Provide a short title (< 10 words)"
- "Describe the task in detail"

**Question Set 2: Scope**
- "What files/systems are involved?"
  - Collect entry point files
  - Identify affected components
- "What's explicitly out of scope?"

**Question Set 3: Success Criteria**
- "How will we know it's complete?"
- "What tests should pass?"
- "What quality requirements?"

**Question Set 4: Constraints**
- "Time budget?" (optional)
- "Token budget?" (optional)
- "Must use specific tools/approaches?" (optional)
- "Files that can't be modified?" (optional)

### Step 3: Format Prime Directive

Build the JSON payload for Verdict:

```json
{
  "title": "<concise title>",
  "goal": "<detailed description with success criteria>",
  "entry_file": "<main file to start from>",
  "constraints": [
    "Must use X",
    "Cannot modify Y",
    "Token budget: Z"
  ],
  "context": {
    "success_criteria": ["criterion 1", "criterion 2"],
    "out_of_scope": ["thing 1", "thing 2"]
  }
}
```

### Step 4: Confirm with User

Present the formatted task as a readable summary:

```
=== Task Summary ===

Title: [title]

Goal:
[detailed goal]

Entry Point: [file]

Success Criteria:
- [criterion 1]
- [criterion 2]

Constraints:
- [constraint 1]

Out of Scope:
- [item 1]

=== Ready to Submit? ===
```

Ask: "Does this look correct? Ready to submit to the Architect?"

If yes → proceed to Step 5
If no → go back and refine

### Step 5: Submit via Verdict CLI

Check that P0 stack is running:

```bash
# Verify services are up
curl -s http://localhost:6120/health > /dev/null 2>&1 || echo "Gateway not running - start with: bash scripts/run_stack.sh"
```

Submit the task:

```bash
./bin/verdict send \
  --title "[title]" \
  --goal "[goal with success criteria]" \
  --entry-file "[entry_file]" \
  --constraints "[constraint1]" "[constraint2]"
```

Capture the task ID from response.

### Step 6: Track Status

Monitor progress and report to user:

**Method 1: Status Polling**
```bash
# Every 30 seconds
./bin/verdict status [task_id]
```

**Method 2: Communication Logs**
```bash
# Real-time tail
./tools/parse_comms_log.py --tail --filter [task_id]
```

**Report Format:**
```
📊 Task Status Update

Status: [in_progress / completed / failed]
Stage: [Architect / PAS / Aider-LCO]
Progress: [X/Y steps complete]

Latest Activity:
- [timestamp] [component]: [message]

Estimated completion: [if available]
```

### Step 7: Review Results

When task completes:

1. **Fetch Results:**
   ```bash
   ./bin/verdict results [task_id]
   ```

2. **Show Changes:**
   - Files modified
   - Diffs (if requested)
   - Test results
   - Token usage / cost

3. **Validate Success:**
   - Check against success criteria
   - Ask user: "Does this meet your requirements?"
   - If no: "What needs adjustment?"

4. **Cleanup:**
   - Archive logs
   - Update task status
   - Offer to commit changes (if satisfied)

## Important Notes

- **Services Required:** Gateway (6120), PAS (6100), Aider-LCO (6130)
- **Check First:** Run `./bin/verdict health` to verify stack is running
- **My Role:** I'm the consultant/frontend - Architect does the heavy lifting
- **Two-Tier Model:** I handle small tasks directly, delegate complex ones to PAS
- **Token Awareness:** PLMS tracks token usage - I'll report costs

## Error Handling

**If services not running:**
```
⚠️  P0 stack not detected. Start services with:
bash scripts/run_stack.sh

Then try /pas-task again.
```

**If task too small:**
```
💡 This task is simple enough for me to handle directly.
Would you like me to:
1. Do it now (faster, no overhead)
2. Submit to PAS anyway (for tracking/logging)
```

**If task unclear:**
```
❓ I need more information to proceed:
- [specific question 1]
- [specific question 2]

Once clarified, I'll format the Prime Directive.
```

## Example Usage

**User:** `/pas-task`

**Me:** "I'll help you submit a task to the P0 stack. What would you like to accomplish?"

**User:** "Add unit tests for the FAISS retrieval module"

**Me:** [Uses AskUserQuestion to gather details]

**Me:** "Here's the task I'll submit:
- Title: Add unit tests for FAISS retrieval
- Entry: src/faiss_retrieval.py
- Success: 80%+ coverage, tests pass
Ready to submit?"

**User:** "Yes"

**Me:** [Submits via Verdict, monitors status, reports completion]

**Me:** "✅ Task complete! 12 tests added, 85% coverage. Token cost: 24.5K. Review changes?"

## References

- **P0 Stack:** `docs/P0_END_TO_END_INTEGRATION.md`
- **Verdict CLI:** `bin/verdict --help`
- **DirEng Contract:** `docs/contracts/DIRENG_SYSTEM_PROMPT.md`
- **PLMS:** `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md`
- **Communication Logs:** `docs/COMMS_LOGGING_GUIDE.md`

---

**Note:** This is a Tier 1 (DirEng) interface to Tier 2 (PAS/Architect). I help you formulate tasks, but the Architect handles complex execution with budget tracking and KPI validation.
