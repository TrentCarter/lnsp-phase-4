# Conversations Log 001

## 2025-09-23: Python 3.11 Version Pinning Implementation

### Task: Implement Python 3.11 version pinning to prevent segfaults

**Objective**: Pin Python to version 3.11 to prevent FAISS/torch ABI/BLAS mismatches that cause segfaults on macOS/ARM.

**Implementation Details**:

1. **Repository Pins (Committed)**:
   - ✅ `.python-version`: Set to `3.11.9`
   - ✅ `pyproject.toml`: Added `[project]` section with `requires-python = ">=3.11,<3.12"`
   - ✅ `scripts/bootstrap_all.sh`: Updated to enforce Python 3.11 with hard-fail if not found
   - ✅ `src/api/retrieve.py`: Added runtime guard to assert Python 3.11 requirement

2. **Bootstrap Script Changes**:
   - Replaced generic `python3` with explicit `python3.11` requirement
   - Added version check with helpful error message for macOS users
   - Added runtime assertion to verify correct Python version in venv
   - Preserved existing LightRAG pinning to `1.4.9rc1`

3. **Runtime Guard**:
   - Added import-time assertion in `src/api/retrieve.py` to fail fast if wrong Python version

**Benefits**:
- Prevents segfaults caused by FAISS/torch wheel incompatibilities
- FAISS/torch wheels are stable on Python 3.11 for macOS/ARM
- Python 3.13 often pulls newer wheels or compiles locally causing ABI mismatches
- Hard pinning prevents environment drift

**Next Steps for User**:
1. Local venv rebuild:
   ```bash
   brew install python@3.11  # if needed
   deactivate 2>/dev/null || true
   rm -rf .venv
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -V  # should print 3.11.x
   pip install -U pip wheel
   pip install -r requirements.txt
   ```

2. VS Code configuration:
   - Cmd-Shift-P → "Python: Select Interpreter" → `.venv/bin/python` (the 3.11 one)

3. Test verification:
   ```bash
   LNSP_TEST_MODE=1 pytest -m "not heavy" -q
   ```

**Files Modified**:
- `.python-version` (new)
- `pyproject.toml` (new)
- `scripts/bootstrap_all.sh` (updated lines 28-40)
- `src/api/retrieve.py` (added runtime guard at top)

---

*This conversation log tracks the implementation of critical Python version pinning to ensure stability and prevent segmentation faults in the LNSP Phase 4 project.*