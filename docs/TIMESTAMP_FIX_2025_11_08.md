# Timestamp Fix (2025-11-08)

## Problem

Actions tab was showing incorrect dates like **"1/21/1970 4:36:57 AM"** instead of the correct date **"11/08/2025 11:08:51 AM"**.

## Root Cause

The database stores timestamps as **Unix seconds** (e.g., `1762618131.642257`), but JavaScript's `Date()` constructor expects **milliseconds**. This is a classic Unix timestamp conversion error.

**Example**:
```javascript
// Database value (Unix seconds)
timestamp = 1762618131.642257

// WRONG (treats seconds as milliseconds → 1970)
new Date(1762618131.642257)  // → 1/21/1970

// CORRECT (multiply by 1000)
new Date(1762618131.642257 * 1000)  // → 11/08/2025
```

## Fixes Applied

### 1. Frontend: Actions Tab (actions.html:734)

**Before**:
```javascript
function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);  // ❌ WRONG
    // ...
}
```

**After**:
```javascript
function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    // FIX: Convert Unix seconds to milliseconds for JavaScript Date
    const date = new Date(timestamp * 1000);  // ✅ CORRECT
    // ...
}
```

### 2. Backend: Sequencer Builder (hmi_app.py:415-426)

**Before**:
```python
# Assumed all timestamps are ISO strings
timestamp_str = action.get('timestamp', datetime.now().isoformat())
try:
    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
except:
    timestamp = datetime.now().timestamp()
```

**After**:
```python
# Handle both Unix timestamp (numeric) and ISO string
timestamp_value = action.get('timestamp', datetime.now().timestamp())
try:
    if isinstance(timestamp_value, (int, float)):
        # Already a Unix timestamp (seconds)
        timestamp = float(timestamp_value)
    else:
        # ISO string - parse and convert to Unix timestamp
        timestamp = datetime.fromisoformat(str(timestamp_value).replace('Z', '+00:00')).timestamp()
except Exception as e:
    logger.warning(f"Error parsing timestamp {timestamp_value}: {e}")
    timestamp = datetime.now().timestamp()
```

## Files Modified

1. **services/webui/templates/actions.html** (line 734)
   - Added `* 1000` to convert Unix seconds → milliseconds

2. **services/webui/hmi_app.py** (lines 415-426)
   - Added type checking to handle both numeric timestamps and ISO strings

## Why This Matters

This bug didn't just affect display - it could have broken real-time updates:

1. **Actions Tab**: Users couldn't see when actions occurred
2. **Sequencer**: Timeline calculations might be wrong if timestamps weren't parsed
3. **SSE Updates**: New actions might not be recognized as "new" if timestamps were wrong

## Testing

Test the fix by refreshing the Actions tab:

```bash
# Open Actions tab in browser
open http://localhost:6101/actions

# Expected:
# - demo-1: 11/08/2025 11:08:51 AM (or current date)
# - NOT: 1/21/1970 4:36:57 AM
```

Test Sequencer with action logs:

```bash
# Run real-time test
./scripts/test_realtime_updates.sh

# Expected:
# - Timeline shows tasks at correct times
# - Time labels (0:00, 5:00, etc.) match actual elapsed time
# - Tasks appear at proper positions on timeline
```

## Related Issues Fixed

This also fixes potential issues with:
- **Tree View**: Timestamps in node tooltips
- **Sequencer**: Task positioning on timeline
- **Real-time updates**: SSE event timing

All views now correctly handle Unix timestamp (seconds) format from the database.

## Notes

- **Database format**: Unix seconds (numeric) with fractional part
- **JavaScript Date**: Expects milliseconds
- **Conversion**: Multiply by 1000 (seconds → milliseconds)
- **Backend**: Now handles both formats (numeric + ISO string) for flexibility

## References

- Unix Time: https://en.wikipedia.org/wiki/Unix_time
- JavaScript Date: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date
- Python datetime: https://docs.python.org/3/library/datetime.html
