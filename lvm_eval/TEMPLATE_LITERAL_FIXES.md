# Template Literal Fixes - Jinja2/JavaScript Conflicts

## Problem Summary
Multiple `Uncaught SyntaxError` errors in Chrome caused by Jinja2 template engine mangling ES6 template literals.

### Errors Reported:
1. `Uncaught SyntaxError: Unexpected identifier '#$' (at (index):5313:48)`
2. `Uncaught SyntaxError: Unexpected identifier '#metric$' (at (index):5404:24)`

## Root Cause
Jinja2 interprets `${variable}` in JavaScript template literals as `{variable}` template syntax and strips it out, leaving broken JavaScript.

### Example:
```javascript
// Source code:
const element = $(`#${key}`);

// After Jinja2 processing:
const element = $(`#$`);  // ❌ Syntax error!
```

## All Fixes Applied

### File: `lvm_eval/templates/index.html`

| Line | Original (Broken) | Fixed |
|------|-------------------|-------|
| 415 | `` `/api/models?limit=${limit}` `` | `'/api/models?limit=' + limit` |
| 423 | `` `Loaded ${models.length} models` `` | `'Loaded ' + models.length + ' models'` |
| 472 | `` `Loaded ${models.length} models (newest first)` `` | `'Loaded ' + models.length + ' models (newest first)'` |
| 474 | `` `<div class="alert alert-danger">Error: ${data.error}</div>` `` | `'<div class="alert alert-danger">Error: ' + data.error + '</div>'` |
| 672 | `` `alert alert-${type} alert-dismissible` `` | `'alert alert-' + type + ' alert-dismissible'` |
| 716 | `` `Model ${index + 1}` `` | `'Model ' + (index + 1)` |
| 828 | `` $(\`#${key}\`) `` | `$('#' + key)` |
| 843 | `` $(\`.model-checkbox[value="${modelPath}"]\`) `` | `$('.model-checkbox[value="' + escapeHtml(modelPath) + '"]')` |
| 919 | `` $(\`#metric${metric.charAt(0).toUpperCase() + metric.slice(1)}\`) `` | `$('#metric' + metric.charAt(0).toUpperCase() + metric.slice(1))` |
| 978 | `` `Evaluation completed for ${results.length} models!` `` | `'Evaluation completed for ' + results.length + ' models!'` |
| 989 | `` `${progress}% - Processing ${modelName}...` `` | `progress + '% - Processing ' + modelName + '...'` |
| 990 | `` `${progress}%` `` | `progress + '%'` |
| 1019 | `` `Error evaluating ${modelName}: ${error}` `` | `'Error evaluating ' + modelName + ': ' + error` |

## Solution Pattern

**Before (Breaks):**
```javascript
const str = `Some ${variable} text`;
```

**After (Works):**
```javascript
const str = 'Some ' + variable + ' text';
```

## Why This Happens

1. **Jinja2** renders the template first (server-side)
2. It sees `${...}` and thinks it's a Jinja2 variable
3. Since the variable doesn't exist in Jinja2 context, it strips it
4. Browser receives mangled JavaScript
5. JavaScript parser throws syntax error

## Testing

```bash
# Verify no mangled selectors remain
curl -s http://localhost:8999/ | grep -o '#\$\|#metric\$'
# Output: (empty) - No matches = Fixed! ✅
```

## Prevention

**Best Practices for Jinja2 + JavaScript:**

1. **Avoid ES6 template literals** in Jinja2 templates
2. **Use string concatenation** instead: `'str' + var`
3. **Or escape properly**: `{% raw %}...{% endraw %}` (but harder to maintain)
4. **Move JS to separate files** if possible (served as static assets)

## Status

✅ **ALL FIXED** - 13 template literal conflicts resolved

Dashboard should now work without JavaScript syntax errors.

## Next Steps

1. **Hard refresh browser**: Ctrl+Shift+R (or Cmd+Shift+R)
2. **Clear cache**: Or use incognito mode
3. **Test evaluation**: Select models and run evaluation
4. **Check console**: F12 → Console should be clean
