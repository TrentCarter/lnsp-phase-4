# LVM Dashboard Improvements - Model Discovery & Control

## Summary

Enhanced the LVM Evaluation Dashboard with comprehensive model discovery and user controls for managing the model list display.

## Changes Made

### 1. **Expanded Model Search** 🔍
- **Before**: Only searched `artifacts/lvm/models` (limited scope)
- **After**: Searches entire project across multiple directories:
  - `artifacts/lvm/models/`
  - `models/`
  - `lvm_eval/models/`
- **Result**: Found **196 total models** (up from ~100)

### 2. **Dynamic Model Limit Control** ⚙️

Added dropdown selector with options:
- 50 models
- 100 models  
- 200 models (default)
- 500 models
- All models (no limit)

### 3. **Refresh Button** 🔄
- Manual refresh to reload model list
- Respects current limit selection
- Useful for detecting newly trained models

### 4. **Model Display Info** ℹ️
- Shows current count: "Showing X models (newest first)"
- Models sorted by modification time (newest first)
- Displays relative path, size in MB, and last modified date

### 5. **API Enhancements** 📡

#### Updated `/api/models` Endpoint
- Accepts `limit` query parameter (e.g., `/api/models?limit=50`)
- Returns metadata:
  ```json
  {
    "status": "success",
    "models": [...],
    "total": 196,
    "limit": 50
  }
  ```

#### Improved Model Discovery Function
```python
def get_available_models(limit=None, search_all=True):
    """
    Args:
        limit: Maximum models to return (None = all)
        search_all: Search entire project vs just artifacts/
    """
```

### 6. **User Preferences** 💾
- Saves selected limit to localStorage
- Restores preference on page reload
- Remembers selected models across refreshes

## UI Updates

### Before
```
[Search box]
[Model list - fixed 50 models]
```

### After
```
[Search box]
[Show: [Dropdown ▼] [🔄 Refresh]]
ℹ️ Showing 196 models (newest first)
[Model list - configurable display]
```

## Testing

### Backend Tests ✅
```bash
# Test with limit
curl "http://localhost:8999/api/models?limit=10"
# Returns: 10 models

# Test without limit  
curl "http://localhost:8999/api/models"
# Returns: 196 models (all)
```

### Model Discovery Results
- **Total Models Found**: 196
- **Newest Model**: `transformer_p5_20251102_095841/stageA/final_model.pt`
- **Search Locations**: 3 directories scanned
- **Sort Order**: Newest first (by modification timestamp)

## Benefits

1. **Better Discoverability**: Users can now see ALL models in the project
2. **Performance Control**: Users can limit display for faster page loads
3. **Always Current**: Refresh button detects newly trained models
4. **User-Friendly**: Dropdown makes it obvious how to control the list
5. **Smart Defaults**: 200 models shown by default (good balance)

## How to Use

1. **Open Dashboard**: http://localhost:8999
2. **Select Limit**: Use "Show: [dropdown]" to choose how many models
3. **Refresh**: Click refresh button to detect new models
4. **Search**: Use search box to filter by name/path

## Technical Notes

- Models sorted by `mtime` (modification time), descending
- Searches recursively through all subdirectories  
- `.pt` and `.pth` files both detected
- Error handling for missing/corrupt model files
- Preserves selected models when changing limit

## Future Enhancements

Potential improvements:
- Filter by model type (transformer, amn, mamba, etc.)
- Sort by size, name, or date
- Group by training session/date
- Model metadata preview (config, performance)
- Bulk actions (select all from date range)

---

**Status**: ✅ Fully implemented and tested
**Dashboard URL**: http://localhost:8999
**Diagnostic URL**: http://localhost:8999/diagnostic
