# Entity Extraction Deduplication Fix

## Problem Identified

The entity extraction system was creating duplicate entities within the same OpportunityId because:

1. **Individual Processing**: Each content piece (description, document1, document2, etc.) was processed by separate async entity extraction workers
2. **Immediate Storage**: Each worker stored entities directly to the database without opportunity-level consolidation
3. **No Deduplication**: The same person appearing in multiple documents would create multiple entity records

**Example**: John Smith's email in both description and 3 documents = 4 duplicate entity records

## Root Cause

When the system moved to async entity extraction, the opportunity-level consolidation logic (`_consolidate_entities_per_opportunity`) was bypassed. The comment in the code confirmed this: *"No longer collecting entities since they're processed asynchronously"*.

## Solution Implemented

**Hybrid Reference Counting + Timeout Approach** (Option 4):

### 1. Reference Counting (Primary Mechanism)
- **Registration**: Before processing an opportunity, count expected entity extraction tasks and register with `EntityExtractionQueue.register_opportunity_tasks()`
- **Accumulation**: Entity workers accumulate entities per opportunity instead of storing immediately
- **Completion Detection**: When `completed_tasks == expected_tasks`, trigger consolidation

### 2. Timeout Safety Net (Fallback)
- **Background Monitor**: Timeout worker thread runs every 30 seconds
- **Configurable Timeout**: `ENTITY_EXTRACTION_COMPLETION_TIMEOUT = 300` seconds (5 minutes)
- **Prevents Hangs**: If worker threads hang, timeout consolidates available entities and logs warning

### 3. Consolidation Logic
- **Same Algorithm**: Reuses existing `_consolidate_entities_per_opportunity()` logic
- **Email Dedup**: Same email address → single entity
- **Name Dedup**: Same name → single entity
- **Database Storage**: Store consolidated entities once per opportunity

## Configuration Added

```python
# config.py
ENTITY_EXTRACTION_COMPLETION_TIMEOUT = 300  # Timeout for opportunity entity consolidation (5 minutes)
```

## Key Changes

### EntityExtractionQueue Class
- Added `opportunity_tracking` dict for reference counting
- Added `register_opportunity_tasks()` method
- Added `_timeout_monitor_worker()` background thread
- Modified `_process_entity_task()` to accumulate instead of store
- Added `_consolidate_opportunity_entities()` for deduplication

### _process_opportunity_simplified Method
- Added task counting logic before processing
- Calls `register_opportunity_tasks()` with expected count

### Statistics Enhanced
- Added `opportunities_completed`, `opportunities_timed_out`, `total_entities_stored` stats
- Updated stat transfer logic to use new consolidated metrics

## Expected Results

**Before Fix**: 
- 1 opportunity with description + 3 documents containing same person = 4 duplicate entities

**After Fix**:
- 1 opportunity with description + 3 documents containing same person = 1 consolidated entity
- Same deduplication logic that previously worked, now properly applied per opportunity

## Testing

Created `debug/test_entity_deduplication.py` to validate:
- Extract entities from multiple content pieces with deliberate duplicates
- Test consolidation logic
- Verify database storage shows deduplicated results
- Confirm John Smith (appearing 4 times) becomes 1 entity record

## Benefits

1. **Eliminates Duplicates**: Same entity consolidation as before async architecture
2. **Robust**: Reference counting + timeout prevents processing hangs
3. **Configurable**: Timeout can be adjusted per environment needs
4. **Backward Compatible**: No breaking changes to existing API or processing flow
5. **Performance**: Still maintains async processing benefits while fixing deduplication

## Next Steps

1. Test with standard parameters (rows 1-35, reprocess=false)
2. Verify entity count reduction in database
3. Monitor timeout logs to ensure reference counting works correctly
4. Adjust `ENTITY_EXTRACTION_COMPLETION_TIMEOUT` if needed based on environment
