# ⚠️ DATABASE RESET SCRIPT DOCUMENTATION ⚠️

## ResetDatabase.py - Complete Production Database Reset

### 🔥 CRITICAL WARNING 🔥

**THIS SCRIPT WILL PERMANENTLY DELETE ALL DATA!**

- **ALL Milvus vector embeddings** (5,312+ records)
- **ALL SQL Server entity extractions** 
- **ALL search indexes and vector data**

### What This Script Does

1. **Milvus Vector Database:**
   - Drops all collections: `boilerplate`, `opportunity_descriptions`, `opportunity_documents`, `opportunity_titles`
   - Deletes all embeddings and search indexes
   - Recreates empty collections using **centralized schema from `setup_vector_db.py`**

2. **SQL Server Database:**
   - Executes: `TRUNCATE TABLE [FBOInternalAPI].[ExtractedEntities]`
   - Removes all person, organization, and entity extractions

### 🏗️ Schema Management

**IMPORTANT:** All Milvus collection schemas are defined in **`setup_vector_db.py`**

- **Schema Source:** `/home/chris/Projects/EmbeddingsService/setup_vector_db.py`
- **Schema Function:** `setup_vector_database()`
- **Schema Updates:** Only modify schemas in `setup_vector_db.py` - ResetDatabase will automatically use the latest definitions

**Schema Details:**
- `opportunity_titles`: 8 fields (id, opportunity_id, embedding, posted_date, importance_score, chunk_index, total_chunks, text_content)
- `opportunity_descriptions`: 8 fields (same as titles)
- `opportunity_documents`: 11 fields (adds file_id, base_importance, file_location, section_type)
- `boilerplate`: 6 fields (id, boilerplate_file, embedding, chunk_index, total_chunks, text_content)

### Safety Features

✅ **Multiple Confirmation Steps:**
- Environment verification
- Explicit destruction confirmation
- Final timestamp confirmation

✅ **Backup Creation:**
- Saves collection metadata before deletion
- Creates timestamped backup file

✅ **Connection Validation:**
- Verifies Milvus and SQL Server connectivity
- Fails safely if connections unavailable

✅ **Cancellation Options:**
- Multiple points to cancel operation
- 5-second countdown with Ctrl+C option

### When to Use This Script

**✅ SAFE TO USE WHEN:**
- Starting completely fresh with new data
- Testing new processing configurations
- Clearing corrupted or inconsistent data
- Development/testing environments

**❌ NEVER USE WHEN:**
- Production system is in active use
- Data has not been backed up elsewhere
- Uncertain about data recovery options
- Processing work represents significant time investment

### How to Run

1. **Ensure Virtual Environment:**
   ```bash
   cd /home/chris/Projects/EmbeddingsService
   source venv/bin/activate
   ```

2. **Run with Python:**
   ```bash
   python ResetDatabase.py
   ```

3. **Follow Safety Prompts:**
   - Confirm environment is correct
   - Type "DELETE ALL DATA" exactly
   - Type "RESET DATABASE NOW" to proceed
   - Wait through 5-second countdown

### Recovery After Reset

After running this script, you'll need to:

1. **Re-process All Documents:**
   ```bash
   python process_documents.py
   ```

2. **Re-extract All Entities:**
   ```bash
   python entity_extractor.py
   ```

3. **Verify API Functionality:**
   ```bash
   curl http://localhost:5000/health
   ```

4. **Check Collection Counts:**
   ```bash
   python check_collections.py
   ```

### Performance Impact

- **Reset Time:** ~30 seconds to 2 minutes
- **Recovery Time:** Hours to days (depending on document volume)
- **GPU Utilization:** Will return to 5,269 sentences/second during re-processing

### Files Created

- `database_reset_backup_YYYYMMDD_HHMMSS.json` - Metadata backup
- Log output showing all operations performed

### Troubleshooting

**If reset fails partially:**
1. Check Milvus container status: `docker ps`
2. Verify SQL Server connectivity
3. Run `create_index.py` manually if collections not recreated
4. Check service logs: `sudo journalctl -u document-embedding-api-v3.service`

**If you regret running it:**
- There is no undo option
- Data must be re-processed from source documents
- Entity extractions must be re-run

### Emergency Contacts

If you accidentally run this script:
1. **STOP** any running processing immediately
2. Check if source documents are still available
3. Estimate re-processing time based on document volume
4. Consider if any external backups exist

---

## 🛡️ REMEMBER: This script is for COMPLETE database resets only!

Only run if you are absolutely certain you want to start fresh with empty databases.
