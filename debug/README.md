# Debug Scripts Directory

This directory contains all debugging, testing, and diagnostic scripts for the EmbeddingsService project. These scripts are NOT part of the main application and are used for development, troubleshooting, and analysis purposes.

## Organization Guidelines

### 🛡️ **IMPORTANT: All debug code must go in this folder**
- Never place debug scripts in the project root
- Keep debug code separate from production code
- Use descriptive filenames that indicate what is being debugged

### 📁 **File Naming Convention**
- Use prefix `debug_` for all debug scripts
- Include the component/feature being debugged: `debug_[component]_[specific_issue].py`
- Examples:
  - `debug_aggregation.py` - Debug search aggregation logic
  - `debug_entity_extraction.py` - Debug entity extraction process
  - `debug_api_search.py` - Debug API search functionality

### 🔧 **Script Structure**
All debug scripts should follow this template:
```python
#!/usr/bin/env python3
"""
Brief description of what this debug script does
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your imports and debug code here

if __name__ == "__main__":
    # Debug execution code
    pass
```

### 📋 **Current Debug Scripts**

| Script | Purpose | Status |
|--------|---------|--------|
| `debug_aggregation.py` | Debug search aggregation process | Active |
| `debug_api_search.py` | Debug API search functionality | Active |
| `debug_direct_search.py` | Debug direct search operations | Active |
| `debug_entity_extraction.py` | Debug entity extraction process | Active |
| `debug_entity_issue.py` | Debug specific entity issues | Active |
| `debug_info_mgmt.py` | Debug information management | Active |
| `debug_production_flow.py` | Debug production workflow | Active |
| `debug_real_entity_flow.py` | Debug real entity processing flow | Active |
| `debug_search.py` | Debug general search functionality | Active |
| `debug_search_issue.py` | Debug specific search issues | Active |
| `debug_specific_title.py` | Debug specific title processing | Active |

### 🚀 **Running Debug Scripts**

1. **From the project root directory:**
   ```bash
   cd /path/to/EmbeddingsService
   python debug/debug_aggregation.py
   ```

2. **With virtual environment:**
   ```bash
   source venv/bin/activate
   python debug/debug_aggregation.py
   ```

3. **Direct execution:**
   ```bash
   cd debug
   python debug_aggregation.py
   ```

### 📝 **Git Handling**
- Debug scripts are tracked in git for team collaboration
- If creating temporary debug files, prefix with `temp_` and add to .gitignore
- Clean up obsolete debug scripts regularly

### 🎯 **Best Practices**
1. **Clear Documentation**: Each script should have a clear docstring explaining its purpose
2. **Minimal Dependencies**: Use only necessary imports
3. **Error Handling**: Include proper error handling and logging
4. **Clean Output**: Use clear, formatted output for debugging information
5. **Self-Contained**: Scripts should be runnable independently

### 🔄 **Maintenance**
- Review debug scripts monthly
- Archive or remove obsolete debugging code
- Update this README when adding new scripts
- Keep debug scripts updated with main codebase changes

---
**For AI Assistants/Agents**: Always place new debug, diagnostic, or testing scripts in this `debug/` directory. Never create debug scripts in the project root or main source directories.
