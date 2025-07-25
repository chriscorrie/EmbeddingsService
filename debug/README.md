# Debug Scripts Directory

This directory contains all debugging, testing, diagnostic, and performance analysis scripts for the EmbeddingsService project. These scripts are NOT part of the main application and are used for development, troubleshooting, validation, and analysis purposes.

## Organization Guidelines

### 🛡️ **IMPORTANT: All debug/test code must go in this folder**
- Never place debug or test scripts in the project root
- Keep debug/test code separate from production code
- Use descriptive filenames that indicate what is being tested or debugged

### 📁 **File Naming Conventions**

#### Debug Scripts
- Use prefix `debug_` for troubleshooting and diagnostic scripts
- Include the component/feature being debugged: `debug_[component]_[specific_issue].py`
- Examples:
  - `debug_aggregation.py` - Debug search aggregation logic
  - `debug_entity_extraction.py` - Debug entity extraction process
  - `debug_api_search.py` - Debug API search functionality

#### Test Scripts
- Use prefix `test_` for feature and functionality testing scripts
- Include the feature being tested: `test_[feature]_[aspect].py`
- Examples:
  - `test_exact_phrase.py` - Test exact phrase search functionality
  - `test_opportunity_search.py` - Test opportunity search features
  - `test_search_implementation.py` - Test search implementation

#### Performance Scripts
- Use descriptive names for performance analysis: `[component]_performance_test.py`
- Use `performance_` prefix for general performance testing
- Examples:
  - `performance_test.py` - General performance testing
  - `gpu_performance_test.py` - GPU-specific performance testing
  - `performance_impact_test.py` - Performance impact analysis

#### Validation Scripts
- Use `[system]_validation_test.py` for system validation
- Examples:
  - `production_validation_test.py` - Production system validation

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

### 📋 **Current Scripts**

#### Debug Scripts
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

#### Test Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `test_exact_phrase.py` | Test exact phrase search functionality | Active |
| `test_opportunity_search.py` | Test opportunity search features | Active |
| `test_producer_consumer.py` | Test producer-consumer patterns | Active |
| `test_scalable_processing.py` | Test scalable processing implementation | Active |
| `test_search_implementation.py` | Test search implementation features | Active |
| `test_threshold_impact.py` | Test threshold impact on search results | Active |

#### Performance & Validation Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `performance_test.py` | General performance testing and comparison | Active |
| `gpu_performance_test.py` | GPU-specific performance testing | Active |
| `performance_impact_test.py` | Performance impact analysis | Active |
| `production_validation_test.py` | Production system validation | Active |

### 🚀 **Running Scripts**

1. **From the project root directory:**
   ```bash
   cd /path/to/EmbeddingsService
   python debug/debug_aggregation.py
   python debug/test_exact_phrase.py
   python debug/performance_test.py
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

### 💡 **Usage Examples**

```bash
# Debug specific issues
python debug/debug_search_issue.py
python debug/debug_entity_extraction.py

# Test functionality
python debug/test_exact_phrase.py
python debug/test_opportunity_search.py

# Performance analysis
python debug/performance_test.py
python debug/gpu_performance_test.py

# System validation
python debug/production_validation_test.py
```

### 📝 **Adding New Scripts**

When adding new scripts to this directory:

1. **Choose the right naming convention** based on the script's purpose
2. **Use the standard import template** for accessing parent directory modules
3. **Update this README** to include your new script in the appropriate table
4. **Document the script's purpose** clearly in the docstring

**Remember**: Never place debug, test, or performance scripts in the project root!

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
