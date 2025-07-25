# Development Guidelines for EmbeddingsService

This document establishes coding standards, organization patterns, and best practices for the EmbeddingsService project.

## 🗂️ Project Organization

### Directory Structure Standards

```
EmbeddingsService/
├── 📁 debug/                 # ALL debug, diagnostic, and testing scripts
├── 📁 logs/                  # ALL log, performance, and benchmark files  
├── 📁 venv/                  # Virtual environment (not tracked)
├── 📄 main application files # Core production code only
└── 📄 configuration files    # Settings and configs
```

### ⚠️ **CRITICAL RULES**

1. **Debug Code Location**: 
   - ✅ `debug/debug_aggregation.py`
   - ❌ `debug_aggregation.py` (root level)

2. **Log File Location**:
   - ✅ `logs/performance_report.json`
   - ❌ `performance_report.json` (root level)

3. **Temporary Files**:
   - ✅ `debug/temp_test.py` (auto-ignored by git)
   - ❌ `temp_test.py` (root level)

## 🧪 Debug Script Guidelines

### Location and Naming
- **Directory**: Always place in `debug/` folder
- **Naming Pattern**: `debug_[component]_[specific_issue].py`
- **Examples**:
  - `debug_search_aggregation.py`
  - `debug_entity_extraction.py`
  - `debug_api_response.py`

### Template Structure
```python
#!/usr/bin/env python3
"""
Brief description of what this debug script investigates
"""

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Your imports
from scalable_processor import ScalableEnhancedProcessor
import logging

def debug_function_name():
    """Clear description of what is being debugged"""
    print("=== Debug: Component Name ===")
    
    # Debug implementation here
    
if __name__ == "__main__":
    debug_function_name()
```

### Execution
- **From project root**: `python debug/debug_script.py`
- **With virtual environment**: `source venv/bin/activate && python debug/debug_script.py`

## 📝 Logging Standards

### File Organization
- **Application logs**: `logs/application_name.log`
- **Performance reports**: `logs/performance_report_*.json`
- **Benchmark results**: `logs/benchmark_results_*.json`
- **Debug output**: `logs/debug_*.log`

### Code Usage
```python
from logging_config import log_file, performance_file, debug_file

# Get proper paths automatically
log_path = log_file('my_component.log')
perf_path = performance_file('performance_test_batch_1_100')
debug_path = debug_file('debug_output.log')
```

## 🚫 Git Ignore Standards

### Automatically Ignored
- `logs/` - All log and performance files
- `debug/temp_*.py` - Temporary debug scripts
- `debug/*_temp.py` - Alternative temp naming
- `debug/scratch_*.py` - Scratch/experimental scripts

### Tracked in Git
- `debug/debug_*.py` - Permanent debug scripts
- `debug/README.md` - Debug documentation
- All main application files

## 🤖 AI Assistant Guidelines

When creating new code as an AI assistant:

### ✅ DO:
- Place all debug scripts in `debug/` folder
- Use logging_config functions for file paths
- Follow naming conventions exactly
- Include clear docstrings and comments
- Create temp files with proper prefixes

### ❌ DON'T:
- Put debug scripts in project root
- Hardcode file paths for logs/performance files
- Create unnamed or poorly documented debug scripts
- Mix debug code with production code

## 🔧 Development Workflow

### Adding New Debug Scripts
1. Create in `debug/` folder with proper naming
2. Use the debug script template
3. Add to `debug/README.md` table if permanent
4. Test from project root directory

### Adding New Features
1. Implement in appropriate production file
2. Create debug script to test the feature
3. Use logging_config for any file outputs
4. Update documentation as needed

### Performance Analysis
1. Use `performance_timer.py` functions
2. Save reports to `logs/` directory
3. Create debug scripts to analyze results
4. Document findings in appropriate README

## 📚 Documentation Standards

### Required Documentation
- `README.md` - Main project documentation
- `debug/README.md` - Debug scripts documentation
- `LOGGING_ORGANIZATION.md` - Logging guidelines
- Inline docstrings for all functions

### Update Requirements
- Update `debug/README.md` when adding permanent debug scripts
- Update main `README.md` for architecture changes
- Update this file for new development patterns

---

**Remember**: Clean organization leads to maintainable code. These guidelines ensure that developers (human or AI) can quickly understand the project structure and contribute effectively.
