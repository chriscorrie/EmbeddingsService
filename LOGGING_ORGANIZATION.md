# Logging and Performance File Organization

This project follows a centralized approach to log and performance file management to keep the repository clean and organized.

## Directory Structure

```
EmbeddingsService/
├── logs/                           # All log and performance files
│   ├── *.log                      # Application logs
│   ├── performance_report_*.json  # Performance reports  
│   ├── benchmark_results*.json    # Benchmark results
│   ├── gpu_optimization_*.json    # GPU optimization results
│   └── *_debug.log               # Debug logs
├── logging_config.py              # Centralized logging configuration
└── demo_logging.py               # Example usage
```

## Usage

### Import the logging configuration:
```python
from logging_config import log_file, performance_file, benchmark_file, debug_file
```

### Get proper file paths:
```python
# All these files will be created in the logs/ directory
log_path = log_file('application.log')
perf_path = performance_file('performance_report_batch_1_100')  
bench_path = benchmark_file('benchmark_results')
debug_path = debug_file('debug_output.log')
```

### Use the context manager:
```python
from logging_config import LogPathManager

with LogPathManager() as log_manager:
    performance_file = log_manager.performance_file('my_performance')
    benchmark_file = log_manager.benchmark_file('my_benchmark')
```

### Set up a logger that writes to logs/:
```python
from logging_config import setup_file_handler

logger = setup_file_handler(
    logger_name="my_app", 
    log_filename="my_application.log"
)
logger.info("This message goes to logs/my_application.log")
```

## Benefits

1. **Clean Repository**: All temporary files are organized in the logs/ directory
2. **Git Ignore**: Performance and log files are automatically ignored by git
3. **Consistent Organization**: All projects use the same structure
4. **Easy Integration**: Simple imports provide the right file paths
5. **Future-Proof**: New log types can be added to the centralized config

## Git Ignore Patterns

The following patterns are automatically ignored:
- `*.log`
- `logs/`
- `performance_report_*.json`
- `benchmark_results*.json`
- `gpu_optimization_results*.json`
- `*_performance.json`
- `*_benchmark.json`

## For New Projects

1. Copy `logging_config.py` to your new project
2. Import and use the path functions instead of hardcoded filenames
3. Add the recommended .gitignore patterns
4. Create a `logs/` directory in your project root

This ensures all your projects maintain clean, organized repositories without cluttering git history with temporary log and performance files.
