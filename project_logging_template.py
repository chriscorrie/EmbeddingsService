#!/usr/bin/env python3
"""
Project Template: Centralized Logging Configuration
Copy this file to any new project to maintain clean organization.

This ensures all log, performance, and benchmark files go to a logs/ directory
and are properly ignored by git.
"""

import os
import logging
from pathlib import Path
from typing import Optional

# Standard log directory name for all projects
LOGS_DIR = "logs"

def ensure_logs_directory() -> Path:
    """Ensure the logs directory exists and return its Path object."""
    logs_path = Path(LOGS_DIR)
    logs_path.mkdir(exist_ok=True)
    return logs_path

def log_file(filename: str) -> str:
    """Get path for a log file in the logs directory."""
    logs_path = ensure_logs_directory()
    return str(logs_path / filename)

def performance_file(filename: str) -> str:
    """Get path for a performance file in the logs directory."""
    if not filename.endswith('.json'):
        filename += '.json'
    logs_path = ensure_logs_directory()
    return str(logs_path / filename)

def benchmark_file(filename: str) -> str:
    """Get path for a benchmark file in the logs directory."""
    if not filename.endswith('.json'):
        filename += '.json'
    logs_path = ensure_logs_directory()
    return str(logs_path / filename)

def debug_file(filename: str) -> str:
    """Get path for a debug file in the logs directory."""
    logs_path = ensure_logs_directory()
    return str(logs_path / filename)

def setup_logger(name: str, log_filename: str, 
                level: int = logging.INFO) -> logging.Logger:
    """Set up a logger that writes to the logs directory."""
    log_path = log_file(log_filename)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Recommended .gitignore patterns for any project
GITIGNORE_PATTERNS = [
    "# Log files",
    "*.log", 
    "logs/",
    "",
    "# Performance reports and benchmarks",
    "performance_report_*.json",
    "benchmark_results*.json", 
    "gpu_optimization_results*.json",
    "*_performance.json",
    "*_benchmark.json"
]

def create_gitignore():
    """Create or update .gitignore with recommended patterns."""
    gitignore_path = Path('.gitignore')
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing_content = f.read()
    else:
        existing_content = ""
    
    # Check if our patterns are already there
    if "logs/" in existing_content and "performance_report_*.json" in existing_content:
        print("✅ .gitignore already contains logging patterns")
        return
    
    # Add our patterns
    with open(gitignore_path, 'a') as f:
        f.write("\n# Added by logging template\n")
        for pattern in GITIGNORE_PATTERNS:
            f.write(f"{pattern}\n")
    
    print("✅ Added logging patterns to .gitignore")

if __name__ == "__main__":
    # Demo usage
    print("🔧 Project Logging Template")
    print("=" * 40)
    
    # Show file paths
    print(f"Log file: {log_file('app.log')}")
    print(f"Performance: {performance_file('performance_test')}")
    print(f"Benchmark: {benchmark_file('benchmark_test')}")
    
    # Create sample logger
    logger = setup_logger("demo", "demo.log")
    logger.info("This is a test log message")
    print(f"Logger created: logs/demo.log")
    
    # Update gitignore
    create_gitignore()
    
    print("\n✅ Project logging template ready!")
    print("📁 All log files will go to logs/ directory")
    print("🚫 Git will ignore temporary files")
