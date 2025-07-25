#!/usr/bin/env python3
"""
Centralized logging configuration for clean project organization.
This module ensures all log and performance files are stored in appropriate directories.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Default directories for different types of files
DEFAULT_LOGS_DIR = "logs"
DEFAULT_PERFORMANCE_DIR = "logs"
DEFAULT_BENCHMARKS_DIR = "logs" 
DEFAULT_DEBUG_DIR = "logs"

def ensure_logs_directory(logs_dir: str = DEFAULT_LOGS_DIR) -> Path:
    """
    Ensure the logs directory exists and return its Path object.
    
    Args:
        logs_dir: Directory name for logs (default: "logs")
        
    Returns:
        Path object for the logs directory
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(exist_ok=True)
    return logs_path

def get_log_file_path(filename: str, logs_dir: str = DEFAULT_LOGS_DIR) -> str:
    """
    Get the full path for a log file, ensuring the directory exists.
    
    Args:
        filename: Name of the log file
        logs_dir: Directory for logs (default: "logs")
        
    Returns:
        Full path to the log file
    """
    logs_path = ensure_logs_directory(logs_dir)
    return str(logs_path / filename)

def get_performance_file_path(filename: str, logs_dir: str = DEFAULT_PERFORMANCE_DIR) -> str:
    """
    Get the full path for a performance report file, ensuring the directory exists.
    
    Args:
        filename: Name of the performance file (should include .json extension)
        logs_dir: Directory for performance files (default: "logs")
        
    Returns:
        Full path to the performance file
    """
    if not filename.endswith('.json'):
        filename += '.json'
    
    logs_path = ensure_logs_directory(logs_dir)
    return str(logs_path / filename)

def get_benchmark_file_path(filename: str, logs_dir: str = DEFAULT_BENCHMARKS_DIR) -> str:
    """
    Get the full path for a benchmark file, ensuring the directory exists.
    
    Args:
        filename: Name of the benchmark file (should include .json extension)
        logs_dir: Directory for benchmark files (default: "logs")
        
    Returns:
        Full path to the benchmark file
    """
    if not filename.endswith('.json'):
        filename += '.json'
        
    logs_path = ensure_logs_directory(logs_dir)
    return str(logs_path / filename)

def setup_file_handler(logger_name: str, log_filename: str, 
                      level: int = logging.INFO, 
                      format_string: Optional[str] = None,
                      logs_dir: str = DEFAULT_LOGS_DIR) -> logging.Logger:
    """
    Set up a logger with file handler that writes to the logs directory.
    
    Args:
        logger_name: Name of the logger
        log_filename: Name of the log file
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        logs_dir: Directory for logs (default: "logs")
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Ensure logs directory exists
    log_file_path = get_log_file_path(log_filename, logs_dir)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    file_handler.setFormatter(formatter)
    
    # Add handler to logger (avoid duplicates)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path 
               for h in logger.handlers):
        logger.addHandler(file_handler)
    
    return logger

class LogPathManager:
    """
    Context manager and utility class for managing log file paths consistently.
    Useful for ensuring all log-related files go to the correct directory.
    """
    
    def __init__(self, base_logs_dir: str = DEFAULT_LOGS_DIR):
        self.base_logs_dir = base_logs_dir
        self.logs_path = ensure_logs_directory(base_logs_dir)
    
    def log_file(self, filename: str) -> str:
        """Get path for a regular log file"""
        return str(self.logs_path / filename)
    
    def performance_file(self, filename: str) -> str:
        """Get path for a performance report file"""
        if not filename.endswith('.json'):
            filename += '.json'
        return str(self.logs_path / filename)
    
    def benchmark_file(self, filename: str) -> str:
        """Get path for a benchmark file"""
        if not filename.endswith('.json'):
            filename += '.json'
        return str(self.logs_path / filename)
    
    def debug_file(self, filename: str) -> str:
        """Get path for a debug file"""
        return str(self.logs_path / filename)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Global instance for easy access
log_paths = LogPathManager()

# Convenience functions that use the global instance
def log_file(filename: str) -> str:
    """Get path for a log file in the logs directory"""
    return log_paths.log_file(filename)

def performance_file(filename: str) -> str:
    """Get path for a performance file in the logs directory"""
    return log_paths.performance_file(filename)

def benchmark_file(filename: str) -> str:
    """Get path for a benchmark file in the logs directory"""
    return log_paths.benchmark_file(filename)

def debug_file(filename: str) -> str:
    """Get path for a debug file in the logs directory"""
    return log_paths.debug_file(filename)

# Project organization patterns
LOGGING_PATTERNS = {
    'performance_reports': 'performance_report_*.json',
    'benchmark_results': 'benchmark_results*.json',
    'gpu_optimization': 'gpu_optimization_results*.json',
    'debug_logs': '*.log',
    'error_logs': 'error_*.log',
    'api_logs': '*_api.log',
    'processing_logs': '*_processing.log'
}

def get_gitignore_patterns() -> list:
    """
    Get recommended gitignore patterns for log and performance files.
    
    Returns:
        List of gitignore patterns
    """
    return [
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

if __name__ == "__main__":
    # Demo usage
    print("🔧 Logging Configuration Demo")
    print("=" * 50)
    
    # Show file paths
    print(f"Log file path: {log_file('demo.log')}")
    print(f"Performance file path: {performance_file('demo_performance')}")
    print(f"Benchmark file path: {benchmark_file('demo_benchmark')}")
    
    # Show gitignore patterns
    print("\n📝 Recommended .gitignore patterns:")
    for pattern in get_gitignore_patterns():
        print(pattern)
    
    print("\n✅ Logs directory created and ready!")
