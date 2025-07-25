#!/usr/bin/env python3
"""
Test and demonstration of the centralized logging configuration.
This shows how all future projects should organize log and performance files.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logging_config import (
    log_file, performance_file, benchmark_file, debug_file,
    setup_file_handler, LogPathManager, get_gitignore_patterns
)

def demo_logging_organization():
    """Demonstrate the logging organization"""
    print("🔧 Centralized Logging Configuration Demo")
    print("=" * 60)
    
    # Show how to get proper file paths
    print("📁 File Path Organization:")
    print(f"   Log file: {log_file('application.log')}")
    print(f"   Performance report: {performance_file('performance_report_1_100')}")
    print(f"   Benchmark results: {benchmark_file('benchmark_results')}")
    print(f"   Debug file: {debug_file('debug_output.log')}")
    
    # Show context manager usage
    print("\n🎯 Context Manager Usage:")
    with LogPathManager() as log_manager:
        print(f"   Performance file: {log_manager.performance_file('test_performance')}")
        print(f"   Benchmark file: {log_manager.benchmark_file('test_benchmark')}")
        print(f"   Debug file: {log_manager.debug_file('test_debug.log')}")
    
    # Show logger setup
    print("\n📝 Logger Setup:")
    logger = setup_file_handler(
        logger_name="demo_logger",
        log_filename="demo_application.log"
    )
    logger.info("This is a test log message - it goes to logs/demo_application.log")
    print(f"   Logger configured: {logger.name}")
    print(f"   Log file: logs/demo_application.log")
    
    # Show gitignore patterns
    print("\n📄 Recommended .gitignore patterns:")
    for pattern in get_gitignore_patterns():
        print(f"   {pattern}")
    
    print("\n✅ All log and performance files will be organized in the logs/ directory!")
    print("🚫 Git will ignore all temporary log and performance files")
    print("📊 Your repository stays clean and organized")

def example_performance_report():
    """Example of how to write a performance report"""
    import json
    from datetime import datetime
    
    # Create sample performance data
    performance_data = {
        "timestamp": datetime.now().isoformat(),
        "test_name": "logging_demo",
        "operations": 100,
        "total_time": 1.234,
        "average_time": 0.01234,
        "results": [
            {"operation": "test_op1", "time": 0.001},
            {"operation": "test_op2", "time": 0.002}
        ]
    }
    
    # Save using the centralized path function
    report_path = performance_file("demo_performance_report")
    with open(report_path, 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"📊 Demo performance report saved to: {report_path}")

if __name__ == "__main__":
    demo_logging_organization()
    example_performance_report()
    
    print("\n" + "=" * 60)
    print("🎉 Logging configuration ready for this and all future projects!")
    print("💡 Use 'from logging_config import log_file, performance_file' in any script")
    print("🔄 All log files will automatically go to the logs/ directory")
