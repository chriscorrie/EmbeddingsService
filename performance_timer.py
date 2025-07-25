#!/usr/bin/env python3
"""
Performance Debug Timer

Thread-safe timing system for micro-benchmarking all processing operations
to identify the exact performance bottlenecks.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

@dataclass
class TimingEntry:
    """Single timing measurement"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    thread_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds"""
        return self.duration * 1000

class PerformanceTimer:
    """
    Thread-safe performance timing system with detailed analytics
    """
    
    def __init__(self, log_file: str = "logs/performance_debug.log"):
        # Ensure logs directory exists
        import os
        os.makedirs("logs", exist_ok=True)
        self.log_file = log_file
        self.timings: List[TimingEntry] = []
        self.active_timers: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.logger = self._setup_logger()
        
        # Operation categories for analysis
        self.categories = {
            'file_io': ['file_read', 'text_extract', 'file_access'],
            'processing': ['text_chunk', 'embedding_generate', 'entity_extract'],
            'database': ['vector_store', 'entity_store', 'sql_query', 'milvus_insert'],
            'comparison': ['boilerplate_filter', 'similarity_calc', 'embedding_compare'],
            'overhead': ['thread_create', 'object_serialize', 'memory_alloc']
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated performance logger"""
        logger = logging.getLogger('performance_timer')
        logger.setLevel(logging.DEBUG)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important timings
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def start_timer(self, operation: str, metadata: Optional[Dict] = None) -> str:
        """
        Start timing an operation
        
        Args:
            operation: Name of the operation being timed
            metadata: Additional context information
            
        Returns:
            timer_id: Unique identifier for this timer
        """
        thread_id = threading.current_thread().name
        timer_id = f"{operation}_{thread_id}_{time.time()}"
        
        with self.lock:
            self.active_timers[timer_id] = time.perf_counter()
            
        self.logger.debug(f"⏱️  START: {operation} | Thread: {thread_id} | Metadata: {metadata}")
        return timer_id
    
    def end_timer(self, timer_id: str, operation: str, metadata: Optional[Dict] = None) -> float:
        """
        End timing an operation and record the result
        
        Args:
            timer_id: Timer identifier from start_timer
            operation: Name of the operation (for validation)
            metadata: Additional context information
            
        Returns:
            duration: Time elapsed in seconds
        """
        end_time = time.perf_counter()
        thread_id = threading.current_thread().name
        
        with self.lock:
            if timer_id not in self.active_timers:
                self.logger.error(f"❌ Timer not found: {timer_id}")
                return 0.0
                
            start_time = self.active_timers.pop(timer_id)
            duration = end_time - start_time
            
            # Create timing entry
            entry = TimingEntry(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                thread_id=thread_id,
                metadata=metadata or {}
            )
            
            self.timings.append(entry)
        
        # Log based on duration (highlight slow operations)
        if duration > 1.0:
            self.logger.warning(f"🐌 SLOW: {operation} took {duration:.3f}s ({duration*1000:.1f}ms) | Thread: {thread_id}")
        elif duration > 0.1:
            self.logger.info(f"⏱️  {operation}: {duration:.3f}s ({duration*1000:.1f}ms) | Thread: {thread_id}")
        else:
            self.logger.debug(f"⚡ FAST: {operation}: {duration:.3f}s ({duration*1000:.1f}ms) | Thread: {thread_id}")
            
        return duration
    
    def time_operation(self, operation: str, metadata: Optional[Dict] = None):
        """
        Context manager for timing operations
        
        Usage:
            with timer.time_operation('embedding_generate', {'batch_size': 32}):
                embeddings = model.encode(texts)
        """
        return TimingContext(self, operation, metadata)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive timing summary"""
        with self.lock:
            if not self.timings:
                return {"error": "No timing data available"}
            
            # Basic statistics
            total_operations = len(self.timings)
            total_time = sum(t.duration for t in self.timings)
            avg_time = total_time / total_operations if total_operations > 0 else 0
            
            # Group by operation
            by_operation = defaultdict(list)
            for timing in self.timings:
                by_operation[timing.operation].append(timing.duration)
            
            # Calculate stats per operation
            operation_stats = {}
            for op, durations in by_operation.items():
                operation_stats[op] = {
                    'count': len(durations),
                    'total_time': sum(durations),
                    'avg_time': sum(durations) / len(durations),
                    'min_time': min(durations),
                    'max_time': max(durations),
                    'total_time_ms': sum(durations) * 1000,
                    'avg_time_ms': (sum(durations) / len(durations)) * 1000
                }
            
            # Group by category
            category_stats = {}
            for category, operations in self.categories.items():
                category_timings = [t for t in self.timings if t.operation in operations]
                if category_timings:
                    total_cat_time = sum(t.duration for t in category_timings)
                    category_stats[category] = {
                        'count': len(category_timings),
                        'total_time': total_cat_time,
                        'percentage': (total_cat_time / total_time) * 100 if total_time > 0 else 0,
                        'operations': list(set(t.operation for t in category_timings))
                    }
            
            # Find bottlenecks (operations taking >10% of total time)
            bottlenecks = []
            for op, stats in operation_stats.items():
                if stats['total_time'] / total_time > 0.1:  # More than 10% of total time
                    bottlenecks.append({
                        'operation': op,
                        'total_time': stats['total_time'],
                        'percentage': (stats['total_time'] / total_time) * 100,
                        'avg_time_ms': stats['avg_time_ms']
                    })
            
            bottlenecks.sort(key=lambda x: x['total_time'], reverse=True)
            
            return {
                'summary': {
                    'total_operations': total_operations,
                    'total_time': total_time,
                    'avg_time': avg_time,
                    'total_time_ms': total_time * 1000,
                    'avg_time_ms': avg_time * 1000
                },
                'by_operation': operation_stats,
                'by_category': category_stats,
                'bottlenecks': bottlenecks,
                'threads_used': list(set(t.thread_id for t in self.timings))
            }
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        summary = self.get_summary()
        
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
        
        print("\n" + "="*80)
        print("🔍 PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall stats
        stats = summary['summary']
        print(f"📊 Overall Statistics:")
        print(f"   Total Operations: {stats['total_operations']}")
        print(f"   Total Time: {stats['total_time']:.3f}s ({stats['total_time_ms']:.1f}ms)")
        print(f"   Average Time per Operation: {stats['avg_time_ms']:.1f}ms")
        print(f"   Threads Used: {len(summary['threads_used'])}")
        
        # Bottlenecks
        if summary['bottlenecks']:
            print(f"\n🐌 PERFORMANCE BOTTLENECKS (>10% of total time):")
            for i, bottleneck in enumerate(summary['bottlenecks'][:5], 1):
                print(f"   {i}. {bottleneck['operation']}: {bottleneck['total_time']:.3f}s "
                      f"({bottleneck['percentage']:.1f}%) - Avg: {bottleneck['avg_time_ms']:.1f}ms")
        
        # Category breakdown
        print(f"\n📂 BY CATEGORY:")
        for category, stats in summary['by_category'].items():
            print(f"   {category.upper()}: {stats['total_time']:.3f}s "
                  f"({stats['percentage']:.1f}%) - {stats['count']} operations")
        
        # Detailed operation stats
        print(f"\n⏱️  DETAILED OPERATION TIMINGS:")
        sorted_ops = sorted(summary['by_operation'].items(), 
                           key=lambda x: x[1]['total_time'], reverse=True)
        
        for op, stats in sorted_ops[:10]:  # Top 10 slowest operations
            print(f"   {op}: {stats['avg_time_ms']:.1f}ms avg "
                  f"(Total: {stats['total_time_ms']:.1f}ms, Count: {stats['count']})")
    
    def save_detailed_report(self, filename: str = "logs/performance_report.json"):
        """Save detailed performance report to JSON"""
        # Ensure logs directory exists
        import os
        os.makedirs("logs", exist_ok=True)
        
        summary = self.get_summary()
        
        # Add raw timing data
        summary['raw_timings'] = [
            {
                'operation': t.operation,
                'duration': t.duration,
                'duration_ms': t.duration_ms,
                'thread_id': t.thread_id,
                'metadata': t.metadata
            }
            for t in self.timings
        ]
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Detailed performance report saved to: {filename}")
    
    def reset(self):
        """Reset all timing data"""
        with self.lock:
            self.timings.clear()
            self.active_timers.clear()
        self.logger.info("🔄 Performance timer reset")

class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, timer: PerformanceTimer, operation: str, metadata: Optional[Dict] = None):
        self.timer = timer
        self.operation = operation
        self.metadata = metadata
        self.timer_id = None
    
    def __enter__(self):
        self.timer_id = self.timer.start_timer(self.operation, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_id:
            self.timer.end_timer(self.timer_id, self.operation, self.metadata)

# Global timer instance
performance_timer = PerformanceTimer()

# Convenience functions
def time_operation(operation: str, metadata: Optional[Dict] = None):
    """Decorator/context manager for timing operations"""
    return performance_timer.time_operation(operation, metadata)

def start_timer(operation: str, metadata: Optional[Dict] = None) -> str:
    """Start a timer for an operation"""
    return performance_timer.start_timer(operation, metadata)

def end_timer(timer_id: str, operation: str, metadata: Optional[Dict] = None) -> float:
    """End a timer for an operation"""
    return performance_timer.end_timer(timer_id, operation, metadata)

def get_summary() -> Dict[str, Any]:
    """Get performance summary"""
    return performance_timer.get_summary()

def print_summary():
    """Print performance summary"""
    performance_timer.print_summary()

def save_report(filename: str = "logs/performance_report.json"):
    """Save detailed performance report"""
    # Ensure logs directory exists
    import os
    os.makedirs("logs", exist_ok=True)
    performance_timer.save_detailed_report(filename)

def reset_timer():
    """Reset all timing data"""
    performance_timer.reset()

if __name__ == "__main__":
    # Test the timer
    import random
    
    print("🧪 Testing Performance Timer...")
    
    # Simulate some operations
    for i in range(5):
        with time_operation('test_operation', {'iteration': i}):
            time.sleep(random.uniform(0.01, 0.1))
        
        with time_operation('slow_operation', {'iteration': i}):
            time.sleep(random.uniform(0.1, 0.3))
    
    # Print results
    print_summary()
    save_report("test_performance_report.json")
