#!/usr/bin/env python3
"""
Performance comparison test between current and optimized processors
"""

import time
import psutil
import os
import sys
from typing import Dict, Any
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def measure_performance(processor_class, start_row: int, end_row: int, name: str) -> Dict[str, Any]:
    """
    Measure performance metrics for a processor
    """
    print(f"\n🧪 Testing {name} processor...")
    
    # Get initial system metrics
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    initial_cpu_percent = process.cpu_percent()
    
    # Start timing
    start_time = time.time()
    
    try:
        # Initialize processor
        processor = processor_class()
        
        # Process the batch
        processor.process_batch(start_row, end_row, replace_existing_records=True)
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Get final system metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Get processor statistics
        stats = getattr(processor, 'stats', {})
        
        return {
            'name': name,
            'processing_time': processing_time,
            'memory_used_mb': memory_used,
            'peak_memory_mb': final_memory,
            'stats': stats,
            'success': True
        }
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'name': name,
            'processing_time': processing_time,
            'memory_used_mb': 0,
            'peak_memory_mb': 0,
            'stats': {},
            'success': False,
            'error': str(e)
        }

def print_performance_comparison(results: list):
    """
    Print detailed performance comparison
    """
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    
    for result in results:
        print(f"\n🔧 {result['name']} Processor:")
        print(f"   ⏱️  Processing Time: {result['processing_time']:.2f} seconds")
        print(f"   💾 Memory Used: {result['memory_used_mb']:.1f} MB")
        print(f"   📈 Peak Memory: {result['peak_memory_mb']:.1f} MB")
        print(f"   ✅ Success: {result['success']}")
        
        if not result['success']:
            print(f"   ❌ Error: {result['error']}")
        
        if result['stats']:
            stats = result['stats']
            print(f"   📋 Statistics:")
            print(f"      • Opportunities: {stats.get('opportunities_processed', 0)}")
            print(f"      • Titles: {stats.get('titles_embedded', 0)}")
            print(f"      • Descriptions: {stats.get('descriptions_embedded', 0)}")
            print(f"      • Documents: {stats.get('documents_embedded', 0)}")
            print(f"      • Total Chunks: {stats.get('total_chunks_generated', 0)}")
            print(f"      • Entities: {stats.get('entities_extracted', 0)}")
            print(f"      • Errors: {stats.get('errors', 0)}")
    
    # Performance comparison
    if len(results) >= 2 and results[0]['success'] and results[1]['success']:
        current_time = results[0]['processing_time']
        optimized_time = results[1]['processing_time']
        
        if optimized_time > 0:
            speedup = current_time / optimized_time
            improvement_percent = ((current_time - optimized_time) / current_time) * 100
            
            print(f"\n🚀 PERFORMANCE IMPROVEMENT:")
            print(f"   ⚡ Speedup: {speedup:.2f}x faster")
            print(f"   📈 Improvement: {improvement_percent:.1f}% faster")
            
            current_memory = results[0]['memory_used_mb']
            optimized_memory = results[1]['memory_used_mb']
            memory_change = optimized_memory - current_memory
            
            print(f"   💾 Memory Change: {memory_change:+.1f} MB")

def run_performance_test():
    """
    Run comprehensive performance test
    """
    print("🎯 Starting Performance Comparison Test")
    print("Testing rows 1-5 for both processors...")
    
    # Test parameters
    start_row = 1
    end_row = 5
    
    results = []
    
    try:
        # Test current processor
        from enhanced_chunked_processor import EnhancedChunkedProcessor
        current_result = measure_performance(
            EnhancedChunkedProcessor, 
            start_row, 
            end_row, 
            "Current (Sequential)"
        )
        results.append(current_result)
        
    except ImportError as e:
        print(f"❌ Could not import current processor: {e}")
        results.append({
            'name': 'Current (Sequential)',
            'processing_time': 0,
            'memory_used_mb': 0,
            'peak_memory_mb': 0,
            'stats': {},
            'success': False,
            'error': f"Import error: {e}"
        })
    
    try:
        # Test optimized processor
        from high_performance_processor import HighPerformanceChunkedProcessor
        optimized_result = measure_performance(
            HighPerformanceChunkedProcessor,
            start_row,
            end_row,
            "Optimized (Parallel)"
        )
        results.append(optimized_result)
        
    except ImportError as e:
        print(f"❌ Could not import optimized processor: {e}")
        results.append({
            'name': 'Optimized (Parallel)',
            'processing_time': 0,
            'memory_used_mb': 0,
            'peak_memory_mb': 0,
            'stats': {},
            'success': False,
            'error': f"Import error: {e}"
        })
    
    # Print results
    print_performance_comparison(results)
    
    return results

if __name__ == "__main__":
    results = run_performance_test()
