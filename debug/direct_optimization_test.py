#!/usr/bin/env python3
"""
Direct GPU Optimization Test

This script directly applies optimizations to the existing processor
and tests performance improvements in real-time.

Based on profiling showing:
- GPU utilization: 1.9% average (massive underutilization)
- Memory utilization: 8.2% of 15.5GB (significant underutilization)
- Target: <35 seconds (currently 67.28s) - 48% improvement needed
"""

import sys
import os
import time
import json
from typing import Dict, Any
from datetime import datetime

# Add project root to path
sys.path.append('/home/chris/Projects/EmbeddingsService')

from scalable_processor import ScalableEnhancedProcessor
import config
import logging

class DirectOptimizationTest:
    """Direct optimization test with runtime config modification"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.baseline_time = 67.28
        self.target_time = 35.0
        
    def apply_runtime_optimizations(self) -> Dict[str, Any]:
        """Apply optimizations directly to config at runtime"""
        
        # Store original values
        original_values = {
            'EMBEDDING_BATCH_SIZE': getattr(config, 'EMBEDDING_BATCH_SIZE', 512),
            'MAX_OPPORTUNITY_WORKERS': getattr(config, 'MAX_OPPORTUNITY_WORKERS', 4),
            'MAX_FILE_WORKERS_PER_OPPORTUNITY': getattr(config, 'MAX_FILE_WORKERS_PER_OPPORTUNITY', 2),
        }
        
        # Apply optimizations directly to config module
        config.EMBEDDING_BATCH_SIZE = 2048  # Increase from 512 to 2048 (300% increase)
        config.MAX_OPPORTUNITY_WORKERS = 6   # Increase from 4 to 6 (50% increase)
        config.MAX_FILE_WORKERS_PER_OPPORTUNITY = 3  # Increase from 2 to 3 (50% increase)
        
        # Set GPU optimizations
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
        os.environ['TORCH_CUDNN_BENCHMARK'] = '1'  # Optimize cuDNN
        
        optimizations_applied = {
            'embedding_batch_size': {'from': original_values['EMBEDDING_BATCH_SIZE'], 'to': config.EMBEDDING_BATCH_SIZE},
            'opportunity_workers': {'from': original_values['MAX_OPPORTUNITY_WORKERS'], 'to': config.MAX_OPPORTUNITY_WORKERS},
            'file_workers': {'from': original_values['MAX_FILE_WORKERS_PER_OPPORTUNITY'], 'to': config.MAX_FILE_WORKERS_PER_OPPORTUNITY},
            'cuda_optimizations': 'enabled'
        }
        
        return original_values, optimizations_applied
    
    def run_optimized_test(self) -> Dict[str, Any]:
        """Run the optimized performance test"""
        print("🚀 Applying runtime optimizations...")
        
        # Apply optimizations
        original_values, optimizations = self.apply_runtime_optimizations()
        
        print("⚡ Optimizations applied:")
        for key, value in optimizations.items():
            if isinstance(value, dict):
                print(f"   {key}: {value['from']} → {value['to']}")
            else:
                print(f"   {key}: {value}")
        print()
        
        try:
            # Create processor with optimized config
            print("🔧 Initializing optimized processor...")
            processor = ScalableEnhancedProcessor()
            
            print("🧪 Starting optimized performance test...")
            print("   Test: 35 opportunities (same as baseline)")
            print("   Target: <35 seconds")
            print()
            
            # Run the same test as baseline
            start_time = time.time()
            
            results = processor.process_scalable_batch_producer_consumer(
                start_row_id=1,
                end_row_id=35,
                replace_existing_records=False,
                task_id="direct_optimization_test"
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate performance metrics
            opportunities_processed = results.get('opportunities_processed', 35)
            documents_processed = results.get('documents_processed', 0)
            chunks_generated = results.get('total_chunks_generated', 0)
            
            seconds_per_opportunity = total_time / opportunities_processed if opportunities_processed > 0 else 0
            improvement_percent = ((self.baseline_time - total_time) / self.baseline_time) * 100
            target_achieved = total_time < self.target_time
            
            test_results = {
                "test_timestamp": datetime.now().isoformat(),
                "baseline_time": self.baseline_time,
                "optimized_time": total_time,
                "target_time": self.target_time,
                "improvement_percent": improvement_percent,
                "target_achieved": target_achieved,
                "opportunities_processed": opportunities_processed,
                "documents_processed": documents_processed,
                "chunks_generated": chunks_generated,
                "seconds_per_opportunity": seconds_per_opportunity,
                "optimizations_applied": optimizations,
                "processor_stats": results
            }
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Optimization test failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": str(e),
                "test_timestamp": datetime.now().isoformat(),
                "baseline_time": self.baseline_time,
                "target_achieved": False,
                "optimizations_applied": optimizations
            }
        
        finally:
            # Restore original config values
            for key, value in original_values.items():
                setattr(config, key, value)
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Analyze and report test results"""
        print("================================================================================")
        print("🎯 DIRECT OPTIMIZATION TEST RESULTS")
        print("================================================================================")
        
        if "error" in results:
            print(f"❌ Test failed: {results['error']}")
            print("\n🔧 Optimizations that were applied:")
            for key, value in results.get('optimizations_applied', {}).items():
                if isinstance(value, dict):
                    print(f"   {key}: {value['from']} → {value['to']}")
                else:
                    print(f"   {key}: {value}")
            return
        
        baseline_time = results['baseline_time']
        optimized_time = results['optimized_time']
        improvement = results['improvement_percent']
        target_achieved = results['target_achieved']
        
        print(f"📊 Performance Comparison:")
        print(f"   Baseline time: {baseline_time:.2f}s")
        print(f"   Optimized time: {optimized_time:.2f}s")
        print(f"   Target time: {results['target_time']:.2f}s")
        print()
        
        print(f"🚀 Performance Results:")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Target achieved: {'✅ YES' if target_achieved else '❌ NO'}")
        print(f"   Seconds per opportunity: {results['seconds_per_opportunity']:.3f}s")
        print()
        
        print(f"📈 Processing Statistics:")
        print(f"   Opportunities: {results['opportunities_processed']}")
        print(f"   Documents: {results['documents_processed']}")
        print(f"   Chunks: {results['chunks_generated']}")
        print()
        
        print("⚙️  Optimizations Applied:")
        for key, value in results['optimizations_applied'].items():
            if isinstance(value, dict):
                print(f"   {key}: {value['from']} → {value['to']}")
            else:
                print(f"   {key}: {value}")
        print()
        
        # Performance assessment with specific guidance
        if target_achieved:
            time_savings = baseline_time - optimized_time
            print(f"🎉 SUCCESS: Target achieved! Saved {time_savings:.1f} seconds")
            print("✅ Ready for production deployment")
            print("💡 Consider implementing these optimizations permanently")
        elif improvement > 40:
            remaining_improvement = ((optimized_time - results['target_time']) / optimized_time) * 100
            print(f"🎯 EXCELLENT: {improvement:.1f}% improvement achieved!")
            print(f"🔧 Need {remaining_improvement:.1f}% more improvement for target")
            print("💡 Try increasing batch size further or adding CUDA streams")
        elif improvement > 20:
            print(f"⚠️  MODERATE: {improvement:.1f}% improvement achieved")
            print("🔍 Consider more aggressive optimizations:")
            print("   - Larger embedding batch sizes (try 4096+)")
            print("   - More concurrent workers")
            print("   - CUDA memory optimization")
        else:
            print(f"❌ MINIMAL: Only {improvement:.1f}% improvement")
            print("🔄 Current optimizations insufficient - need different approach:")
            print("   - Check GPU utilization during processing")
            print("   - Investigate I/O bottlenecks")
            print("   - Consider algorithmic optimizations")
        
        print("================================================================================")
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/direct_optimization_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filename

def main():
    """Main test execution"""
    print("================================================================================")
    print("🧪 DIRECT GPU OPTIMIZATION TEST")
    print("================================================================================")
    print("Approach: Runtime configuration optimization")
    print("Target: <35 seconds for 35 opportunities")
    print("Baseline: 67.28 seconds (1.922s per opportunity)")
    print("Improvement needed: 48% reduction")
    print()
    
    tester = DirectOptimizationTest()
    
    # Run the optimization test
    results = tester.run_optimized_test()
    
    # Analyze and report results
    tester.analyze_results(results)
    
    # Save results
    if "error" not in results:
        results_file = tester.save_results(results)
        print(f"💾 Results saved: {results_file}")
    
    print()
    print("🏁 Direct optimization test complete!")

if __name__ == "__main__":
    main()
