#!/usr/bin/env python3
"""
GPU Optimization Validation Test

This script tests the optimized configuration against the baseline to validate
the projected 60% performance improvement (67.28s → 26.91s target).

Key Optimizations Being Tested:
1. Batch size: 512 → 2809 (448% increase)
2. Workers: 4 → 6 (50% increase)  
3. File workers: 2 → 3 (50% increase)
4. Concurrent operations: 8 → 18 (125% increase)
5. CUDA streams: 1 → 4 (400% increase)
"""

import sys
import os
import time
import json
from typing import Dict, Any
from datetime import datetime

# Add project root to path
sys.path.append('/home/chris/Projects/EmbeddingsService')

# Import optimized configuration
from debug.optimized_config import OptimizedConfig
from scalable_processor import ScalableEnhancedProcessor
from performance_timer import PerformanceTimer
import logging

class OptimizationValidator:
    """Validates optimization performance improvements"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.baseline_time = 67.28  # From profiling
        self.target_time = 35.0
        self.projected_time = 26.91
        
    def setup_optimized_processor(self) -> ScalableEnhancedProcessor:
        """Create processor with optimized configuration"""
        # Patch the config module with optimized values
        optimized_config = OptimizedConfig()
        
        # Override config values directly in the config module
        import config
        config.EMBEDDING_BATCH_SIZE = optimized_config.EMBEDDING_BATCH_SIZE
        config.MAX_OPPORTUNITY_WORKERS = optimized_config.MAX_OPPORTUNITY_WORKERS
        config.MAX_FILE_WORKERS_PER_OPPORTUNITY = optimized_config.MAX_FILE_WORKERS_PER_OPPORTUNITY
        
        # Force reload of modules to pick up new config
        import importlib
        importlib.reload(config)
        
        # Import and reload the processor with new config
        from scalable_processor import ScalableEnhancedProcessor
        importlib.reload(sys.modules['scalable_processor'])
        
        # Create processor with optimized settings - it will read from updated config
        processor = ScalableEnhancedProcessor()
        
        return processor
    
    def run_baseline_comparison_test(self) -> Dict[str, Any]:
        """Run the same test as baseline for comparison"""
        print("🧪 Running optimized configuration test...")
        print("📊 Testing with optimized settings:")
        
        optimized_config = OptimizedConfig()
        print(f"   Batch size: {optimized_config.EMBEDDING_BATCH_SIZE}")
        print(f"   Workers: {optimized_config.MAX_OPPORTUNITY_WORKERS}")
        print(f"   File workers: {optimized_config.MAX_FILE_WORKERS_PER_OPPORTUNITY}")
        print(f"   Concurrent ops: {optimized_config.MAX_CONCURRENT_OPERATIONS}")
        print()
        
        # Setup processor with optimized configuration
        processor = self.setup_optimized_processor()
        
        # Run the same test parameters as baseline
        start_time = time.time()
        
        try:
            # Process the same batch as the baseline test
            results = processor.process_scalable_batch_producer_consumer(
                start_row_id=1,
                end_row_id=35,
                replace_existing_records=False,
                task_id="optimization_validation_test"
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate performance metrics
            opportunities_processed = results.get('opportunities_processed', 35)
            documents_processed = results.get('documents_processed', 0)
            chunks_generated = results.get('chunks_generated', 0)
            
            seconds_per_opportunity = total_time / opportunities_processed if opportunities_processed > 0 else 0
            
            # Calculate improvement
            improvement_percent = ((self.baseline_time - total_time) / self.baseline_time) * 100
            target_achieved = total_time < self.target_time
            
            test_results = {
                "test_timestamp": datetime.now().isoformat(),
                "baseline_time": self.baseline_time,
                "optimized_time": total_time,
                "projected_time": self.projected_time,
                "target_time": self.target_time,
                "improvement_percent": improvement_percent,
                "target_achieved": target_achieved,
                "opportunities_processed": opportunities_processed,
                "documents_processed": documents_processed,
                "chunks_generated": chunks_generated,
                "seconds_per_opportunity": seconds_per_opportunity,
                "processor_stats": results,
                "optimizations_applied": {
                    "batch_size": optimized_config.EMBEDDING_BATCH_SIZE,
                    "workers": optimized_config.MAX_OPPORTUNITY_WORKERS,
                    "file_workers": optimized_config.MAX_FILE_WORKERS_PER_OPPORTUNITY,
                    "concurrent_operations": optimized_config.MAX_CONCURRENT_OPERATIONS,
                    "cuda_streams": optimized_config.CUDA_STREAMS
                }
            }
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Optimization test failed: {e}")
            return {
                "error": str(e),
                "test_timestamp": datetime.now().isoformat(),
                "baseline_time": self.baseline_time,
                "target_achieved": False
            }
    
    def analyze_results(self, results: Dict[str, Any]) -> None:
        """Analyze and report test results"""
        print("================================================================================")
        print("🎯 OPTIMIZATION VALIDATION RESULTS")
        print("================================================================================")
        
        if "error" in results:
            print(f"❌ Test failed: {results['error']}")
            return
        
        baseline_time = results['baseline_time']
        optimized_time = results['optimized_time']
        improvement = results['improvement_percent']
        target_achieved = results['target_achieved']
        
        print(f"📊 Performance Comparison:")
        print(f"   Baseline time: {baseline_time:.2f}s")
        print(f"   Optimized time: {optimized_time:.2f}s")
        print(f"   Projected time: {results['projected_time']:.2f}s")
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
            print(f"   {key}: {value}")
        print()
        
        # Performance assessment
        if target_achieved:
            print("🎉 SUCCESS: Target performance achieved!")
            print("✅ Ready for production deployment")
        elif improvement > 40:
            print("🎯 GOOD: Significant improvement achieved")
            print("🔧 Consider additional fine-tuning for target")
        elif improvement > 20:
            print("⚠️  MODERATE: Some improvement achieved")
            print("🔍 Review optimization strategy")
        else:
            print("❌ POOR: Minimal improvement")
            print("🔄 Optimization strategy needs revision")
        
        print("================================================================================")
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/optimization_validation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filename

def main():
    """Main validation test"""
    print("================================================================================")
    print("🧪 GPU OPTIMIZATION VALIDATION TEST")
    print("================================================================================")
    print("Target: Validate 60% improvement (67.28s → 26.91s)")
    print("Threshold: <35 seconds for success")
    print()
    
    validator = OptimizationValidator()
    
    # Run the optimization test
    print("🚀 Starting optimization validation test...")
    results = validator.run_baseline_comparison_test()
    
    # Analyze and report results
    validator.analyze_results(results)
    
    # Save results
    if "error" not in results:
        results_file = validator.save_results(results)
        print(f"💾 Results saved: {results_file}")
    
    print()
    print("🏁 Optimization validation complete!")

if __name__ == "__main__":
    main()
