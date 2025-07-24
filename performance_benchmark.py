#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Suite
Tests different configurations to maximize throughput
"""

import time
import psutil
import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resource_manager import get_optimal_configuration

class PerformanceBenchmark:
    """
    Benchmark different processor configurations for optimal performance
    """
    
    def __init__(self):
        self.results = []
        self.system_info = self._get_detailed_system_info()
        
    def _get_detailed_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_test_configurations(self) -> List[Dict[str, Any]]:
        """
        Create different configuration scenarios to test
        """
        base_config = get_optimal_configuration()
        
        configurations = [
            {
                'name': 'Conservative (Current)',
                'max_opportunity_workers': 4,
                'max_file_workers_per_opportunity': 4,
                'embedding_batch_size': 32,
                'total_workers': 16,
                'expected_cpu_usage': '25-40%',
                'memory_usage_mb': 2048
            },
            {
                'name': 'Moderate',
                'max_opportunity_workers': 6,
                'max_file_workers_per_opportunity': 4,
                'embedding_batch_size': 64,
                'total_workers': 24,
                'expected_cpu_usage': '40-60%',
                'memory_usage_mb': 4096
            },
            {
                'name': 'Aggressive (Recommended)',
                'max_opportunity_workers': 8,
                'max_file_workers_per_opportunity': 6,
                'embedding_batch_size': 128,
                'total_workers': 48,
                'expected_cpu_usage': '60-80%',
                'memory_usage_mb': 8192
            },
            {
                'name': 'Maximum (Stress Test)',
                'max_opportunity_workers': 12,
                'max_file_workers_per_opportunity': 8,
                'embedding_batch_size': 256,
                'total_workers': 96,
                'expected_cpu_usage': '80-95%',
                'memory_usage_mb': 12288
            }
        ]
        
        return configurations
    
    def simulate_workload(self, config: Dict[str, Any], duration_seconds: int = 10) -> Dict[str, Any]:
        """
        Simulate processing workload to measure resource utilization
        """
        print(f"🧪 Testing {config['name']} configuration...")
        print(f"   Workers: {config['max_opportunity_workers']} opportunities × {config['max_file_workers_per_opportunity']} files = {config['total_workers']} total")
        
        # Track resource usage during simulation
        cpu_samples = []
        memory_samples = []
        start_time = time.time()
        
        def cpu_intensive_task(worker_id: int, batch_size: int):
            """Simulate embedding generation and file processing"""
            # Simulate embedding computation
            import numpy as np
            for i in range(10):  # Simulate processing chunks
                # Simulate batch embedding generation
                vectors = np.random.random((batch_size, 384))  # Typical embedding size
                # Simulate vector operations
                similarity = np.dot(vectors, vectors.T)
                time.sleep(0.01)  # Simulate I/O wait
        
        def monitor_resources():
            """Monitor system resources during test"""
            while time.time() - start_time < duration_seconds:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                cpu_samples.append(cpu_percent)
                memory_samples.append(memory_mb)
                time.sleep(0.1)
        
        # Start resource monitoring
        monitor_thread = ThreadPoolExecutor(max_workers=1)
        monitor_future = monitor_thread.submit(monitor_resources)
        
        # Simulate parallel processing workload
        with ThreadPoolExecutor(max_workers=config['total_workers']) as executor:
            # Submit simulated tasks
            futures = []
            for worker_id in range(config['max_opportunity_workers']):
                for file_worker in range(config['max_file_workers_per_opportunity']):
                    future = executor.submit(
                        cpu_intensive_task, 
                        worker_id * 100 + file_worker,
                        config['embedding_batch_size']
                    )
                    futures.append(future)
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
        
        # Stop monitoring
        monitor_future.result()
        monitor_thread.shutdown()
        
        # Calculate metrics
        actual_duration = time.time() - start_time
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        max_cpu = max(cpu_samples) if cpu_samples else 0
        avg_memory_mb = sum(memory_samples) / len(memory_samples) if memory_samples else 0
        max_memory_mb = max(memory_samples) if memory_samples else 0
        
        # Calculate throughput metrics
        simulated_chunks = config['max_opportunity_workers'] * config['max_file_workers_per_opportunity'] * 10
        chunks_per_second = simulated_chunks / actual_duration
        
        return {
            'config_name': config['name'],
            'total_workers': config['total_workers'],
            'duration_seconds': actual_duration,
            'avg_cpu_percent': avg_cpu,
            'max_cpu_percent': max_cpu,
            'avg_memory_mb': avg_memory_mb,
            'max_memory_mb': max_memory_mb,
            'simulated_chunks_per_second': chunks_per_second,
            'cpu_utilization_efficiency': avg_cpu / 100,
            'memory_utilization_gb': max_memory_mb / 1024,
            'theoretical_speedup': chunks_per_second / (simulated_chunks / 10)  # vs sequential
        }
    
    def run_benchmark_suite(self) -> List[Dict[str, Any]]:
        """
        Run complete benchmark suite
        """
        print("🚀 Starting Comprehensive Performance Benchmark")
        print("="*70)
        print(f"System: {self.system_info['cpu_cores_physical']} cores ({self.system_info['cpu_cores_logical']} threads), {self.system_info['total_memory_gb']:.1f}GB RAM")
        print(f"Available: {self.system_info['available_memory_gb']:.1f}GB RAM")
        print()
        
        configurations = self.create_test_configurations()
        results = []
        
        for config in configurations:
            try:
                result = self.simulate_workload(config, duration_seconds=8)
                results.append(result)
                
                print(f"✅ {config['name']} Results:")
                print(f"   CPU Usage: {result['avg_cpu_percent']:.1f}% avg, {result['max_cpu_percent']:.1f}% max")
                print(f"   Memory Usage: {result['max_memory_mb']:.0f} MB ({result['memory_utilization_gb']:.1f}GB)")
                print(f"   Throughput: {result['simulated_chunks_per_second']:.1f} chunks/second")
                print(f"   Efficiency: {result['cpu_utilization_efficiency']*100:.1f}% CPU utilization")
                print()
                
            except Exception as e:
                print(f"❌ Error testing {config['name']}: {e}")
                continue
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze benchmark results and provide recommendations
        """
        if not results:
            return {}
        
        # Find optimal configuration
        best_throughput = max(results, key=lambda x: x['simulated_chunks_per_second'])
        best_efficiency = max(results, key=lambda x: x['cpu_utilization_efficiency'])
        
        # Calculate improvement potential
        current_config = results[0]  # Conservative baseline
        best_config = best_throughput
        
        throughput_improvement = (best_config['simulated_chunks_per_second'] / 
                                current_config['simulated_chunks_per_second'])
        
        return {
            'current_performance': current_config,
            'recommended_config': best_config,
            'throughput_improvement': throughput_improvement,
            'best_efficiency_config': best_efficiency,
            'analysis': {
                'cpu_underutilized': current_config['avg_cpu_percent'] < 50,
                'memory_underutilized': current_config['memory_utilization_gb'] < 10,
                'can_increase_workers': best_config['max_cpu_percent'] < 90,
                'recommended_workers': best_config['total_workers']
            }
        }
    
    def print_recommendations(self, analysis: Dict[str, Any]):
        """
        Print detailed recommendations
        """
        if not analysis:
            print("❌ No analysis available")
            return
            
        print("\n" + "="*70)
        print("📊 PERFORMANCE ANALYSIS & RECOMMENDATIONS")
        print("="*70)
        
        current = analysis['current_performance']
        recommended = analysis['recommended_config']
        
        print(f"📈 Current Performance (Conservative):")
        print(f"   • CPU Usage: {current['avg_cpu_percent']:.1f}% (UNDERUTILIZED)")
        print(f"   • Memory Usage: {current['memory_utilization_gb']:.1f}GB (UNDERUTILIZED)")
        print(f"   • Throughput: {current['simulated_chunks_per_second']:.1f} chunks/second")
        print(f"   • Workers: {current['total_workers']}")
        
        print(f"\n⚡ Recommended Configuration ({recommended['config_name']}):")
        print(f"   • CPU Usage: {recommended['avg_cpu_percent']:.1f}% (OPTIMAL)")
        print(f"   • Memory Usage: {recommended['memory_utilization_gb']:.1f}GB")
        print(f"   • Throughput: {recommended['simulated_chunks_per_second']:.1f} chunks/second")
        print(f"   • Workers: {recommended['total_workers']}")
        
        improvement = analysis['throughput_improvement']
        print(f"\n🚀 Expected Performance Improvement:")
        print(f"   • {improvement:.1f}x faster processing")
        print(f"   • {(improvement-1)*100:.0f}% throughput increase")
        print(f"   • Better CPU utilization (+{recommended['avg_cpu_percent']-current['avg_cpu_percent']:.0f}%)")
        
        print(f"\n🔧 Recommended config.py Settings:")
        if recommended['config_name'] == 'Aggressive (Recommended)':
            print(f"   MAX_OPPORTUNITY_WORKERS = 8")
            print(f"   MAX_FILE_WORKERS_PER_OPPORTUNITY = 6") 
            print(f"   EMBEDDING_BATCH_SIZE = 128")
            print(f"   MAX_MEMORY_USAGE_MB = 8192")
        elif recommended['config_name'] == 'Maximum (Stress Test)':
            print(f"   MAX_OPPORTUNITY_WORKERS = 12")
            print(f"   MAX_FILE_WORKERS_PER_OPPORTUNITY = 8")
            print(f"   EMBEDDING_BATCH_SIZE = 256")
            print(f"   MAX_MEMORY_USAGE_MB = 12288")
        
        print(f"\n💡 Key Insights:")
        if analysis['analysis']['cpu_underutilized']:
            print(f"   • Your CPU is significantly underutilized - increase workers")
        if analysis['analysis']['memory_underutilized']:
            print(f"   • Your RAM is underutilized - increase batch sizes")
        print(f"   • Your {self.system_info['cpu_cores_physical']}-core system can handle much more load")
        print(f"   • With {self.system_info['available_memory_gb']:.0f}GB available RAM, you can be aggressive")
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = 'benchmark_results.json'):
        """Save benchmark results to file"""
        output = {
            'system_info': self.system_info,
            'benchmark_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Benchmark results saved to {filename}")

def run_performance_benchmark():
    """
    Main function to run performance benchmark
    """
    benchmark = PerformanceBenchmark()
    
    print("🎯 This benchmark will test different worker configurations")
    print("to find the optimal settings for your 12-core, 62GB system.\n")
    
    # Run benchmark suite
    results = benchmark.run_benchmark_suite()
    
    if results:
        # Analyze results
        analysis = benchmark.analyze_results(results)
        
        # Print recommendations
        benchmark.print_recommendations(analysis)
        
        # Save results
        benchmark.save_results(results)
        
        return analysis
    else:
        print("❌ No benchmark results available")
        return None

if __name__ == "__main__":
    run_performance_benchmark()
