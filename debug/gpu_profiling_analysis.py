#!/usr/bin/env python3
"""
GPU Profiling Analysis Script
Comprehensive analysis of GPU utilization patterns during document processing
to identify optimization opportunities for sub-1-second-per-opportunity performance.

Target: <35 seconds for 35 opportunities (<1 second per opportunity)
Current baseline: ~67.6 seconds (1.93 seconds per opportunity)
"""

import sys
import os
import time
import threading
import json
from datetime import datetime
import subprocess
import psutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scalable_processor import ScalableEnhancedProcessor

class GPUProfiler:
    """Comprehensive GPU profiling during document processing"""
    
    def __init__(self):
        self.gpu_stats = []
        self.system_stats = []
        self.processing_phases = []
        self.monitoring = False
        self.start_time = None
        
    def get_gpu_info(self):
        """Get current GPU utilization and memory usage"""
        try:
            # Get GPU stats using nvidia-smi
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpu_line = result.stdout.strip()
                utilization, memory_used, memory_total, temperature = gpu_line.split(', ')
                return {
                    'timestamp': time.time(),
                    'gpu_utilization_percent': int(utilization),
                    'memory_used_mb': int(memory_used),
                    'memory_total_mb': int(memory_total),
                    'memory_percent': round((int(memory_used) / int(memory_total)) * 100, 2),
                    'temperature_c': int(temperature)
                }
        except Exception as e:
            print(f"Warning: Could not get GPU stats: {e}")
            return None
    
    def get_system_info(self):
        """Get current system resource usage"""
        try:
            return {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': round(psutil.virtual_memory().used / (1024**3), 2),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
        except Exception as e:
            print(f"Warning: Could not get system stats: {e}")
            return None
    
    def monitor_resources(self):
        """Background monitoring of GPU and system resources"""
        while self.monitoring:
            gpu_info = self.get_gpu_info()
            system_info = self.get_system_info()
            
            if gpu_info:
                self.gpu_stats.append(gpu_info)
            if system_info:
                self.system_stats.append(system_info)
            
            time.sleep(0.5)  # Sample every 500ms for detailed profiling
    
    def log_processing_phase(self, phase_name, details=None):
        """Log a processing phase with timestamp"""
        phase_info = {
            'timestamp': time.time(),
            'elapsed_since_start': time.time() - self.start_time if self.start_time else 0,
            'phase': phase_name,
            'details': details or {}
        }
        self.processing_phases.append(phase_info)
        print(f"⏱️  {phase_info['elapsed_since_start']:.2f}s: {phase_name}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")
    
    def start_monitoring(self):
        """Start background resource monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        self.gpu_stats = []
        self.system_stats = []
        self.processing_phases = []
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        monitor_thread.start()
        print("🔍 GPU profiling started...")
    
    def stop_monitoring(self):
        """Stop monitoring and generate analysis report"""
        self.monitoring = False
        time.sleep(1)  # Allow final samples
        print("⏹️  GPU profiling stopped")
    
    def analyze_results(self):
        """Analyze collected profiling data"""
        if not self.gpu_stats:
            print("❌ No GPU data collected")
            return {}
        
        # GPU utilization analysis
        gpu_utilizations = [stat['gpu_utilization_percent'] for stat in self.gpu_stats]
        gpu_memory_percents = [stat['memory_percent'] for stat in self.gpu_stats]
        
        # System resource analysis
        cpu_utilizations = [stat['cpu_percent'] for stat in self.system_stats if stat]
        memory_percents = [stat['memory_percent'] for stat in self.system_stats if stat]
        
        analysis = {
            'profiling_duration': time.time() - self.start_time,
            'samples_collected': len(self.gpu_stats),
            'gpu_analysis': {
                'peak_utilization_percent': max(gpu_utilizations),
                'avg_utilization_percent': round(sum(gpu_utilizations) / len(gpu_utilizations), 2),
                'min_utilization_percent': min(gpu_utilizations),
                'peak_memory_mb': max([stat['memory_used_mb'] for stat in self.gpu_stats]),
                'avg_memory_mb': round(sum([stat['memory_used_mb'] for stat in self.gpu_stats]) / len(self.gpu_stats), 2),
                'peak_memory_percent': max(gpu_memory_percents),
                'avg_memory_percent': round(sum(gpu_memory_percents) / len(gpu_memory_percents), 2),
                'peak_temperature_c': max([stat['temperature_c'] for stat in self.gpu_stats]),
                'total_memory_gb': round(self.gpu_stats[0]['memory_total_mb'] / 1024, 2)
            },
            'system_analysis': {
                'peak_cpu_percent': max(cpu_utilizations) if cpu_utilizations else 0,
                'avg_cpu_percent': round(sum(cpu_utilizations) / len(cpu_utilizations), 2) if cpu_utilizations else 0,
                'peak_memory_percent': max(memory_percents) if memory_percents else 0,
                'avg_memory_percent': round(sum(memory_percents) / len(memory_percents), 2) if memory_percents else 0
            },
            'processing_phases': self.processing_phases,
            'optimization_opportunities': self.identify_optimization_opportunities()
        }
        
        return analysis
    
    def identify_optimization_opportunities(self):
        """Identify specific optimization opportunities from profiling data"""
        opportunities = []
        
        if not self.gpu_stats:
            return opportunities
        
        gpu_utilizations = [stat['gpu_utilization_percent'] for stat in self.gpu_stats]
        gpu_memory_percents = [stat['memory_percent'] for stat in self.gpu_stats]
        
        avg_gpu_util = sum(gpu_utilizations) / len(gpu_utilizations)
        max_gpu_util = max(gpu_utilizations)
        avg_memory_util = sum(gpu_memory_percents) / len(gpu_memory_percents)
        
        # Low GPU utilization opportunity
        if avg_gpu_util < 50:
            opportunities.append({
                'type': 'LOW_GPU_UTILIZATION',
                'severity': 'HIGH',
                'description': f'Average GPU utilization only {avg_gpu_util:.1f}% - significant underutilization',
                'recommendation': 'Increase batch sizes, improve pipeline efficiency, or implement concurrent processing'
            })
        
        # Memory underutilization opportunity
        if avg_memory_util < 20:
            opportunities.append({
                'type': 'LOW_MEMORY_UTILIZATION',
                'severity': 'MEDIUM',
                'description': f'Average GPU memory usage only {avg_memory_util:.1f}% - memory underutilized',
                'recommendation': 'Increase batch sizes or process multiple streams concurrently'
            })
        
        # Intermittent usage pattern
        if max_gpu_util > 80 and avg_gpu_util < 30:
            opportunities.append({
                'type': 'INTERMITTENT_USAGE',
                'severity': 'HIGH',
                'description': f'GPU usage is bursty (peak {max_gpu_util}%, avg {avg_gpu_util:.1f}%)',
                'recommendation': 'Implement better pipeline batching to maintain consistent GPU utilization'
            })
        
        return opportunities
    
    def save_detailed_report(self, filename=None):
        """Save comprehensive profiling report to logs directory"""
        if not filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"logs/gpu_profiling_report_{timestamp}.json"
        
        os.makedirs('logs', exist_ok=True)
        
        analysis = self.analyze_results()
        detailed_report = {
            'report_metadata': {
                'timestamp': datetime.now().isoformat(),
                'target_performance': '<35 seconds for 35 opportunities',
                'current_baseline': '~67.6 seconds',
                'improvement_needed': '48.2% reduction required'
            },
            'analysis': analysis,
            'raw_gpu_stats': self.gpu_stats,
            'raw_system_stats': self.system_stats
        }
        
        with open(filename, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        print(f"📊 Detailed profiling report saved: {filename}")
        return filename

def profile_standard_processing():
    """Profile the standard 35-opportunity processing test"""
    print("="*80)
    print("🎯 GPU PROFILING: STANDARD 35-OPPORTUNITY TEST")
    print("="*80)
    print("Target: <35 seconds total (<1 second per opportunity)")
    print("Current baseline: ~67.6 seconds")
    print()
    
    profiler = GPUProfiler()
    
    # Get initial GPU state
    initial_gpu = profiler.get_gpu_info()
    if initial_gpu:
        print(f"🖥️  Initial GPU State:")
        print(f"   Memory: {initial_gpu['memory_used_mb']}/{initial_gpu['memory_total_mb']} MB ({initial_gpu['memory_percent']}%)")
        print(f"   Temperature: {initial_gpu['temperature_c']}°C")
        print()
    
    # Initialize processor
    profiler.log_processing_phase("Initializing ScalableEnhancedProcessor")
    processor = ScalableEnhancedProcessor()
    profiler.log_processing_phase("Processor initialized")
    
    # Start profiling
    profiler.start_monitoring()
    
    try:
        # Profile the standard test
        start_time = time.time()
        profiler.log_processing_phase("Starting batch processing", {'start_row': 1, 'end_row': 35, 'reprocess': False})
        
        result = processor.process_scalable_batch(
            start_row_id=1,
            end_row_id=35,
            replace_existing_records=False,
            task_id=f"gpu_profile_{int(time.time())}"
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        profiler.log_processing_phase("Batch processing completed", {
            'total_time_seconds': round(total_time, 2),
            'opportunities_processed': result.get('opportunities_processed', 0),
            'documents_processed': result.get('documents_processed', 0),
            'chunks_generated': result.get('total_chunks_generated', 0),
            'seconds_per_opportunity': round(total_time / max(result.get('opportunities_processed', 1), 1), 3)
        })
        
    except Exception as e:
        profiler.log_processing_phase("ERROR during processing", {'error': str(e)})
        raise
    finally:
        profiler.stop_monitoring()
    
    # Analyze results
    print("\n" + "="*80)
    print("📊 PROFILING ANALYSIS")
    print("="*80)
    
    analysis = profiler.analyze_results()
    
    # Performance summary
    opportunities_processed = result.get('opportunities_processed', 0)
    seconds_per_opp = total_time / max(opportunities_processed, 1)
    improvement_needed = ((total_time - 35) / total_time) * 100 if total_time > 35 else 0
    
    print(f"⏱️  Performance Results:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Opportunities: {opportunities_processed}")
    print(f"   Seconds per opportunity: {seconds_per_opp:.3f}")
    print(f"   Target: <1.000 second per opportunity")
    print(f"   Improvement needed: {improvement_needed:.1f}% reduction")
    print()
    
    # GPU utilization summary
    gpu_analysis = analysis.get('gpu_analysis', {})
    print(f"🖥️  GPU Utilization:")
    print(f"   Peak: {gpu_analysis.get('peak_utilization_percent', 0)}%")
    print(f"   Average: {gpu_analysis.get('avg_utilization_percent', 0)}%")
    print(f"   Memory peak: {gpu_analysis.get('peak_memory_mb', 0)} MB ({gpu_analysis.get('peak_memory_percent', 0)}%)")
    print(f"   Memory average: {gpu_analysis.get('avg_memory_mb', 0)} MB ({gpu_analysis.get('avg_memory_percent', 0)}%)")
    print(f"   Temperature peak: {gpu_analysis.get('peak_temperature_c', 0)}°C")
    print()
    
    # Optimization opportunities
    opportunities = analysis.get('optimization_opportunities', [])
    if opportunities:
        print(f"🎯 Optimization Opportunities ({len(opportunities)} found):")
        for i, opp in enumerate(opportunities, 1):
            print(f"   {i}. {opp['type']} ({opp['severity']} priority)")
            print(f"      {opp['description']}")
            print(f"      💡 {opp['recommendation']}")
        print()
    else:
        print("✅ No obvious optimization opportunities detected")
        print()
    
    # Save detailed report
    report_file = profiler.save_detailed_report()
    
    print("="*80)
    print("📈 Next Steps for GPU Optimization:")
    print("1. Review detailed report for timing patterns")
    print("2. Experiment with larger batch sizes if memory allows")
    print("3. Investigate CUDA stream optimization")
    print("4. Consider concurrent processing streams")
    print(f"5. Target improvement: {improvement_needed:.1f}% reduction needed")
    print("="*80)
    
    return analysis, report_file

if __name__ == "__main__":
    try:
        analysis, report_file = profile_standard_processing()
        print(f"\n✅ GPU profiling completed successfully!")
        print(f"📊 Report saved: {report_file}")
    except Exception as e:
        print(f"\n❌ GPU profiling failed: {e}")
        import traceback
        traceback.print_exc()
