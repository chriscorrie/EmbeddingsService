#!/usr/bin/env python3
"""
GPU Optimization Parameter Sweep
Tests various combinations of workers, batch sizes, and other parameters
to find optimal configuration for maximum throughput
"""

import config
import time
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import os
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

def single_worker_test(batch_size, test_size=2000):
    """Test single worker performance with given batch size"""
    test_texts = [
        'Federal procurement requires comprehensive document analysis and review.',
        'Contract compliance verification involves detailed clause examination.',
        'GPU acceleration enables processing of enormous document collections.',
        'Blackwell architecture delivers exceptional parallel processing performance.',
    ] * (test_size // 4)
    
    start_time = time.time()
    model = SentenceTransformer(config.EMBEDDING_MODEL, device='cuda')
    embeddings = model.encode(test_texts, batch_size=batch_size, show_progress_bar=False)
    end_time = time.time()
    
    processing_time = end_time - start_time
    sentences_per_second = len(test_texts) / processing_time
    gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
    
    return {
        'batch_size': batch_size,
        'test_size': len(test_texts),
        'processing_time': processing_time,
        'sentences_per_second': sentences_per_second,
        'gpu_memory_mb': gpu_memory_mb,
        'embedding_shape': embeddings.shape
    }

def worker_function(worker_id, texts_queue, results_queue, batch_size):
    """Worker function for multi-worker tests"""
    model = SentenceTransformer(config.EMBEDDING_MODEL, device='cuda')
    total_processed = 0
    start_time = time.time()
    
    while True:
        try:
            texts = texts_queue.get(timeout=1)
            if texts is None:  # Poison pill
                break
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            total_processed += len(texts)
            texts_queue.task_done()
        except queue.Empty:
            break
    
    end_time = time.time()
    processing_time = end_time - start_time
    sentences_per_second = total_processed / processing_time if processing_time > 0 else 0
    
    results_queue.put({
        'worker_id': worker_id,
        'total_processed': total_processed,
        'processing_time': processing_time,
        'sentences_per_second': sentences_per_second
    })

def multi_worker_test(num_workers, batch_size, test_size=2000):
    """Test multi-worker performance with given parameters"""
    test_texts = [
        'Federal procurement requires comprehensive document analysis and review.',
        'Contract compliance verification involves detailed clause examination.',
        'GPU acceleration enables processing of enormous document collections.',
        'Blackwell architecture delivers exceptional parallel processing performance.',
    ] * (test_size // 4)
    
    # Split texts into chunks for workers
    chunk_size = len(test_texts) // num_workers
    text_chunks = [test_texts[i:i + chunk_size] for i in range(0, len(test_texts), chunk_size)]
    
    texts_queue = queue.Queue()
    results_queue = queue.Queue()
    
    # Add text chunks to queue
    for chunk in text_chunks:
        texts_queue.put(chunk)
    
    # Add poison pills
    for _ in range(num_workers):
        texts_queue.put(None)
    
    start_time = time.time()
    
    # Start workers
    workers = []
    for i in range(num_workers):
        worker = threading.Thread(target=worker_function, args=(i, texts_queue, results_queue, batch_size))
        worker.start()
        workers.append(worker)
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join()
    
    end_time = time.time()
    
    # Collect results
    total_processed = 0
    worker_results = []
    while not results_queue.empty():
        result = results_queue.get()
        worker_results.append(result)
        total_processed += result['total_processed']
    
    overall_time = end_time - start_time
    overall_sentences_per_second = total_processed / overall_time
    gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
    
    return {
        'num_workers': num_workers,
        'batch_size': batch_size,
        'test_size': total_processed,
        'processing_time': overall_time,
        'sentences_per_second': overall_sentences_per_second,
        'gpu_memory_mb': gpu_memory_mb,
        'worker_results': worker_results
    }

def run_parameter_sweep():
    """Run comprehensive parameter sweep tests"""
    
    print("🚀 Starting GPU Optimization Parameter Sweep...")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)")
    
    # Test configurations
    batch_sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    worker_counts = [1, 2, 3, 4, 6, 8]
    test_size = 2000  # Sentences per test
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': {
            'gpu': torch.cuda.get_device_name(0),
            'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda
        },
        'single_worker_tests': [],
        'multi_worker_tests': []
    }
    
    print("\\n🔬 Testing Single Worker Performance...")
    print("Batch Size | Sentences/sec | Time(s) | GPU Memory(MB)")
    print("-" * 55)
    
    # Single worker tests with varying batch sizes
    for batch_size in batch_sizes:
        try:
            print(f"Testing batch size {batch_size}...", end=" ")
            result = single_worker_test(batch_size, test_size)
            results['single_worker_tests'].append(result)
            print(f"{result['batch_size']:8d} | {result['sentences_per_second']:11.0f} | {result['processing_time']:6.2f} | {result['gpu_memory_mb']:12.1f}")
        except Exception as e:
            print(f"FAILED: {e}")
            results['single_worker_tests'].append({
                'batch_size': batch_size,
                'error': str(e)
            })
    
    # Find best single worker batch size
    best_single = max([r for r in results['single_worker_tests'] if 'error' not in r], 
                     key=lambda x: x['sentences_per_second'])
    optimal_batch_size = best_single['batch_size']
    
    print(f"\\n🏆 Best Single Worker: {optimal_batch_size} batch size = {best_single['sentences_per_second']:.0f} sentences/sec")
    
    print("\\n🔬 Testing Multi-Worker Performance...")
    print("Workers | Batch Size | Sentences/sec | Time(s) | GPU Memory(MB)")
    print("-" * 65)
    
    # Multi-worker tests
    for num_workers in worker_counts:
        try:
            print(f"Testing {num_workers} workers...", end=" ")
            result = multi_worker_test(num_workers, optimal_batch_size, test_size)
            results['multi_worker_tests'].append(result)
            print(f"{result['num_workers']:7d} | {result['batch_size']:10d} | {result['sentences_per_second']:11.0f} | {result['processing_time']:6.2f} | {result['gpu_memory_mb']:12.1f}")
        except Exception as e:
            print(f"FAILED: {e}")
            results['multi_worker_tests'].append({
                'num_workers': num_workers,
                'batch_size': optimal_batch_size,
                'error': str(e)
            })
    
    # Find optimal configuration
    best_config = max([r for r in results['multi_worker_tests'] if 'error' not in r], 
                     key=lambda x: x['sentences_per_second'])
    
    print(f"\\n🎯 OPTIMAL CONFIGURATION:")
    print(f"   Workers: {best_config['num_workers']}")
    print(f"   Batch Size: {best_config['batch_size']}")
    print(f"   Performance: {best_config['sentences_per_second']:.0f} sentences/second")
    print(f"   GPU Memory: {best_config['gpu_memory_mb']:.1f} MB")
    print(f"   Daily Capacity: {best_config['sentences_per_second'] * 86400:.0f} sentences/day")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Save results to file
    with open('logs/gpu_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Update performance log
    update_performance_log(results, best_config)
    
    return results, best_config

def update_performance_log(results, best_config):
    """Update the performance log with optimization results"""
    
    log_entry = f'''
#### Optimization Parameter Sweep
- **Date**: {results["timestamp"]}
- **Test Type**: Comprehensive parameter optimization
- **Single Worker Best**: {max([r for r in results['single_worker_tests'] if 'error' not in r], key=lambda x: x['sentences_per_second'])['sentences_per_second']:.0f} sentences/sec (batch size {max([r for r in results['single_worker_tests'] if 'error' not in r], key=lambda x: x['sentences_per_second'])['batch_size']})
- **Multi-Worker Best**: {best_config['sentences_per_second']:.0f} sentences/sec ({best_config['num_workers']} workers, batch size {best_config['batch_size']})
- **Optimal Configuration**: {best_config['num_workers']} workers × {best_config['batch_size']} batch size
- **GPU Memory**: {best_config['gpu_memory_mb']:.1f} MB
- **Daily Capacity**: {best_config['sentences_per_second'] * 86400:.0f} sentences/day
- **Status**: Parameter sweep completed - optimal configuration identified

'''
    
    # Append to log
    try:
        with open('logs/gpu_performance_log.md', 'a') as f:
            f.write(log_entry)
        print(f"\n📝 Results logged to logs/gpu_performance_log.md and logs/gpu_optimization_results.json")
    except Exception as e:
        print(f"Failed to update log: {e}")

if __name__ == "__main__":
    try:
        results, best_config = run_parameter_sweep()
        print(f"\\n✅ Parameter sweep completed successfully!")
        print(f"📊 Results saved to logs/gpu_optimization_results.json")
    except Exception as e:
        print(f"\\n❌ Parameter sweep failed: {e}")
        import traceback
        traceback.print_exc()
