#!/usr/bin/env python3
"""
GPU Performance Test Script
Automatically tests GPU performance and logs results to gpu_performance_log.md
"""

import config
import time
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import os

def run_gpu_performance_test():
    """Run GPU performance test and log results"""
    
    # Test configuration
    test_texts = [
        'Federal procurement requires comprehensive document analysis and review.',
        'Contract compliance verification involves detailed clause examination.',
        'GPU acceleration enables processing of enormous document collections.',
        'Blackwell architecture delivers exceptional parallel processing performance.',
        'Advanced document processing supports complex regulatory requirements.',
        'Large-scale text analysis benefits significantly from GPU optimization.',
        'Federal contracting documents contain critical compliance information.',
        'Efficient processing workflows enable rapid document analysis.',
    ] * 500  # 4000 sentences - MASSIVE test
    
    print(f'🚀 Testing MAXIMUM GPU Performance...')
    print(f'⚡ MAXIMUM GPU Configuration:')
    print(f'   Base batch size: {config.EMBEDDING_BATCH_SIZE}')
    print(f'   GPU multiplier: {config.GPU_BATCH_SIZE_MULTIPLIER}')
    print(f'   GPU batch size: {config.EMBEDDING_BATCH_SIZE * config.GPU_BATCH_SIZE_MULTIPLIER}')
    print(f'   Large doc batch: {config.LARGE_DOC_EMBEDDING_BATCH_SIZE}')
    print(f'\\n🔥 Testing MAXIMUM performance with {len(test_texts)} sentences...')
    print(f'   Target batch size: {config.EMBEDDING_BATCH_SIZE * config.GPU_BATCH_SIZE_MULTIPLIER}')
    
    start_time = time.time()
    
    model = SentenceTransformer(config.EMBEDDING_MODEL, device='cuda')
    max_batch_size = config.EMBEDDING_BATCH_SIZE * config.GPU_BATCH_SIZE_MULTIPLIER
    embeddings = model.encode(test_texts, batch_size=max_batch_size, show_progress_bar=False)
    
    end_time = time.time()
    processing_time = end_time - start_time
    sentences_per_second = len(test_texts) / processing_time
    
    # Collect results
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_size': len(test_texts),
        'batch_size': max_batch_size,
        'base_batch': config.EMBEDDING_BATCH_SIZE,
        'multiplier': config.GPU_BATCH_SIZE_MULTIPLIER,
        'processing_time': processing_time,
        'sentences_per_second': sentences_per_second,
        'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2,
        'vram_usage_percent': (torch.cuda.memory_allocated() / (15.5 * 1024**3)) * 100,
        'embedding_shape': embeddings.shape,
        'large_doc_batch': config.LARGE_DOC_EMBEDDING_BATCH_SIZE
    }
    
    # Print results
    print(f'✅ MAXIMUM GPU processing complete!')
    print(f'📊 Performance: {results["test_size"]} sentences in {results["processing_time"]:.2f}s')
    print(f'🏆 Speed: {results["sentences_per_second"]:.0f} sentences/second')
    print(f'📐 Embedding shape: {results["embedding_shape"]}')
    print(f'💾 GPU Memory: {results["gpu_memory_mb"]:.1f} MB')
    print(f'📈 VRAM Usage: {results["vram_usage_percent"]:.1f}% of 15.5GB')
    print(f'🔥 Batch size used: {results["batch_size"]}')
    print(f'🚀 SPEEDUP vs original 377: {results["sentences_per_second"]/377:.1f}x faster!')
    print(f'🚀 SPEEDUP vs optimized 1636: {results["sentences_per_second"]/1636:.1f}x faster!')
    
    # Log to file
    log_results_to_file(results)
    
    return results

def log_results_to_file(results):
    """Append test results to the performance log"""
    
    log_entry = f'''
#### Test 3: Maximum Performance Configuration
- **Date**: {results["timestamp"]}
- **Batch Size**: {results["batch_size"]} ({results["base_batch"]} × {results["multiplier"]}) - **{results["batch_size"]//4096:.0f}x larger than previous**
- **Test Size**: {results["test_size"]} sentences - **{results["test_size"]/1200:.1f}x larger test**
- **Performance**: **{results["sentences_per_second"]:.0f} sentences/second** 🚀
- **Processing Time**: {results["processing_time"]:.2f}s
- **GPU Memory**: {results["gpu_memory_mb"]:.1f} MB ({results["vram_usage_percent"]:.1f}% of 15.5GB)
- **Speedup vs baseline (377)**: **{results["sentences_per_second"]/377:.1f}x faster**
- **Speedup vs optimized (1636)**: **{results["sentences_per_second"]/1636:.1f}x faster**
- **Cache Size**: {results["cache_size"]}
- **Large Doc Batch**: {results["large_doc_batch"]}
- **Status**: {"Maximum performance achieved!" if results["sentences_per_second"] > 2000 else "Good performance, room for improvement"}

'''
    
    # Read existing log
    try:
        with open('gpu_performance_log.md', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        content = "# GPU Performance Optimization Log\\n"
    
    # Find where to insert new entry (after "### Performance Test Results")
    insert_point = content.find("### Performance Test Results")
    if insert_point != -1:
        # Find the end of existing entries
        next_section = content.find("### ", insert_point + 1)
        if next_section != -1:
            # Insert before next section
            new_content = content[:next_section] + log_entry + "\\n" + content[next_section:]
        else:
            # Append to end
            new_content = content + log_entry
    else:
        # Append to end
        new_content = content + "\\n### Performance Test Results\\n" + log_entry
    
    # Write updated log
    with open('gpu_performance_log.md', 'w') as f:
        f.write(new_content)
    
    print(f"\\n📝 Results logged to gpu_performance_log.md")

if __name__ == "__main__":
    try:
        results = run_gpu_performance_test()
        print(f"\\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        # Log the error too
        with open('gpu_performance_log.md', 'a') as f:
            f.write(f"\\n#### Test Failed - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"- **Error**: {str(e)}\\n")
            f.write(f"- **Attempted Batch Size**: {config.EMBEDDING_BATCH_SIZE * config.GPU_BATCH_SIZE_MULTIPLIER}\\n\\n")
