#!/usr/bin/env python3
"""
Realistic Content Batch Size Optimization
Find optimal batch size for realistic federal procurement document sentences
"""

import config
import time
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import random

def generate_realistic_sentences(count=1000):
    """Generate realistic document sentences similar to federal procurement documents"""
    
    templates = [
        "The contractor shall provide {service} in accordance with the Statement of Work attached hereto as Exhibit {letter} and comply with all applicable federal regulations including but not limited to the Federal Acquisition Regulation (FAR) and agency-specific procurement guidelines.",
        "All deliverables must comply with Federal Acquisition Regulation (FAR) {section} requirements and undergo comprehensive quality assurance testing to ensure compliance with government standards and specifications.",
        "Performance under this contract is subject to the terms and conditions specified in {document} including milestone deliverables, performance metrics, and quality standards as outlined in the Performance Work Statement.",
        "The Government reserves the right to inspect all work performed under this contract and may conduct periodic reviews, audits, and evaluations to ensure contractor compliance with technical specifications.",
        "Payment shall be made within {days} days of receipt of properly submitted invoices accompanied by all required documentation including proof of deliverable acceptance and compliance certifications.",
        "This procurement is set aside for small business concerns under NAICS code {code} and contractors must maintain their small business certification throughout the period of performance.",
        "The period of performance shall commence on {date} and continue through {end_date} with options for additional performance periods subject to government approval and funding availability.",
        "All personnel working on this contract must maintain appropriate security clearances and undergo background investigations as required by government security protocols and agency-specific requirements.",
        "The contractor must submit monthly progress reports detailing completed milestones, upcoming deliverables, resource utilization, and any potential risks or issues that may impact contract performance.",
        "Quality assurance procedures must be implemented throughout the contract performance period including regular testing, documentation reviews, and compliance audits to ensure deliverable quality meets government standards.",
    ]
    
    services = ["professional consulting services", "technical support and maintenance", "information technology services", "engineering and design services"]
    letters = ["A", "B", "C", "D"]
    sections = ["15.203", "12.301", "8.405", "16.505"]
    documents = ["the base contract and all amendments", "Appendix 1 and related technical specifications", "the master agreement and performance work statement"]
    days = ["30", "45", "60"]
    codes = ["541511", "541512", "541330", "541618"]
    dates = ["January 1, 2025", "February 15, 2025", "March 1, 2025"]
    
    sentences = []
    for i in range(count):
        template = random.choice(templates)
        sentence = template.format(
            service=random.choice(services),
            letter=random.choice(letters),
            section=random.choice(sections),
            document=random.choice(documents),
            days=random.choice(days),
            code=random.choice(codes),
            date=random.choice(dates),
            end_date=random.choice(dates)
        )
        sentences.append(sentence)
    
    return sentences

def test_batch_size_with_realistic_content(batch_size, test_size=5000):
    """Test specific batch size with realistic content"""
    
    # Generate realistic test content
    test_sentences = generate_realistic_sentences(test_size)
    avg_length = sum(len(s) for s in test_sentences) / len(test_sentences)
    
    print(f"Testing batch size {batch_size} with {test_size} sentences (avg length: {avg_length:.0f} chars)")
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        start_time = time.time()
        model = SentenceTransformer(config.EMBEDDING_MODEL, device='cuda')
        
        # Monitor memory before processing
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        embeddings = model.encode(
            test_sentences, 
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=False
        )
        
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        sentences_per_second = len(test_sentences) / processing_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        result = {
            'batch_size': batch_size,
            'test_size': len(test_sentences),
            'avg_sentence_length': avg_length,
            'processing_time': processing_time,
            'sentences_per_second': sentences_per_second,
            'peak_memory_mb': peak_memory,
            'initial_memory_mb': initial_memory,
            'status': 'SUCCESS'
        }
        
        print(f"  ✅ SUCCESS: {sentences_per_second:.0f} sentences/sec, {peak_memory:.0f} MB peak memory")
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ OUT OF MEMORY: {e}")
        return {
            'batch_size': batch_size,
            'status': 'OUT_OF_MEMORY',
            'error': str(e)
        }
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return {
            'batch_size': batch_size,
            'status': 'ERROR',
            'error': str(e)
        }

def find_optimal_realistic_batch_size():
    """Find optimal batch size for realistic content"""
    
    print("🔍 Finding Optimal Batch Size for Realistic Federal Procurement Content")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)")
    
    # Test progressively smaller batch sizes
    batch_sizes = [16384, 8192, 4096, 2048, 1024, 512, 256]
    results = []
    
    print("\\nBatch Size | Status | Sentences/sec | Peak Memory(MB) | Avg Length")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        result = test_batch_size_with_realistic_content(batch_size)
        results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"{result['batch_size']:9d} | SUCCESS   | {result['sentences_per_second']:11.0f} | {result['peak_memory_mb']:13.0f} | {result['avg_sentence_length']:8.0f}")
        else:
            print(f"{result['batch_size']:9d} | {result['status']:9s} | {'N/A':11s} | {'N/A':13s} | {'N/A':8s}")
    
    # Find best successful result
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    
    if successful_results:
        best_result = max(successful_results, key=lambda x: x['sentences_per_second'])
        
        print(f"\\n🏆 OPTIMAL CONFIGURATION FOR REALISTIC CONTENT:")
        print(f"   Batch Size: {best_result['batch_size']}")
        print(f"   Performance: {best_result['sentences_per_second']:.0f} sentences/second")
        print(f"   Peak Memory: {best_result['peak_memory_mb']:.0f} MB")
        print(f"   Daily Capacity: {best_result['sentences_per_second'] * 86400:.0f} sentences/day")
        
        # Update config recommendation
        print(f"\\n📝 RECOMMENDED CONFIG UPDATE:")
        print(f"   EMBEDDING_BATCH_SIZE = {best_result['batch_size']}")
        
        return best_result
    else:
        print("\\n❌ No successful batch sizes found!")
        return None

if __name__ == "__main__":
    try:
        optimal_config = find_optimal_realistic_batch_size()
        if optimal_config:
            print(f"\\n✅ Optimal batch size found: {optimal_config['batch_size']}")
            print(f"🎯 Expected performance: {optimal_config['sentences_per_second']:.0f} sentences/second")
        
    except Exception as e:
        print(f"\\n❌ Batch size optimization failed: {e}")
        import traceback
        traceback.print_exc()
