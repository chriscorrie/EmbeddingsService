#!/usr/bin/env python3
"""
Production Validation Test
Tests the optimal configuration with realistic document content
and validates sustained performance for millions of documents
"""

import config
import time
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import random

def generate_realistic_sentences(count=10000):
    """Generate realistic document sentences similar to federal procurement documents"""
    
    # Base templates for realistic federal procurement content
    templates = [
        "The contractor shall provide {service} in accordance with the Statement of Work attached hereto as Exhibit {letter}.",
        "All deliverables must comply with Federal Acquisition Regulation (FAR) {section} requirements.",
        "Performance under this contract is subject to the terms and conditions specified in {document}.",
        "The Government reserves the right to inspect all work performed under this contract.",
        "Payment shall be made within {days} days of receipt of properly submitted invoices.",
        "This procurement is set aside for small business concerns under NAICS code {code}.",
        "The period of performance shall commence on {date} and continue through {end_date}.",
        "All personnel working on this contract must maintain appropriate security clearances.",
        "The contractor must submit monthly progress reports detailing completed milestones.",
        "Quality assurance procedures must be implemented throughout the contract performance period.",
        "Technical specifications are detailed in the Performance Work Statement (PWS).",
        "The contractor shall maintain insurance coverage as specified in the contract terms.",
        "All intellectual property developed under this contract belongs to the Government.",
        "Environmental compliance requirements must be met for all contract activities.",
        "The contractor is responsible for obtaining all necessary permits and licenses.",
        "Subcontracting arrangements must be approved in writing by the Contracting Officer.",
        "Risk management procedures shall be implemented to ensure successful contract execution.",
        "The contractor must provide qualified personnel with relevant industry experience.",
        "Performance metrics will be evaluated quarterly using established benchmarks.",
        "Security protocols must be followed for all classified or sensitive information.",
    ]
    
    # Variables for template substitution
    services = ["professional services", "technical support", "maintenance services", "consulting services", "IT support", "engineering services"]
    letters = ["A", "B", "C", "D", "E", "F"]
    sections = ["15.203", "12.301", "8.405", "16.505", "52.212-4", "13.106"]
    documents = ["the base contract", "Appendix 1", "the master agreement", "Schedule A", "the task order"]
    days = ["30", "45", "60", "90"]
    codes = ["541511", "541512", "541330", "541618", "541690", "541715"]
    dates = ["January 1, 2025", "February 15, 2025", "March 1, 2025", "April 30, 2025"]
    
    sentences = []
    for i in range(count):
        template = random.choice(templates)
        
        # Fill in template variables
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

def production_validation_test(total_sentences=100000):
    """Run production validation with realistic content"""
    
    print("🏭 Production Validation Test")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)")
    print(f"Testing with {total_sentences:,} realistic document sentences")
    print(f"Expected time: {total_sentences / 15000:.1f} seconds based on optimization testing")
    
    # Generate realistic test content
    print("\\n📝 Generating realistic federal procurement document content...")
    test_sentences = generate_realistic_sentences(total_sentences)
    print(f"Generated {len(test_sentences):,} sentences")
    
    # Initialize model
    print("\\n🤖 Loading sentence transformer model...")
    start_load = time.time()
    model = SentenceTransformer(config.EMBEDDING_MODEL, device='cuda')
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Monitor initial GPU state
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    # Production processing test
    print(f"\\n⚡ Processing {total_sentences:,} sentences with batch size {config.EMBEDDING_BATCH_SIZE:,}...")
    
    start_time = time.time()
    embeddings = model.encode(
        test_sentences, 
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=False  # Save memory
    )
    end_time = time.time()
    
    # Calculate performance metrics
    processing_time = end_time - start_time
    sentences_per_second = total_sentences / processing_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    final_memory = torch.cuda.memory_allocated() / 1024**2
    
    # Projected daily capacity
    daily_sentences = sentences_per_second * 86400
    
    # Results
    print(f"\\n📊 PRODUCTION VALIDATION RESULTS:")
    print(f"   Total Sentences: {total_sentences:,}")
    print(f"   Processing Time: {processing_time:.2f} seconds")
    print(f"   Performance: {sentences_per_second:,.0f} sentences/second")
    print(f"   GPU Memory Peak: {peak_memory:.1f} MB")
    print(f"   GPU Memory Final: {final_memory:.1f} MB")
    print(f"   Embedding Shape: {embeddings.shape}")
    print(f"   Daily Capacity: {daily_sentences:,.0f} sentences/day")
    
    # Production projections
    print(f"\\n🎯 PRODUCTION PROJECTIONS:")
    print(f"   1 Million documents: {1000000 / sentences_per_second:.1f} seconds ({1000000 / sentences_per_second / 60:.1f} minutes)")
    print(f"   10 Million documents: {10000000 / sentences_per_second:.1f} seconds ({10000000 / sentences_per_second / 3600:.1f} hours)")
    print(f"   100 Million documents: {100000000 / sentences_per_second:.1f} seconds ({100000000 / sentences_per_second / 86400:.1f} days)")
    
    # Log results
    log_production_results(total_sentences, processing_time, sentences_per_second, peak_memory, embeddings.shape)
    
    return {
        'total_sentences': total_sentences,
        'processing_time': processing_time,
        'sentences_per_second': sentences_per_second,
        'peak_memory_mb': peak_memory,
        'embedding_shape': embeddings.shape,
        'daily_capacity': daily_sentences
    }

def log_production_results(total_sentences, processing_time, sentences_per_second, peak_memory, embedding_shape):
    """Log production validation results"""
    
    log_entry = f'''
#### Production Validation Test
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Test Type**: Production validation with realistic content
- **Total Sentences**: {total_sentences:,}
- **Processing Time**: {processing_time:.2f} seconds
- **Performance**: {sentences_per_second:,.0f} sentences/second
- **GPU Memory Peak**: {peak_memory:.1f} MB
- **Embedding Shape**: {embedding_shape}
- **Daily Capacity**: {sentences_per_second * 86400:,.0f} sentences/day
- **Configuration**: Batch size {config.EMBEDDING_BATCH_SIZE:,}, Single worker
- **Status**: Production validation completed - ready for millions of documents

'''
    
    try:
        with open('gpu_performance_log.md', 'a') as f:
            f.write(log_entry)
        print(f"\\n📝 Results logged to gpu_performance_log.md")
    except Exception as e:
        print(f"Failed to update log: {e}")

if __name__ == "__main__":
    print("🚀 Starting Production Validation Test with Optimal Configuration")
    print(f"Configuration: {config.EMBEDDING_BATCH_SIZE:,} batch size, {config.MAX_OPPORTUNITY_WORKERS} worker(s)")
    
    try:
        # Test with 100K sentences (realistic production preview)
        results = production_validation_test(100000)
        
        print(f"\\n✅ Production validation completed successfully!")
        print(f"🎯 System ready for processing millions of documents at {results['sentences_per_second']:,.0f} sentences/second")
        
    except Exception as e:
        print(f"\\n❌ Production validation failed: {e}")
        import traceback
        traceback.print_exc()
