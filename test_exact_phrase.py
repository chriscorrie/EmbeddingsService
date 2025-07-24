#!/usr/bin/env python3
"""
Test the exact phrase "Information Collection Management"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from scalable_processor import ScalableEnhancedProcessor
from pymilvus import connections, Collection
import logging

def test_exact_phrase():
    """Test exact phrase search"""
    
    print("=== Testing Exact Phrase: 'Information Collection Management' ===")
    
    # Connect directly
    connections.connect(alias="default", host="localhost", port="19530")
    
    # Get processor to generate embedding
    processor = ScalableEnhancedProcessor()
    
    exact_phrase = "Information Collection Management"
    full_title = "RFP - Intelligence Application (Intel Apps) Information Collection Management (ICM) - W56KGY24R0002"
    target_id = "F19A27DC-E75B-4F76-B268-00C16CCFF02B"
    
    print(f"Exact phrase: '{exact_phrase}'")
    print(f"Target title: '{full_title}'")
    print(f"Target ID: '{target_id}'")
    print()
    
    # Generate embeddings for comparison
    phrase_embedding = processor.encode_with_pool([exact_phrase], normalize_embeddings=True)[0]
    title_embedding = processor.encode_with_pool([full_title], normalize_embeddings=True)[0]
    
    # Manual similarity calculation
    import numpy as np
    manual_similarity = np.dot(phrase_embedding, title_embedding)
    print(f"Manual similarity (phrase vs full title): {manual_similarity:.4f}")
    print(f"Should easily pass 0.7 threshold? {manual_similarity >= 0.7}")
    print()
    
    # Test direct search
    collection = Collection("opportunity_titles")
    collection.load()
    
    search_params = {
        "metric_type": "COSINE", 
        "params": {"nprobe": 10}
    }
    
    print("=== Direct Search with Exact Phrase ===")
    try:
        results = collection.search(
            data=[phrase_embedding],
            anns_field="embedding",
            param=search_params,
            limit=30,
            output_fields=["opportunity_id", "text_content", "posted_date"]
        )
        
        print(f"Raw results: {len(results[0])}")
        
        found_target = False
        
        for i, result in enumerate(results[0]):
            score = result.distance
            opp_id = result.entity.get('opportunity_id')
            text = result.entity.get('text_content', '')
            
            if i < 10:  # Show top 10
                print(f"  {i+1}. Score: {score:.4f}, ID: {opp_id}")
                if 'Information' in text:
                    print(f"      Text: {text[:150]}...")
            
            if opp_id == target_id:
                found_target = True
                print(f"  *** FOUND TARGET: Position {i+1}, Score: {score:.4f}")
                print(f"      Above 0.7 threshold? {score >= 0.7}")
                print(f"      Full text: {text}")
                break
        
        if not found_target:
            print(f"  *** TARGET NOT FOUND in {len(results[0])} results")
            
        # Show the threshold filtering
        high_scores = [r for r in results[0] if r.distance >= 0.7]
        print(f"Results >= 0.7 threshold: {len(high_scores)}")
        
        if high_scores:
            print("Top results above 0.7:")
            for i, result in enumerate(high_scores[:5]):
                print(f"  {i+1}. Score: {result.distance:.4f}, ID: {result.entity.get('opportunity_id')}")
                
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test API call
    print("=== Testing API Search ===")
    try:
        results = processor.search_similar_documents(
            query=exact_phrase,
            limit=10,
            title_similarity_threshold=0.7,
            description_similarity_threshold=0.7,
            document_similarity_threshold=0.7,
            boost_factor=1
        )
        
        print(f"API search returned {len(results)} results")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['aggregated_score']:.4f}, ID: {result['opportunity_id']}")
            if result['opportunity_id'] == target_id:
                print(f"      *** FOUND OUR TARGET!")
            
    except Exception as e:
        print(f"API search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exact_phrase()
