#!/usr/bin/env python3
"""
Direct test of search functionality with exact debugging
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from scalable_processor import ScalableEnhancedProcessor
from pymilvus import connections, Collection
import logging

def direct_search_test():
    """Direct search test"""
    
    print("=== Direct Search Test ===")
    
    # Connect directly
    connections.connect(alias="default", host="localhost", port="19530")
    
    # Get processor to generate embedding
    processor = ScalableEnhancedProcessor()
    
    query = "information management"
    embedding = processor.encode_with_pool([query], normalize_embeddings=True)[0]
    
    print(f"Query: '{query}'")
    print(f"Embedding length: {len(embedding)}")
    print(f"Embedding sample: {embedding[:5]}")
    print()
    
    # Test titles collection directly
    collection = Collection("opportunity_titles")
    collection.load()
    
    search_params = {
        "metric_type": "COSINE", 
        "params": {"nprobe": 10}
    }
    
    print("=== Direct Collection Search ===")
    try:
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=30,
            output_fields=["opportunity_id", "text_content", "posted_date"]
        )
        
        print(f"Raw results: {len(results[0])}")
        
        target_id = "F19A27DC-E75B-4F76-B268-00C16CCFF02B"
        found_target = False
        
        for i, result in enumerate(results[0]):
            score = result.distance
            opp_id = result.entity.get('opportunity_id')
            text = result.entity.get('text_content', '')
            
            if i < 10:  # Show top 10
                print(f"  {i+1}. Score: {score:.4f}, ID: {opp_id}")
                if 'Information' in text:
                    print(f"      Text: {text[:100]}...")
            
            if opp_id == target_id:
                found_target = True
                print(f"  *** FOUND TARGET: Position {i+1}, Score: {score:.4f}")
                print(f"      Above 0.7? {score >= 0.7}")
                print(f"      Text: {text}")
        
        if not found_target:
            print(f"  *** TARGET NOT FOUND in {len(results[0])} results")
            
        # Show the threshold filtering
        high_scores = [r for r in results[0] if r.distance >= 0.7]
        print(f"Results >= 0.7 threshold: {len(high_scores)}")
        
        medium_scores = [r for r in results[0] if r.distance >= 0.5]
        print(f"Results >= 0.5 threshold: {len(medium_scores)}")
        
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    direct_search_test()
