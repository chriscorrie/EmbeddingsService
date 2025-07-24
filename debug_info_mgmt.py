#!/usr/bin/env python3
"""
Debug script to investigate the "information management" search issue
"""

from pymilvus import connections, Collection
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_information_management_search():
    """Debug the information management search"""
    try:
        # Connect to Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        print("Connected to Milvus")
        
        # Get titles collection
        titles_collection = Collection("opportunity_titles")
        titles_collection.load()
        
        print(f"Titles collection has {titles_collection.num_entities} entities")
        
        # Search for records containing "Information Collection Management"
        print("\n=== Searching for 'Information Collection Management' ===")
        
        try:
            results = titles_collection.query(
                expr='text_content like "Information Collection%"',
                output_fields=["opportunity_id", "text_content"],
                limit=5
            )
            
            print(f"Found {len(results)} records containing 'Information Collection':")
            for result in results:
                print(f"  - {result['opportunity_id']}: {result['text_content']}")
                
        except Exception as e:
            print(f"LIKE query failed: {e}")
            
        # Now test semantic similarity
        print("\n=== Testing semantic similarity search ===")
        from scalable_processor import ScalableEnhancedProcessor
        
        processor = ScalableEnhancedProcessor()
        
        # Generate embeddings for both phrases
        query_embedding = processor.encode_with_pool(["information management"], normalize_embeddings=True)[0]
        target_embedding = processor.encode_with_pool(["Information Collection Management"], normalize_embeddings=True)[0]
        
        # Calculate direct similarity
        import numpy as np
        cosine_sim = np.dot(query_embedding, target_embedding)
        print(f"Direct cosine similarity between 'information management' and 'Information Collection Management': {cosine_sim:.4f}")
        
        # Search in titles collection
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = titles_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=10,
            output_fields=["opportunity_id", "text_content"]
        )
        
        print(f"\nTop 10 similarity search results for 'information management':")
        for hit in results[0]:
            score = hit.distance  # For COSINE, distance IS the similarity score
            print(f"  - Score: {score:.4f}, Text: '{hit.entity['text_content']}'")
            
        # Look specifically for our target
        target_found = False
        for hit in results[0]:
            if "Information Collection Management" in hit.entity['text_content']:
                score = hit.distance
                print(f"\n🎯 FOUND TARGET: Score: {score:.4f}, Text: '{hit.entity['text_content']}'")
                target_found = True
                break
                
        if not target_found:
            print("\n❌ Target 'Information Collection Management' NOT found in top 10 results")
            
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_information_management_search()
