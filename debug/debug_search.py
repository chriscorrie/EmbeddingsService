#!/usr/bin/env python3
"""
Debug script to investigate the "ARMED FORCES CAREER CENTER" search issue
"""

from pymilvus import connections, Collection
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_armed_forces_search():
    """Debug the Armed Forces Career Center search"""
    try:
        # Connect to Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        print("Connected to Milvus")
        
        # Get titles collection
        titles_collection = Collection("opportunity_titles")
        titles_collection.load()
        
        print(f"Titles collection has {titles_collection.num_entities} entities")
        
        # Search for exact text match first
        print("\n=== Searching for exact text matches ===")
        
        # Query for records containing "ARMED FORCES"
        expr = 'text_content like "ARMED FORCES%"'
        try:
            results = titles_collection.query(
                expr=expr,
                output_fields=["opportunity_id", "text_content"],
                limit=10
            )
            
            print(f"Found {len(results)} records containing 'ARMED FORCES':")
            for result in results:
                print(f"  - {result['opportunity_id']}: {result['text_content']}")
                
        except Exception as e:
            print(f"LIKE query failed: {e}")
            
        # Try a broader search
        print("\n=== Getting some sample titles to see what's in the database ===")
        try:
            sample_results = titles_collection.query(
                expr="",  # Get all
                output_fields=["opportunity_id", "text_content"],
                limit=10
            )
            
            print(f"Sample titles from database:")
            for result in sample_results:
                print(f"  - {result['opportunity_id']}: {result['text_content']}")
                
        except Exception as e:
            print(f"Sample query failed: {e}")
            
        # Now let's test the actual similarity search
        print("\n=== Testing embedding similarity search ===")
        from scalable_processor import ScalableEnhancedProcessor
        
        processor = ScalableEnhancedProcessor()
        
        # Generate embedding for our query
        query_embedding = processor.encode_with_pool(["ARMED FORCES CAREER CENTER"], normalize_embeddings=True)[0]
        print(f"Generated embedding for query (length: {len(query_embedding)})")
        
        # Search in titles collection directly
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = titles_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=5,
            output_fields=["opportunity_id", "text_content"]
        )
        
        print(f"Top 5 similarity search results:")
        for hit in results[0]:
            score = 1.0 - hit.distance  # Convert distance to similarity
            print(f"  - Score: {score:.4f}, ID: {hit.entity['opportunity_id']}, Text: '{hit.entity['text_content']}'")
            
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_armed_forces_search()
