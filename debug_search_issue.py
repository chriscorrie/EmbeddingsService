#!/usr/bin/env python3
"""
Debug script to investigate the search issue with "information collection management"
"""

from pymilvus import connections, Collection
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scalable_processor import ScalableEnhancedProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_search_issue():
    """Debug the search issue"""
    try:
        # Connect to Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        print("Connected to Milvus")
        
        # Initialize processor for embeddings
        processor = ScalableEnhancedProcessor()
        
        # Get the opportunity ID from our search result
        opportunity_id = "F19A27DC-E75B-4F76-B268-00C16CCFF02B"
        
        # Check titles collection
        titles_collection = Collection("opportunity_titles")
        titles_collection.load()
        
        print(f"\n=== Checking opportunity {opportunity_id} in titles collection ===")
        
        # Query for this specific opportunity
        results = titles_collection.query(
            expr=f'opportunity_id == "{opportunity_id}"',
            output_fields=["opportunity_id", "text_content", "chunk_index", "total_chunks"],
            limit=10
        )
        
        print(f"Found {len(results)} title records for this opportunity:")
        for result in results:
            print(f"  Chunk {result['chunk_index']}/{result['total_chunks']}: {result['text_content']}")
            
        # Now check if any title contains the phrase we're looking for
        print(f"\n=== Searching for phrase containing 'information collection management' ===")
        
        # Try different variations of the search
        search_variations = [
            "information collection management",
            "Information Collection Management", 
            "collection management",
            "information management"
        ]
        
        for search_term in search_variations:
            print(f"\n--- Searching for '{search_term}' ---")
            try:
                # Query by text content
                results = titles_collection.query(
                    expr=f'text_content like "%{search_term}%"',
                    output_fields=["opportunity_id", "text_content"],
                    limit=5
                )
                
                print(f"Found {len(results)} records containing '{search_term}':")
                for result in results:
                    print(f"  - {result['opportunity_id']}: {result['text_content']}")
                    
            except Exception as e:
                print(f"Text search failed for '{search_term}': {e}")
        
        # Test semantic similarity with different queries
        print(f"\n=== Testing semantic similarity ===")
        
        query_phrases = [
            "information collection management",
            "Information Collection Management",
            "information management", 
            "collection management",
            "data collection management"
        ]
        
        for query in query_phrases:
            print(f"\n--- Testing query: '{query}' ---")
            
            # Generate embedding
            query_embedding = processor.encode_with_pool([query], normalize_embeddings=True)[0]
            
            # Search with COSINE similarity
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            search_results = titles_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=3,
                output_fields=["opportunity_id", "text_content"]
            )
            
            print(f"Top 3 results for '{query}':")
            for i, hit in enumerate(search_results[0]):
                score = hit.distance  # For COSINE, distance IS the similarity score
                print(f"  {i+1}. Score: {score:.4f} - {hit.entity.get('opportunity_id')}: {hit.entity.get('text_content')}")
        
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_search_issue()
