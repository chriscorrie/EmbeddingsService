#!/usr/bin/env python3
"""
Test the exact title we know exists
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from scalable_processor import ScalableEnhancedProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_specific_title():
    """Test the specific title we know exists"""
    
    print("=== Testing Specific Known Title ===")
    
    processor = ScalableEnhancedProcessor()
    
    query = "information management"
    known_title = "RFP - Intelligence Application (Intel Apps) Information Collection Management (ICM) - W56KGY24R0002"
    known_id = "F19A27DC-E75B-4F76-B268-00C16CCFF02B"
    
    print(f"Query: '{query}'")
    print(f"Known title: '{known_title}'")
    print(f"Known ID: '{known_id}'")
    print()
    
    # First, let's see if we can find the specific title in the database
    print("=== Searching for the known title directly ===")
    try:
        # Search for the exact title text
        title_results = processor._search_collection_by_embedding(
            collection_name="opportunity_titles",
            embedding=processor.encode_with_pool([known_title], normalize_embeddings=True)[0],
            date_filter=None,
            limit=10
        )
        
        print(f"Self-search results: {len(title_results)}")
        if title_results:
            print(f"Top result: Score={title_results[0].distance:.4f}, ID={title_results[0].entity.get('opportunity_id')}")
        
    except Exception as e:
        print(f"Direct title search failed: {e}")
    
    print()
    
    # Now search with our query
    print("=== Searching with our query ===")
    try:
        query_embedding = processor.encode_with_pool([query], normalize_embeddings=True)[0]
        
        # Search titles
        title_results = processor._search_collection_by_embedding(
            collection_name="opportunity_titles",
            embedding=query_embedding,
            date_filter=None,
            limit=30
        )
        
        print(f"Query search results: {len(title_results)}")
        
        # Look for our specific ID
        found_our_title = False
        for i, result in enumerate(title_results):
            opp_id = result.entity.get('opportunity_id')
            score = result.distance
            
            if i < 10:  # Show top 10
                print(f"  {i+1}. Score: {score:.4f}, ID: {opp_id}")
                if 'Information' in result.entity.get('text_content', ''):
                    print(f"      Text: {result.entity.get('text_content', '')[:150]}...")
            
            if opp_id == known_id:
                found_our_title = True
                print(f"  *** FOUND OUR TARGET: Position {i+1}, Score: {score:.4f}")
                print(f"      Above 0.7 threshold? {score >= 0.7}")
                print(f"      Text: {result.entity.get('text_content', '')}")
        
        if not found_our_title:
            print(f"  *** TARGET ID {known_id} NOT FOUND in top {len(title_results)} results")
        
    except Exception as e:
        print(f"Query search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Manual similarity calculation
    print("=== Manual Similarity Calculation ===")
    try:
        query_emb = processor.encode_with_pool([query], normalize_embeddings=True)[0]
        title_emb = processor.encode_with_pool([known_title], normalize_embeddings=True)[0]
        
        import numpy as np
        similarity = np.dot(query_emb, title_emb)
        
        print(f"Manual similarity: {similarity:.4f}")
        print(f"Above 0.7 threshold? {similarity >= 0.7}")
        
    except Exception as e:
        print(f"Manual calculation failed: {e}")

if __name__ == "__main__":
    test_specific_title()
