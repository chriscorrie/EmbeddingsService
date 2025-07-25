#!/usr/bin/env python3
"""
Debug the actual API search to see what's happening
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from scalable_processor import ScalableEnhancedProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_search():
    """Debug the search functionality step by step"""
    
    print("=== Debugging API Search for 'information management' ===")
    
    processor = ScalableEnhancedProcessor()
    
    # Test parameters matching the API call
    query = "information management"
    limit = 10
    title_threshold = 0.7
    description_threshold = 0.7
    document_threshold = 0.7
    boost_factor = 1
    
    print(f"Query: '{query}'")
    print(f"Title threshold: {title_threshold}")
    print(f"Description threshold: {description_threshold}")
    print(f"Document threshold: {document_threshold}")
    print(f"Limit: {limit}")
    print()
    
    # Step 1: Check if we can find the title directly
    print("=== Step 1: Direct title search ===")
    try:
        query_embedding = processor.encode_with_pool([query], normalize_embeddings=True)[0]
        title_results = processor._search_collection_by_embedding(
            collection_name="opportunity_titles",
            embedding=query_embedding,
            limit=limit,
            similarity_threshold=title_threshold
        )
        print(f"Title search returned {len(title_results)} results")
        for i, result in enumerate(title_results):
            print(f"  {i+1}. Score: {result.distance:.4f}, ID: {result.entity.get('opportunity_id')}, Text: {result.entity.get('text_content', '')[:100]}...")
    except Exception as e:
        print(f"Title search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Step 2: Check descriptions
    print("=== Step 2: Description search ===")
    try:
        query_embedding = processor.encode_with_pool([query], normalize_embeddings=True)[0]
        desc_results = processor._search_collection_by_embedding(
            collection_name="opportunity_descriptions",
            embedding=query_embedding,
            limit=limit,
            similarity_threshold=description_threshold
        )
        print(f"Description search returned {len(desc_results)} results")
        for i, result in enumerate(desc_results):
            print(f"  {i+1}. Score: {result.distance:.4f}, ID: {result.entity.get('opportunity_id')}, Text: {result.entity.get('text_content', '')[:100]}...")
    except Exception as e:
        print(f"Description search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Step 3: Check documents
    print("=== Step 3: Document search ===")
    try:
        query_embedding = processor.encode_with_pool([query], normalize_embeddings=True)[0]
        doc_results = processor._search_collection_by_embedding(
            collection_name="opportunity_documents",
            embedding=query_embedding,
            limit=limit,
            similarity_threshold=document_threshold
        )
        print(f"Document search returned {len(doc_results)} results")
        for i, result in enumerate(doc_results[:5]):  # Show first 5
            print(f"  {i+1}. Score: {result.distance:.4f}, ID: {result.entity.get('opportunity_id')}, Text: {result.entity.get('text_content', '')[:100]}...")
    except Exception as e:
        print(f"Document search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Step 4: Full similarity search
    print("=== Step 4: Full similarity search ===")
    try:
        results = processor.search_similar_documents(
            query=query,
            limit=limit,
            title_similarity_threshold=title_threshold,
            description_similarity_threshold=description_threshold,
            document_similarity_threshold=document_threshold,
            boost_factor=boost_factor
        )
        print(f"Full search returned {len(results)} results")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result['aggregated_score']:.4f}, ID: {result['opportunity_id']}")
            print(f"     Title score: {result.get('title_score', 'N/A')}")
            print(f"     Desc score: {result.get('description_score', 'N/A')}")
            print(f"     Doc score: {result.get('document_score', 'N/A')}")
    except Exception as e:
        print(f"Full search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_search()
