#!/usr/bin/env python3
"""
Simple debug to see what's happening in the search aggregation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

from scalable_processor import ScalableEnhancedProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_aggregation():
    """Debug the search aggregation process"""
    
    print("=== Debugging Search Aggregation ===")
    
    processor = ScalableEnhancedProcessor()
    
    # Test parameters matching the API call
    query = "information management"
    limit = 10
    title_threshold = 0.7
    description_threshold = 0.7
    document_threshold = 0.7
    boost_factor = 1
    
    print(f"Query: '{query}'")
    print(f"Thresholds: title={title_threshold}, desc={description_threshold}, doc={document_threshold}")
    print()
    
    # Generate embedding
    query_embedding = processor.encode_with_pool([query], normalize_embeddings=True)[0]
    
    # Search titles directly
    print("=== Title Collection Search ===")
    try:
        title_results = processor._search_collection_by_embedding(
            collection_name="opportunity_titles",
            embedding=query_embedding,
            date_filter=None,
            limit=30  # Get more to see the threshold effect
        )
        print(f"Raw title results: {len(title_results)}")
        for i, result in enumerate(title_results[:5]):
            score = result.distance  # For COSINE, distance IS the similarity
            print(f"  {i+1}. Score: {score:.4f}, Above 0.7? {score >= 0.7}, ID: {result.entity.get('opportunity_id')}")
            print(f"      Text: {result.entity.get('text_content', '')[:100]}...")
    except Exception as e:
        print(f"Title search failed: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Check what the aggregation function does
    print("=== Testing Aggregation Logic ===")
    
    # Simulate the task structure
    tasks = [
        {"collection": "opportunity_titles", "threshold": title_threshold},
        {"collection": "opportunity_descriptions", "threshold": description_threshold}, 
        {"collection": "opportunity_documents", "threshold": document_threshold}
    ]
    
    all_scores = []
    
    for task in tasks:
        try:
            results = processor._search_collection_by_embedding(
                collection_name=task["collection"],
                embedding=query_embedding,
                date_filter=None,
                limit=30
            )
            
            print(f"=== {task['collection']} ===")
            print(f"Raw results: {len(results)}")
            
            # Filter by threshold
            filtered_results = [r for r in results if r.distance >= task["threshold"]]
            print(f"Above threshold ({task['threshold']}): {len(filtered_results)}")
            
            if filtered_results:
                print("Top filtered results:")
                for i, result in enumerate(filtered_results[:3]):
                    score = result.distance
                    opp_id = result.entity.get('opportunity_id')
                    print(f"  {i+1}. Score: {score:.4f}, ID: {opp_id}")
                    
                    # Add to all_scores for aggregation
                    all_scores.append({
                        'opportunity_id': opp_id,
                        'collection': task["collection"],
                        'score': score,
                        'distance': score,
                        'entity': result.entity
                    })
            else:
                print("No results above threshold!")
                
        except Exception as e:
            print(f"Error searching {task['collection']}: {e}")
    
    print()
    print(f"=== Total Scores Collected: {len(all_scores)} ===")
    
    # Group by opportunity_id 
    opportunity_scores = {}
    for score_data in all_scores:
        opp_id = score_data['opportunity_id']
        if opp_id not in opportunity_scores:
            opportunity_scores[opp_id] = {
                'opportunity_id': opp_id,
                'title_scores': [],
                'description_scores': [],
                'document_scores': []
            }
        
        collection = score_data['collection']
        if 'titles' in collection:
            opportunity_scores[opp_id]['title_scores'].append(score_data['score'])
        elif 'descriptions' in collection:
            opportunity_scores[opp_id]['description_scores'].append(score_data['score'])
        elif 'documents' in collection:
            opportunity_scores[opp_id]['document_scores'].append(score_data['score'])
    
    print(f"Opportunities with scores: {len(opportunity_scores)}")
    for opp_id, scores in opportunity_scores.items():
        print(f"  {opp_id}: titles={len(scores['title_scores'])}, desc={len(scores['description_scores'])}, docs={len(scores['document_scores'])}")
        if scores['title_scores']:
            print(f"    Title scores: {scores['title_scores'][:3]}")
        if scores['description_scores']:
            print(f"    Desc scores: {scores['description_scores'][:3]}")
        if scores['document_scores']:
            print(f"    Doc scores: {scores['document_scores'][:3]}")

if __name__ == "__main__":
    debug_aggregation()
