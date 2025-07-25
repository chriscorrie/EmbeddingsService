#!/usr/bin/env python3
"""
Test the impact of different thresholds on search results
"""

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_threshold_impact():
    """Test how different thresholds affect search results"""
    try:
        # Connect to Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        
        # Initialize embedding model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test query
        query_text = "information collection management"
        query_embedding = model.encode([query_text]).tolist()
        
        # Get title collection
        titles_collection = Collection("opportunity_titles")
        titles_collection.load()
        
        print(f"Testing query: '{query_text}'")
        print("="*60)
        
        # Test different thresholds
        thresholds = [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
        
        for threshold in thresholds:
            print(f"\n🔍 Testing threshold: {threshold}")
            
            # Search with current threshold
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = titles_collection.search(
                data=query_embedding,
                anns_field="embedding", 
                param=search_params,
                limit=10,
                expr=None,
                output_fields=["opportunity_id", "text_content"]
            )
            
            # Filter by threshold and show results
            filtered_results = []
            all_results = []
            
            for hits in results:
                for hit in hits:
                    all_results.append((hit.score, hit.entity.get("text_content", "N/A")))
                    if hit.score >= threshold:
                        filtered_results.append((hit.score, hit.entity.get("text_content", "N/A")))
            
            print(f"   Total raw results: {len(all_results)}")
            print(f"   Results above threshold {threshold}: {len(filtered_results)}")
            
            # Show top 3 results regardless of threshold
            print(f"   Top 3 raw scores:")
            for i, (score, content) in enumerate(all_results[:3]):
                content_preview = content[:80] + "..." if len(content) > 80 else content
                threshold_status = "✅" if score >= threshold else "❌"
                print(f"     {i+1}. {score:.4f} {threshold_status} {content_preview}")
            
            # Show results that pass threshold
            if filtered_results:
                print(f"   Results passing threshold:")
                for i, (score, content) in enumerate(filtered_results[:3]):
                    content_preview = content[:80] + "..." if len(content) > 80 else content
                    print(f"     {i+1}. {score:.4f} ✅ {content_preview}")
            else:
                print(f"   ❌ No results pass threshold {threshold}")
        
        print("\n" + "="*60)
        print("Summary: Check which threshold allows 'Information Collection Management' to be found")
        
    except Exception as e:
        logger.error(f"Error testing thresholds: {e}")
        raise

if __name__ == "__main__":
    test_threshold_impact()
