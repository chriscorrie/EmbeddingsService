#!/usr/bin/env python3
"""
Enhanced search with exact phrase boosting for better precision
"""

import re
from typing import List, Dict, Any
from pymilvus import connections, Collection
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scalable_processor import ScalableEnhancedProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSearchProcessor:
    """Enhanced search processor with exact phrase boosting"""
    
    def __init__(self):
        self.processor = ScalableEnhancedProcessor()
        connections.connect(alias="default", host="localhost", port="19530")
        
        # Load collections
        self.collections = {
            'titles': Collection("opportunity_titles"),
            'descriptions': Collection("opportunity_descriptions"),
            'documents': Collection("opportunity_documents")
        }
        
        for collection in self.collections.values():
            collection.load()
    
    def enhanced_similarity_search(self, query: str, limit: int = 10,
                                 base_threshold: float = 0.35,
                                 exact_phrase_boost: float = 0.3) -> List[Dict[str, Any]]:
        """
        Enhanced search with exact phrase matching and boosting
        
        Args:
            query: Search query
            limit: Maximum results
            base_threshold: Base similarity threshold
            exact_phrase_boost: Score boost for exact phrase matches
        """
        # First, perform standard semantic search with lower threshold
        results = self.processor.search_similar_documents(
            query=query,
            limit=limit * 2,  # Get more results for filtering
            title_similarity_threshold=base_threshold,
            description_similarity_threshold=base_threshold,
            document_similarity_threshold=base_threshold
        )
        
        # Now apply exact phrase boosting
        enhanced_results = []
        query_lower = query.lower()
        
        for result in results:
            opportunity_id = result['opportunity_id']
            enhanced_result = result.copy()
            
            # Check for exact phrase matches in titles
            title_boost = self._check_exact_phrase_in_collection(
                opportunity_id, query_lower, 'titles'
            )
            
            # Check for exact phrase matches in descriptions  
            desc_boost = self._check_exact_phrase_in_collection(
                opportunity_id, query_lower, 'descriptions'
            )
            
            # Check for exact phrase matches in documents
            doc_boost = self._check_exact_phrase_in_collection(
                opportunity_id, query_lower, 'documents'
            )
            
            # Apply boosts
            if title_boost:
                enhanced_result['title_score'] = min(1.0, 
                    enhanced_result['title_score'] + exact_phrase_boost)
                enhanced_result['exact_title_match'] = True
            else:
                enhanced_result['exact_title_match'] = False
                
            if desc_boost:
                enhanced_result['description_score'] = min(1.0,
                    enhanced_result['description_score'] + exact_phrase_boost)
                enhanced_result['exact_description_match'] = True  
            else:
                enhanced_result['exact_description_match'] = False
                
            if doc_boost:
                enhanced_result['document_score'] = min(1.0,
                    enhanced_result['document_score'] + exact_phrase_boost)
                enhanced_result['exact_document_match'] = True
            else:
                enhanced_result['exact_document_match'] = False
                
            # Recalculate combined score
            enhanced_result['combined_score'] = (
                enhanced_result['title_score'] + 
                enhanced_result['description_score'] + 
                enhanced_result['document_score']
            )
            
            enhanced_results.append(enhanced_result)
        
        # Sort by enhanced combined score
        enhanced_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return enhanced_results[:limit]
    
    def _check_exact_phrase_in_collection(self, opportunity_id: str, 
                                        query_lower: str, 
                                        collection_key: str) -> bool:
        """Check if exact phrase exists in collection for given opportunity"""
        try:
            collection = self.collections[collection_key]
            
            # Query all chunks for this opportunity
            if collection_key == 'documents':
                results = collection.query(
                    expr=f'opportunity_id == "{opportunity_id}"',
                    output_fields=["text_content"],
                    limit=100
                )
            else:
                results = collection.query(
                    expr=f'opportunity_id == "{opportunity_id}"',
                    output_fields=["text_content"], 
                    limit=10
                )
            
            # Check each chunk for exact phrase
            for result in results:
                text_content = result.get('text_content', '').lower()
                if query_lower in text_content:
                    return True
                    
            return False
            
        except Exception as e:
            logger.warning(f"Error checking exact phrase in {collection_key}: {e}")
            return False

def test_enhanced_search():
    """Test the enhanced search functionality"""
    enhancer = EnhancedSearchProcessor()
    
    query = "information collection management"
    
    print(f"Testing enhanced search for: '{query}'")
    print("="*60)
    
    # Test with different configurations
    configs = [
        {"base_threshold": 0.35, "exact_phrase_boost": 0.3, "name": "Moderate Boost"},
        {"base_threshold": 0.35, "exact_phrase_boost": 0.5, "name": "High Boost"},
        {"base_threshold": 0.4, "exact_phrase_boost": 0.3, "name": "Higher Threshold + Moderate Boost"}
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        print(f"Base threshold: {config['base_threshold']}, Boost: {config['exact_phrase_boost']}")
        
        results = enhancer.enhanced_similarity_search(
            query=query,
            limit=5,
            base_threshold=config['base_threshold'],
            exact_phrase_boost=config['exact_phrase_boost']
        )
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            exact_matches = []
            if result.get('exact_title_match'):
                exact_matches.append('Title')
            if result.get('exact_description_match'):
                exact_matches.append('Description') 
            if result.get('exact_document_match'):
                exact_matches.append('Document')
                
            exact_str = f" [EXACT: {', '.join(exact_matches)}]" if exact_matches else ""
            
            print(f"  {i}. Score: {result['combined_score']:.4f} - "
                  f"T:{result['title_score']:.3f} "
                  f"D:{result['description_score']:.3f} "
                  f"Doc:{result['document_score']:.3f} "
                  f"- {result['opportunity_id']}{exact_str}")

if __name__ == "__main__":
    test_enhanced_search()
