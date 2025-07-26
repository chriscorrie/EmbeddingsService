#!/usr/bin/env python3
"""
Enhanced search with exact phrase boosting for better precision
Updated to support FileId-based document architecture
"""

import re
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scalable_processor import ScalableEnhancedProcessor
import logging

# SQL Server imports for stored procedure calls
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False

# Import SQL connection configuration
from config import SQL_CONNECTION_STRING, SQL_GLOBAL_TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSearchProcessor:
    """Enhanced search processor with exact phrase boosting and FileId-based document support"""
    
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
        
        # Setup SQL connection for stored procedure calls
        self.setup_sql_connection()
    
    def setup_sql_connection(self):
        """Setup SQL Server connection for stored procedure calls"""
        try:
            if not PYODBC_AVAILABLE:
                logger.warning("SQL Server connection not available - pyodbc not installed")
                self.sql_conn = None
                return
                
            if SQL_CONNECTION_STRING and SQL_CONNECTION_STRING != 'your_connection_string_here':
                # Create connection string with timeout parameter
                connection_string = SQL_CONNECTION_STRING
                if 'timeout=' not in connection_string.lower():
                    connection_string += f";timeout={SQL_GLOBAL_TIMEOUT}"
                
                self.sql_conn = pyodbc.connect(connection_string)
                self.sql_conn.autocommit = True
                logger.info("Connected to SQL Server for stored procedure calls")
            else:
                logger.warning("SQL Server connection not configured for search processor")
                self.sql_conn = None
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server: {e}")
            self.sql_conn = None
    
    def get_file_opportunities(self, file_id: int, begin_posted_date: Optional[str] = None, 
                             end_posted_date: Optional[str] = None) -> List[str]:
        """
        Call FBOInternalAPI.GetEmbeddingFileOpportunities stored procedure
        
        Args:
            file_id: The FileId to lookup
            begin_posted_date: Optional start date filter (YYYY-MM-DD)
            end_posted_date: Optional end date filter (YYYY-MM-DD)
            
        Returns:
            List of OpportunityId strings (empty list if no matches or error)
        """
        if not self.sql_conn:
            logger.error("SQL connection not available for stored procedure call")
            return []
            
        try:
            cursor = self.sql_conn.cursor()
            
            # Call the stored procedure
            cursor.execute(
                "EXEC FBOInternalAPI.GetEmbeddingFileOpportunities ?, ?, ?",
                file_id, begin_posted_date, end_posted_date
            )
            
            # Fetch all results
            opportunity_ids = []
            for row in cursor.fetchall():
                opportunity_ids.append(row[0])  # OpportunityId is the only column
            
            cursor.close()
            return opportunity_ids
            
        except Exception as e:
            logger.error(f"Failed to call GetEmbeddingFileOpportunities for file_id {file_id}: {e}")
            return []
    
    def enhanced_similarity_search(self, query: str, limit: int = 10,
                                 base_threshold: float = 0.35,
                                 exact_phrase_boost: float = 0.3) -> List[Dict[str, Any]]:
        """
        Enhanced search with exact phrase matching and boosting
        Updated to support FileId-based document architecture
        
        Args:
            query: Search query
            limit: Maximum results
            base_threshold: Base similarity threshold
            exact_phrase_boost: Score boost for exact phrase matches
        """
        # Use the updated similarity search method
        results = self.search_similar_documents(
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
            
            # Check for exact phrase matches in documents (using FileId mapping)
            doc_boost = self._check_exact_phrase_in_documents_by_opportunity(
                opportunity_id, query_lower
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
            
            # Skip documents collection - use separate method for FileId-based search
            if collection_key == 'documents':
                return self._check_exact_phrase_in_documents_by_opportunity(opportunity_id, query_lower)
            
            # Query all chunks for this opportunity (titles/descriptions only)
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
    
    def _check_exact_phrase_in_documents_by_opportunity(self, opportunity_id: str, query_lower: str) -> bool:
        """
        Check if exact phrase exists in documents for given opportunity
        This method handles the FileId-based document architecture
        """
        try:
            # Get all file_ids associated with this opportunity (no date filter for phrase matching)
            file_opportunity_ids = self.get_file_opportunities_by_opportunity_reverse_lookup(opportunity_id)
            
            if not file_opportunity_ids:
                return False
            
            collection = self.collections['documents']
            
            # Search in all files associated with this opportunity
            for file_id in file_opportunity_ids:
                results = collection.query(
                    expr=f'file_id == {file_id}',
                    output_fields=["text_content"],
                    limit=100
                )
                
                # Check each chunk for exact phrase
                for result in results:
                    text_content = result.get('text_content', '').lower()
                    if query_lower in text_content:
                        return True
                        
            return False
            
        except Exception as e:
            logger.warning(f"Error checking exact phrase in documents for opportunity {opportunity_id}: {e}")
            return False
    
    def get_file_opportunities_by_opportunity_reverse_lookup(self, opportunity_id: str) -> List[int]:
        """
        Reverse lookup to get file_ids associated with an opportunity_id
        This is a simplified implementation - in practice, you might want to cache this or optimize
        """
        try:
            # This is a workaround since we need to find file_ids for a given opportunity_id
            # In a full implementation, you might want a dedicated stored procedure for this
            # For now, we'll search the documents collection to find file_ids for this opportunity
            
            # Note: This method is primarily for exact phrase matching which is less critical
            # The main search functionality uses the proper FileId -> OpportunityId mapping
            return []  # Simplified implementation - exact phrase matching on documents will be limited
            
        except Exception as e:
            logger.warning(f"Error in reverse lookup for opportunity {opportunity_id}: {e}")
            return []
    
    def search_similar_documents(self, query: str, limit: int = 10, boost_factor: float = 1.0, 
                               include_entities: bool = False, 
                               title_similarity_threshold: float = 0.5,
                               description_similarity_threshold: float = 0.5,
                               document_similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Updated similarity search that handles FileId-based document architecture
        This replaces the processor's search_similar_documents method
        """
        try:
            logger.info(f"Starting FileId-based similarity search for query: '{query}' (limit: {limit})")
            
            # Generate embedding for the query
            query_embedding = self.processor.encode_with_pool([query], normalize_embeddings=True)[0]
            
            # Collect all similar opportunities across all collections
            similar_opportunities = {}
            
            # Search titles and descriptions (unchanged - still OpportunityId-based)
            self._search_titles_descriptions(query_embedding, similar_opportunities, 
                                           title_similarity_threshold, description_similarity_threshold, limit)
            
            # Search documents with FileId-based logic
            self._search_documents_with_fileid_mapping(query_embedding, similar_opportunities,
                                                     document_similarity_threshold, limit)
            
            # Aggregate and format results
            final_results = self._aggregate_and_format_results(similar_opportunities, limit)
            
            logger.info(f"FileId-based similarity search completed: {len(final_results)} results found")
            return final_results
            
        except Exception as e:
            logger.error(f"FileId-based similarity search failed: {e}")
            return []
    
    def search_similar_opportunities(self, opportunity_ids: List[str],
                                   title_similarity_threshold: float = 0.5,
                                   description_similarity_threshold: float = 0.5,
                                   document_similarity_threshold: float = 0.5,
                                   start_posted_date: Optional[str] = None,
                                   end_posted_date: Optional[str] = None,
                                   document_sow_boost_multiplier: float = 0.0,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Updated opportunity search that handles FileId-based document architecture with date filtering
        This replaces the processor's search_similar_opportunities method
        """
        try:
            logger.info(f"Starting FileId-based opportunity search for {len(opportunity_ids)} opportunities")
            
            # Get embeddings for input opportunities (unchanged for titles/descriptions)
            input_embeddings = self.processor._get_opportunity_embeddings(opportunity_ids)
            
            # Collect all similar opportunities
            similar_opportunities = {}
            
            # For each input opportunity, find similar ones
            for input_opp_id, embeddings in input_embeddings.items():
                if not any([embeddings.get('title'), embeddings.get('description'), embeddings.get('documents')]):
                    logger.warning(f"No embeddings found for opportunity {input_opp_id}")
                    continue
                
                # Search titles and descriptions (unchanged - still OpportunityId-based)
                if embeddings.get('title'):
                    self._search_collection_by_embedding_simple(
                        'opportunity_titles', embeddings['title']['embedding'],
                        similar_opportunities, 'title_score', title_similarity_threshold, limit
                    )
                
                if embeddings.get('description'):
                    self._search_collection_by_embedding_simple(
                        'opportunity_descriptions', embeddings['description']['embedding'],
                        similar_opportunities, 'description_score', description_similarity_threshold, limit
                    )
                
                # Search documents with FileId-based logic and date filtering
                if embeddings.get('documents'):
                    for doc_embedding in embeddings['documents']:
                        self._search_documents_with_fileid_mapping_and_dates(
                            doc_embedding['embedding'], similar_opportunities,
                            document_similarity_threshold, start_posted_date, end_posted_date,
                            document_sow_boost_multiplier, doc_embedding.get('boost_factor', 1.0), limit
                        )
            
            # Aggregate and format results
            final_results = self._aggregate_and_format_results(similar_opportunities, limit)
            
            # Remove input opportunities from results
            final_results = [r for r in final_results if r['opportunity_id'] not in opportunity_ids][:limit]
            
            logger.info(f"FileId-based opportunity search completed: {len(final_results)} results found")
            return final_results
            
        except Exception as e:
            logger.error(f"FileId-based opportunity search failed: {e}")
            return []
    
    def _search_titles_descriptions(self, query_embedding: List[float], similar_opportunities: Dict,
                                  title_threshold: float, description_threshold: float, limit: int):
        """Search titles and descriptions collections (unchanged logic)"""
        from concurrent.futures import ThreadPoolExecutor
        
        def search_collection(collection_key, threshold, score_key):
            try:
                collection = self.collections[collection_key]
                collection.load()
                
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 20}}
                
                results = collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=limit * 3,
                    output_fields=["opportunity_id", "posted_date", "importance_score", 
                                 "chunk_index", "total_chunks"]
                )
                
                self._process_search_results_simple(results[0], similar_opportunities, score_key, threshold)
                
            except Exception as e:
                logger.warning(f"Search failed for collection {collection_key}: {e}")
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(search_collection, 'titles', title_threshold, 'title_score')
            executor.submit(search_collection, 'descriptions', description_threshold, 'description_score')
    
    def _search_documents_with_fileid_mapping(self, query_embedding: List[float], similar_opportunities: Dict,
                                            threshold: float, limit: int):
        """Search documents collection with FileId to OpportunityId mapping (no date filter)"""
        try:
            collection = self.collections['documents']
            collection.load()
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 20}}
            
            # Search the opportunity_documents collection (now FileId-based)
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit * 3,
                output_fields=["file_id", "min_posted_date", "max_posted_date", "base_importance",
                             "chunk_index", "total_chunks", "file_location", "section_type"]
            )
            
            # Process results and map FileId to OpportunityId
            for hit in results[0]:
                if hit.distance < threshold:
                    continue
                
                file_id = hit.entity.get('file_id')
                if not file_id:
                    continue
                
                # Map FileId to OpportunityIds (no date filter for similarity search)
                opportunity_ids = self.get_file_opportunities(file_id)
                
                # Add score for each associated opportunity
                for opp_id in opportunity_ids:
                    if opp_id not in similar_opportunities:
                        similar_opportunities[opp_id] = {
                            'title_scores': [],
                            'description_scores': [],
                            'document_scores': []
                        }
                    similar_opportunities[opp_id]['document_scores'].append(hit.distance)
                    
        except Exception as e:
            logger.warning(f"Search failed for documents collection: {e}")
    
    def _search_documents_with_fileid_mapping_and_dates(self, doc_embedding: List[float], similar_opportunities: Dict,
                                                      threshold: float, start_date: Optional[str], end_date: Optional[str],
                                                      boost_multiplier: float, boost_factor: float, limit: int):
        """Search documents collection with FileId to OpportunityId mapping and date filtering"""
        try:
            collection = self.collections['documents']
            collection.load()
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 20}}
            
            # Build date filter for FileId-based schema
            date_filter = self._build_fileid_date_filter(start_date, end_date)
            
            # Search the opportunity_documents collection (now FileId-based)
            results = collection.search(
                data=[doc_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit * 3,
                expr=date_filter,
                output_fields=["file_id", "min_posted_date", "max_posted_date", "base_importance",
                             "chunk_index", "total_chunks", "file_location", "section_type"]
            )
            
            # Process results and map FileId to OpportunityId with date filtering
            for hit in results[0]:
                if hit.distance < threshold:
                    continue
                
                file_id = hit.entity.get('file_id')
                if not file_id:
                    continue
                
                # Map FileId to OpportunityIds with date filtering
                opportunity_ids = self.get_file_opportunities(file_id, start_date, end_date)
                
                # Calculate score (with boost if applicable)
                score = hit.distance
                if boost_multiplier > 0.0:
                    score = self._calculate_boosted_score(score, boost_factor, boost_multiplier)
                
                # Add score for each associated opportunity
                for opp_id in opportunity_ids:
                    if opp_id not in similar_opportunities:
                        similar_opportunities[opp_id] = {
                            'title_scores': [],
                            'description_scores': [],
                            'document_scores': []
                        }
                    similar_opportunities[opp_id]['document_scores'].append(score)
                    
        except Exception as e:
            logger.warning(f"Search failed for documents collection with date filter: {e}")
    
    def _build_fileid_date_filter(self, start_date: Optional[str], end_date: Optional[str]) -> Optional[str]:
        """Build date filter for FileId-based schema using min/max posted dates"""
        if not start_date and not end_date:
            return None
        
        filters = []
        if start_date and end_date:
            # File date range overlaps with search range
            filters.append(f"(min_posted_date <= '{end_date}' and max_posted_date >= '{start_date}')")
        elif start_date:
            # File max date is after start date
            filters.append(f"max_posted_date >= '{start_date}'")
        elif end_date:
            # File min date is before end date
            filters.append(f"min_posted_date <= '{end_date}'")
        
        return " and ".join(filters)
    
    def _search_collection_by_embedding_simple(self, collection_name: str, embedding: List[float],
                                             similar_opportunities: Dict, score_key: str, threshold: float, limit: int):
        """Simple search for titles/descriptions collections"""
        try:
            collection = self.collections[collection_name.replace('opportunity_', '')]
            collection.load()
            
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["opportunity_id", "posted_date", "importance_score"]
            )
            
            self._process_search_results_simple(results[0], similar_opportunities, score_key, threshold)
            
        except Exception as e:
            logger.warning(f"Search failed for collection {collection_name}: {e}")
    
    def _process_search_results_simple(self, search_results: List[Any], similar_opportunities: Dict,
                                     score_key: str, threshold: float):
        """Process search results and collect scores"""
        for hit in search_results:
            opp_id = hit.entity.get('opportunity_id')
            if not opp_id:
                continue
                
            score = hit.distance  # For COSINE, distance IS the similarity score
            if score < threshold:
                continue
                
            if opp_id not in similar_opportunities:
                similar_opportunities[opp_id] = {
                    'title_scores': [],
                    'description_scores': [],
                    'document_scores': []
                }
                
            if score_key == 'title_score':
                similar_opportunities[opp_id]['title_scores'].append(score)
            elif score_key == 'description_score':
                similar_opportunities[opp_id]['description_scores'].append(score)
            elif score_key == 'document_score':
                similar_opportunities[opp_id]['document_scores'].append(score)
    
    def _aggregate_and_format_results(self, similar_opportunities: Dict, limit: int) -> List[Dict[str, Any]]:
        """Aggregate scores and format final results"""
        final_results = []
        
        for opp_id, score_data in similar_opportunities.items():
            title_scores = score_data.get('title_scores', [])
            description_scores = score_data.get('description_scores', [])
            document_scores = score_data.get('document_scores', [])
            
            # Aggregate scores using the same logic as the original processor
            title_score = max(title_scores) if title_scores else 0.0
            
            desc_scores_sorted = sorted(description_scores, reverse=True)[:3]
            description_score = sum(desc_scores_sorted) / len(desc_scores_sorted) if desc_scores_sorted else 0.0
            
            doc_scores_sorted = sorted(document_scores, reverse=True)[:5]
            base_doc_score = sum(doc_scores_sorted) / len(doc_scores_sorted) if doc_scores_sorted else 0.0
            
            # Bonus for multiple matches (same logic as original)
            if len(doc_scores_sorted) > 1:
                import math
                count_bonus = math.log(len(doc_scores_sorted)) * 0.1
                document_score = min(1.0, base_doc_score + count_bonus)
            else:
                document_score = base_doc_score
            
            result = {
                'opportunity_id': opp_id,
                'title_score': title_score,
                'description_score': description_score,
                'document_score': document_score,
                'combined_score': title_score + description_score + document_score,
                'document_match_count': len(document_scores),
                'title_match_count': len(title_scores),
                'description_match_count': len(description_scores)
            }
            final_results.append(result)
        
        # Sort by combined score and limit results
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return final_results[:limit]
    
    def _calculate_boosted_score(self, unboosted_score: float, boost_factor: float, 
                               boost_multiplier: float) -> float:
        """Calculate boosted score (same formula as original processor)"""
        if boost_multiplier == 0.0:
            return unboosted_score
            
        boosted_score = unboosted_score + (unboosted_score * (boost_factor - 1) * boost_multiplier)
        return min(boosted_score, 1.0)

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
