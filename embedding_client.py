#!/usr/bin/env python3
"""
Python client library for the Enhanced Document Embedding REST API
Provides easy-to-use methods for all API endpoints
"""

import requests
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbeddingClient:
    """Client for Document Embedding REST API"""
    
    def __init__(self, base_url: str = "http://localhost:5000/api/v1", timeout: int = 30):
        """
        Initialize the client
        
        Args:
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API service is healthy
        
        Returns:
            Health status information
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def similarity_search(self, 
                         query: str,
                         search_type: str = "both",
                         limit: int = 10,
                         boost_multiplier: float = 1.0,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform similarity search
        
        Args:
            query: Search query text
            search_type: "titles", "descriptions", or "both"
            limit: Maximum number of results
            boost_multiplier: Multiplier for section-based boosting
            start_date: Filter results from this date (YYYY-MM-DD)
            end_date: Filter results to this date (YYYY-MM-DD)
            
        Returns:
            Search results with similarity scores
        """
        data = {
            "query": query,
            "search_type": search_type,
            "limit": limit,
            "boost_multiplier": boost_multiplier
        }
        
        if start_date:
            data["start_date"] = start_date
        if end_date:
            data["end_date"] = end_date
        
        try:
            response = self.session.post(f"{self.base_url}/similarity-search", 
                                       json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def create_embeddings(self,
                         start_row_id: int = 1,
                         end_row_id: int = 1000,
                         batch_size: int = 100,
                         force_restart: bool = False) -> Dict[str, Any]:
        """
        Start asynchronous embedding generation
        
        Args:
            start_row_id: Starting opportunity ID
            end_row_id: Ending opportunity ID
            batch_size: Number of opportunities per batch
            force_restart: Force restart if processing is already running
            
        Returns:
            Request information with request_id
        """
        data = {
            "start_row_id": start_row_id,
            "end_row_id": end_row_id,
            "batch_size": batch_size,
            "force_restart": force_restart
        }
        
        try:
            response = self.session.post(f"{self.base_url}/create-embeddings", 
                                       json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Create embeddings failed: {e}")
            raise
    
    def get_processing_status(self) -> Dict[str, Any]:
        """
        Get current processing status
        
        Returns:
            Processing status information
        """
        try:
            response = self.session.get(f"{self.base_url}/processing-status", 
                                      timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Get processing status failed: {e}")
            raise
    
    def wait_for_completion(self, 
                           max_wait_time: int = 3600,
                           check_interval: int = 5,
                           callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Wait for embedding generation to complete
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            check_interval: Time between status checks in seconds
            callback: Optional callback function called on each status update
            
        Returns:
            Final processing status
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.get_processing_status()
            
            if callback:
                callback(status)
            
            if not status['is_processing']:
                logger.info("Processing completed!")
                return status
            
            time.sleep(check_interval)
        
        logger.warning(f"Timeout after {max_wait_time} seconds")
        return self.get_processing_status()
    
    def search_with_boost(self, 
                         query: str, 
                         boost_multiplier: float = 2.0,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Convenience method for search with boosting
        
        Args:
            query: Search query
            boost_multiplier: Boost multiplier for section importance
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        result = self.similarity_search(
            query=query,
            boost_multiplier=boost_multiplier,
            limit=limit
        )
        return result.get('results', [])
    
    def process_embeddings_and_wait(self,
                                   start_row_id: int = 1,
                                   end_row_id: int = 1000,
                                   batch_size: int = 100,
                                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Start embedding generation and wait for completion
        
        Args:
            start_row_id: Starting opportunity ID
            end_row_id: Ending opportunity ID
            batch_size: Batch size for processing
            progress_callback: Optional callback for progress updates
            
        Returns:
            Final processing status
        """
        # Start embedding generation
        result = self.create_embeddings(
            start_row_id=start_row_id,
            end_row_id=end_row_id,
            batch_size=batch_size
        )
        
        logger.info(f"Started embedding generation with request ID: {result['request_id']}")
        
        # Wait for completion
        return self.wait_for_completion(callback=progress_callback)
    
    def get_errors(self) -> List[Dict[str, Any]]:
        """
        Get processing errors from current or last run
        
        Returns:
            List of error information
        """
        status = self.get_processing_status()
        return status.get('errors', [])
    
    def is_processing(self) -> bool:
        """
        Check if embedding generation is currently running
        
        Returns:
            True if processing is active
        """
        status = self.get_processing_status()
        return status.get('is_processing', False)
    
    def get_progress(self) -> float:
        """
        Get current processing progress percentage
        
        Returns:
            Progress percentage (0.0 - 100.0)
        """
        status = self.get_processing_status()
        return status.get('progress_percentage', 0.0)

# Example usage and utility functions
def example_usage():
    """Example usage of the client library"""
    
    # Initialize client
    client = DocumentEmbeddingClient()
    
    # Check health
    try:
        health = client.health_check()
        print(f"API Health: {health['status']}")
    except Exception as e:
        print(f"API not available: {e}")
        return
    
    # Perform similarity search
    try:
        results = client.similarity_search(
            query="cloud computing services",
            boost_multiplier=1.5,
            limit=5
        )
        print(f"Found {len(results['results'])} results")
        
        for result in results['results']:
            print(f"ID: {result['id']}, Score: {result['score']:.4f}")
    except Exception as e:
        print(f"Search failed: {e}")
    
    # Start embedding generation
    try:
        if not client.is_processing():
            embedding_result = client.create_embeddings(
                start_row_id=1,
                end_row_id=10,
                batch_size=5
            )
            print(f"Started embedding generation: {embedding_result['request_id']}")
            
            # Monitor progress
            def progress_callback(status):
                print(f"Progress: {status['progress_percentage']:.1f}% "
                      f"({status['processed_opportunities']}/{status['total_opportunities']})")
            
            final_status = client.wait_for_completion(
                max_wait_time=300,
                callback=progress_callback
            )
            
            print(f"Final status: {final_status['processed_opportunities']} processed")
            
            # Check for errors
            errors = client.get_errors()
            if errors:
                print(f"Processing errors: {len(errors)}")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"  - {error['opportunity_id']}: {error['error_message']}")
        else:
            print("Embedding generation already in progress")
    except Exception as e:
        print(f"Embedding generation failed: {e}")

if __name__ == '__main__':
    example_usage()
