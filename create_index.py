#!/usr/bin/env python3
"""
Create index for the vector database collection
"""

from pymilvus import connections, Collection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_index():
    """Create index for efficient searching"""
    try:
        # Connect to Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        logger.info("Connected to Milvus database")
        
        # Get collection
        collection = Collection("document_embeddings")
        
        # Create index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        logger.info("Index created successfully")
        
        # Load collection into memory
        collection.load()
        logger.info("Collection loaded into memory")
        
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        raise

if __name__ == "__main__":
    create_index()
