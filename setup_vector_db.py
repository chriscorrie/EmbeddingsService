#!/usr/bin/env python3
"""
Enhanced vector database setup for v3 API with search functionality
"""

from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_vector_database():
    """Setup vector database with collections needed for search functionality"""
    try:
        connections.connect(alias="default", host="localhost", port="19530")
        logger.info("Connected to Milvus database")
        
        # Create collections for different content types
        collections_config = {
            "opportunity_titles": {
                "description": "Opportunity Title Embeddings with Chunking",
                "fields": [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="opportunity_id", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
                    FieldSchema(name="posted_date", dtype=DataType.INT64),  # Unix timestamp in milliseconds
                    FieldSchema(name="importance_score", dtype=DataType.FLOAT),
                    FieldSchema(name="chunk_index", dtype=DataType.INT32),
                    FieldSchema(name="total_chunks", dtype=DataType.INT32),
                    FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=2000),
                ]
            },
            "opportunity_descriptions": {
                "description": "Opportunity Description Embeddings with Chunking", 
                "fields": [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="opportunity_id", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
                    FieldSchema(name="posted_date", dtype=DataType.INT64),  # Unix timestamp in milliseconds
                    FieldSchema(name="importance_score", dtype=DataType.FLOAT),
                    FieldSchema(name="chunk_index", dtype=DataType.INT32),
                    FieldSchema(name="total_chunks", dtype=DataType.INT32),
                    FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=2000),
                ]
            },
            "opportunity_documents": {
                "description": "Opportunity Document Embeddings with File ID-based Deduplication",
                "fields": [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="file_id", dtype=DataType.INT64),
                    FieldSchema(name="min_posted_date", dtype=DataType.INT64),  # Unix timestamp in milliseconds
                    FieldSchema(name="max_posted_date", dtype=DataType.INT64),  # Unix timestamp in milliseconds
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
                    FieldSchema(name="base_importance", dtype=DataType.FLOAT),
                    FieldSchema(name="chunk_index", dtype=DataType.INT32),
                    FieldSchema(name="total_chunks", dtype=DataType.INT32),
                    FieldSchema(name="text_content", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="file_location", dtype=DataType.VARCHAR, max_length=500),
                    FieldSchema(name="section_type", dtype=DataType.VARCHAR, max_length=50),
                ]
            }
        }
        
        created_collections = {}
        
        for collection_name, config in collections_config.items():
            try:
                # Drop existing collection if it exists
                if utility.has_collection(collection_name):
                    utility.drop_collection(collection_name)
                    logger.info(f"Dropped existing collection: {collection_name}")
                
                # Create new collection
                schema = CollectionSchema(config["fields"], config["description"])
                collection = Collection(name=collection_name, schema=schema)
                created_collections[collection_name] = collection
                
                logger.info(f"Created collection: {collection_name}")
                
                # Create COSINE index for embedding fields (better for search)
                embedding_collections = ["opportunity_titles", "opportunity_descriptions", "opportunity_documents"]
                if collection_name in embedding_collections:
                    # Enhanced index parameters for better search precision
                    index_params = {
                        "metric_type": "COSINE",  # Use COSINE for similarity search
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 256}  # Increased for better precision/recall balance
                    }
                    
                    collection.create_index(
                        field_name="embedding",
                        index_params=index_params
                    )
                    
                    logger.info(f"Created COSINE embedding index for collection: {collection_name}")
                
                # Create scalar indices for fast lookups
                if collection_name == "opportunity_documents":
                    # Index on file_id for document lookups (primary lookup key)
                    collection.create_index(
                        field_name="file_id",
                        index_params={"index_type": "STL_SORT"}
                    )
                    logger.info(f"Created file_id index for collection: {collection_name}")
                
                # Create scalar index on opportunity_id for titles and descriptions
                if collection_name in ["opportunity_titles", "opportunity_descriptions"]:
                    collection.create_index(
                        field_name="opportunity_id", 
                        index_params={"index_type": "Trie"}
                    )
                    logger.info(f"Created opportunity_id index for collection: {collection_name}")
                
            except Exception as e:
                logger.error(f"Failed to create collection {collection_name}: {e}")
                raise
        
        # Note: Not loading collections into memory here - they will auto-load when data is inserted
        # This avoids "AmbiguousIndexName" errors with multiple indexes
        logger.info("Collections created with proper schemas and indexes")
        
        logger.info("Vector database setup complete")
        logger.info(f"Created collections: {list(created_collections.keys())}")
        return created_collections
        
    except Exception as e:
        logger.error(f"Failed to setup vector database: {e}")
        raise

def test_connection():
    """Test Milvus connection"""
    try:
        connections.connect(alias="default", host="localhost", port="19530")
        logger.info("✅ Successfully connected to Milvus")
        
        # List existing collections
        collections = utility.list_collections()
        logger.info(f"Existing collections: {collections}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to connect to Milvus: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Vector Database for v3 API")
    print("="*50)
    
    # Test connection first
    if test_connection():
        # Setup the database
        collections = setup_vector_database()
        print("✅ Vector database setup completed successfully!")
        print(f"Created {len(collections)} collections")
    else:
        print("❌ Failed to connect to Milvus. Please check if Milvus is running.")
