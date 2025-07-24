#!/usr/bin/env python3
"""
Complete Document Processing Pipeline
Processes documents, generates embeddings, and stores them in Milvus vector database
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from process_documents import extract_text_from_file
from generate_embeddings import generate_embeddings
from config import (
    DOCUMENTS_PATH, 
    EMBEDDING_MODEL, 
    SQL_CONNECTION_STRING,
    VECTOR_DB_PATH,
    DOCUMENT_PATH_TO_REPLACE,
    DOCUMENT_PATH_REPLACEMENT_VALUE
)

# Import required libraries
from pymilvus import connections, Collection, utility
import pyodbc
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.setup_milvus_connection()
        self.setup_sql_connection()
        
    def setup_milvus_connection(self):
        """Setup connection to Milvus vector database"""
        try:
            connections.connect(alias="default", host="localhost", port="19530")
            logger.info("Connected to Milvus database")
            
            # Get the collection
            if utility.has_collection("document_embeddings"):
                self.collection = Collection("document_embeddings")
                logger.info("Connected to document_embeddings collection")
            else:
                logger.error("Collection 'document_embeddings' not found")
                raise Exception("Vector database collection not found")
                
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
            
    def setup_sql_connection(self):
        """Setup connection to SQL Server"""
        try:
            # Skip SQL Server connection for now
            logger.info("Skipping SQL Server connection (not configured)")
            self.sql_conn = None
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server: {e}")
            self.sql_conn = None
            
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for a document from SQL Server"""
        if not self.sql_conn:
            return {"fileDate": str(datetime.now()), "fileParentId": "unknown"}
            
        try:
            cursor = self.sql_conn.cursor()
            # This is a placeholder query - adjust based on your SQL Server schema
            query = """
            SELECT CreatedDate, ParentId 
            FROM DocumentMetadata 
            WHERE FilePath = ?
            """
            cursor.execute(query, (file_path,))
            result = cursor.fetchone()
            
            if result:
                return {
                    "fileDate": str(result[0]),
                    "fileParentId": str(result[1])
                }
            else:
                return {
                    "fileDate": str(datetime.now()),
                    "fileParentId": "unknown"
                }
                
        except Exception as e:
            logger.warning(f"Failed to get metadata for {file_path}: {e}")
            return {"fileDate": str(datetime.now()), "fileParentId": "unknown"}
            
    def replace_document_path(self, original_path: str) -> str:
        """
        Replace document path prefix for accessing files from different locations
        
        Args:
            original_path: Original file path from database
            
        Returns:
            Updated file path with replaced prefix
        """
        # Check if replacement is configured
        if not DOCUMENT_PATH_TO_REPLACE or not DOCUMENT_PATH_REPLACEMENT_VALUE:
            return original_path
            
        # Check if path contains the prefix to replace
        if original_path.startswith(DOCUMENT_PATH_TO_REPLACE):
            # Replace the prefix
            new_path = original_path.replace(DOCUMENT_PATH_TO_REPLACE, DOCUMENT_PATH_REPLACEMENT_VALUE, 1)
            logger.debug(f"Path replaced: {original_path} -> {new_path}")
            return new_path
        else:
            # Return original path if no replacement needed
            return original_path
            
    def process_document(self, file_path: str, doc_id: int) -> bool:
        """Process a single document with path replacement support"""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Apply path replacement
            actual_file_path = self.replace_document_path(file_path)
            
            # Extract text
            text = extract_text_from_file(actual_file_path)
            if not text:
                logger.warning(f"No text extracted from {actual_file_path} (original: {file_path})")
                return False
                
            # Generate embeddings
            embeddings = generate_embeddings([text], EMBEDDING_MODEL)
            if embeddings is None or len(embeddings) == 0:
                logger.warning(f"No embeddings generated for {actual_file_path} (original: {file_path})")
                return False
                
            # Convert to list if it's a numpy array
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            
            embedding_vector = embeddings[0] if isinstance(embeddings, list) else embeddings
                
            # Get metadata
            metadata = self.get_document_metadata(file_path)
            
            # Prepare data for insertion
            data = [
                [doc_id],  # id
                [embedding_vector],  # embedding
                [metadata["fileDate"]],  # fileDate
                [metadata["fileParentId"]]  # fileParentId
            ]
            
            # Insert into Milvus
            self.collection.insert(data)
            logger.info(f"Inserted document {doc_id} into vector database")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return False
            
    def process_documents_folder(self, folder_path: str, max_documents: int = None):
        """Process all documents in a folder"""
        logger.info(f"Processing documents from: {folder_path}")
        
        supported_extensions = {'.pdf', '.docx', '.txt', '.xlsx', '.pptx'}
        processed_count = 0
        success_count = 0
        doc_id = 1
        
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if max_documents and processed_count >= max_documents:
                    break
                    
                file_path = os.path.join(root, file)
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in supported_extensions:
                    if self.process_document(file_path, doc_id):
                        success_count += 1
                    processed_count += 1
                    doc_id += 1
                    
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} documents, {success_count} successful")
                        
        logger.info(f"Processing complete: {success_count}/{processed_count} documents processed successfully")
        
        # Flush the collection to ensure data is written
        self.collection.flush()
        logger.info("Data flushed to vector database")
        
    def search_similar_documents(self, query_text: str, top_k: int = 10):
        """Search for similar documents"""
        try:
            # Generate embedding for query
            query_embeddings = generate_embeddings([query_text], EMBEDDING_MODEL)
            
            # Convert to list if it's a numpy array
            if hasattr(query_embeddings, 'tolist'):
                query_embeddings = query_embeddings.tolist()
            
            query_embedding = query_embeddings[0] if isinstance(query_embeddings, list) else query_embeddings
            
            # Search parameters
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=None
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None

def main():
    """Main function to run the document processing pipeline"""
    logger.info("Starting document processing pipeline")
    
    # Create processor instance
    processor = DocumentProcessor()
    
    # Process documents (limit to 1000 for testing)
    processor.process_documents_folder(DOCUMENTS_PATH, max_documents=1000)
    
    # Example search
    logger.info("Testing search functionality")
    results = processor.search_similar_documents("sample query", top_k=5)
    if results:
        logger.info(f"Search returned {len(results[0])} results")
    
    logger.info("Document processing pipeline complete")

if __name__ == "__main__":
    main()
