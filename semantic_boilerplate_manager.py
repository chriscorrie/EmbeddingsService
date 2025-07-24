#!/usr/bin/env python3
"""
Semantic boilerplate removal using embeddings comparison
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import Collection

logger = logging.getLogger(__name__)

class SemanticBoilerplateManager:
    """
    Manages semantic boilerplate removal using embedding similarity
    """
    
    def __init__(self, embeddings_model, similarity_threshold: float = 0.9):
        """
        Initialize the boilerplate manager
        
        Args:
            embeddings_model: Sentence transformer model for generating embeddings
            similarity_threshold: Cosine similarity threshold for boilerplate detection
        """
        self.embeddings_model = embeddings_model
        self.similarity_threshold = similarity_threshold
        self.boilerplate_embeddings_cache: List[np.ndarray] = []
        self.boilerplate_matrix: np.ndarray = None  # Vectorized matrix for fast similarity
        self.boilerplate_collection = None
        self._cache_loaded = False
        
    def setup_boilerplate_collection(self, collection: Collection):
        """Set up the boilerplate collection reference"""
        self.boilerplate_collection = collection
        
    def process_boilerplate_documents(self, boilerplate_docs_path: str, chunker) -> int:
        """
        Process boilerplate documents and store their embeddings
        
        Args:
            boilerplate_docs_path: Path to directory containing boilerplate documents
            chunker: SemanticChunker instance for chunking boilerplate docs
            
        Returns:
            Number of boilerplate chunks processed
        """
        if not os.path.exists(boilerplate_docs_path):
            logger.warning(f"Boilerplate documents path not found: {boilerplate_docs_path}")
            return 0
            
        if not self.boilerplate_collection:
            logger.error("Boilerplate collection not set up")
            return 0
            
        # Import here to avoid circular imports
        from process_documents import extract_text_from_file
        
        total_chunks = 0
        
        # Process each boilerplate document
        for filename in os.listdir(boilerplate_docs_path):
            file_path = os.path.join(boilerplate_docs_path, filename)
            
            if not os.path.isfile(file_path):
                continue
                
            try:
                # Extract text
                text = extract_text_from_file(file_path)
                if not text:
                    logger.warning(f"No text extracted from boilerplate file: {filename}")
                    continue
                
                # Chunk the boilerplate document (NO boilerplate removal here!)
                chunks = chunker.chunk_text(text)
                
                if not chunks:
                    logger.warning(f"No chunks generated from boilerplate file: {filename}")
                    continue
                
                # Generate embeddings for each chunk
                chunk_data = []
                for i, chunk in enumerate(chunks):
                    embedding = self.embeddings_model.encode(chunk, normalize_embeddings=True)
                    
                    chunk_data.append([
                        filename,                          # boilerplate_file
                        embedding.tolist(),                # embedding  
                        i,                                 # chunk_index
                        len(chunks),                       # total_chunks
                        chunk[:2000]                       # chunk_text (truncated to fit schema)
                    ])
                
                # Insert chunks into boilerplate collection
                if chunk_data:
                    # Transpose data for Milvus format
                    data = [
                        [item[0] for item in chunk_data],  # boilerplate_file
                        [item[1] for item in chunk_data],  # embedding
                        [item[2] for item in chunk_data],  # chunk_index
                        [item[3] for item in chunk_data],  # total_chunks
                        [item[4] for item in chunk_data],  # chunk_text
                    ]
                    
                    self.boilerplate_collection.insert(data)
                    total_chunks += len(chunks)
                    logger.info(f"Processed boilerplate file {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing boilerplate file {filename}: {e}")
                continue
        
        # Flush to ensure data is written
        if total_chunks > 0:
            self.boilerplate_collection.flush()
            logger.info(f"Processed {total_chunks} boilerplate chunks total")
            
        return total_chunks
    
    def load_boilerplate_embeddings_cache(self):
        """Load all boilerplate embeddings into memory cache for fast comparison"""
        if self._cache_loaded:
            return
            
        if not self.boilerplate_collection:
            logger.warning("Boilerplate collection not available - no caching")
            return
            
        try:
            # Query all boilerplate embeddings
            results = self.boilerplate_collection.query(
                expr="chunk_index >= 0",  # Get all records
                output_fields=["embedding"],
                limit=10000  # Should be enough for boilerplate docs
            )
            
            # Convert to numpy arrays and cache
            self.boilerplate_embeddings_cache = []
            for result in results:
                embedding = np.array(result["embedding"])
                self.boilerplate_embeddings_cache.append(embedding)
            
            # OPTIMIZATION: Pre-compute vectorized matrix for fast similarity calculations
            if self.boilerplate_embeddings_cache:
                self.boilerplate_matrix = np.vstack(self.boilerplate_embeddings_cache)
                logger.info(f"Created boilerplate matrix: {self.boilerplate_matrix.shape}")
            else:
                self.boilerplate_matrix = None
            
            self._cache_loaded = True
            logger.info(f"Cached {len(self.boilerplate_embeddings_cache)} boilerplate embeddings")
            
        except Exception as e:
            logger.error(f"Error loading boilerplate embeddings cache: {e}")
            self.boilerplate_embeddings_cache = []
            self.boilerplate_matrix = None
    
    def is_boilerplate_chunk(self, chunk_embedding: np.ndarray) -> bool:
        """
        Check if a chunk embedding is similar to boilerplate (OPTIMIZED VERSION)
        
        Args:
            chunk_embedding: Embedding vector for the chunk to check
            
        Returns:
            True if chunk is likely boilerplate, False otherwise
        """
        if self.boilerplate_matrix is None or len(self.boilerplate_embeddings_cache) == 0:
            # No boilerplate embeddings available - don't filter
            return False
        
        try:
            # FAST: Vectorized batch similarity calculation
            similarities = cosine_similarity(
                chunk_embedding.reshape(1, -1),    # Shape: (1, embedding_dim)
                self.boilerplate_matrix            # Shape: (N, embedding_dim)
            )[0]  # Result: array of N similarities
            
            # FAST: Early termination with vectorized max
            max_similarity = np.max(similarities)
            is_boilerplate = max_similarity >= self.similarity_threshold
            
            if is_boilerplate:
                logger.debug(f"Chunk identified as boilerplate (similarity: {max_similarity:.3f})")
                
            return is_boilerplate
            
        except Exception as e:
            logger.error(f"Error in vectorized boilerplate check: {e}")
            # Fallback to no filtering on error
            return False
    
    def filter_non_boilerplate_chunks(self, chunks: List[str]) -> List[Tuple[str, np.ndarray]]:
        """
        Filter out boilerplate chunks and return non-boilerplate chunks with embeddings
        
        Args:
            chunks: List of text chunks to filter
            
        Returns:
            List of (chunk_text, embedding) tuples for non-boilerplate chunks
        """
        if not chunks:
            return []
            
        non_boilerplate_chunks = []
        boilerplate_count = 0
        
        for chunk in chunks:
            # Generate embedding for this chunk
            chunk_embedding = self.embeddings_model.encode(chunk, normalize_embeddings=True)
            
            # Check if it's boilerplate
            if not self.is_boilerplate_chunk(chunk_embedding):
                non_boilerplate_chunks.append((chunk, chunk_embedding))
            else:
                boilerplate_count += 1
        
        logger.debug(f"Filtered {boilerplate_count} boilerplate chunks, kept {len(non_boilerplate_chunks)} chunks")
        return non_boilerplate_chunks


def create_boilerplate_manager(embeddings_model, similarity_threshold: float = 0.9) -> SemanticBoilerplateManager:
    """
    Factory function to create a semantic boilerplate manager
    
    Args:
        embeddings_model: Sentence transformer model
        similarity_threshold: Similarity threshold for boilerplate detection
        
    Returns:
        Configured SemanticBoilerplateManager instance
    """
    return SemanticBoilerplateManager(embeddings_model, similarity_threshold)
