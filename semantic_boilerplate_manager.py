#!/usr/bin/env python3
"""
Semantic boilerplate removal using embeddings comparison
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from sklearn.metrics.pairwise import cosine_similarity

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
        self._cache_loaded = False
        
    def setup_boilerplate_collection(self, collection):
        """Legacy method - no longer needed since we don't use database storage"""
        pass  # No-op: boilerplate is now cached directly in memory
        
    def process_boilerplate_documents(self, boilerplate_docs_path: str, chunker) -> int:
        """
        Process boilerplate documents and cache their embeddings directly in memory
        
        Args:
            boilerplate_docs_path: Path to directory containing boilerplate documents
            chunker: SemanticChunker instance for chunking boilerplate docs
            
        Returns:
            Number of boilerplate chunks processed
        """
        if not os.path.exists(boilerplate_docs_path):
            logger.warning(f"Boilerplate documents path not found: {boilerplate_docs_path}")
            return 0
            
        # Import here to avoid circular imports
        from process_documents import extract_text_from_file
        
        total_chunks = 0
        embeddings_cache = []
        
        logger.info(f"Processing boilerplate documents from: {boilerplate_docs_path}")
        
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
                
                # Generate embeddings for each chunk and cache directly in memory
                for i, chunk in enumerate(chunks):
                    embedding = self.embeddings_model.encode(chunk, normalize_embeddings=True)
                    embeddings_cache.append(embedding)
                
                total_chunks += len(chunks)
                logger.info(f"Processed boilerplate file {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing boilerplate file {filename}: {e}")
                continue
        
        # Cache embeddings directly in memory
        if embeddings_cache:
            self.boilerplate_embeddings_cache = embeddings_cache
            self.boilerplate_matrix = np.vstack(embeddings_cache)
            self._cache_loaded = True
            logger.info(f"Cached {len(embeddings_cache)} boilerplate embeddings directly in memory")
            logger.info(f"Created boilerplate matrix: {self.boilerplate_matrix.shape}")
        else:
            logger.warning("No boilerplate embeddings were processed")
            
        return total_chunks
    
    def load_boilerplate_embeddings_cache(self):
        """Load boilerplate embeddings - now a no-op since embeddings are cached during processing"""
        if self._cache_loaded:
            logger.debug("Boilerplate embeddings already cached in memory")
            return
            
        logger.warning("load_boilerplate_embeddings_cache() called but no embeddings cached. "
                      "Ensure process_boilerplate_documents() was called first.")
    
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
        OPTIMIZED: Batch embedding generation + vectorized similarity computation
        
        Args:
            chunks: List of text chunks to filter
            
        Returns:
            List of (chunk_text, embedding) tuples for non-boilerplate chunks
        """
        if not chunks:
            return []
            
        # OPTIMIZATION 1: Batch generate ALL chunk embeddings at once
        logger.debug(f"Batch generating embeddings for {len(chunks)} chunks...")
        chunk_embeddings = self.embeddings_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        
        # If no boilerplate loaded, return all chunks
        if self.boilerplate_matrix is None or len(self.boilerplate_embeddings_cache) == 0:
            logger.debug("No boilerplate embeddings available - keeping all chunks")
            return [(chunk, embedding) for chunk, embedding in zip(chunks, chunk_embeddings)]
        
        # OPTIMIZATION 2: Vectorized batch similarity computation
        logger.debug(f"Computing similarity for {len(chunks)} chunks vs {len(self.boilerplate_embeddings_cache)} boilerplate chunks...")
        try:
            # Compute similarity between ALL chunks and ALL boilerplate embeddings at once
            # Shape: (num_chunks, num_boilerplate_embeddings)
            similarities = cosine_similarity(chunk_embeddings, self.boilerplate_matrix)
            
            # For each chunk, get the maximum similarity with any boilerplate
            max_similarities = np.max(similarities, axis=1)  # Shape: (num_chunks,)
            
            # Determine which chunks are NOT boilerplate (vectorized comparison)
            non_boilerplate_mask = max_similarities < self.similarity_threshold
            
        except Exception as e:
            logger.error(f"Error in vectorized boilerplate similarity computation: {e}")
            # Fallback: keep all chunks on error
            non_boilerplate_mask = np.ones(len(chunks), dtype=bool)
        
        # Filter chunks and embeddings using the mask
        non_boilerplate_chunks = []
        boilerplate_count = 0
        
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            if non_boilerplate_mask[i]:
                non_boilerplate_chunks.append((chunk, embedding))
            else:
                boilerplate_count += 1
                logger.debug(f"Chunk identified as boilerplate (similarity: {max_similarities[i]:.3f})")
        
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
