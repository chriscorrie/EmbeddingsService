#!/usr/bin/env python3
"""
Chunk Embedding Cache System for Phase 2 Optimizations

This module provides intelligent caching of chunk embeddings to avoid
recomputing embeddings for similar or identical chunks across documents.
"""

import hashlib
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ChunkEmbeddingCache:
    """
    Thread-safe LRU cache for chunk embeddings with similarity-based lookup
    
    Features:
    - Exact hash matching for identical chunks
    - Cosine similarity matching for near-identical chunks  
    - Thread-safe operations with read-write locks
    - LRU eviction policy to manage memory usage
    - Statistics tracking for cache hit/miss analysis
    """
    
    def __init__(self, max_size: int = 10000, similarity_threshold: float = 0.98, embedding_model=None):
        """
        Initialize the embedding cache
        
        Args:
            max_size: Maximum number of cached embeddings
            similarity_threshold: Cosine similarity threshold for considering chunks equivalent
            embedding_model: SentenceTransformer model instance for similarity calculations
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model  # Use shared model instance
        
        # Cache storage: hash -> (chunk_text, embedding, access_count)
        self._exact_cache = OrderedDict()
        
        # For similarity search: list of (hash, embedding) tuples  
        self._similarity_index = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'exact_hits': 0,
            'similarity_hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_lookups': 0
        }
        
        logger.info(f"Initialized chunk embedding cache: max_size={max_size}, similarity_threshold={similarity_threshold}")
    
    def _compute_hash(self, chunk_text: str) -> str:
        """Compute hash for chunk text"""
        return hashlib.md5(chunk_text.strip().encode('utf-8')).hexdigest()
    
    def get_embedding(self, chunk_text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for chunk text
        
        Args:
            chunk_text: The text chunk to lookup
            
        Returns:
            Cached embedding array if found, None otherwise
        """
        with self._lock:
            self.stats['total_lookups'] += 1
            
            chunk_hash = self._compute_hash(chunk_text)
            
            # First try exact hash match
            if chunk_hash in self._exact_cache:
                # Move to end (LRU update)
                cached_data = self._exact_cache.pop(chunk_hash)
                self._exact_cache[chunk_hash] = cached_data
                
                # Update access count
                text, embedding, access_count = cached_data
                self._exact_cache[chunk_hash] = (text, embedding, access_count + 1)
                
                self.stats['exact_hits'] += 1
                logger.debug(f"Cache EXACT hit for hash {chunk_hash[:8]}...")
                return embedding.copy()
            
            # Try similarity search if we have embeddings to compare against
            if self._similarity_index and len(chunk_text.strip()) > 20 and self.embedding_model:  # Only for substantial chunks
                # Use the shared embedding model instead of creating a new one
                try:
                    query_embedding = self.embedding_model.encode([chunk_text.strip()], normalize_embeddings=True)[0]
                    
                    # Check similarity against cached embeddings
                    similarities = []
                    for cached_hash, cached_embedding in self._similarity_index:
                        sim = cosine_similarity([query_embedding], [cached_embedding])[0][0]
                        similarities.append((sim, cached_hash))
                    
                    # Find best match above threshold
                    best_sim, best_hash = max(similarities) if similarities else (0, None)
                    
                    if best_sim >= self.similarity_threshold:
                        # Use the cached embedding
                        cached_data = self._exact_cache[best_hash]
                        text, embedding, access_count = cached_data
                        
                        # Update access (move to end)
                        self._exact_cache.pop(best_hash)
                        self._exact_cache[best_hash] = (text, embedding, access_count + 1)
                        
                        self.stats['similarity_hits'] += 1
                        logger.debug(f"Cache SIMILARITY hit: {best_sim:.3f} for hash {best_hash[:8]}...")
                        return embedding.copy()
                        
                except Exception as e:
                    # Similarity search failed, continue to cache miss
                    logger.debug(f"Similarity search failed: {e}")
            
            # Cache miss
            self.stats['misses'] += 1
            logger.debug(f"Cache miss for hash {chunk_hash[:8]}...")
            return None
    
    def store_embedding(self, chunk_text: str, embedding: np.ndarray):
        """
        Store embedding in cache
        
        Args:
            chunk_text: The chunk text
            embedding: The computed embedding
        """
        with self._lock:
            chunk_hash = self._compute_hash(chunk_text)
            
            # Don't store if already exists
            if chunk_hash in self._exact_cache:
                return
            
            # Evict if at capacity
            if len(self._exact_cache) >= self.max_size:
                # Remove least recently used (first item)
                old_hash, old_data = self._exact_cache.popitem(last=False)
                
                # Remove from similarity index
                self._similarity_index = [(h, e) for h, e in self._similarity_index if h != old_hash]
                
                self.stats['evictions'] += 1
                logger.debug(f"Cache evicted hash {old_hash[:8]}...")
            
            # Store new embedding
            self._exact_cache[chunk_hash] = (chunk_text[:500], embedding.copy(), 1)  # Truncate text for memory
            
            # Add to similarity index (if embedding is reasonable size)
            if len(embedding) <= 1024:  # Don't index huge embeddings
                self._similarity_index.append((chunk_hash, embedding.copy()))
            
            logger.debug(f"Cache stored hash {chunk_hash[:8]}... (cache size: {len(self._exact_cache)})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_lookups = self.stats['total_lookups']
            if total_lookups > 0:
                hit_rate = (self.stats['exact_hits'] + self.stats['similarity_hits']) / total_lookups
                return {
                    **self.stats,
                    'cache_size': len(self._exact_cache),
                    'hit_rate': hit_rate,
                    'exact_hit_rate': self.stats['exact_hits'] / total_lookups,
                    'similarity_hit_rate': self.stats['similarity_hits'] / total_lookups,
                    'miss_rate': self.stats['misses'] / total_lookups
                }
            else:
                return {
                    **self.stats,
                    'cache_size': len(self._exact_cache),
                    'hit_rate': 0.0,
                    'exact_hit_rate': 0.0,
                    'similarity_hit_rate': 0.0,
                    'miss_rate': 0.0
                }
    
    def clear(self):
        """Clear all cached data"""
        with self._lock:
            self._exact_cache.clear()
            self._similarity_index.clear()
            # Reset stats
            for key in self.stats:
                self.stats[key] = 0
            logger.info("Cache cleared")

# Global cache instance
_global_cache = None
_cache_lock = threading.Lock()

def get_global_cache(embedding_model=None) -> ChunkEmbeddingCache:
    """Get the global chunk embedding cache instance"""
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                from config import CHUNK_CACHE_SIZE, CHUNK_CACHE_SIMILARITY_THRESHOLD
                _global_cache = ChunkEmbeddingCache(
                    max_size=CHUNK_CACHE_SIZE,
                    similarity_threshold=CHUNK_CACHE_SIMILARITY_THRESHOLD,
                    embedding_model=embedding_model
                )
    elif embedding_model is not None and _global_cache.embedding_model is None:
        # Update with embedding model if not already set
        _global_cache.embedding_model = embedding_model
    return _global_cache

def clear_global_cache():
    """Clear the global cache"""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
