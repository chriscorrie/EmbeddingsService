#!/usr/bin/env python3
"""
Text chunking utilities for semantic document processing
"""

import re
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Semantic text chunker that splits text by paragraphs and sentences
    with configurable overlap for better context preservation
    """
    
    def __init__(self, target_chunk_size: int = 1500, overlap_percentage: float = 0.2):
        """
        Initialize the semantic chunker
        
        Args:
            target_chunk_size: Target size in characters for each chunk
            overlap_percentage: Percentage of overlap between consecutive chunks (0.0 to 1.0)
        """
        self.target_chunk_size = target_chunk_size
        self.overlap_percentage = overlap_percentage
        self.overlap_size = int(target_chunk_size * overlap_percentage)
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text semantically with overlap
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks with overlap
        """
        if not text or len(text.strip()) == 0:
            return []
            
        # If text is smaller than target size, return as single chunk
        if len(text) <= self.target_chunk_size:
            return [text.strip()]
        
        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(text)
        
        # Build chunks from paragraphs
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed target size
            if len(current_chunk) + len(paragraph) > self.target_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Handle very long paragraphs by splitting at sentence level
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.target_chunk_size * 1.5:  # 50% tolerance
                final_chunks.extend(self._split_long_chunk(chunk))
            else:
                final_chunks.append(chunk)
        
        logger.debug(f"Chunked text of {len(text)} chars into {len(final_chunks)} chunks")
        return final_chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines, single newlines, or major punctuation followed by whitespace
        paragraphs = re.split(r'\n\s*\n|\n(?=\s*[A-Z])|(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean and filter paragraphs
        clean_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 10:  # Filter out very short paragraphs
                clean_paragraphs.append(para)
        
        return clean_paragraphs
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= self.overlap_size:
            return text + "\n\n"
        
        # Try to find a good breaking point (sentence end) within overlap region
        overlap_start = len(text) - self.overlap_size
        overlap_text = text[overlap_start:]
        
        # Look for sentence boundaries in the overlap region
        sentence_breaks = [m.end() for m in re.finditer(r'[.!?]\s+', overlap_text)]
        
        if sentence_breaks:
            # Use the last sentence break as the start of overlap
            best_break = sentence_breaks[-1]
            return overlap_text[best_break:] + "\n\n"
        else:
            # No sentence break found, use word boundary
            words = overlap_text.split()
            if len(words) > 5:
                # Take last 70% of words to avoid cutting mid-sentence
                word_start = int(len(words) * 0.3)
                return " ".join(words[word_start:]) + "\n\n"
            else:
                return overlap_text + "\n\n"
    
    def _split_long_chunk(self, chunk: str) -> List[str]:
        """Split a chunk that's too long at sentence boundaries"""
        sentences = re.split(r'(?<=[.!?])\s+', chunk)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.target_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


def create_chunker(target_size: int = 1500, overlap: float = 0.2) -> SemanticChunker:
    """
    Factory function to create a semantic chunker
    
    Args:
        target_size: Target chunk size in characters
        overlap: Overlap percentage (0.0 to 1.0)
        
    Returns:
        Configured SemanticChunker instance
    """
    return SemanticChunker(target_size, overlap)


if __name__ == "__main__":
    # Test the chunker
    chunker = create_chunker()
    
    sample_text = """
    This is a sample document with multiple paragraphs for testing the semantic chunker.
    It should be able to handle different types of content and maintain context through overlap.
    
    The first paragraph talks about the general approach to chunking and why it's important
    for maintaining semantic meaning in document processing pipelines.
    
    The second paragraph discusses the technical implementation details. This includes
    how we split by paragraphs first, then by sentences if needed. The overlap mechanism
    ensures that context is preserved between chunks.
    
    Finally, the third paragraph covers the testing methodology and validation approaches
    that we use to ensure the chunker works correctly across different document types.
    """
    
    chunks = chunker.chunk_text(sample_text)
    
    print(f"Original text: {len(sample_text)} characters")
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
