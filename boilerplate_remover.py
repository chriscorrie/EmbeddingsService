#!/usr/bin/env python3
"""
Boilerplate removal module for cleaning documents before embedding generation
"""

import os
import re
from typing import List, Set
from pathlib import Path
from difflib import SequenceMatcher
from process_documents import extract_text_from_file
import logging

logger = logging.getLogger(__name__)

class BoilerplateRemover:
    def __init__(self, boilerplate_docs_path: str = None):
        """
        Initialize boilerplate remover
        
        Args:
            boilerplate_docs_path: Path to directory containing boilerplate documents
        """
        self.boilerplate_patterns = []
        self.boilerplate_texts = []
        self.min_similarity_threshold = 0.8  # Minimum similarity to consider as boilerplate
        
        if boilerplate_docs_path and os.path.exists(boilerplate_docs_path):
            self.load_boilerplate_documents(boilerplate_docs_path)
    
    def load_boilerplate_documents(self, boilerplate_docs_path: str):
        """Load boilerplate documents and extract patterns"""
        logger.info(f"Loading boilerplate documents from: {boilerplate_docs_path}")
        
        boilerplate_count = 0
        for root, dirs, files in os.walk(boilerplate_docs_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    text = extract_text_from_file(file_path)
                    if text:
                        # Clean and normalize text
                        cleaned_text = self._clean_text(text)
                        if len(cleaned_text) > 50:  # Only consider substantial text
                            self.boilerplate_texts.append(cleaned_text)
                            boilerplate_count += 1
                            logger.debug(f"Loaded boilerplate from: {file}")
                except Exception as e:
                    logger.warning(f"Failed to load boilerplate from {file}: {e}")
        
        logger.info(f"Loaded {boilerplate_count} boilerplate documents")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for comparison"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-]', '', text)
        return text.strip().lower()
    
    def _find_similar_segments(self, text: str, boilerplate: str) -> List[tuple]:
        """Find similar segments between text and boilerplate"""
        matcher = SequenceMatcher(None, text, boilerplate)
        matches = []
        
        for block in matcher.get_matching_blocks():
            if block.size > 50:  # Only consider substantial matches
                similarity = block.size / min(len(text), len(boilerplate))
                if similarity > self.min_similarity_threshold:
                    matches.append((block.a, block.a + block.size, similarity))
        
        return matches
    
    def remove_boilerplate(self, text: str) -> str:
        """Remove boilerplate content from text"""
        if not text or not self.boilerplate_texts:
            return text
        
        original_text = text
        cleaned_text = self._clean_text(text)
        
        # Track segments to remove
        segments_to_remove = []
        
        for boilerplate in self.boilerplate_texts:
            matches = self._find_similar_segments(cleaned_text, boilerplate)
            segments_to_remove.extend(matches)
        
        # Sort segments by position (reverse order to avoid index shifting)
        segments_to_remove.sort(key=lambda x: x[0], reverse=True)
        
        # Remove duplicate/overlapping segments
        filtered_segments = []
        for segment in segments_to_remove:
            overlap = False
            for existing in filtered_segments:
                if (segment[0] <= existing[1] and segment[1] >= existing[0]):
                    overlap = True
                    break
            if not overlap:
                filtered_segments.append(segment)
        
        # Remove segments from original text
        result_text = original_text
        for start, end, similarity in filtered_segments:
            logger.debug(f"Removing boilerplate segment (similarity: {similarity:.2f})")
            # Map cleaned text positions back to original text (approximate)
            result_text = result_text[:start] + result_text[end:]
        
        # Clean up remaining text
        result_text = re.sub(r'\s+', ' ', result_text).strip()
        
        if len(result_text) < len(original_text) * 0.3:
            logger.warning("Boilerplate removal resulted in very short text, returning original")
            return original_text
        
        return result_text
    
    def add_boilerplate_pattern(self, pattern: str):
        """Add a regex pattern for boilerplate removal"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            self.boilerplate_patterns.append(compiled_pattern)
            logger.info(f"Added boilerplate pattern: {pattern}")
        except re.error as e:
            logger.error(f"Invalid regex pattern: {pattern} - {e}")
    
    def remove_pattern_boilerplate(self, text: str) -> str:
        """Remove boilerplate using regex patterns"""
        result = text
        
        for pattern in self.boilerplate_patterns:
            result = pattern.sub('', result)
        
        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        return result
    
    def process_text(self, text: str) -> str:
        """Complete boilerplate removal process"""
        if not text:
            return text
        
        # First remove pattern-based boilerplate
        text = self.remove_pattern_boilerplate(text)
        
        # Then remove document-based boilerplate
        text = self.remove_boilerplate(text)
        
        return text

# Common boilerplate patterns for government documents
COMMON_GOVERNMENT_PATTERNS = [
    r'This\s+is\s+a\s+combined\s+synopsis/solicitation.*?(?=\n\n|\Z)',
    r'NAICS\s+Code\s*:?\s*\d+.*?(?=\n\n|\Z)',
    r'Set\s+Aside\s*:?\s*.*?(?=\n\n|\Z)',
    r'Response\s+Date\s*:?\s*.*?(?=\n\n|\Z)',
    r'Point\s+of\s+Contact\s*:?\s*.*?(?=\n\n|\Z)',
    r'Contracting\s+Officer\s*:?\s*.*?(?=\n\n|\Z)',
    r'Contract\s+Specialist\s*:?\s*.*?(?=\n\n|\Z)',
    r'This\s+procurement\s+is\s+unrestricted.*?(?=\n\n|\Z)',
    r'The\s+government\s+reserves\s+the\s+right.*?(?=\n\n|\Z)',
    r'All\s+responsible\s+sources\s+may\s+submit.*?(?=\n\n|\Z)',
]

def create_default_boilerplate_remover(boilerplate_docs_path: str = None) -> BoilerplateRemover:
    """Create a boilerplate remover with default government patterns"""
    remover = BoilerplateRemover(boilerplate_docs_path)
    
    # Add common government boilerplate patterns
    for pattern in COMMON_GOVERNMENT_PATTERNS:
        remover.add_boilerplate_pattern(pattern)
    
    return remover
