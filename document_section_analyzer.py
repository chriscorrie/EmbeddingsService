#!/usr/bin/env python3
"""
Document Section Analyzer
Identifies sections in government procurement documents and provides search-time boosting
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SectionInfo:
    """Information about a document section"""
    section_type: str
    confidence: float
    start_position: int
    end_position: int
    content: str
    importance_score: float = 1.0

class DocumentSectionAnalyzer:
    """
    Analyzes government procurement documents to identify key sections
    and provides search-time boosting capabilities
    """
    
    def __init__(self):
        """Initialize the section analyzer with government procurement patterns"""
        
        # Define section patterns for government procurement documents
        self.section_patterns = {
            'statement_of_work': [
                r'statement\s+of\s+work',
                r'scope\s+of\s+work',
                r'work\s+statement',
                r'SOW\b',
                r'statement\s+of\s+requirements'
            ],
            'performance_work_statement': [
                r'performance\s+work\s+statement',
                r'PWS\b',
                r'performance\s+statement',
                r'work\s+performance\s+statement'
            ],
            'technical_requirements': [
                r'technical\s+requirements',
                r'technical\s+specifications',
                r'tech\s+specs',
                r'technical\s+approach',
                r'technical\s+solution'
            ],
            'evaluation_criteria': [
                r'evaluation\s+criteria',
                r'evaluation\s+factors',
                r'selection\s+criteria',
                r'award\s+criteria',
                r'evaluation\s+standards'
            ],
            'contract_terms': [
                r'contract\s+terms',
                r'terms\s+and\s+conditions',
                r'contractual\s+requirements',
                r'contract\s+provisions',
                r'agreement\s+terms'
            ],
            'deliverables': [
                r'deliverables',
                r'deliverable\s+items',
                r'work\s+products',
                r'project\s+deliverables',
                r'required\s+deliverables'
            ],
            'timeline': [
                r'timeline',
                r'schedule',
                r'project\s+schedule',
                r'delivery\s+schedule',
                r'performance\s+period',
                r'milestones'
            ],
            'budget_cost': [
                r'budget',
                r'cost\s+estimate',
                r'pricing',
                r'financial\s+requirements',
                r'cost\s+proposal',
                r'budget\s+estimate'
            ],
            'security_requirements': [
                r'security\s+requirements',
                r'clearance\s+requirements',
                r'security\s+clearance',
                r'access\s+requirements',
                r'security\s+provisions'
            ],
            'quality_assurance': [
                r'quality\s+assurance',
                r'QA\s+requirements',
                r'quality\s+control',
                r'testing\s+requirements',
                r'acceptance\s+criteria'
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for section_type, patterns in self.section_patterns.items():
            self.compiled_patterns[section_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        
        # Define importance weights for different sections
        self.importance_weights = {
            'statement_of_work': 2.0,
            'performance_work_statement': 1.8,
            'technical_requirements': 1.5,
            'evaluation_criteria': 1.4,
            'deliverables': 1.3,
            'security_requirements': 1.2,
            'contract_terms': 1.1,
            'timeline': 1.0,
            'budget_cost': 0.9,
            'quality_assurance': 0.8,
            'general': 0.5
        }
    
    def identify_section(self, text: str) -> Dict[str, Any]:
        """
        Identify the section type of a given text
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with section information
        """
        if not text:
            return {
                'section_type': 'general',
                'confidence': 0.0,
                'importance_score': 0.5
            }
        
        best_match = None
        best_confidence = 0.0
        
        # Check each section type
        for section_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text.lower())
                if matches:
                    # Calculate confidence based on number of matches and text length
                    confidence = min(len(matches) / (len(text) / 100), 1.0)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = section_type
        
        # Default to general if no specific section identified
        if best_match is None:
            best_match = 'general'
            best_confidence = 0.1
        
        return {
            'section_type': best_match,
            'confidence': best_confidence,
            'importance_score': self.importance_weights.get(best_match, 0.5)
        }
    
    def analyze_document_sections(self, text: str) -> List[SectionInfo]:
        """
        Analyze a document and identify all sections
        
        Args:
            text: The full document text
            
        Returns:
            List of SectionInfo objects
        """
        sections = []
        
        # Split text into paragraphs for section analysis
        paragraphs = text.split('\n\n')
        current_position = 0
        
        for paragraph in paragraphs:
            if paragraph.strip():
                section_info = self.identify_section(paragraph)
                
                end_position = current_position + len(paragraph)
                
                section = SectionInfo(
                    section_type=section_info['section_type'],
                    confidence=section_info['confidence'],
                    start_position=current_position,
                    end_position=end_position,
                    content=paragraph,
                    importance_score=section_info['importance_score']
                )
                
                sections.append(section)
                current_position = end_position + 2  # Account for \n\n
        
        return sections
    
    def adjust_search_scores(self, search_results: List[Any], boost_factors: Dict[str, float]) -> List[Any]:
        """
        Adjust search scores based on section types and boost factors
        
        Args:
            search_results: List of search results from Milvus
            boost_factors: Dictionary mapping section types to boost multipliers
            
        Returns:
            List of search results with adjusted scores
        """
        adjusted_results = []
        
        for result in search_results:
            # Create a copy of the result to avoid modifying the original
            adjusted_result = result
            
            # Get section type from result
            section_type = 'general'  # default
            if hasattr(result, 'entity') and result.entity:
                section_type = result.entity.get('section_type', 'general')
            
            # Apply boost factor if specified
            boost_factor = boost_factors.get(section_type, 1.0)
            
            # Adjust the score
            original_score = result.score
            adjusted_score = min(original_score * boost_factor, 1.0)  # Cap at 1.0
            
            # Update the score
            adjusted_result.score = adjusted_score
            
            adjusted_results.append(adjusted_result)
        
        # Re-sort by adjusted scores
        adjusted_results.sort(key=lambda x: x.score, reverse=True)
        
        return adjusted_results
    
    def get_section_boost_recommendations(self, query: str) -> Dict[str, float]:
        """
        Get recommended boost factors based on query content
        
        Args:
            query: The search query
            
        Returns:
            Dictionary of recommended boost factors
        """
        query_lower = query.lower()
        recommendations = {}
        
        # Analyze query for section-specific terms
        if any(term in query_lower for term in ['work', 'scope', 'requirements']):
            recommendations['statement_of_work'] = 2.0
            recommendations['performance_work_statement'] = 1.8
            recommendations['technical_requirements'] = 1.5
        
        if any(term in query_lower for term in ['deliver', 'output', 'product']):
            recommendations['deliverables'] = 1.5
        
        if any(term in query_lower for term in ['schedule', 'timeline', 'deadline']):
            recommendations['timeline'] = 1.3
        
        if any(term in query_lower for term in ['security', 'clearance', 'access']):
            recommendations['security_requirements'] = 1.4
        
        if any(term in query_lower for term in ['evaluation', 'criteria', 'selection']):
            recommendations['evaluation_criteria'] = 1.4
        
        # Default boost for high-importance sections if no specific matches
        if not recommendations:
            recommendations = {
                'statement_of_work': 1.5,
                'performance_work_statement': 1.3,
                'technical_requirements': 1.2
            }
        
        return recommendations
    
    def print_section_analysis(self, text: str):
        """Print detailed section analysis for debugging"""
        sections = self.analyze_document_sections(text)
        
        print(f"\n📋 Document Section Analysis")
        print("=" * 50)
        
        for i, section in enumerate(sections, 1):
            print(f"\n{i}. Section Type: {section.section_type}")
            print(f"   Confidence: {section.confidence:.2f}")
            print(f"   Importance Score: {section.importance_score:.2f}")
            print(f"   Position: {section.start_position}-{section.end_position}")
            print(f"   Content Preview: {section.content[:100]}...")
        
        print(f"\nTotal Sections Found: {len(sections)}")

if __name__ == "__main__":
    # Example usage
    analyzer = DocumentSectionAnalyzer()
    
    # Test with sample text
    sample_text = """
    Statement of Work
    
    This procurement requires the contractor to provide software development services.
    The scope of work includes developing a web application with the following technical requirements:
    
    Technical Requirements
    
    The system must be built using modern web technologies and support 1000 concurrent users.
    Security requirements include encryption of all data in transit and at rest.
    
    Deliverables
    
    The contractor must deliver the following items by the specified deadlines:
    1. System architecture document
    2. Working prototype
    3. Final production system
    """
    
    print("🔍 Testing Document Section Analyzer")
    analyzer.print_section_analysis(sample_text)
    
    # Test boost recommendations
    query = "software development technical requirements"
    recommendations = analyzer.get_section_boost_recommendations(query)
    print(f"\n🚀 Boost Recommendations for '{query}':")
    for section, boost in recommendations.items():
        print(f"   {section}: {boost}x")
