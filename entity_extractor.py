"""
Entity extraction module for document embedding system
Extracts people, organizations, contact information from text
"""

import spacy
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass
class EntityComponents:
    """Individual components found in text"""
    names: List[Tuple[str, int, int, float]] = field(default_factory=list)  # (text, start, end, confidence)
    emails: List[Tuple[str, int, int, float]] = field(default_factory=list)
    phones: List[Tuple[str, int, int, float]] = field(default_factory=list)
    titles: List[Tuple[str, int, int, float]] = field(default_factory=list)
    organizations: List[Tuple[str, int, int, float]] = field(default_factory=list)

@dataclass
class LinkedEntity:
    """Linked entity with all associated attributes"""
    opportunity_id: str
    file_id: Optional[int]
    source_type: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    title: Optional[str] = None
    organization: Optional[str] = None
    confidence_score: float = 0.0
    context_text: Optional[str] = None
    extraction_method: str = "combined"

class EntityExtractor:
    """
    Extract and link related entity components into unified entities
    Optimized for speed and accuracy
    """
    
    def __init__(self, enable_gpu: bool = False):
        self.nlp = None
        self.enable_gpu = enable_gpu
        self._load_model()
        self._compile_patterns()
        
    def _validate_name(self, name_text: str) -> bool:
        """Validate name quality - ensure real names, not single words"""
        if not name_text or len(name_text.strip()) < 2:
            return False
        
        name_lower = name_text.lower().strip()
        words = name_text.split()
        
        # Reject single words unless they're clearly names
        if len(words) == 1:
            # Allow some single-word names that are clearly personal names
            single_name_exceptions = {
                'madonna', 'cher', 'prince', 'elvis', 'bono', 'sting', 'adele', 'beyonce'
            }
            if name_lower not in single_name_exceptions:
                return False
        
        # Must be at least 2 characters
        if len(name_text) < 2:
            return False
        
        # Must contain only letters, spaces, hyphens, apostrophes, and periods
        if not re.match(r"^[A-Za-z\s\-'.]+$", name_text):
            return False
        
        # Check for common non-name words
        non_name_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'none', 'null', 'undefined', 'test', 'example', 'sample', 'page', 'section',
            'document', 'file', 'attachment', 'appendix', 'exhibit', 'pursuant', 'accordance',
            'reference', 'paragraph', 'clause', 'contractor', 'government', 'agency',
            'department', 'office', 'bureau', 'division', 'branch', 'unit', 'team',
            'group', 'organization', 'company', 'corporation', 'inc', 'llc', 'ltd'
        }
        
        # Reject if any word is in the non-name blacklist
        for word in words:
            if word.lower().strip('.,-') in non_name_words:
                return False
        
        # Reject obvious patterns
        if any(pattern in name_lower for pattern in ['email', 'phone', 'address', 'contact']):
            return False
        
        return True

    def _validate_title(self, title_text: str) -> bool:
        """Validate title quality - ensure complete titles"""
        if not title_text or len(title_text.strip()) < 3:
            return False
        
        title_lower = title_text.lower().strip()
        
        # Check blacklist for incomplete titles
        if title_lower in self.title_blacklist:
            return False
        
        # Length constraints
        if len(title_text) < 3 or len(title_text) > 60:
            return False
        
        # Must contain at least one letter
        if not any(c.isalpha() for c in title_text):
            return False
        
        # For most government titles, require at least 2 words (exceptions for CEO, CTO, etc.)
        words = title_text.split()
        single_word_exceptions = {'ceo', 'cto', 'cfo', 'coo', 'president', 'director', 'administrator'}
        if len(words) == 1 and title_lower not in single_word_exceptions:
            return False
        
        # Reject obvious garbage patterns
        if 'none' in title_lower or 'null' in title_lower:
            return False
        
        return True

    def _validate_email_name_consistency(self, email: str, name: str) -> bool:
        """Check if email and name are reasonably consistent"""
        if not email or not name:
            return True  # Can't validate, but don't reject
        
        email_local = email.split('@')[0].lower()
        name_parts = name.lower().replace('.', '').replace('-', '').split()
        
        if len(name_parts) < 2:
            return True  # Single names are hard to validate
        
        first_name = name_parts[0]
        last_name = name_parts[-1]
        
        # Common email patterns to check
        patterns_to_check = [
            first_name + last_name,          # johnsmith
            first_name + '.' + last_name,    # john.smith  
            first_name[0] + last_name,       # jsmith
            last_name + first_name[0],       # smithj
            first_name[:3] + last_name[:3],  # johsmi
            last_name + first_name[:3],      # smithjoh
        ]
        
        # Also check full first name variations (Timothy -> Tim)
        if len(first_name) > 4:
            short_first = first_name[:3]
            patterns_to_check.extend([
                short_first + last_name,     # timsmith
                short_first + '.' + last_name, # tim.smith
                short_first[0] + last_name,  # tsmith
                last_name + short_first[0],  # smitht
            ])
        
        # Check if any pattern matches
        for pattern in patterns_to_check:
            if pattern in email_local or email_local in pattern:
                return True
        
        # More lenient check - any name part in email
        for part in name_parts:
            if len(part) >= 3 and (part in email_local or email_local in part):
                return True
        
        return False  # No reasonable match found

    def _load_model(self):
        """Load spaCy model optimized for speed"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            
            # Optimize for speed - disable unnecessary components (only if they exist)
            available_pipes = self.nlp.pipe_names
            pipes_to_disable = []
            
            for pipe_name in ['parser', 'lemmatizer', 'textcat']:
                if pipe_name in available_pipes:
                    pipes_to_disable.append(pipe_name)
            
            if pipes_to_disable:
                self.nlp.disable_pipes(pipes_to_disable)
                logger.debug(f"Disabled spaCy pipes: {pipes_to_disable}")
            
            if self.enable_gpu:
                try:
                    spacy.prefer_gpu()
                    logger.info("GPU acceleration enabled for entity extraction")
                except:
                    logger.warning("GPU acceleration not available")
            
            logger.info("spaCy model loaded for entity extraction")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
    
    def _compile_patterns(self):
        """Compile regex patterns for fast extraction"""
        
        # Processing statistics
        self.stats = {
            'texts_processed': 0,
            'entities_extracted': 0,
            'processing_time': 0
        }
        
        # Email patterns (high confidence)
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        )
        
        # Phone patterns (multiple formats)
        self.phone_patterns = [
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # 555-123-4567
            re.compile(r'\(\d{3}\)\s?\d{3}[-.]?\d{4}\b'),  # (555) 123-4567
            re.compile(r'\b\d{3}\s\d{3}\s\d{4}\b'),       # 555 123 4567
        ]
        
        # Title patterns - focused on complete government/contractor titles
        self.title_patterns = [
            # Complete government titles (two-word minimum for most)
            re.compile(r'\b(Contracting Officer(?:\s+Representative)?)\b', re.IGNORECASE),
            re.compile(r'\b(Program Manager)\b', re.IGNORECASE),
            re.compile(r'\b(Project Manager)\b', re.IGNORECASE),
            re.compile(r'\b(Technical Manager)\b', re.IGNORECASE),
            re.compile(r'\b(Contract Specialist)\b', re.IGNORECASE),
            re.compile(r'\b(Procurement Analyst)\b', re.IGNORECASE),
            re.compile(r'\b(Acquisition Officer)\b', re.IGNORECASE),
            re.compile(r'\b(Program Director)\b', re.IGNORECASE),
            re.compile(r'\b(Technical Director)\b', re.IGNORECASE),
            re.compile(r'\b(Business Manager)\b', re.IGNORECASE),
            re.compile(r'\b(Director of (?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*))\b', re.IGNORECASE),
            re.compile(r'\b(Deputy Director)\b', re.IGNORECASE),
            re.compile(r'\b(Assistant Director)\b', re.IGNORECASE),
            re.compile(r'\b(Branch Chief)\b', re.IGNORECASE),
            re.compile(r'\b(Division Chief)\b', re.IGNORECASE),
            re.compile(r'\b(Senior (?:Advisor|Analyst|Specialist|Engineer|Manager|Director))\b', re.IGNORECASE),
            re.compile(r'\b(Chief (?:Technology|Information|Financial|Operating|Executive) Officer)\b', re.IGNORECASE),
            re.compile(r'\b(Vice President)\b', re.IGNORECASE),
            re.compile(r'\b(Senior Vice President)\b', re.IGNORECASE),
            re.compile(r'\b(Executive Vice President)\b', re.IGNORECASE),
            re.compile(r'\b(Principal (?:Engineer|Scientist|Analyst|Consultant))\b', re.IGNORECASE),
            re.compile(r'\b(Lead (?:Engineer|Scientist|Analyst|Developer))\b', re.IGNORECASE),
        ]
        
        # Title validation - blacklist for incomplete titles
        self.title_blacklist = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'contracting', 'officer', 'program', 'project', 'manager', 'director', 'chief',  # Incomplete titles
            'senior', 'assistant', 'deputy', 'lead', 'principal',  # Incomplete modifiers  
            'business', 'technical', 'procurement', 'acquisition',  # Incomplete descriptors
            'none', 'null', 'undefined', 'test', 'example', 'sample'
        }
    
    def extract_entities(self, text: str, opportunity_id: str, source_type: str, 
                        file_id: Optional[int] = None) -> List[LinkedEntity]:
        """
        Extract and link entities from text
        
        Args:
            text: Input text to process
            opportunity_id: GUID of the opportunity
            source_type: 'title', 'description', or 'document'
            file_id: File ID if from document (BIGINT)
            
        Returns:
            List of LinkedEntity objects
        """
        start_time = time.time()
        
        if not text or len(text.strip()) < 3:
            return []
        
        # Step 1: Extract all individual components
        components = self._extract_components(text)
        
        # Step 2: Link components into unified entities
        linked_entities = self._link_components(components, text, opportunity_id, source_type, file_id)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['texts_processed'] += 1
        self.stats['entities_extracted'] += len(linked_entities)
        self.stats['processing_time'] += processing_time
        
        return linked_entities
    
    def _extract_components(self, text: str) -> EntityComponents:
        """Extract all individual entity components from text"""
        components = EntityComponents()
        
        # Preprocess text to fix broken emails with spaces
        text = self._repair_broken_emails(text)
        
        # Extract emails (high confidence)
        for match in self.email_pattern.finditer(text):
            components.emails.append((match.group(), match.start(), match.end(), 0.95))
        
        # Extract phone numbers
        for pattern in self.phone_patterns:
            for match in pattern.finditer(text):
                # Avoid duplicates
                if not any(abs(match.start() - existing[1]) < 5 for existing in components.phones):
                    components.phones.append((match.group(), match.start(), match.end(), 0.85))
        
        # Extract titles using patterns with validation
        for pattern in self.title_patterns:
            for match in pattern.finditer(text):
                title_text = match.group(1) if match.groups() else match.group()
                
                # Validate title before adding
                if self._validate_title(title_text):
                    components.titles.append((title_text.strip(), match.start(), match.end(), 0.80))
        
        # Extract names using spaCy with validation (NO organizations)
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Validate name before adding
                if self._validate_name(ent.text):
                    components.names.append((ent.text, ent.start_char, ent.end_char, 0.75))
        
        return components
    
    def _repair_broken_emails(self, text: str) -> str:
        """Fix broken email addresses that have spaces inserted or truncated domains"""
        import re
        
        # Pattern 1: Fix emails with spaces: "name@domain .com" or "name@domain. com" etc.
        broken_email_pattern = re.compile(
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,})',
            re.IGNORECASE
        )
        
        def fix_email(match):
            broken_email = match.group(1)
            # Remove all spaces within the email
            fixed_email = re.sub(r'\s+', '', broken_email)
            return fixed_email
        
        # Fix broken emails with spaces
        text = broken_email_pattern.sub(fix_email, text)
        
        # Pattern 2: Handle complex broken patterns like "user @ domain . com"
        complex_broken_pattern = re.compile(
            r'([A-Za-z0-9._%+-]+)\s*@\s*([A-Za-z0-9.-]+)\s*\.\s*([A-Za-z]{2,})',
            re.IGNORECASE
        )
        
        def fix_complex_email(match):
            user, domain, tld = match.groups()
            return f"{user}@{domain}.{tld}"
        
        text = complex_broken_pattern.sub(fix_complex_email, text)
        
        # Pattern 3: Fix common truncated military domains (.mi -> .mil, .ar -> .army, etc.)
        truncated_mil_pattern = re.compile(
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*army)\.mi\b',
            re.IGNORECASE
        )
        
        def fix_truncated_mil(match):
            email_prefix = match.group(1)
            return f"{email_prefix}.mil"
        
        text = truncated_mil_pattern.sub(fix_truncated_mil, text)
        
        # Pattern 4: Fix mail.mi -> mail.mil (common military pattern)
        mail_mi_pattern = re.compile(
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*mail)\.mi\b',
            re.IGNORECASE
        )
        
        text = mail_mi_pattern.sub(fix_truncated_mil, text)
        
        # Pattern 5: Fix other common truncated domains
        common_fixes = {
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*army)\.ar\b': r'\1.army',
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*gov)\.go\b': r'\1.gov', 
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*com)\.co\b': r'\1.com',
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*net)\.ne\b': r'\1.net',
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]*org)\.or\b': r'\1.org'
        }
        
        for pattern, replacement in common_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Pattern 6: Remove extensions after Top Level Domains
        # This handles cases like: nina.m.bushnell.civ@mail.mil.DRAFT -> nina.m.bushnell.civ@mail.mil
        tld_extension_pattern = re.compile(
            r'([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(mil|gov|com|net|org|edu|army|navy|af|marines))(\.[A-Z0-9._%+-]+)',
            re.IGNORECASE
        )
        
        def fix_tld_extension(match):
            email_with_tld = match.group(1)
            # Return just the email up to the TLD, removing the extension
            return email_with_tld
        
        text = tld_extension_pattern.sub(fix_tld_extension, text)
        
        return text
    
    def _link_components(self, components: EntityComponents, text: str, 
                        opportunity_id: str, source_type: str, file_id: Optional[int]) -> List[LinkedEntity]:
        """Link related components into unified entities"""
        
        linked_entities = []
        
        # Strategy 1: Proximity-based linking
        proximity_entities = self._link_by_proximity(components, text, opportunity_id, source_type, file_id)
        linked_entities.extend(proximity_entities)
        
        # Strategy 2: Create entities for unlinked high-confidence components
        orphan_entities = self._create_orphan_entities(components, proximity_entities, text, opportunity_id, source_type, file_id)
        linked_entities.extend(orphan_entities)
        
        return linked_entities
    
    def _link_by_proximity(self, components: EntityComponents, text: str,
                          opportunity_id: str, source_type: str, file_id: Optional[int]) -> List[LinkedEntity]:
        """Link components that appear close to each other in the text"""
        
        linked_entities = []
        used_components = set()
        
        # Create position-sorted list of all components (NO ORGANIZATIONS)
        all_components = []
        
        for name, start, end, conf in components.names:
            all_components.append(('name', name, start, end, conf))
        for email, start, end, conf in components.emails:
            all_components.append(('email', email, start, end, conf))
        for phone, start, end, conf in components.phones:
            all_components.append(('phone_number', phone, start, end, conf))
        for title, start, end, conf in components.titles:
            all_components.append(('title', title, start, end, conf))
        
        # Sort by position
        all_components.sort(key=lambda x: x[2])  # Sort by start position
        
        # Link components within proximity window
        proximity_window = 200  # characters
        
        for i, (comp_type, comp_text, start, end, conf) in enumerate(all_components):
            if (comp_type, start, end) in used_components:
                continue
            
            # Start a new entity
            entity = LinkedEntity(
                opportunity_id=opportunity_id,
                file_id=file_id,
                source_type=source_type,
                confidence_score=conf
            )
            
            # Set the initial component
            setattr(entity, comp_type, comp_text)
            used_components.add((comp_type, start, end))
            
            # Look for nearby components to link
            for j in range(i + 1, len(all_components)):
                other_type, other_text, other_start, other_end, other_conf = all_components[j]
                
                # Stop if too far away
                if other_start - end > proximity_window:
                    break
                
                # Skip if already used
                if (other_type, other_start, other_end) in used_components:
                    continue
                
                # Skip if we already have this type
                if getattr(entity, other_type) is not None:
                    continue
                
                # Link this component
                setattr(entity, other_type, other_text)
                used_components.add((other_type, other_start, other_end))
                
                # Update confidence (weighted average)
                entity.confidence_score = (entity.confidence_score + other_conf) / 2
            
            # Add context
            context_start = max(0, start - 50)
            context_end = min(len(text), end + 50)
            entity.context_text = text[context_start:context_end].strip()
            
            # Validate email-name consistency if both exist
            if entity.email and entity.name:
                if not self._validate_email_name_consistency(entity.email, entity.name):
                    # Split inconsistent entities into separate email and name entities
                    # Create email-only entity
                    email_entity = LinkedEntity(
                        opportunity_id=opportunity_id,
                        file_id=file_id,
                        source_type=source_type,
                        email=entity.email,
                        phone_number=entity.phone_number,  # Associate phone with email
                        title=entity.title,               # Associate title with email
                        confidence_score=entity.confidence_score * 0.8,  # Slightly reduced confidence
                        context_text=entity.context_text,
                        extraction_method=entity.extraction_method
                    )
                    linked_entities.append(email_entity)
                    
                    # Create name-only entity
                    name_entity = LinkedEntity(
                        opportunity_id=opportunity_id,
                        file_id=file_id,
                        source_type=source_type,
                        name=entity.name,
                        confidence_score=entity.confidence_score * 0.6,  # More reduced confidence
                        context_text=entity.context_text,
                        extraction_method=entity.extraction_method
                    )
                    linked_entities.append(name_entity)
                    continue  # Skip adding the combined entity
            
            # Only include entities that have either a name OR an email
            if entity.name or entity.email:
                linked_entities.append(entity)
        
        return linked_entities
    
    def _create_orphan_entities(self, components: EntityComponents, existing_entities: List[LinkedEntity],
                               text: str, opportunity_id: str, source_type: str, file_id: Optional[int]) -> List[LinkedEntity]:
        """Create entities for high-confidence components that weren't linked"""
        
        orphan_entities = []
        
        # Create entities for unlinked emails (always valuable)
        for email, start, end, conf in components.emails:
            if not self._is_component_used(email, existing_entities):
                entity = LinkedEntity(
                    opportunity_id=opportunity_id,
                    file_id=file_id,
                    source_type=source_type,
                    email=email,
                    confidence_score=conf,
                    context_text=self._get_context(text, start, end)
                )
                orphan_entities.append(entity)
        
        # Create entities for unlinked phone numbers (always valuable)
        for phone, start, end, conf in components.phones:
            if not self._is_component_used(phone, existing_entities):
                entity = LinkedEntity(
                    opportunity_id=opportunity_id,
                    file_id=file_id,
                    source_type=source_type,
                    phone_number=phone,
                    confidence_score=conf,
                    context_text=self._get_context(text, start, end)
                )
                orphan_entities.append(entity)
        
        # Create entities for high-confidence unlinked names (only if validated)
        for name, start, end, conf in components.names:
            if conf >= 0.7 and not self._is_component_used(name, existing_entities):
                # Validate name before creating entity
                if self._validate_name(name):
                    entity = LinkedEntity(
                        opportunity_id=opportunity_id,
                        file_id=file_id,
                        source_type=source_type,
                        name=name,
                        confidence_score=conf,
                        context_text=self._get_context(text, start, end)
                    )
                    orphan_entities.append(entity)
        
        return orphan_entities
    
    def _is_component_used(self, comp_text: str, existing_entities: List[LinkedEntity]) -> bool:
        """Check if a component is already used in existing entities"""
        for entity in existing_entities:
            if (entity.name == comp_text or 
                entity.email == comp_text or 
                entity.phone_number == comp_text or 
                entity.title == comp_text):
                return True
        return False
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context around an entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if self.stats['texts_processed'] == 0:
            return self.stats
        
        return {
            **self.stats,
            'avg_entities_per_text': self.stats['entities_extracted'] / self.stats['texts_processed'],
            'avg_processing_time': self.stats['processing_time'] / self.stats['texts_processed']
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'texts_processed': 0,
            'entities_extracted': 0,
            'processing_time': 0
        }
