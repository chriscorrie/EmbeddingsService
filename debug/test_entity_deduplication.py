#!/usr/bin/env python3
"""
Test script to validate entity extraction deduplication is working correctly

This script tests the new hybrid reference counting + timeout approach for
opportunity-level entity consolidation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from entity_extractor import EntityExtractor, LinkedEntity
from sql_entity_manager import SQLEntityManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_entity_deduplication():
    """Test that entity deduplication works correctly at the opportunity level"""
    
    print("="*80)
    print("🧪 TESTING ENTITY EXTRACTION DEDUPLICATION")
    print("="*80)
    
    # Test opportunity ID
    test_opportunity_id = "test-12345678-1234-1234-1234-123456789abc"
    
    try:
        # Initialize components
        entity_extractor = EntityExtractor()
        entity_manager = SQLEntityManager()
        
        print(f"📋 Test Opportunity ID: {test_opportunity_id}")
        
        # Clean up any existing test data
        print("🧹 Cleaning up existing test data...")
        entity_manager.delete_opportunity_entities(test_opportunity_id)
        
        # Test content with deliberate duplicates
        description = """
        This is a test opportunity description. 
        Contact John Smith at john.smith@example.com or call 555-123-4567.
        John Smith is the Program Manager for this contract.
        """
        
        document1 = """
        Document 1 content with the same contact information.
        Please reach out to John Smith at john.smith@example.com for questions.
        You can also call John Smith at 555-123-4567.
        John Smith serves as the Technical Manager.
        """
        
        document2 = """
        Document 2 with additional contacts.
        Primary contact: John Smith (john.smith@example.com, 555-123-4567)
        Secondary contact: Jane Doe (jane.doe@example.com, 555-987-6543)
        John Smith is the Contract Specialist for this procurement.
        """
        
        print("📝 Extracting entities from individual content pieces...")
        
        # Extract entities from each piece individually (simulating old behavior)
        desc_entities = entity_extractor.extract_entities(description, test_opportunity_id, 'description')
        doc1_entities = entity_extractor.extract_entities(document1, test_opportunity_id, 'document', file_id=1001)
        doc2_entities = entity_extractor.extract_entities(document2, test_opportunity_id, 'document', file_id=1002)
        
        print(f"✅ Description entities: {len(desc_entities)}")
        for entity in desc_entities:
            print(f"   - {entity.name or 'No name'} | {entity.email or 'No email'} | {entity.phone_number or 'No phone'}")
        
        print(f"✅ Document 1 entities: {len(doc1_entities)}")
        for entity in doc1_entities:
            print(f"   - {entity.name or 'No name'} | {entity.email or 'No email'} | {entity.phone_number or 'No phone'}")
        
        print(f"✅ Document 2 entities: {len(doc2_entities)}")
        for entity in doc2_entities:
            print(f"   - {entity.name or 'No name'} | {entity.email or 'No email'} | {entity.phone_number or 'No phone'}")
        
        # Combine all entities
        all_entities = desc_entities + doc1_entities + doc2_entities
        print(f"\n📊 Total entities before consolidation: {len(all_entities)}")
        
        # Test the consolidation logic (same as in EntityExtractionQueue)
        print("🔄 Testing consolidation logic...")
        
        opportunity_groups = {}
        for entity in all_entities:
            opp_id = entity.opportunity_id
            if opp_id not in opportunity_groups:
                opportunity_groups[opp_id] = []
            opportunity_groups[opp_id].append(entity)
        
        consolidated = []
        for opp_id, opp_entities in opportunity_groups.items():
            seen_emails = set()
            seen_names = set()
            opportunity_entities = []
            
            print(f"   Processing {len(opp_entities)} entities for opportunity {opp_id}")
            
            for entity in opp_entities:
                if entity.email and entity.email.strip():
                    email_key = entity.email.lower().strip()
                    if email_key not in seen_emails:
                        seen_emails.add(email_key)
                        opportunity_entities.append(entity)
                        print(f"   ✅ Added entity with email: {entity.email}")
                    else:
                        print(f"   ❌ Duplicate email filtered: {entity.email}")
                elif entity.name and entity.name.strip():
                    name_key = entity.name.lower().strip()
                    if name_key not in seen_names:
                        seen_names.add(name_key)
                        opportunity_entities.append(entity)
                        print(f"   ✅ Added entity with name: {entity.name}")
                    else:
                        print(f"   ❌ Duplicate name filtered: {entity.name}")
                else:
                    print(f"   ❌ Entity has no email or name")
            
            consolidated.extend(opportunity_entities)
        
        print(f"\n📊 Entities after consolidation: {len(consolidated)}")
        print("Final consolidated entities:")
        for i, entity in enumerate(consolidated, 1):
            print(f"   {i}. {entity.name or 'No name'} | {entity.email or 'No email'} | {entity.phone_number or 'No phone'} | {entity.title or 'No title'}")
        
        # Test database storage
        print(f"\n💾 Storing {len(consolidated)} consolidated entities to database...")
        stored_count = entity_manager.store_entities(consolidated)
        print(f"✅ Successfully stored {stored_count} entities")
        
        # Verify database contents
        print("🔍 Verifying database contents...")
        db_entities = entity_manager.get_opportunity_entities(test_opportunity_id)
        print(f"📊 Found {len(db_entities)} entities in database:")
        
        for i, entity in enumerate(db_entities, 1):
            print(f"   {i}. {entity.get('Name', 'No name')} | {entity.get('Email', 'No email')} | {entity.get('PhoneNumber', 'No phone')} | {entity.get('Title', 'No title')}")
        
        # Expected results validation
        print("\n🎯 VALIDATION RESULTS:")
        
        expected_entities = 2  # John Smith + Jane Doe (deduplicated)
        if len(db_entities) == expected_entities:
            print(f"✅ SUCCESS: Found expected {expected_entities} unique entities")
        else:
            print(f"❌ FAILURE: Expected {expected_entities} entities, but found {len(db_entities)}")
        
        # Check for John Smith deduplication
        john_entities = [e for e in db_entities if e.get('Name') and 'john smith' in e.get('Name', '').lower()]
        if len(john_entities) == 1:
            print("✅ SUCCESS: John Smith properly deduplicated (1 entity)")
        else:
            print(f"❌ FAILURE: Found {len(john_entities)} John Smith entities (should be 1)")
        
        # Check for Jane Doe
        jane_entities = [e for e in db_entities if e.get('Name') and 'jane doe' in e.get('Name', '').lower()]
        if len(jane_entities) == 1:
            print("✅ SUCCESS: Jane Doe found (1 entity)")
        else:
            print(f"❌ FAILURE: Found {len(jane_entities)} Jane Doe entities (should be 1)")
        
        print("\n" + "="*80)
        if len(db_entities) == expected_entities and len(john_entities) == 1 and len(jane_entities) == 1:
            print("🎉 OVERALL TEST RESULT: SUCCESS - Entity deduplication working correctly!")
        else:
            print("💥 OVERALL TEST RESULT: FAILURE - Entity deduplication not working properly!")
        print("="*80)
        
        return len(db_entities) == expected_entities
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test data
        try:
            print(f"\n🧹 Cleaning up test data for opportunity {test_opportunity_id}...")
            entity_manager.delete_opportunity_entities(test_opportunity_id)
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup failed: {e}")

if __name__ == "__main__":
    success = test_entity_deduplication()
    sys.exit(0 if success else 1)
