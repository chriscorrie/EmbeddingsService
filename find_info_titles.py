#!/usr/bin/env python3
"""
Simple search for information-related titles
"""

from pymilvus import connections, Collection

def find_information_titles():
    try:
        connections.connect(alias="default", host="localhost", port="19530")
        
        titles_collection = Collection("opportunity_titles")
        titles_collection.load()
        
        print(f"Total entities in titles collection: {titles_collection.num_entities}")
        
        # Get all records and search manually
        print("\n=== Getting all records to search manually ===")
        results = titles_collection.query(
            expr="",  # Get all
            output_fields=["opportunity_id", "text_content"],
            limit=100  # Get all 100
        )
        
        print(f"Retrieved {len(results)} records")
        
        # Search for information-related titles
        info_matches = []
        for result in results:
            text = result['text_content'].lower()
            if 'information' in text:
                info_matches.append(result)
                
        print(f"\nFound {len(info_matches)} titles containing 'information':")
        for match in info_matches:
            print(f"  - {match['opportunity_id']}: {match['text_content']}")
            
        # Also check for management
        mgmt_matches = []
        for result in results:
            text = result['text_content'].lower()
            if 'management' in text:
                mgmt_matches.append(result)
                
        print(f"\nFound {len(mgmt_matches)} titles containing 'management':")
        for match in mgmt_matches[:5]:  # Show first 5
            print(f"  - {match['opportunity_id']}: {match['text_content']}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    find_information_titles()
