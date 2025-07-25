#!/usr/bin/env python3
"""
Simple script to check vector database collection counts
"""

from pymilvus import connections, Collection, utility

def check_collections():
    try:
        # Connect to Milvus
        connections.connect(alias='default', host='localhost', port='19530')
        
        # Get all collections
        collections = utility.list_collections()
        print('📊 Vector Database Collection Counts:')
        print('=' * 60)
        
        total_entities = 0
        for collection_name in sorted(collections):
            try:
                collection = Collection(collection_name)
                # DO NOT LOAD - just get count from metadata
                
                # Get entity count without loading into memory
                count = collection.num_entities
                total_entities += count
                
                # Format large numbers nicely
                if count > 1000000:
                    count_str = f'{count:,} ({count/1000000:.1f}M)'
                elif count > 1000:
                    count_str = f'{count:,} ({count/1000:.1f}K)'
                else:
                    count_str = f'{count:,}'
                    
                print(f'{collection_name:30} : {count_str:>15} entities')
                
            except Exception as e:
                print(f'{collection_name:30} : ❌ Error - {e}')
        
        print('=' * 60)
        print(f'{'Total entities':30} : {total_entities:,}')
        
        # Check if document_opportunity_mapping still exists
        if 'document_opportunity_mapping' in collections:
            print('\n⚠️  WARNING: document_opportunity_mapping collection still exists!')
            print('   This collection should be dropped as it is no longer used.')
        
    except Exception as e:
        print(f'❌ Connection error: {e}')

if __name__ == "__main__":
    check_collections()
