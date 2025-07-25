#!/usr/bin/env python3
"""
DATABASE RESET SCRIPT

⚠️  WARNING: This will PERMANENTLY DELETE ALL DATA:
- ALL Milvus vector collections (embeddings, search indexes)  
- SQL Server ExtractedEntities table (all entity data)

This operation CANNOT be undone!
"""

import os
import sys
import json
from datetime import datetime
import pyodbc
from pymilvus import connections, utility, Collection

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

class DatabaseResetManager:
    """Manages complete database reset operations"""
    
    def __init__(self):
        self.milvus_connected = False
        self.sql_connected = False
        
    def get_user_confirmation(self):
        """Get user confirmation for reset"""
        print("\n� Environment Check:")
        print(f"   📍 Working Directory: {os.getcwd()}")
        print(f"   🗄️  Milvus Vector DB: {config.VECTOR_DB_PATH}")
        print(f"   💾 SQL Server: {config.SQL_SERVER_CONNECTION_STRING.split(';')[1].split('=')[1]}")
        
        confirm_env = input("\nIs this the correct environment to reset? (yes/no): ").strip().lower()
        if confirm_env != 'yes':
            print("❌ Operation cancelled")
            return False
            
        print("\n⚠️  This will DELETE ALL embeddings and entity data!")
        final_confirm = input("Type 'RESET' to proceed with database reset: ").strip()
        if final_confirm != 'RESET':
            print("❌ Operation cancelled")
            return False
            
        return True
        
    def connect_to_milvus(self):
        """Connect to Milvus vector database"""
        try:
            print("🔌 Connecting to Milvus...")
            connections.connect('default', host='localhost', port='19530')
            self.milvus_connected = True
            print("✅ Connected to Milvus vector database")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            return False
            
    def connect_to_sql_server(self):
        """Connect to SQL Server database"""
        try:
            print("🔌 Connecting to SQL Server...")
            self.sql_connection = pyodbc.connect(config.SQL_SERVER_CONNECTION_STRING)
            self.sql_connected = True
            print("✅ Connected to SQL Server database")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to SQL Server: {e}")
            return False
            
    def backup_collection_info(self):
        """Backup collection information before deletion"""
        try:
            print("📋 Backing up collection information...")
            collections = utility.list_collections()
            
            backup_info = {
                'timestamp': datetime.now().isoformat(),
                'collections': {}
            }
            
            for collection_name in collections:
                try:
                    collection = Collection(collection_name)
                    backup_info['collections'][collection_name] = {
                        'entity_count': collection.num_entities,
                        'schema': str(collection.schema)
                    }
                    print(f"   📁 {collection_name}: {collection.num_entities:,} entities")
                except Exception as e:
                    print(f"   ❌ {collection_name}: Error backing up - {e}")
                    
            # Save backup info to file
            import json
            backup_file = f"database_reset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(backup_info, f, indent=2)
                
            print(f"✅ Backup information saved to {backup_file}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to backup collection info: {e}")
            return False
            
    def drop_milvus_collections(self):
        """Drop all Milvus collections"""
        try:
            print("\n🗑️  DROPPING MILVUS COLLECTIONS...")
            collections = utility.list_collections()
            
            if not collections:
                print("   ℹ️  No collections found to drop")
                return True
                
            for collection_name in collections:
                try:
                    print(f"   🗑️  Dropping collection: {collection_name}")
                    utility.drop_collection(collection_name)
                    print(f"   ✅ Dropped: {collection_name}")
                except Exception as e:
                    print(f"   ❌ Failed to drop {collection_name}: {e}")
                    return False
                    
            print("✅ All Milvus collections dropped successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to drop Milvus collections: {e}")
            return False
            
    def recreate_milvus_collections(self):
        """Recreate Milvus collections using the centralized schema setup script"""
        try:
            print("\n🏗️  RECREATING MILVUS COLLECTIONS...")
            print("   📋 Using centralized schema from setup_vector_db.py")
            
            # Import and call the centralized setup function
            from setup_vector_db import setup_vector_database
            
            print("   🏗️  Creating collections with proper schemas...")
            created_collections = setup_vector_database()
            
            if created_collections:
                print(f"   ✅ Successfully created {len(created_collections)} collections:")
                for collection_name in created_collections.keys():
                    print(f"      • {collection_name}")
                print("✅ All Milvus collections recreated successfully")
                return True
            else:
                print("   ❌ No collections were created")
                return False
                
        except Exception as e:
            print(f"❌ Failed to recreate Milvus collections: {e}")
            print("   💡 Ensure setup_vector_db.py is available and working")
            return False
            
    def reset_sql_entities_table(self):
        """Reset the SQL Server ExtractedEntities table"""
        try:
            print("\n🗑️  RESETTING SQL SERVER ENTITIES TABLE...")
            
            cursor = self.sql_connection.cursor()
            
            # Get current count before deletion
            cursor.execute("SELECT COUNT(*) FROM [FBOInternalAPI].[ExtractedEntities]")
            count_before = cursor.fetchone()[0]
            print(f"   📊 Current entity records: {count_before:,}")
            
            # Truncate the table
            print("   🗑️  Truncating [FBOInternalAPI].[ExtractedEntities] table...")
            cursor.execute("TRUNCATE TABLE [FBOInternalAPI].[ExtractedEntities]")
            self.sql_connection.commit()
            
            # Verify truncation
            cursor.execute("SELECT COUNT(*) FROM [FBOInternalAPI].[ExtractedEntities]")
            count_after = cursor.fetchone()[0]
            
            if count_after == 0:
                print(f"✅ Successfully deleted {count_before:,} entity records")
                return True
            else:
                print(f"❌ Truncation failed - {count_after} records remain")
                return False
                
        except Exception as e:
            print(f"❌ Failed to reset SQL entities table: {e}")
            return False
            
    def cleanup_connections(self):
        """Clean up database connections"""
        try:
            if self.milvus_connected:
                connections.disconnect('default')
                print("🔌 Disconnected from Milvus")
                
            if self.sql_connected:
                self.sql_connection.close()
                print("🔌 Disconnected from SQL Server")
                
        except Exception as e:
            print(f"⚠️  Warning during cleanup: {e}")
            
    def run_complete_reset(self):
        """Execute the complete database reset process"""
        print("🚀 STARTING COMPLETE DATABASE RESET")
        print("=" * 50)
        
        success = True
        
        # Step 1: Connect to databases
        if not self.connect_to_milvus():
            success = False
        if not self.connect_to_sql_server():
            success = False
            
        if not success:
            print("❌ Failed to connect to required databases")
            return False
            
        # Step 2: Backup collection information
        if not self.backup_collection_info():
            print("⚠️  Backup failed - continuing with reset...")
            
        # Step 3: Drop Milvus collections
        if not self.drop_milvus_collections():
            success = False
            
        # Step 4: Reset SQL entities table
        if not self.reset_sql_entities_table():
            success = False
            
        # Step 5: Recreate Milvus collections
        if not self.recreate_milvus_collections():
            print("⚠️  Collection recreation failed - you may need to run create_index.py")
            
        # Cleanup
        self.cleanup_connections()
        
        if success:
            print("\n✅ DATABASE RESET COMPLETED SUCCESSFULLY!")
            print("\n📋 Next Steps:")
            print("   1. Run document processing to rebuild embeddings")
            print("   2. Run entity extraction to rebuild entity data")
            print("   3. Verify API functionality with test searches")
        else:
            print("\n❌ DATABASE RESET FAILED")
            print("   Some operations may have completed partially")
            print("   Check the output above for specific errors")
            
        return success

def main():
    """Main entry point"""
    print(__doc__)
    
    reset_manager = DatabaseResetManager()
    
    # Get confirmation
    if not reset_manager.get_user_confirmation():
        return
        
    print("\n🚀 Starting database reset...")
    
    # Execute the reset
    success = reset_manager.run_complete_reset()
    
    if success:
        print("\n✅ Database reset completed successfully!")
    else:
        print("\n❌ Database reset failed - check errors above")
        sys.exit(1)

if __name__ == '__main__':
    main()
