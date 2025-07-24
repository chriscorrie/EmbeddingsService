"""
SQL Entity Manager for storing and retrieving extracted entities
Handles database operations for entity extraction
"""

import pyodbc
import logging
from typing import List, Dict, Any, Optional, Set
from entity_extractor import LinkedEntity
import uuid

logger = logging.getLogger(__name__)

class SQLEntityManager:
    """
    Manages entity storage and retrieval in SQL Server
    Tracks processed files to avoid reprocessing
    Uses thread-safe connection management
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # Test the connection to ensure it's valid
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to ensure it's valid"""
        try:
            conn = pyodbc.connect(self.connection_string)
            conn.autocommit = True
            conn.close()
            logger.info("SQL Server connection string validated for entity management")
        except Exception as e:
            logger.error(f"Failed to validate SQL Server connection: {e}")
            raise
    
    def _get_connection(self):
        """Get a new thread-safe connection for each operation"""
        try:
            conn = pyodbc.connect(self.connection_string)
            conn.autocommit = True  # Enable autocommit to avoid transaction management issues
            return conn
        except Exception as e:
            logger.error(f"Failed to create SQL Server connection: {e}")
            raise
    
    def get_processed_files(self, opportunity_ids: List[str]) -> Set[int]:
        """
        Get set of file IDs that have already been processed for entity extraction
        
        Args:
            opportunity_ids: List of opportunity IDs to check
            
        Returns:
            Set of file IDs that have already been processed
        """
        if not opportunity_ids:
            return set()
        
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create parameter placeholders
            placeholders = ','.join(['?' for _ in opportunity_ids])
            
            query = f"""
                SELECT DISTINCT file_id 
                FROM FBOInternalAPI.ExtractedEntities 
                WHERE file_id IS NOT NULL 
                AND opportunity_id IN ({placeholders})
            """
            
            cursor.execute(query, opportunity_ids)
            rows = cursor.fetchall()
            
            processed_files = {row[0] for row in rows if row[0] is not None}
            
            logger.debug(f"Found {len(processed_files)} already processed files")
            return processed_files
            
        except Exception as e:
            logger.error(f"Error getting processed files: {e}")
            return set()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def store_entities(self, entities: List[LinkedEntity]) -> int:
        """
        Store extracted entities in SQL Server
        
        Args:
            entities: List of LinkedEntity objects to store
            
        Returns:
            Number of entities stored
        """
        if not entities:
            return 0
        
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare batch insert (Organization column removed)
            insert_sql = """
                INSERT INTO FBOInternalAPI.ExtractedEntities (
                    OpportunityId, FileId, SourceType, Name, Email, 
                    PhoneNumber, Title, ConfidenceScore, 
                    ExtractionMethod
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            batch_data = []
            for entity in entities:
                # Handle file_id conversion - ensure it's either None or a valid integer
                file_id = None
                if entity.file_id is not None:
                    try:
                        file_id = int(entity.file_id)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid file_id '{entity.file_id}' for entity, setting to None")
                        file_id = None
                
                batch_data.append((
                    entity.opportunity_id,
                    file_id,  # Use the properly converted file_id
                    entity.source_type,
                    entity.name,
                    entity.email,
                    entity.phone_number,
                    entity.title,
                    entity.confidence_score,
                    entity.extraction_method
                ))
            
            # Execute batch insert
            cursor.executemany(insert_sql, batch_data)
            
            stored_count = len(batch_data)
            logger.info(f"Stored {stored_count} entities in database")
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing entities: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def delete_opportunity_entities(self, opportunity_id: str):
        """
        Delete all entities for a specific opportunity
        
        Args:
            opportunity_id: GUID of the opportunity to delete entities for
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM FBOInternalAPI.ExtractedEntities 
                WHERE OpportunityId = ?
            """, (opportunity_id,))
            
            rows_deleted = cursor.rowcount
            
            logger.debug(f"Deleted {rows_deleted} entities for opportunity {opportunity_id}")
            
        except Exception as e:
            logger.error(f"Error deleting entities for opportunity {opportunity_id}: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_opportunity_entities(self, opportunity_id: str) -> List[Dict[str, Any]]:
        """
        Get all entities for a specific opportunity
        
        Args:
            opportunity_id: GUID of the opportunity
            
        Returns:
            List of entity dictionaries
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT EntityId, OpportunityId, FileId, SourceType, 
                       Name, Email, PhoneNumber, Title, 
                       ConfidenceScore, ExtractionMethod
                FROM FBOInternalAPI.ExtractedEntities
                WHERE OpportunityId = ?
                ORDER BY ConfidenceScore DESC
            """, (opportunity_id,))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            entities = []
            for row in rows:
                entity = dict(zip(columns, row))
                # Convert UUID to string for JSON serialization
                if entity['opportunity_id']:
                    entity['opportunity_id'] = str(entity['opportunity_id'])
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error getting entities for opportunity {opportunity_id}: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def search_entities(self, search_term: str, entity_types: Optional[List[str]] = None,
                       opportunity_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for entities across opportunities
        
        Args:
            search_term: Text to search for
            entity_types: List of entity types to search in ('name', 'email', 'phone_number', 'title')
            opportunity_id: Optional opportunity ID to limit search
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build search conditions
            where_clauses = []
            params = []
            
            if opportunity_id:
                where_clauses.append("opportunity_id = ?")
                params.append(opportunity_id)
            
            # Search across specified entity types or all
            if entity_types:
                type_conditions = []
                for entity_type in entity_types:
                    if entity_type in ['name', 'email', 'phone_number', 'title']:
                        type_conditions.append(f"{entity_type} LIKE ?")
                        params.append(f"%{search_term}%")
                if type_conditions:
                    where_clauses.append(f"({' OR '.join(type_conditions)})")
            else:
                # Search all entity types (Organization removed)
                type_conditions = [
                    "name LIKE ?",
                    "email LIKE ?",
                    "phone_number LIKE ?",
                    "title LIKE ?"
                ]
                where_clauses.append(f"({' OR '.join(type_conditions)})")
                params.extend([f"%{search_term}%"] * 4)
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            query = f"""
                SELECT TOP {limit}
                    EntityId, OpportunityId, FileId, SourceType,
                    Name, Email, PhoneNumber, Title,
                    ConfidenceScore
                FROM FBOInternalAPI.ExtractedEntities
                WHERE {where_clause}
                ORDER BY ConfidenceScore DESC
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            results = []
            for row in rows:
                entity = dict(zip(columns, row))
                if entity['opportunity_id']:
                    entity['opportunity_id'] = str(entity['opportunity_id'])
                results.append(entity)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about extracted entities
        
        Returns:
            Dictionary with entity statistics
        """
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Total entities
            cursor.execute("SELECT COUNT(*) FROM FBOInternalAPI.ExtractedEntities")
            total_entities = cursor.fetchone()[0]
            
            # Entities by attribute
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN name IS NOT NULL THEN 1 END) as with_name,
                    COUNT(CASE WHEN email IS NOT NULL THEN 1 END) as with_email,
                    COUNT(CASE WHEN PhoneNumber IS NOT NULL THEN 1 END) as with_phone,
                    COUNT(CASE WHEN Title IS NOT NULL THEN 1 END) as with_title,
                    COUNT(CASE WHEN Name IS NOT NULL AND Email IS NOT NULL THEN 1 END) as name_and_email,
                    COUNT(CASE WHEN Name IS NOT NULL AND PhoneNumber IS NOT NULL THEN 1 END) as name_and_phone,
                    AVG(ConfidenceScore) as avg_confidence
                FROM FBOInternalAPI.ExtractedEntities
            """)
            
            stats_row = cursor.fetchone()
            
            # Entities by source type
            cursor.execute("""
                SELECT SourceType, COUNT(*) as count
                FROM FBOInternalAPI.ExtractedEntities
                GROUP BY SourceType
                ORDER BY count DESC
            """)
            source_counts = dict(cursor.fetchall())
            
            return {
                'total_entities': total_entities,
                'entities_with_name': stats_row[0],
                'entities_with_email': stats_row[1],
                'entities_with_phone': stats_row[2],
                'entities_with_title': stats_row[3],
                'complete_name_email': stats_row[4],
                'complete_name_phone': stats_row[5],
                'average_confidence': float(stats_row[6]) if stats_row[6] else 0.0,
                'source_type_counts': source_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting entity statistics: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def close(self):
        """Close database connections - no-op since we use fresh connections per operation"""
        logger.info("SQLEntityManager uses per-operation connections - no persistent connection to close")
