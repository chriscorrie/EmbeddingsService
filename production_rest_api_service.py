#!/usr/bin/env python3
"""
Enhanced Production RESTful Web Service with Integrated OpenAPI/Swagger Documentation
Combines the psearch_result_model = api.model('SimilaritySearchResult', {
    'opportunity_id': fields.String(description='Opportunity GUID'),
    'title_score': fields.Float(description='Title similarity score'),
    'description_score': fields.Float(description='Description similarity score'),
    'document_score': fields.Float(description='Document similarity score'),
    'combined_score': fields.Float(description='Combined score across all content types'),
    'document_match_count': fields.Integer(description='Number of matching document chunks'),
    'title_match_count': fields.Integer(description='Number of matching title chunks'),
    'description_match_count': fields.Integer(description='Number of matching description chunks')
})API functionality with proper OpenAPI documentation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import uuid
import sys
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our Phase 2 optimized modules
from scalable_processor import ScalableEnhancedProcessor
from enhanced_search_processor import EnhancedSearchProcessor
from document_section_analyzer import DocumentSectionAnalyzer

# Import production configuration
try:
    from production_config import (
        HOST, PORT, DEBUG, setup_logging, get_server_info,
        MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE, MAX_SEARCH_LIMIT, DEFAULT_SEARCH_LIMIT,
        MAX_CONCURRENT_REQUESTS, PROCESSING_TIMEOUT, API_PREFIX
    )
    # Setup production logging
    logger = setup_logging()
    logger.info("Using production configuration with Phase 2 optimizations")
except ImportError:
    # Fallback to basic configuration
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    API_PREFIX = '/api/v1'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=['*'])

# Initialize Flask-RESTX API with OpenAPI documentation
api = Api(
    app,
    version='2.0.0',
    title='Document Embedding Production API',
    description='Production RESTful API for document embedding generation and processing with Phase 2 optimizations',
    doc='/docs/',  # Swagger UI will be available at /docs/
    contact='Chris Corrie',
    license='MIT',
    prefix=API_PREFIX
)

# Create namespaces for logical grouping
health_ns = api.namespace('health', description='Health check operations')
embedding_ns = api.namespace('embeddings', description='Embedding generation operations')
status_ns = api.namespace('status', description='Processing status operations')
search_ns = api.namespace('search', description='Search operations')

# Define OpenAPI models for documentation
process_embeddings_model = api.model('ProcessEmbeddingsRequest', {
    'start_row_id': fields.Integer(required=True, description='Starting row ID for processing', example=1),
    'end_row_id': fields.Integer(required=True, description='Ending row ID for processing', example=100),
    'reprocess': fields.Boolean(default=False, description='Whether to reprocess existing records', example=True)
})

process_embeddings_response_model = api.model('ProcessEmbeddingsResponse', {
    'task_id': fields.String(description='Task identifier for tracking processing status'),
    'message': fields.String(description='Status message'),
    'total_opportunities': fields.Integer(description='Total number of opportunities to process'),
    'phase2_optimizations': fields.String(description='Optimization features enabled')
})

processing_status_model = api.model('ProcessingStatus', {
    'task_id': fields.String(description='Task identifier'),
    'status': fields.String(description='Current processing status'),
    'opportunities_processed': fields.Integer(description='Number of opportunities processed'),
    'total_opportunities': fields.Integer(description='Total opportunities to process'),
    'progress_percentage': fields.Float(description='Processing progress as percentage'),
    'start_time': fields.DateTime(description='Processing start time'),
    'end_time': fields.DateTime(description='Processing end time (null if still running)'),
    'elapsed_time': fields.Float(description='Elapsed processing time in seconds'),
    'estimated_remaining': fields.Float(description='Estimated remaining time in seconds'),
    'phase2_optimizations': fields.String(description='Optimization features enabled')
})

health_status_model = api.model('HealthStatus', {
    'status': fields.String(description='API health status'),
    'timestamp': fields.DateTime(description='Health check timestamp'),
    'version': fields.String(description='API version'),
    'processor_initialized': fields.Boolean(description='Whether the processor is initialized'),
    'features': fields.List(fields.String, description='Enabled features')
})

# Error response model for consistent error handling
error_response_model = api.model('ErrorResponse', {
    'error': fields.String(required=True, description='Error message describing what went wrong'),
    'error_code': fields.String(required=True, description='Machine-readable error code'),
    'details': fields.Raw(description='Additional error details (optional)'),
    'timestamp': fields.DateTime(description='When the error occurred'),
    'request_id': fields.String(description='Unique identifier for this request (for debugging)')
})

# Search models
similarity_search_model = api.model('SimilaritySearchRequest', {
    'query': fields.String(required=True, description='Search query text', example='machine learning software development'),
    'limit': fields.Integer(default=10, description='Maximum number of results to return'),
    'title_similarity_threshold': fields.Float(default=0.5, description='Minimum similarity score for title matches'),
    'description_similarity_threshold': fields.Float(default=0.5, description='Minimum similarity score for description matches'),
    'document_similarity_threshold': fields.Float(default=0.5, description='Minimum similarity score for document matches'),
    'boost_factor': fields.Float(default=1.0, description='Legacy boost factor (maintained for compatibility)'),
    'include_entities': fields.Boolean(default=False, description='Legacy entity extraction flag (maintained for compatibility)')
})

search_result_metadata_model = api.model('SearchResultMetadata', {
    'opportunity_id': fields.String(description='Opportunity GUID'),
    'file_id': fields.Integer(description='File identifier'),
    'file_location': fields.String(description='File location path'),
    'section_type': fields.String(description='Document section type'),
    'posted_date': fields.DateTime(description='Date opportunity was posted'),
    'title': fields.String(description='Opportunity title'),
    'description': fields.String(description='Opportunity description')
})

search_result_model = api.model('SearchResult', {
    'id': fields.String(description='Result identifier'),
    'text': fields.String(description='Matching text content'),
    'score': fields.Float(description='Similarity score (0.0-1.0)'),
    'boost_factor': fields.Float(description='Boost factor applied'),
    'boosted_score': fields.Float(description='Score after boosting'),
    'source': fields.String(description='Source collection name'),
    'metadata': fields.Nested(search_result_metadata_model, description='Result metadata'),
    'entities': fields.Raw(description='Extracted entities (if requested)')
})

similarity_search_response_model = api.model('SimilaritySearchResponse', {
    'results': fields.List(fields.Nested(search_result_model), description='Aggregated search results'),
    'total_results': fields.Integer(description='Total number of results found'),
    'query': fields.String(description='Original search query'),
    'processing_time_ms': fields.Float(description='Query processing time in milliseconds')
})

opportunity_search_model = api.model('OpportunitySearchRequest', {
    'opportunity_ids': fields.List(fields.String, required=True, description='List of opportunity GUIDs', 
                                 example=['12345678-1234-1234-1234-123456789abc', '87654321-4321-4321-4321-cba987654321']),
    'title_similarity_threshold': fields.Float(default=0.5, description='Minimum similarity score for title matches'),
    'description_similarity_threshold': fields.Float(default=0.5, description='Minimum similarity score for description matches'),
    'document_similarity_threshold': fields.Float(default=0.5, description='Minimum similarity score for document matches'),
    'start_posted_date': fields.String(description='Start date filter (YYYY-MM-DD format)', example='2024-01-01'),
    'end_posted_date': fields.String(description='End date filter (YYYY-MM-DD format)', example='2024-12-31'),
    'document_sow_boost_multiplier': fields.Float(default=0.0, description='Document SOW boost multiplier (0.0 or higher, where 1.0 = no boost, >1.0 = boost, <1.0 = penalty)'),
    'limit': fields.Integer(default=100, description='Maximum number of results to return')
})

opportunity_search_result_model = api.model('OpportunitySearchResult', {
    'opportunity_id': fields.String(description='Similar opportunity GUID'),
    'title_score': fields.Float(description='Title similarity score'),
    'description_score': fields.Float(description='Description similarity score'),
    'document_score': fields.Float(description='Document similarity score'),
    'document_match_count': fields.Integer(description='Number of matching document chunks'),
    'title_match_count': fields.Integer(description='Number of matching title chunks'),
    'description_match_count': fields.Integer(description='Number of matching description chunks')
})

opportunity_search_response_model = api.model('OpportunitySearchResponse', {
    'results': fields.List(fields.Nested(opportunity_search_result_model), description='Similar opportunities'),
    'total_results': fields.Integer(description='Total number of results found'),
    'request_id': fields.String(description='Request identifier'),
    'processing_time_ms': fields.Float(description='Processing time in milliseconds')
})

# Global variables for processing management
processor = None
search_processor = None
analyzer = None
processing_tasks = {}
processing_lock = threading.Lock()

@dataclass
class ProcessingTask:
    """Data class for tracking processing tasks with completely isolated stats"""
    task_id: str
    status: str
    opportunities_processed: int = 0
    total_opportunities: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    phase2_optimizations: str = "enabled"
    # Process-specific isolated stats (completely separate from shared processor)
    isolated_stats: Dict = None
    
    def __post_init__(self):
        if self.isolated_stats is None:
            self.isolated_stats = {
                'opportunities_processed': 0,  # Task-specific counter
                'documents_processed': 0,
                'documents_skipped': 0,
                'total_chunks_generated': 0,
                'entities_extracted': 0,
                'errors': 0
            }
    
    def increment_progress(self):
        """Increment the task-specific progress counter"""
        self.isolated_stats['opportunities_processed'] += 1
        self.opportunities_processed = self.isolated_stats['opportunities_processed']

def create_error_response(error_message: str, error_code: str, status_code: int = 400, details: Optional[Dict] = None, request_id: Optional[str] = None):
    """Create a standardized error response with proper structure and logging"""
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    error_response = {
        'error': error_message,
        'error_code': error_code,
        'timestamp': datetime.now().isoformat(),
        'request_id': request_id
    }
    
    if details:
        error_response['details'] = details
    
    # Log the error for debugging
    logger.error(f"API Error [{request_id}]: {error_code} - {error_message}")
    if details:
        logger.error(f"Error details [{request_id}]: {details}")
    
    return error_response, status_code

def initialize_services():
    """Initialize the processor and analyzer services"""
    global processor, analyzer, search_processor
    
    try:
        logger.info("Initializing scalable enhanced processor...")
        processor = ScalableEnhancedProcessor()
        
        logger.info("Initializing enhanced search processor...")
        search_processor = EnhancedSearchProcessor()
        
        logger.info("Initializing document section analyzer...")
        from document_section_analyzer import DocumentSectionAnalyzer
        analyzer = DocumentSectionAnalyzer()
        
        logger.info("All services initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

def update_task_progress(task_id: str, opportunities_processed: int):
    """Update task progress in real-time"""
    try:
        with processing_lock:
            if task_id in processing_tasks:
                processing_tasks[task_id].opportunities_processed = opportunities_processed
                logger.debug(f"Updated task {task_id} progress: {opportunities_processed} opportunities processed")
    except Exception as e:
        logger.warning(f"Failed to update progress for task {task_id}: {e}")

def start_progress_monitor(task_id: str):
    """Start background thread to monitor and update progress every 10 seconds"""
    def monitor():
        while True:
            try:
                with processing_lock:
                    task = processing_tasks.get(task_id)
                    if not task or task.status not in ["running"]:
                        logger.debug(f"Progress monitor stopping for task {task_id} (status: {task.status if task else 'not found'})")
                        break
                
                # Get current stats from processor if available
                if processor:
                    try:
                        # Try to get task-specific stats first
                        if hasattr(processor, 'task_specific_stats') and processor.task_specific_stats is not None:
                            # Sync ALL task-specific stats for real-time updates
                            processor_stats = processor.task_specific_stats.copy()
                            logger.debug(f"Using task-specific stats for {task_id}: {processor_stats}")
                            
                            # Update task progress with all current stats
                            with processing_lock:
                                if task_id in processing_tasks:
                                    task = processing_tasks[task_id]
                                    task.opportunities_processed = processor_stats.get('opportunities_processed', 0)
                                    # Update isolated stats with ALL current processor stats
                                    task.isolated_stats.update(processor_stats)
                        else:
                            # Fallback to shared stats
                            current_stats = processor.get_current_stats()
                            current_processed = current_stats.get('opportunities_processed', 0)
                            logger.debug(f"Using shared stats for {task_id}: {current_processed} opportunities")
                            update_task_progress(task_id, current_processed)
                    except Exception as e:
                        logger.debug(f"Progress monitor error for task {task_id}: {e}")
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.warning(f"Progress monitor exception for task {task_id}: {e}")
                break
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor, daemon=True, name=f"ProgressMonitor-{task_id[:8]}")
    monitor_thread.start()
    logger.info(f"Started progress monitor thread for task {task_id}")

def process_embeddings_background(task_id: str, start_row_id: int, end_row_id: int, reprocess: bool = False):
    """Background processing function"""
    global processing_tasks, processor
    
    start_time = time.time()
    
    try:
        # Update task status
        with processing_lock:
            if task_id in processing_tasks:
                processing_tasks[task_id].status = "running"
                processing_tasks[task_id].start_time = datetime.now()
        
        logger.info(f"Starting background processing for task {task_id}: rows {start_row_id}-{end_row_id}, reprocess={reprocess}")
        
        # Create completely task-specific progress tracking (no shared processor dependency)
        task_progress_tracker = {
            'opportunities_processed': 0,
            'documents_processed': 0,
            'documents_embedded': 0,
            'documents_skipped': 0,
            'titles_embedded': 0,
            'descriptions_embedded': 0,
            'total_chunks_generated': 0,
            'boilerplate_chunks_filtered': 0,
            'entities_extracted': 0,
            'errors': 0
        }
        
        def task_specific_progress_callback(processed_count=None):
            """Completely task-specific progress callback that doesn't rely on shared processor stats"""
            with processing_lock:
                if task_id in processing_tasks:
                    task = processing_tasks[task_id]
                    
                    # Update opportunities count from callback
                    if processed_count is not None:
                        task_progress_tracker['opportunities_processed'] = processed_count
                    else:
                        task_progress_tracker['opportunities_processed'] += 1
                    
                    # Also sync all current processor stats if available
                    if processor and hasattr(processor, 'task_specific_stats') and processor.task_specific_stats is not None:
                        # Merge current processor stats into our tracker
                        task_progress_tracker.update(processor.task_specific_stats)
                    
                    # Update the task with all isolated stats
                    task.opportunities_processed = task_progress_tracker['opportunities_processed']
                    task.isolated_stats.update(task_progress_tracker)
                    
                    logger.debug(f"Task {task_id} isolated progress: {task.opportunities_processed}/{task.total_opportunities} ops, {task_progress_tracker.get('documents_embedded', 0)} docs embedded")

        # Set the task-specific progress callback on the processor
        processor.progress_callback = task_specific_progress_callback
        
        # Process the batch using Phase 2 optimized processor
        result = processor.process_scalable_batch(
            start_row_id, 
            end_row_id, 
            replace_existing_records=reprocess,
            task_id=task_id
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Update final task status
        with processing_lock:
            if task_id in processing_tasks:
                task = processing_tasks[task_id]
                
                # Log the final stats before updating
                logger.info(f"Task {task_id} final stats from scalable processor: {result}")
                
                task.status = "completed"
                task.end_time = datetime.now()
                task.opportunities_processed = result.get('opportunities_processed', 0)
                
                # Update the isolated stats with final comprehensive results
                if task.isolated_stats and result:
                    task.isolated_stats.update({
                        'opportunities_processed': result.get('opportunities_processed', 0),
                        'documents_processed': result.get('documents_processed', 0),
                        'documents_embedded': result.get('documents_embedded', 0),
                        'documents_skipped': result.get('documents_skipped', 0),
                        'titles_embedded': result.get('titles_embedded', 0),
                        'descriptions_embedded': result.get('descriptions_embedded', 0),
                        'total_chunks_generated': result.get('total_chunks_generated', 0),
                        'boilerplate_chunks_filtered': result.get('boilerplate_chunks_filtered', 0),
                        'entities_extracted': result.get('entities_extracted', 0),
                        'errors': result.get('errors', 0),
                        'processing_time': result.get('processing_time', 0.0)
                    })
                    logger.info(f"Task {task_id} updated isolated_stats: {task.isolated_stats}")
                else:
                    logger.warning(f"Task {task_id} - No isolated_stats or result to update")
                
        logger.info(f"✓ Background processing completed for task {task_id} in {elapsed_time:.2f}s")
        logger.info(f"  Processed {result.get('opportunities_processed', 0)} opportunities")
        
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.error(f"✗ Background processing failed for task {task_id}: {e}")
        traceback.print_exc()
        
        # Update task with error status
        with processing_lock:
            if task_id in processing_tasks:
                task = processing_tasks[task_id]
                task.status = "failed"
                task.end_time = datetime.now()
                task.error_message = str(e)
    
    finally:
        # Clear progress callback to prevent memory leaks
        if processor:
            processor.progress_callback = None

# Health check endpoint
@health_ns.route('')
class HealthCheck(Resource):
    @health_ns.doc('health_check')
    @health_ns.response(200, 'Success', health_status_model)
    @health_ns.response(500, 'Internal Server Error', error_response_model)
    def get(self):
        """Health check endpoint"""
        try:
            server_info = get_server_info() if 'get_server_info' in globals() else {}
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0-phase2',
                'processor_initialized': processor is not None,
                'features': ['parallel_entity_extraction', 'bulk_vector_operations', 'offline_processing', 'performance_monitoring']
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'version': 'unknown',
                'processor_initialized': False,
                'features': []
            }, 500

# Process embeddings endpoint
@embedding_ns.route('/process-embeddings')
class ProcessEmbeddings(Resource):
    @embedding_ns.doc('process_embeddings')
    @embedding_ns.expect(process_embeddings_model)
    @embedding_ns.response(202, 'Processing Started', process_embeddings_response_model)
    @embedding_ns.response(400, 'Bad Request', error_response_model)
    @embedding_ns.response(500, 'Internal Server Error', error_response_model)
    def post(self):
        """Process embeddings for a range of opportunities"""
        try:
            data = request.get_json()
            
            # Validate input
            if not data:
                return create_error_response('No JSON data provided', 'MISSING_REQUEST_DATA')
            
            start_row_id = data.get('start_row_id')
            end_row_id = data.get('end_row_id')
            reprocess = data.get('reprocess', False)
            
            if start_row_id is None or end_row_id is None:
                return create_error_response('start_row_id and end_row_id are required', 'MISSING_REQUIRED_FIELDS')
            
            if not isinstance(start_row_id, int) or not isinstance(end_row_id, int):
                return create_error_response('start_row_id and end_row_id must be integers', 'INVALID_DATA_TYPE')
            
            if start_row_id < 1 or end_row_id < 1:
                return create_error_response('start_row_id and end_row_id must be positive integers', 'INVALID_RANGE')
            
            if start_row_id > end_row_id:
                return create_error_response('start_row_id must be less than or equal to end_row_id', 'INVALID_RANGE')
            
            # Check batch size limits
            batch_size = end_row_id - start_row_id + 1
            max_batch = MAX_BATCH_SIZE if 'MAX_BATCH_SIZE' in globals() else 1000
            if batch_size > max_batch:
                return create_error_response(
                    f'Batch size ({batch_size}) exceeds maximum allowed ({max_batch})', 
                    'BATCH_SIZE_EXCEEDED',
                    details={'requested_batch_size': batch_size, 'max_batch_size': max_batch}
                )
            
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Get total opportunities count
            total_opportunities = processor.get_opportunities_count(start_row_id, end_row_id)
            
            # Create task
            task = ProcessingTask(
                task_id=task_id,
                status="queued",
                total_opportunities=total_opportunities
            )
            
            # Store task
            with processing_lock:
                processing_tasks[task_id] = task
            
            # Start background processing
            thread = threading.Thread(
                target=process_embeddings_background,
                args=(task_id, start_row_id, end_row_id, reprocess),
                daemon=True
            )
            thread.start()
            
            logger.info(f"Started processing task {task_id} for rows {start_row_id}-{end_row_id} (reprocess={reprocess})")
            
            return {
                'task_id': task_id,
                'message': f'Processing started for rows {start_row_id}-{end_row_id}',
                'total_opportunities': total_opportunities,
                'phase2_optimizations': 'enabled'
            }, 202
            
        except Exception as e:
            logger.error(f"Error processing embeddings: {e}")
            traceback.print_exc()
            return create_error_response('Internal server error', 'INTERNAL_ERROR', details={'exception': str(e)})

# Processing status endpoints
@status_ns.route('/processing-status/<string:task_id>')
class ProcessingStatusById(Resource):
    @status_ns.doc('get_processing_status_by_id')
    @status_ns.response(200, 'Success', processing_status_model)
    @status_ns.response(404, 'Task Not Found', error_response_model)
    @status_ns.response(500, 'Internal Server Error', error_response_model)
    def get(self, task_id):
        """Get processing status for a specific task"""
        try:
            with processing_lock:
                task = processing_tasks.get(task_id)
            
            if not task:
                return create_error_response('Task not found', 'TASK_NOT_FOUND', status_code=404)
            
            # Calculate progress and timing
            progress_percentage = 0.0
            elapsed_time = 0.0
            estimated_remaining = None
            
            if task.start_time:
                # Calculate elapsed time based on task completion status
                if task.end_time:
                    # Task completed - use fixed end time
                    elapsed_time = (task.end_time - task.start_time).total_seconds()
                else:
                    # Task still running - use current time
                    elapsed_time = (datetime.now() - task.start_time).total_seconds()
                
                if task.total_opportunities > 0:
                    progress_percentage = (task.opportunities_processed / task.total_opportunities) * 100
                    
                    if task.opportunities_processed > 0 and task.status == "running":
                        rate = task.opportunities_processed / elapsed_time
                        remaining_opportunities = task.total_opportunities - task.opportunities_processed
                        estimated_remaining = remaining_opportunities / rate if rate > 0 else None
            
            # Get task-specific isolated stats (completely separate from shared processor)
            processor_stats = task.isolated_stats.copy() if task.isolated_stats else {}
            
            response = {
                'task_id': task.task_id,
                'status': task.status,
                'opportunities_processed': task.opportunities_processed,
                'total_opportunities': task.total_opportunities,
                'progress_percentage': progress_percentage,
                'start_time': task.start_time.isoformat() if task.start_time else None,
                'end_time': task.end_time.isoformat() if task.end_time else None,
                'elapsed_time': elapsed_time,
                'estimated_remaining': estimated_remaining,
                'phase2_optimizations': task.phase2_optimizations
            }
            
            # Add processor stats if available
            if processor_stats:
                response['processor_stats'] = processor_stats
                
            return response
            
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return create_error_response('Internal server error', 'STATUS_ERROR', details={'exception': str(e)})

@status_ns.route('/processing-status')
class ProcessingStatusAll(Resource):
    @status_ns.doc('get_all_processing_status')
    @status_ns.response(500, 'Internal Server Error', error_response_model)
    def get(self):
        """Get status of all processing tasks"""
        try:
            with processing_lock:
                tasks_info = []
                for task_id, task in processing_tasks.items():
                    # Calculate progress
                    progress_percentage = 0.0
                    elapsed_time = 0.0
                    
                    if task.start_time:
                        # Calculate elapsed time based on task completion status
                        if task.end_time:
                            # Task completed - use fixed end time
                            elapsed_time = (task.end_time - task.start_time).total_seconds()
                        else:
                            # Task still running - use current time
                            elapsed_time = (datetime.now() - task.start_time).total_seconds()
                            
                        if task.total_opportunities > 0:
                            progress_percentage = (task.opportunities_processed / task.total_opportunities) * 100
                    
                    tasks_info.append({
                        'task_id': task.task_id,
                        'status': task.status,
                        'opportunities_processed': task.opportunities_processed,
                        'total_opportunities': task.total_opportunities,
                        'progress_percentage': progress_percentage,
                        'start_time': task.start_time.isoformat() if task.start_time else None,
                        'end_time': task.end_time.isoformat() if task.end_time else None,
                        'elapsed_time': elapsed_time,
                        'phase2_optimizations': task.phase2_optimizations
                    })
            
            return {
                'tasks': tasks_info,
                'total_tasks': len(tasks_info)
            }
            
        except Exception as e:
            logger.error(f"Error getting all processing status: {e}")
            return create_error_response('Internal server error', 'STATUS_ERROR', details={'exception': str(e)})

# Search endpoints
@search_ns.route('/similarity-search')
class SimilaritySearch(Resource):
    @search_ns.doc('similarity_search')
    @search_ns.expect(similarity_search_model)
    @search_ns.response(400, 'Bad Request', error_response_model)
    @search_ns.response(500, 'Internal Server Error', error_response_model)
    # @search_ns.marshal_with(similarity_search_response_model)  # Temporarily disable marshalling
    def post(self):
        """Perform aggregated similarity search using a text query"""
        try:
            start_time = time.time()
            data = request.get_json()
            
            # Validate input
            if not data:
                return create_error_response('No JSON data provided', 'MISSING_REQUEST_DATA')
            
            query = data.get('query', '')
            if not query or not query.strip():
                return create_error_response('Query parameter is required and cannot be empty', 'MISSING_QUERY')
            
            limit = data.get('limit', 10)
            title_threshold = data.get('title_similarity_threshold', 0.5)
            description_threshold = data.get('description_similarity_threshold', 0.5)
            document_threshold = data.get('document_similarity_threshold', 0.5)
            
            # Legacy parameters (kept for backward compatibility)
            boost_factor = data.get('boost_factor', 1.0)
            include_entities = data.get('include_entities', False)
            
            # Validate parameters
            if not isinstance(limit, int) or limit < 1 or limit > 100:
                return create_error_response('Limit must be an integer between 1 and 100', 'INVALID_LIMIT', 
                                           details={'provided_limit': limit, 'valid_range': '1-100'})
            
            for threshold, name in [(title_threshold, 'title_similarity_threshold'),
                                  (description_threshold, 'description_similarity_threshold'),
                                  (document_threshold, 'document_similarity_threshold')]:
                if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
                    return create_error_response(f'{name} must be a number between 0.0 and 1.0', 'INVALID_THRESHOLD',
                                               details={'parameter': name, 'provided_value': threshold, 'valid_range': '0.0-1.0'})
            
            # Perform aggregated search using the enhanced search processor
            results = search_processor.search_similar_documents(
                query=query,
                limit=limit,
                title_similarity_threshold=title_threshold,
                description_similarity_threshold=description_threshold,
                document_similarity_threshold=document_threshold,
                boost_factor=boost_factor,  # Legacy parameter
                include_entities=include_entities  # Legacy parameter
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Aggregated similarity search completed for query '{query}': {len(results)} results in {processing_time:.2f}ms")
            
            return {
                'results': results,
                'total_results': len(results),
                'query': query,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Aggregated similarity search failed: {e}")
            traceback.print_exc()
            return create_error_response('Internal server error', 'SEARCH_ERROR', details={'exception': str(e)})

@search_ns.route('/opportunity-search')
class OpportunitySearch(Resource):
    @search_ns.doc('opportunity_search')
    @search_ns.expect(opportunity_search_model)
    @search_ns.response(200, 'Success', opportunity_search_response_model)
    @search_ns.response(400, 'Bad Request', error_response_model)
    @search_ns.response(500, 'Internal Server Error', error_response_model)
    def post(self):
        """Perform opportunity-based similarity search using opportunity GUIDs"""
        try:
            start_time = time.time()
            data = request.get_json()
            
            # Validate input
            if not data:
                return create_error_response('No JSON data provided', 'MISSING_REQUEST_DATA')
            
            opportunity_ids = data.get('opportunity_ids', [])
            if not opportunity_ids or not isinstance(opportunity_ids, list):
                return create_error_response('opportunity_ids must be a non-empty list', 'INVALID_OPPORTUNITY_IDS')
            
            # Validate GUID format
            import uuid
            try:
                for opp_id in opportunity_ids:
                    uuid.UUID(opp_id)  # This will raise ValueError if not a valid GUID
            except (ValueError, TypeError):
                return create_error_response('All opportunity_ids must be valid GUIDs', 'INVALID_GUID_FORMAT')
            
            # Extract parameters with defaults
            title_threshold = data.get('title_similarity_threshold', 0.5)
            description_threshold = data.get('description_similarity_threshold', 0.5)
            document_threshold = data.get('document_similarity_threshold', 0.5)
            start_date = data.get('start_posted_date')
            end_date = data.get('end_posted_date')
            boost_multiplier = data.get('document_sow_boost_multiplier', 0.0)
            limit = data.get('limit', 100)
            
            # Validate thresholds
            for threshold, name in [(title_threshold, 'title_similarity_threshold'),
                                  (description_threshold, 'description_similarity_threshold'),
                                  (document_threshold, 'document_similarity_threshold')]:
                if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
                    return create_error_response(f'{name} must be a number between 0.0 and 1.0', 'INVALID_THRESHOLD',
                                               details={'parameter': name, 'provided_value': threshold, 'valid_range': '0.0-1.0'})
            
            if not isinstance(boost_multiplier, (int, float)) or boost_multiplier < 0.0:
                return create_error_response('document_sow_boost_multiplier must be a non-negative number', 'INVALID_BOOST_MULTIPLIER',
                                           details={'provided_value': boost_multiplier, 'valid_range': '>=0.0'})
            
            if not isinstance(limit, int) or limit < 1 or limit > 1000:
                return create_error_response('Limit must be an integer between 1 and 1000', 'INVALID_LIMIT',
                                           details={'provided_limit': limit, 'valid_range': '1-1000'})
            
            # Validate date format if provided
            if start_date:
                try:
                    datetime.strptime(start_date, '%Y-%m-%d')
                except ValueError:
                    return create_error_response('start_posted_date must be in YYYY-MM-DD format', 'INVALID_DATE_FORMAT',
                                               details={'provided_date': start_date, 'expected_format': 'YYYY-MM-DD'})
            
            if end_date:
                try:
                    datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    return create_error_response('end_posted_date must be in YYYY-MM-DD format', 'INVALID_DATE_FORMAT',
                                               details={'provided_date': end_date, 'expected_format': 'YYYY-MM-DD'})
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            logger.info(f"Starting opportunity search {request_id} for {len(opportunity_ids)} opportunities")
            
            # Perform search using the enhanced search processor
            results = search_processor.search_similar_opportunities(
                opportunity_ids=opportunity_ids,
                title_similarity_threshold=title_threshold,
                description_similarity_threshold=description_threshold,
                document_similarity_threshold=document_threshold,
                start_posted_date=start_date,
                end_posted_date=end_date,
                document_sow_boost_multiplier=boost_multiplier,
                limit=limit
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Opportunity search {request_id} completed: {len(results)} results in {processing_time:.2f}ms")
            
            return {
                'results': results,
                'total_results': len(results),
                'request_id': request_id,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            logger.error(f"Opportunity search failed: {e}")
            traceback.print_exc()
            return create_error_response('Internal server error', 'SEARCH_ERROR', details={'exception': str(e)})

# Legacy endpoints for backward compatibility (without OpenAPI docs)
@app.route('/health', methods=['GET'])
def legacy_health():
    """Legacy health endpoint for backward compatibility"""
    try:
        server_info = get_server_info() if 'get_server_info' in globals() else {}
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0-phase2',
            'processor_initialized': processor is not None,
            'features': ['parallel_entity_extraction', 'bulk_vector_operations', 'offline_processing', 'performance_monitoring']
        })
    except Exception as e:
        logger.error(f"Legacy health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# Note: Root route is handled by Flask-RESTX automatically

if __name__ == '__main__':
    # Initialize services
    if not initialize_services():
        logger.error("Failed to initialize services. Exiting.")
        sys.exit(1)
    
    # Start the Flask application
    logger.info(f"Starting Production API v2.0.0 with Phase 2 optimizations on {HOST}:{PORT}")
    logger.info(f"API documentation available at: http://{HOST}:{PORT}/docs/")
    logger.info(f"Health check available at: http://{HOST}:{PORT}/health")
    
    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
        threaded=True
    )
