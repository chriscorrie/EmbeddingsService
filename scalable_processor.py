#!/usr/bin/env python3
"""
Scalable Enhanced Document Processor with Intelligent Resource Management
"""

import os
import sys
import logging
import time
import threading
import queue
import numpy as np
from typing import List, Dict, Set, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional imports with graceful fallbacks
try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False
    
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional psutil import with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import performance timer
from performance_timer import time_operation, start_timer, end_timer, print_summary, save_report

# Import our modules
from process_documents import extract_text_from_file
from semantic_chunker import create_chunker
from semantic_boilerplate_manager import create_boilerplate_manager
from entity_extractor import LinkedEntity
from resource_manager import get_optimal_configuration
from config import (
    DOCUMENTS_PATH,
    DOCUMENT_PATH_TO_REPLACE,
    DOCUMENT_PATH_REPLACEMENT_VALUE,
    EMBEDDING_MODEL,
    SQL_CONNECTION_STRING,
    BOILERPLATE_SIMILARITY_THRESHOLD,
    BOILERPLATE_DOCS_PATH,
    ENABLE_ENTITY_EXTRACTION,
    ENTITY_CONF_THRESHOLD,
    ENABLE_PARALLEL_PROCESSING,
    MAX_OPPORTUNITY_WORKERS,
    MAX_FILE_WORKERS_PER_OPPORTUNITY,
    ENABLE_MEMORY_MONITORING,
    EMBEDDING_BATCH_SIZE,
    ENTITY_BATCH_SIZE,
    VECTOR_INSERT_BATCH_SIZE,
    ENABLE_OPPORTUNITY_BATCH_COMMITS,
    ENABLE_PRODUCER_CONSUMER_ARCHITECTURE,
    ENABLE_EMBEDDING_MODEL_POOL,
    EMBEDDING_MODEL_POOL_SIZE,
    EMBEDDING_TIMEOUT_SECONDS,
    ENABLE_GPU_ACCELERATION,
    GPU_DEVICE,
    GPU_BATCH_SIZE_MULTIPLIER,
    FALLBACK_TO_CPU
)
# Optional imports with graceful fallbacks
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scalable_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Opportunity:
    """Data class representing an opportunity with its metadata and documents"""
    def __init__(self, opportunity_id: str, title: str, description: str, posted_date: str = None):
        self.opportunity_id = opportunity_id
        self.title = title
        self.description = description
        self.posted_date = posted_date
        self.documents = []  # List of Document objects
    
    def add_document(self, document):
        """Add a document to this opportunity"""
        self.documents.append(document)

class Document:
    """Data class representing a document with its metadata and optional pre-loaded text"""
    def __init__(self, file_id: int, file_location: str, file_size_bytes: int = None, text_content: str = None):
        self.file_id = file_id
        self.file_location = file_location
        self.file_size_bytes = file_size_bytes
        self.text_content = text_content  # Pre-loaded text data (None if not loaded)
        self.load_error = None  # Track any errors during loading
    
    def is_text_loaded(self) -> bool:
        """Check if text content is already loaded"""
        return self.text_content is not None
    
    def get_memory_footprint_mb(self) -> float:
        """Get approximate memory footprint in MB"""
        if self.text_content:
            return len(self.text_content.encode('utf-8')) / (1024 * 1024)
        return 0.0

class EntityExtractionQueue:
    """
    Asynchronous entity extraction queue that runs independently from main processing pipeline
    """
    
    def __init__(self, entity_extractor, entity_manager, enable_entity_extraction: bool = True):
        """
        Initialize the entity extraction queue
        
        Args:
            entity_extractor: The entity extractor instance
            entity_manager: The entity manager for database operations
            enable_entity_extraction: Whether entity extraction is enabled
        """
        self.entity_extractor = entity_extractor
        self.entity_manager = entity_manager
        self.enable_entity_extraction = enable_entity_extraction
        
        if not self.enable_entity_extraction:
            logger.info("Entity extraction disabled - queue will not process tasks")
            return
        
        # Entity extraction queue and workers
        self.entity_queue = queue.Queue(maxsize=1000)  # Large queue for async processing
        self.worker_count = 2  # Dedicated entity extraction workers
        self.workers = []
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats_lock = threading.Lock()
        self.stats = {
            'tasks_queued': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'entities_extracted': 0
        }
        
        # Start workers
        self._start_workers()
        logger.info(f"✅ EntityExtractionQueue initialized with {self.worker_count} workers")
    
    def _start_workers(self):
        """Start entity extraction worker threads"""
        for i in range(self.worker_count):
            worker = threading.Thread(
                target=self._entity_worker,
                args=(i,),
                name=f"EntityWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _entity_worker(self, worker_id: int):
        """Entity extraction worker thread"""
        logger.info(f"EntityWorker-{worker_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task = self.entity_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                # Process entity extraction task
                self._process_entity_task(task, worker_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"EntityWorker-{worker_id} error: {e}")
                with self.stats_lock:
                    self.stats['tasks_failed'] += 1
            finally:
                try:
                    self.entity_queue.task_done()
                except ValueError:
                    pass
        
        logger.info(f"EntityWorker-{worker_id} shutdown")
    
    def _process_entity_task(self, task: Dict, worker_id: int):
        """Process a single entity extraction task"""
        try:
            with time_operation('async_entity_extraction', {
                'file_id': task.get('file_id'),
                'text_length': len(task['text_content']),
                'worker_id': worker_id
            }):
                # Extract entities
                entities = self.entity_extractor.extract_entities(
                    task['text_content'],
                    task['opportunity_id'],
                    task['content_type'],
                    task.get('file_id')
                )
                
                # Store entities if any were found
                if entities:
                    # Filter and consolidate entities
                    filtered_entities = self._filter_entities_by_confidence(entities)
                    if filtered_entities:
                        stored_count = self.entity_manager.store_entities(filtered_entities)
                        
                        with self.stats_lock:
                            self.stats['entities_extracted'] += stored_count
                        
                        logger.debug(f"EntityWorker-{worker_id}: Stored {stored_count} entities for {task['content_type']} {task.get('file_id', task['opportunity_id'])}")
                
                with self.stats_lock:
                    self.stats['tasks_completed'] += 1
                    
        except Exception as e:
            logger.error(f"EntityWorker-{worker_id}: Error processing entity task: {e}")
            with self.stats_lock:
                self.stats['tasks_failed'] += 1
    
    def _filter_entities_by_confidence(self, entities: List) -> List:
        """Filter entities by confidence threshold"""
        try:
            from config import ENTITY_CONF_THRESHOLD
            return [entity for entity in entities if entity.confidence_score >= ENTITY_CONF_THRESHOLD]
        except ImportError:
            return entities  # If no threshold configured, return all
    
    def submit_task(self, text_content: str, opportunity_id: str, content_type: str, file_id: int = None):
        """
        Submit an entity extraction task (non-blocking)
        
        Args:
            text_content: Text to extract entities from
            opportunity_id: Opportunity ID
            content_type: Type of content ('document', 'description', etc.)
            file_id: Optional file ID
        """
        if not self.enable_entity_extraction:
            return
        
        # Skip very short text
        if len(text_content.strip()) < 50:
            return
        
        task = {
            'text_content': text_content,
            'opportunity_id': opportunity_id,
            'content_type': content_type,
            'file_id': file_id,
            'submitted_at': time.time()
        }
        
        try:
            self.entity_queue.put_nowait(task)
            with self.stats_lock:
                self.stats['tasks_queued'] += 1
            logger.debug(f"Entity extraction task queued for {content_type} {file_id or opportunity_id}")
        except queue.Full:
            logger.warning(f"Entity extraction queue full, dropping task for {content_type} {file_id or opportunity_id}")
    
    def get_stats(self) -> Dict:
        """Get entity extraction statistics"""
        with self.stats_lock:
            return self.stats.copy()
    
    def shutdown(self, timeout: float = 30.0):
        """Shutdown entity extraction queue"""
        if not self.enable_entity_extraction:
            return
        
        logger.info("Shutting down EntityExtractionQueue...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Add shutdown sentinels
        for _ in self.workers:
            try:
                self.entity_queue.put_nowait(None)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)
        
        # Log final stats
        final_stats = self.get_stats()
        logger.info(f"EntityExtractionQueue shutdown complete. Final stats: {final_stats}")

class ScalableEnhancedProcessor:
    """
    Enhanced document processor with intelligent resource scaling
    """
    
    def __init__(self, custom_config: Dict = None, progress_callback=None):
        """
        Initialize the scalable processor with intelligent resource management
        
        Args:
            custom_config: Override default resource configuration
            progress_callback: Optional callback function for real-time progress updates
        """
        # Store progress callback for real-time status updates
        self.progress_callback = progress_callback
        # Check for aggressive configuration bypass
        try:
            from config import BYPASS_RESOURCE_MANAGER, FORCE_AGGRESSIVE_CONFIG
            bypass_resource_manager = BYPASS_RESOURCE_MANAGER
            force_aggressive = FORCE_AGGRESSIVE_CONFIG
        except ImportError:
            bypass_resource_manager = False
            force_aggressive = False
        
        if bypass_resource_manager and force_aggressive:
            # Use aggressive configuration directly from config.py
            logger.info("🚀 AGGRESSIVE MODE: Bypassing conservative resource manager")
            self.opportunity_workers = MAX_OPPORTUNITY_WORKERS
            self.file_workers_per_opportunity = MAX_FILE_WORKERS_PER_OPPORTUNITY
            self.batch_sizes = {
                'embedding_batch_size': EMBEDDING_BATCH_SIZE,
                'entity_batch_size': ENTITY_BATCH_SIZE,
                'vector_batch_size': VECTOR_INSERT_BATCH_SIZE
            }
            self.resource_config = {
                'system_healthy': True,
                'health_message': 'Aggressive mode - bypassing checks'
            }
        else:
            # Get optimal configuration for this system
            self.resource_config = get_optimal_configuration()
            
            # Override with custom configuration if provided
            if custom_config:
                self.resource_config.update(custom_config)
            
            # Set worker counts based on optimal configuration
            if ENABLE_PARALLEL_PROCESSING:
                self.opportunity_workers = min(
                    self.resource_config['optimal_workers']['opportunity_workers'],
                    MAX_OPPORTUNITY_WORKERS
                )
                self.file_workers_per_opportunity = min(
                    self.resource_config['optimal_workers']['file_workers_per_opportunity'],
                    MAX_FILE_WORKERS_PER_OPPORTUNITY
                )
            else:
                self.opportunity_workers = 1
                self.file_workers_per_opportunity = 1
            
            # Dynamic batch sizes based on available memory
            self.batch_sizes = self.resource_config['batch_sizes']
        
        logger.info(f"🚀 Scalable Processor Initialized:")
        logger.info(f"   Opportunity Workers: {self.opportunity_workers}")
        logger.info(f"   File Workers/Opportunity: {self.file_workers_per_opportunity}")
        logger.info(f"   Embedding Batch Size: {self.batch_sizes['embedding_batch_size']}")
        logger.info(f"   Total Parallel Capacity: {self.opportunity_workers * self.file_workers_per_opportunity} concurrent operations")
        
        # Initialize components
        self.setup_milvus_connection()
        self.setup_sql_connection()
        self.setup_embeddings()
        
        # Entity extraction setup (only if enabled)
        self.enable_entity_extraction = ENABLE_ENTITY_EXTRACTION
        if self.enable_entity_extraction:
            self.setup_entity_extraction()
            # Initialize asynchronous entity extraction queue
            self.entity_queue = EntityExtractionQueue(
                self.entity_extractor, 
                self.entity_manager, 
                self.enable_entity_extraction
            )
            logger.info("✅ Entity extraction enabled with async queue")
        else:
            self.entity_extractor = None
            self.entity_manager = None
            self.entity_queue = None
            logger.info("❌ Entity extraction disabled")
        
        # Initialize processing components (always use config file for boilerplate path)
        boilerplate_docs_path = BOILERPLATE_DOCS_PATH
            
        self.chunker = create_chunker(target_size=1500, overlap=0.2)
        self.boilerplate_manager = create_boilerplate_manager(
            self.embeddings, 
            BOILERPLATE_SIMILARITY_THRESHOLD
        )
        
        # Process boilerplate documents (no database collection needed)
        if os.path.exists(boilerplate_docs_path):
            logger.info("Processing boilerplate documents...")
            bp_chunks = self.boilerplate_manager.process_boilerplate_documents(
                boilerplate_docs_path, 
                self.chunker
            )
            logger.info(f"Processed {bp_chunks} boilerplate chunks")
            self.boilerplate_manager.load_boilerplate_embeddings_cache()
        else:
            logger.warning(f"Boilerplate documents path not found: {boilerplate_docs_path}")
        
        # Track processed opportunities
        self.processed_opportunities: Set[str] = set()
        
        # Batch commit tracking for vector database flushing
        self.vector_batch_size = VECTOR_INSERT_BATCH_SIZE
        self.enable_batch_commits = ENABLE_OPPORTUNITY_BATCH_COMMITS
        
        self.opportunities_since_last_flush = 0  # Track opportunities processed since last flush
        self.batch_commit_lock = threading.Lock()  # Thread-safe batch counting
        
        # Thread-safe file processing cache (protects against race conditions)
        self.processed_files: Set[int] = set()  # Track processed file IDs (session + database)
        self.processed_files_lock = threading.Lock()  # Protects processed_files set
        
        # Thread-safe statistics
        self.stats_lock = threading.Lock()
        self.stats = {
            'opportunities_processed': 0,
            'titles_embedded': 0,
            'descriptions_embedded': 0,
            'documents_embedded': 0,
            'documents_skipped': 0,
            'documents_already_processed': 0,  # Files skipped due to previous processing
            'documents_race_condition_skipped': 0,  # Files skipped due to thread race conditions
            'total_chunks_generated': 0,
            'boilerplate_chunks_filtered': 0,
            'entities_extracted': 0,
            'errors': 0,
            'processing_time': 0.0,
            'peak_memory_mb': 0.0,
            'avg_cpu_percent': 0.0,
            # Queue monitoring statistics
            'queue_max_size': 0,
            'queue_avg_size': 0.0,
            'queue_samples': 0,
            'producer_throughput_mb_per_sec': 0.0,
            'file_load_errors': 0
        }
        
        # Memory monitoring setup
        self.memory_monitor = None
        if ENABLE_MEMORY_MONITORING:
            self.setup_memory_monitoring()
    
    def setup_memory_monitoring(self):
        """Setup memory monitoring during processing"""
        if not PSUTIL_AVAILABLE:
            logger.warning("Memory monitoring disabled - psutil not available")
            return
            
        def monitor_resources():
            max_memory = 0
            cpu_samples = []
            
            while getattr(self, '_monitoring_active', False):
                try:
                    memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                    cpu_percent = psutil.cpu_percent()
                    
                    max_memory = max(max_memory, memory_mb)
                    cpu_samples.append(cpu_percent)
                    
                    with self.stats_lock:
                        self.stats['peak_memory_mb'] = max_memory
                        if cpu_samples:
                            self.stats['avg_cpu_percent'] = sum(cpu_samples) / len(cpu_samples)
                    
                    time.sleep(1)  # Sample every second
                except Exception as e:
                    logger.debug(f"Memory monitoring error: {e}")
                    break
        
        self._monitoring_active = True
        self.memory_monitor = threading.Thread(target=monitor_resources, daemon=True)
        self.memory_monitor.start()
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring"""
        self._monitoring_active = False
        if self.memory_monitor:
            self.memory_monitor.join(timeout=2)
    
    def setup_milvus_connection(self):
        """Setup Milvus connection"""
        try:
            connections.connect(alias="default", host="localhost", port="19530")
            logger.info("Connected to Milvus database")
            
            self.collections = {
                'titles': Collection("opportunity_titles"),
                'descriptions': Collection("opportunity_descriptions"),
                'opportunity_documents': Collection("opportunity_documents")
            }
            
            logger.info("Connected to all collections")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def setup_sql_connection(self):
        """Setup SQL Server connection"""
        try:
            if not PYODBC_AVAILABLE:
                logger.warning("SQL Server connection not available - pyodbc not installed")
                self.sql_conn = None
                return
                
            if SQL_CONNECTION_STRING and SQL_CONNECTION_STRING != 'your_connection_string_here':
                self.sql_conn = pyodbc.connect(SQL_CONNECTION_STRING)
                # Enable autocommit to avoid transaction management issues with FreeTDS
                self.sql_conn.autocommit = True
                logger.info("Connected to SQL Server")
            else:
                logger.warning("SQL Server connection not configured")
                self.sql_conn = None
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server: {e}")
            self.sql_conn = None
    
    def setup_entity_extraction(self):
        """Setup entity extraction components"""
        try:
            from entity_extractor import EntityExtractor
            from sql_entity_manager import SQLEntityManager
            
            self.entity_extractor = EntityExtractor()
            self.entity_manager = SQLEntityManager(SQL_CONNECTION_STRING)
            logger.info(f"Entity extraction enabled")
        except ImportError as e:
            logger.error(f"Entity extraction dependencies not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to setup entity extraction: {e}")
            raise
    
    def setup_embeddings(self):
        """Setup embeddings model with GPU acceleration and multi-process pool for performance"""
        try:
            # Detect and configure device (GPU vs CPU)
            device = self._detect_optimal_device()
            
            self.embeddings = SentenceTransformer(EMBEDDING_MODEL, device=device)
            logger.info(f"Loaded embedding model: {EMBEDDING_MODEL} on device: {device}")
            
            # Adjust batch size for GPU
            self.effective_batch_size = EMBEDDING_BATCH_SIZE
            if device.startswith('cuda') and ENABLE_GPU_ACCELERATION:
                self.effective_batch_size = EMBEDDING_BATCH_SIZE * GPU_BATCH_SIZE_MULTIPLIER
                logger.info(f"🚀 GPU detected: Increased batch size to {self.effective_batch_size}")
            
            # Initialize embedding model pool for high-performance concurrent processing
            if ENABLE_EMBEDDING_MODEL_POOL:
                if device.startswith('cuda'):
                    # For GPU, use single device with larger batches instead of multi-process
                    self.embedding_pool = None
                    self.use_pool = False
                    logger.info("🚀 GPU mode: Using single-process with large batches for optimal GPU utilization")
                else:
                    # For CPU, use multi-process pool
                    self.embedding_pool = self.embeddings.start_multi_process_pool(
                        target_devices=['cpu'] * EMBEDDING_MODEL_POOL_SIZE
                    )
                    logger.info(f"✅ CPU mode: Embedding model pool initialized with {EMBEDDING_MODEL_POOL_SIZE} processes")
                    self.use_pool = True
            else:
                self.embedding_pool = None
                self.use_pool = False
                logger.info(f"Single-process embedding model initialized on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _detect_optimal_device(self) -> str:
        """Detect optimal device for embedding model (GPU vs CPU)"""
        if not ENABLE_GPU_ACCELERATION:
            return 'cpu'
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using CPU")
            return 'cpu'
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                logger.info(f"🚀 GPU detected: {gpu_name} with {gpu_memory:.1f}GB VRAM")
                
                if gpu_memory >= 8.0:  # Minimum 8GB for large batches
                    return GPU_DEVICE if GPU_DEVICE else 'cuda'
                else:
                    logger.warning(f"GPU has only {gpu_memory:.1f}GB VRAM, falling back to CPU")
                    return 'cpu'
            else:
                logger.info("No CUDA GPU available, using CPU")
                return 'cpu'
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}, falling back to CPU")
            return 'cpu' if FALLBACK_TO_CPU else GPU_DEVICE
    
    def encode_with_pool(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings using pool if available, or GPU-optimized batches"""
        # Use effective batch size (larger for GPU)
        batch_size = kwargs.get('batch_size', self.effective_batch_size)
        
        if self.use_pool and self.embedding_pool:
            # Use multi-process pool for CPU
            return self.embeddings.encode_multi_process(
                texts, 
                pool=self.embedding_pool,
                normalize_embeddings=kwargs.get('normalize_embeddings', True),
                batch_size=batch_size
            )
        else:
            # Direct model encoding (single GPU or CPU)
            return self.embeddings.encode(
                texts, 
                batch_size=batch_size,
                normalize_embeddings=kwargs.get('normalize_embeddings', True),
                show_progress_bar=kwargs.get('show_progress_bar', False),
                convert_to_tensor=False  # Keep as numpy arrays for consistency
            )
    
    def _reset_stats(self):
        """Reset all statistics for a new processing run"""
        with self.stats_lock:
            self.stats = {
                'opportunities_processed': 0,
                'titles_embedded': 0,
                'descriptions_embedded': 0,
                'documents_embedded': 0,
                'documents_skipped': 0,
                'documents_already_processed': 0,  # Files skipped due to previous processing
                'documents_race_condition_skipped': 0,  # Files skipped due to thread race conditions
                'total_chunks_generated': 0,
                'boilerplate_chunks_filtered': 0,
                'entities_extracted': 0,
                'errors': 0,
                'processing_time': 0.0,
                'peak_memory_mb': 0.0,
                'avg_cpu_percent': 0.0,
                # Queue monitoring statistics
                'queue_max_size': 0,
                'queue_avg_size': 0.0,
                'queue_samples': 0,
                'file_load_errors': 0,
                'producer_throughput_mb_per_sec': 0.0
            }
            logger.debug("Statistics reset for new processing run")

    def _update_stats(self, key: str, increment: int = 1):
        """Thread-safe statistics update with task-specific tracking when available"""
        with self.stats_lock:
            # Update shared stats (for legacy compatibility)
            self.stats[key] += increment
            
            # Update task-specific stats if available
            if hasattr(self, 'task_specific_stats') and self.task_specific_stats is not None:
                if key in self.task_specific_stats:
                    self.task_specific_stats[key] += increment
                
                # Use task-specific counter for progress callback
                if key == 'opportunities_processed' and self.progress_callback:
                    try:
                        self.progress_callback(self.task_specific_stats['opportunities_processed'])
                    except Exception as e:
                        logger.warning(f"Task-specific progress callback failed: {e}")
            else:
                # Fallback to shared stats for legacy compatibility
                if key == 'opportunities_processed' and self.progress_callback:
                    try:
                        self.progress_callback(self.stats['opportunities_processed'])
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
    
    def get_current_stats(self):
        """Get current processing statistics (thread-safe)"""
        with self.stats_lock:
            return self.stats.copy()
    
    def _flush_all_vector_collections(self):
        """Flush all vector collections as a batch commit operation"""
        try:
            with time_operation('batch_commit_flush'):
                for collection_name, collection in self.collections.items():
                    collection.flush()
            logger.debug(f"Successfully flushed all {len(self.collections)} vector collections")
        except Exception as e:
            logger.error(f"Error during batch commit flush: {e}")
            raise
    
    def get_opportunities_count(self, start_row_id: int, end_row_id: int) -> int:
        """
        Get count of distinct opportunities in the specified row range
        
        Args:
            start_row_id: Starting row ID (inclusive)
            end_row_id: Ending row ID (inclusive)
            
        Returns:
            Count of distinct opportunities
        """
        if not self.sql_conn:
            logger.warning("SQL Server connection not available")
            return 0
            
        cursor = None
        try:
            cursor = self.sql_conn.cursor()
            
            # Original FBOInternalAPI count query with proper joins
            count_query = """
            SELECT COUNT(DISTINCT A.OpportunityId)
            FROM FBOInternalAPI.Opportunities A 
            LEFT OUTER JOIN FBOInternalAPI.Descriptions B ON A.opportunityId=B.opportunityId 
            LEFT OUTER JOIN FBOInternalAPI.OpportunityAttachments C ON A.opportunityId=C.OpportunityId AND C.deletedFlag=0 
            LEFT OUTER JOIN FBOInternalAPI.OpportunityExtractedFiles D ON C.ExtractedFileId=D.fileId
            WHERE A.rowID >= ? and A.rowID <= ?
            """
            
            cursor.execute(count_query, (start_row_id, end_row_id))
            count = cursor.fetchone()[0]
            
            logger.info(f"Found {count} distinct opportunities in row range {start_row_id} to {end_row_id}")
            return count
            
        except Exception as e:
            logger.error(f"Error getting opportunities count: {e}")
            return 0
        finally:
            if cursor:
                cursor.close()

    def process_scalable_batch_producer_consumer(self, start_row_id: int, end_row_id: int, replace_existing_records: bool = False, task_id: str = None):
        """
        Process batch using producer/consumer architecture for optimal performance
        """
        # Initialize task-specific stats if task_id is provided
        if task_id:
            # Create completely isolated stats for this task
            self.task_specific_stats = {
                'opportunities_processed': 0,
                'documents_processed': 0,
                'documents_skipped': 0,
                'total_chunks_generated': 0,
                'entities_extracted': 0,
                'errors': 0
            }
            self.current_task_id = task_id
            logger.info(f"🔧 Initialized task-specific stats for task {task_id}")
        else:
            self.task_specific_stats = None
            self.current_task_id = None
        
        # Reset shared statistics for this processing run (for legacy compatibility)
        self._reset_stats()
        
        with time_operation('process_scalable_batch_producer_consumer', {'start_row': start_row_id, 'end_row': end_row_id}):
            start_time = time.time()
            logger.info(f"🚀 Starting producer/consumer processing for rows {start_row_id} to {end_row_id}")
            
            # Create bounded queue for opportunities
            opportunity_queue = queue.Queue(maxsize=100)
            
            # Consumer thread control
            consumer_count = getattr(self, 'opportunity_workers', 2)
            consumers_active = threading.Event()
            consumers_active.set()
            
            # Error handling
            processing_errors = []
            error_lock = threading.Lock()
            
            def add_error(error_msg: str):
                with error_lock:
                    processing_errors.append(error_msg)
                    logger.error(error_msg)
            
            # Producer thread function
            def producer_thread():
                """Producer thread that iterates through SQL results and creates Opportunity objects"""
                try:
                    if not self.sql_conn:
                        add_error("SQL Server connection not available")
                        return
                    
                    cursor = self.sql_conn.cursor()
                    
                    # Execute the SQL query
                    selection_query = """
                    SELECT A.OpportunityId, A.Title, B.description as Description, 
                           D.fileId, D.fileLocation, A.postedDate, D.fileSizeBytes
                    FROM FBOInternalAPI.Opportunities A 
                    LEFT OUTER JOIN FBOInternalAPI.Descriptions B ON A.opportunityId=B.opportunityId 
                    LEFT OUTER JOIN FBOInternalAPI.OpportunityAttachments C ON A.opportunityId=C.OpportunityId AND C.deletedFlag=0 
                    LEFT OUTER JOIN FBOInternalAPI.OpportunityExtractedFiles D ON C.ExtractedFileId=D.fileId
                    WHERE A.rowID >= ? and A.rowID <= ?
                    ORDER BY A.rowID
                    """
                    
                    cursor.execute(selection_query, (start_row_id, end_row_id))
                    
                    current_opportunity = None
                    opportunities_produced = 0
                    total_text_size_mb = 0.0
                    producer_start_time = time.time()
                    
                    for row in cursor.fetchall():
                        opportunity_id = row[0]
                        title = row[1] if row[1] else ''
                        description = row[2] if row[2] else ''
                        file_id = row[3] if row[3] else None
                        file_location = row[4] if row[4] else None
                        posted_date = row[5].isoformat() if row[5] else None
                        file_size_bytes = row[6] if row[6] else None
                        
                        # Log file size correlation data
                        if file_size_bytes is not None:
                            logger.debug(f"File size correlation: OpportunityId={opportunity_id}, FileId={file_id}, SizeBytes={file_size_bytes}")
                        else:
                            logger.debug(f"File size correlation: OpportunityId={opportunity_id}, FileId={file_id}, SizeBytes=NULL")
                        
                        # Check if this is a new opportunity
                        if current_opportunity is None or current_opportunity.opportunity_id != opportunity_id:
                            # Queue the previous opportunity if it exists
                            if current_opportunity is not None:
                                opportunity_queue.put(current_opportunity)
                                opportunities_produced += 1
                                
                                # Monitor queue size
                                queue_size = opportunity_queue.qsize()
                                with self.stats_lock:
                                    self.stats['queue_max_size'] = max(self.stats['queue_max_size'], queue_size)
                                    self.stats['queue_samples'] += 1
                                    self.stats['queue_avg_size'] = ((self.stats['queue_avg_size'] * (self.stats['queue_samples'] - 1)) + queue_size) / self.stats['queue_samples']
                                
                                if opportunities_produced % 5 == 0:
                                    logger.info(f"Producer: {opportunities_produced} opportunities queued, queue size: {queue_size}")
                            
                            # Create new opportunity
                            current_opportunity = Opportunity(opportunity_id, title, description, posted_date)
                        
                        # Add document to current opportunity if file data exists
                        if file_id is not None and file_location is not None:
                            # PRE-LOAD DOCUMENT TEXT IN PRODUCER THREAD
                            file_path = self.replace_document_path(file_location)
                            text_content = None
                            load_error = None
                            
                            if os.path.exists(file_path):
                                try:
                                    with time_operation('producer_file_load', {'file_id': file_id, 'file_size_bytes': file_size_bytes}):
                                        text_content = extract_text_from_file(file_path)
                                        if text_content:
                                            text_size_mb = len(text_content.encode('utf-8')) / (1024 * 1024)
                                            total_text_size_mb += text_size_mb
                                            logger.debug(f"Producer loaded file {file_id}: {text_size_mb:.2f}MB text")
                                        else:
                                            logger.warning(f"Producer: No text extracted from file: {file_path}")
                                except Exception as e:
                                    load_error = str(e)
                                    logger.error(f"Producer: Error loading file {file_path}: {e}")
                                    with self.stats_lock:
                                        self.stats['file_load_errors'] += 1
                            else:
                                load_error = f"File not found: {file_path}"
                                logger.warning(f"Producer: {load_error}")
                                with self.stats_lock:
                                    self.stats['file_load_errors'] += 1
                            
                            # Create document with pre-loaded text
                            document = Document(file_id, file_location, file_size_bytes, text_content)
                            document.load_error = load_error
                            current_opportunity.add_document(document)
                    
                    # Don't forget the last opportunity
                    if current_opportunity is not None:
                        opportunity_queue.put(current_opportunity)
                        opportunities_produced += 1
                    
                    # Calculate producer throughput
                    producer_time = time.time() - producer_start_time
                    if producer_time > 0:
                        throughput_mb_per_sec = total_text_size_mb / producer_time
                        with self.stats_lock:
                            self.stats['producer_throughput_mb_per_sec'] = throughput_mb_per_sec
                        logger.info(f"Producer throughput: {throughput_mb_per_sec:.2f} MB/sec ({total_text_size_mb:.2f}MB in {producer_time:.1f}s)")
                    
                    cursor.close()
                    logger.info(f"Producer finished: {opportunities_produced} opportunities queued, {total_text_size_mb:.2f}MB text pre-loaded")
                    
                except Exception as e:
                    add_error(f"Producer thread error: {e}")
                finally:
                    # Signal that producer is done by putting None sentinel
                    for _ in range(consumer_count):
                        opportunity_queue.put(None)
            
            # Consumer worker function
            def consumer_worker(worker_id: int):
                """Consumer worker that processes opportunities from the queue"""
                try:
                    while consumers_active.is_set():
                        try:
                            # Get opportunity from queue with timeout
                            opportunity = opportunity_queue.get(timeout=1.0)
                            
                            # Check for sentinel value (None means producer is done)
                            if opportunity is None:
                                logger.debug(f"Consumer {worker_id} received shutdown signal")
                                break
                            
                            # Process the opportunity (this handles the stats update)
                            self._process_opportunity_simplified(opportunity, replace_existing_records)
                            
                        except queue.Empty:
                            # Timeout occurred, check if we should continue
                            continue
                        except Exception as e:
                            add_error(f"Consumer {worker_id} error processing opportunity: {e}")
                        finally:
                            try:
                                opportunity_queue.task_done()
                            except ValueError:
                                # task_done() called more times than items in queue
                                pass
                
                except Exception as e:
                    add_error(f"Consumer {worker_id} fatal error: {e}")
                finally:
                    # Final flush for this consumer
                    try:
                        logger.debug(f"Consumer {worker_id} performing final flush...")
                        self._flush_all_vector_collections()
                        logger.info(f"Consumer {worker_id} finished processing")
                    except Exception as e:
                        add_error(f"Consumer {worker_id} final flush error: {e}")
            
            # Start producer thread
            producer = threading.Thread(target=producer_thread, name="ProducerThread")
            producer.start()
            
            # Start consumer workers
            consumers = []
            for i in range(consumer_count):
                consumer = threading.Thread(target=consumer_worker, args=(i,), name=f"ConsumerWorker-{i}")
                consumer.start()
                consumers.append(consumer)
            
            # Wait for producer to finish
            producer.join()
            logger.info("Producer thread completed")
            
            # Wait for all consumers to finish
            for consumer in consumers:
                consumer.join()
            logger.info("All consumer workers completed")
            
            # Stop memory monitoring and cleanup
            self.stop_memory_monitoring()
            
            # Entity extraction cleanup
            if self.enable_entity_extraction and self.entity_queue:
                logger.info("Shutting down entity extraction queue...")
                self.entity_queue.shutdown(timeout=30.0)
            
            # Final processing time
            processing_time = time.time() - start_time
            self.stats['processing_time'] = processing_time
            
            # Log any errors that occurred
            if processing_errors:
                logger.error(f"Processing completed with {len(processing_errors)} errors:")
                for error in processing_errors:
                    logger.error(f"  - {error}")
            
            # Print performance summary
            logger.info("🔍 PRODUCER/CONSUMER ANALYSIS COMPLETE - See detailed timing below:")
            print_summary()
            # Ensure logs directory exists for performance report
            os.makedirs("logs", exist_ok=True)
            # Use task_id for unique performance reports, fallback to row range if no task_id
            logger.info(f"🔍 DEBUG: Performance report task_id='{task_id}', start_row_id={start_row_id}, end_row_id={end_row_id}")
            if task_id:
                report_filename = f"logs/performance_report_{task_id}.json"
                logger.info(f"🔍 Task-specific performance report: {report_filename}")
            else:
                report_filename = f"logs/performance_report_{start_row_id}_{end_row_id}.json"
                logger.info(f"🔍 Legacy performance report: {report_filename}")
            save_report(report_filename)
            
            self._log_final_stats(processing_time)
            
            # Return processing statistics (use task-specific stats when available)
            if hasattr(self, 'task_specific_stats') and self.task_specific_stats is not None:
                stats_to_return = self.task_specific_stats.copy()
                stats_to_return.update({
                    'processing_time': processing_time,
                    'errors': self.task_specific_stats['errors'] + len(processing_errors)
                })
                logger.info(f"🔍 Returning task-specific stats: {self.task_specific_stats['opportunities_processed']} opportunities")
            else:
                stats_to_return = {
                    'opportunities_processed': self.stats['opportunities_processed'],
                    'documents_embedded': self.stats['documents_embedded'],
                    'documents_skipped': self.stats['documents_skipped'],
                    'total_chunks_generated': self.stats['total_chunks_generated'],
                    'processing_time': processing_time,
                    'errors': self.stats['errors'] + len(processing_errors)
                }
                logger.info(f"🔍 Returning shared stats: {self.stats['opportunities_processed']} opportunities")
            
            return stats_to_return

    def _process_opportunity_simplified(self, opportunity: Opportunity, replace_existing_records: bool = False):
        """Process a single opportunity using simplified architecture (no intra-opportunity parallelization)"""
        with time_operation('process_opportunity_simplified', {'opportunity_id': opportunity.opportunity_id, 'document_count': len(opportunity.documents)}):
            
            # Process title and description (fast, keep sequential)
            if opportunity.opportunity_id not in self.processed_opportunities:
                if opportunity.title:
                    with time_operation('title_embedding'):
                        self.process_title_embedding(
                            opportunity.opportunity_id, 
                            opportunity.title, 
                            opportunity.posted_date, 
                            replace_existing_records
                        )
                        self._update_stats('titles_embedded', 1)
                
                if opportunity.description:
                    with time_operation('description_embedding'):
                        self.process_description_embedding(
                            opportunity.opportunity_id, 
                            opportunity.description, 
                            opportunity.posted_date, 
                            replace_existing_records
                        )
                        self._update_stats('descriptions_embedded', 1)
                    
                    # Submit description entity extraction asynchronously
                    if self.enable_entity_extraction and self.entity_queue:
                        self.entity_queue.submit_task(
                            text_content=opportunity.description,
                            opportunity_id=opportunity.opportunity_id,
                            content_type='description'
                        )
                
                self.processed_opportunities.add(opportunity.opportunity_id)
            
            # Process documents sequentially (simplified - no intra-opportunity parallelization)
            for document in opportunity.documents:
                try:
                    # No longer collecting entities since they're processed asynchronously
                    self._process_single_file_simplified(
                        document, 
                        opportunity.opportunity_id, 
                        opportunity.posted_date, 
                        replace_existing_records
                    )
                except Exception as e:
                    logger.error(f"Error processing document {document.file_location}: {e}")
                    self._update_stats('errors', 1)
            
            # Only increment progress counter after ALL processing is complete for this opportunity
            self._update_stats('opportunities_processed', 1)

    def _process_single_file_simplified(self, document: Document, opportunity_id: str, posted_date: str, replace_existing_records: bool = False):
        """Process a single document using simplified architecture"""
        if document.file_id is None:
            logger.warning(f"Skipping document with NULL file_id: {document.file_location}")
            return []
        
        # Create context dict for performance logging with file size
        context = {'file_id': document.file_id}
        if document.file_size_bytes is not None:
            context['file_size_bytes'] = document.file_size_bytes
            context['file_size_mb'] = round(document.file_size_bytes / (1024 * 1024), 2) if document.file_size_bytes > 0 else 0
        
        with time_operation('process_single_file_simplified', context):
            # Thread-safe file claim
            if not self._try_claim_file_for_processing(document.file_id, replace_existing_records):
                logger.debug(f"File {document.file_id} already processed")
                return []
            
            # Check for file load errors from producer
            if document.load_error:
                logger.warning(f"Skipping file {document.file_id} due to load error: {document.load_error}")
                return []

            try:
                # Use pre-loaded text content if available, otherwise fallback to file reading
                if document.is_text_loaded():
                    text_content = document.text_content
                    logger.debug(f"Using pre-loaded text for file {document.file_id} ({document.get_memory_footprint_mb():.2f}MB)")
                else:
                    # Fallback: Extract text content (this should be rare with optimized producer)
                    file_path = self.replace_document_path(document.file_location)
                    if not os.path.exists(file_path):
                        logger.warning(f"File not found: {file_path}")
                        return []
                    
                    with time_operation('file_read_fallback', {'file_path': file_path, 'file_size_bytes': document.file_size_bytes}):
                        text_content = extract_text_from_file(file_path)
                        logger.warning(f"Consumer fallback file read for {document.file_id} - producer should have loaded this")
                
                if not text_content:
                    logger.warning(f"No text content available for file {document.file_id}")
                    return []
                
                # Process embeddings (this is now the main work for consumers)
                with time_operation('text_chunk', {'text_length': len(text_content), 'file_size_bytes': document.file_size_bytes}):
                    chunks = self.chunker.chunk_text(text_content)
                    self._update_stats('total_chunks_generated', len(chunks))
                
                # Filter boilerplate
                with time_operation('boilerplate_filter', {'chunk_count': len(chunks), 'file_size_bytes': document.file_size_bytes}):
                    chunks_with_embeddings = self.boilerplate_manager.filter_non_boilerplate_chunks(chunks)
                    boilerplate_filtered = len(chunks) - len(chunks_with_embeddings)
                    self._update_stats('boilerplate_chunks_filtered', boilerplate_filtered)
                
                if chunks_with_embeddings:
                    # Process embeddings (primary GPU work) - using pre-computed embeddings from boilerplate filtering
                    with time_operation('embedding_generate', {'chunk_count': len(chunks_with_embeddings), 'file_size_bytes': document.file_size_bytes}):
                        # Convert Document to file_info dict for compatibility
                        file_info = {
                            'file_id': document.file_id,
                            'file_location': document.file_location,
                            'file_size_bytes': document.file_size_bytes
                        }
                        self._process_embedding_chunks_batch(chunks_with_embeddings, file_info, opportunity_id, posted_date)
                        self._update_stats('documents_embedded', 1)
                else:
                    self._update_stats('documents_skipped', 1)
                
                # Submit entity extraction task asynchronously (non-blocking)
                if self.enable_entity_extraction and self.entity_queue and len(text_content.strip()) > 50:
                    self.entity_queue.submit_task(
                        text_content=text_content,
                        opportunity_id=opportunity_id,
                        content_type='document',
                        file_id=document.file_id
                    )
                
                return []  # No longer returning entities since processing is async
                
            except Exception as e:
                logger.error(f"Error processing file {document.file_location}: {e}")
                return []

    def process_scalable_batch(self, start_row_id: int, end_row_id: int, replace_existing_records: bool = False, task_id: str = None):
        """
        Process batch using producer/consumer architecture for optimal performance
        """
        logger.info("Using producer/consumer architecture for optimal performance")
        return self.process_scalable_batch_producer_consumer(start_row_id, end_row_id, replace_existing_records, task_id)

    def _process_scalable_batch_legacy(self, start_row_id: int, end_row_id: int, replace_existing_records: bool = False, task_id: str = None):
        """
        Process batch with intelligent scaling based on system resources (legacy data grouping approach)
        """
        # Reset statistics for this processing run
        self._reset_stats()
        
        with time_operation('process_scalable_batch', {'start_row': start_row_id, 'end_row': end_row_id}):
            start_time = time.time()
            logger.info(f"🚀 Starting scalable processing for rows {start_row_id} to {end_row_id}")
            logger.info(f"📊 Using {self.opportunity_workers} opportunity workers, {self.file_workers_per_opportunity} file workers each")
            
            # Check system health before starting
            if not self.resource_config['system_healthy']:
                logger.warning(f"⚠️  System resource warning: {self.resource_config['health_message']}")
                logger.warning("Proceeding with reduced performance expectations...")
            
            # Get data from SQL Server
            with time_operation('sql_data_retrieval', {'row_count': end_row_id - start_row_id}):
                opportunities_data = self.get_opportunities_data(start_row_id, end_row_id)
            
            if not opportunities_data:
                logger.warning("No data retrieved from SQL Server")
                return {
                    'opportunities_processed': 0,
                    'documents_embedded': 0,
                    'documents_skipped': 0,
                    'total_chunks_generated': 0,
                    'processing_time': 0.0,
                    'errors': 0
                }
            
            # Group by opportunity_id
            with time_operation('data_grouping'):
                grouped_data = self._group_opportunities_data(opportunities_data)
            
            logger.info(f"📋 Processing {len(grouped_data)} unique opportunities")
            
            # Process opportunities with intelligent scaling
            if ENABLE_PARALLEL_PROCESSING and self.opportunity_workers > 1:
                self._process_opportunities_parallel(grouped_data, replace_existing_records)
            else:
                self._process_opportunities_sequential(grouped_data, replace_existing_records)
            
            # Stop memory monitoring
            with time_operation('cleanup_operations'):
                self.stop_memory_monitoring()
                
                # Entity extraction cleanup
                if self.enable_entity_extraction and self.entity_queue:
                    logger.info("Shutting down entity extraction queue...")
                    self.entity_queue.shutdown(timeout=30.0)
                
                # Cleanup embedding model pool
                if self.use_pool and self.embedding_pool:
                    logger.info("Shutting down embedding model pool...")
                    self.embeddings.stop_multi_process_pool(self.embedding_pool)
                    self.embedding_pool = None
                    self.use_pool = False
                
                # Final batch commit - flush any remaining operations
                if self.enable_batch_commits and self.opportunities_since_last_flush > 0:
                    logger.info(f"💾 Final commit: flushing all vector collections for remaining {self.opportunities_since_last_flush} opportunities")
                    self._flush_all_vector_collections()
                
                # Flush all collections (final safety flush)
                for collection_name, collection in self.collections.items():
                    collection.flush()
            
            # Final statistics
            processing_time = time.time() - start_time
            self.stats['processing_time'] = processing_time
            
            # Print performance summary
            logger.info("🔍 PERFORMANCE ANALYSIS COMPLETE - See detailed timing below:")
            print_summary()
            # Ensure logs directory exists for performance report
            os.makedirs("logs", exist_ok=True)
            # Use task_id for unique performance reports, fallback to row range if no task_id
            report_filename = f"logs/performance_report_{task_id}.json" if task_id else f"logs/performance_report_{start_row_id}_{end_row_id}.json"
            save_report(report_filename)
        
        self._log_final_stats(processing_time)
        
        # Return processing statistics for API service
        return {
            'opportunities_processed': self.stats['opportunities_processed'],
            'documents_embedded': self.stats['documents_embedded'],
            'documents_skipped': self.stats['documents_skipped'],
            'total_chunks_generated': self.stats['total_chunks_generated'],
            'processing_time': processing_time,
            'errors': self.stats['errors']
        }
    
    def _process_opportunities_parallel(self, grouped_data: Dict, replace_existing_records: bool):
        """Process opportunities in parallel with intelligent scaling"""
        with ThreadPoolExecutor(max_workers=self.opportunity_workers) as executor:
            # Submit all opportunity processing tasks
            future_to_opp = {
                executor.submit(
                    self._process_single_opportunity, 
                    opportunity_id, 
                    data, 
                    replace_existing_records
                ): opportunity_id
                for opportunity_id, data in grouped_data.items()
            }
            
            # Collect results with progress tracking
            completed = 0
            total = len(future_to_opp)
            
            for future in as_completed(future_to_opp):
                opportunity_id = future_to_opp[future]
                try:
                    future.result()
                    completed += 1
                    
                    if completed % 5 == 0 or completed == total:
                        progress = (completed / total) * 100
                        logger.info(f"📈 Progress: {completed}/{total} opportunities ({progress:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Error processing opportunity {opportunity_id}: {e}")
                    self._update_stats('errors', 1)
    
    def _process_opportunities_sequential(self, grouped_data: Dict, replace_existing_records: bool):
        """Process opportunities sequentially (fallback mode)"""
        logger.info("Processing opportunities sequentially (parallel processing disabled)")
        
        for i, (opportunity_id, data) in enumerate(grouped_data.items(), 1):
            try:
                self._process_single_opportunity(opportunity_id, data, replace_existing_records)
                
                # Batch commit logic - flush all collections after processing batch of opportunities
                if self.enable_batch_commits:
                    with self.batch_commit_lock:
                        self.opportunities_since_last_flush += 1
                        
                        if self.opportunities_since_last_flush >= self.vector_batch_size:
                            logger.info(f"💾 Committing batch: flushing all vector collections after {self.opportunities_since_last_flush} opportunities")
                            self._flush_all_vector_collections()
                            self.opportunities_since_last_flush = 0
                
                if i % 5 == 0:
                    progress = (i / len(grouped_data)) * 100
                    logger.info(f"📈 Progress: {i}/{len(grouped_data)} opportunities ({progress:.1f}%)")
                    
            except Exception as e:
                logger.error(f"Error processing opportunity {opportunity_id}: {e}")
                self._update_stats('errors', 1)
    
    def _process_single_opportunity(self, opportunity_id: str, data: Dict, replace_existing_records: bool):
        """Process a single opportunity with file-level parallelization"""
        with time_operation('process_single_opportunity', {'opportunity_id': opportunity_id, 'file_count': len(data.get('files', []))}):
            
            # Process title and description (fast, keep sequential)
            if opportunity_id not in self.processed_opportunities:
                if data['title']:
                    with time_operation('title_embedding'):
                        success = self.process_title_embedding(opportunity_id, data['title'], data['posted_date'], replace_existing_records)
                        if success:
                            self._update_stats('titles_embedded', 1)
                
                if data['description']:
                    with time_operation('description_embedding'):
                        success = self.process_description_embedding(opportunity_id, data['description'], data['posted_date'], replace_existing_records)
                        if success:
                            self._update_stats('descriptions_embedded', 1)
                    
                    # Submit description entity extraction asynchronously
                    if self.enable_entity_extraction and self.entity_queue:
                        self.entity_queue.submit_task(
                            data['description'], 
                            opportunity_id, 
                            'description'
                        )
                
                self.processed_opportunities.add(opportunity_id)
            
            # Process files with configurable parallelization
            if data['files']:
                file_worker_count = min(self.file_workers_per_opportunity, len(data['files']))
                
                if file_worker_count > 1:
                    with time_operation('files_parallel_processing', {'file_count': len(data['files']), 'workers': file_worker_count}):
                        self._process_files_parallel(data['files'], opportunity_id, data['posted_date'], file_worker_count, replace_existing_records)
                else:
                    with time_operation('files_sequential_processing', {'file_count': len(data['files'])}):
                        self._process_files_sequential(data['files'], opportunity_id, data['posted_date'], replace_existing_records)
            
            # Only increment progress counter after ALL processing is complete for this opportunity
            self._update_stats('opportunities_processed', 1)
    
    def _process_files_parallel(self, files: List[Dict], opportunity_id: str, posted_date: str, worker_count: int, replace_existing_records: bool = False):
        """Process files in parallel for a single opportunity"""
        
        with ThreadPoolExecutor(max_workers=worker_count) as file_executor:
            file_futures = {
                file_executor.submit(
                    self._process_single_file, 
                    file_info, 
                    opportunity_id, 
                    posted_date,
                    replace_existing_records
                ): file_info
                for file_info in files if file_info.get('file_id') is not None
            }
            
            for future in as_completed(file_futures):
                file_info = file_futures[future]
                try:
                    future.result()  # No longer collecting entities
                except Exception as e:
                    logger.error(f"Error processing file {file_info['file_location']}: {e}")
                    self._update_stats('errors', 1)
    
    def _process_files_sequential(self, files: List[Dict], opportunity_id: str, posted_date: str, replace_existing_records: bool = False):
        """Process files sequentially for a single opportunity"""
        
        for file_info in files:
            if file_info.get('file_id') is None:
                logger.warning(f"Skipping file with NULL file_id: {file_info.get('file_location')}")
                continue
            try:
                self._process_single_file(file_info, opportunity_id, posted_date, replace_existing_records)  # No longer collecting entities
            except Exception as e:
                logger.error(f"Error processing file {file_info['file_location']}: {e}")
                self._update_stats('errors', 1)
    
    def _try_claim_file_for_processing(self, file_id: int, replace_existing_records: bool = False) -> bool:
        """
        Simplified file claim - no longer prevents duplicate processing since same file can be processed for different opportunities
        
        Args:
            file_id: File ID to claim
            replace_existing_records: If True, delete existing vector DB records and reprocess
        """
        with time_operation('file_claim_check', {'file_id': file_id}):
            with self.processed_files_lock:
                # Only check if we want to replace existing records
                if replace_existing_records:
                    try:
                        if 'opportunity_documents' in self.collections:
                            # Query for any existing chunks for this file_id
                            results = self.collections['opportunity_documents'].query(
                                expr=f"file_id == {file_id}",
                                output_fields=["file_id"],
                                limit=1  # We only need to know if ANY chunks exist
                            )
                            
                            if results:
                                # DELETE existing records and process as new
                                logger.info(f"Deleting existing vector DB records for file {file_id} (replace_existing_records=True)")
                                self._delete_existing_file_records(file_id)
                                
                    except Exception as e:
                        logger.warning(f"Error checking vector database for file {file_id}: {e}")
                        # If database check fails, proceed to claim the file to be safe
                
                # ALWAYS CLAIM: Allow the same file to be processed multiple times for different opportunities
                # Note: We still use the session cache to avoid processing the same file multiple times in the same batch
                if file_id not in self.processed_files:
                    self.processed_files.add(file_id)
                    logger.debug(f"File {file_id} claimed for processing by thread {threading.current_thread().name}")
                    return True
                else:
                    # File already processed in this session/batch
                    logger.debug(f"File {file_id} already processed in this session")
                    return False

    def _delete_existing_file_records(self, file_id: int):
        """Delete all existing vector database records for a file ID"""
        try:
            # Delete from opportunity_documents collection
            if 'opportunity_documents' in self.collections:
                delete_result = self.collections['opportunity_documents'].delete(expr=f"file_id == {file_id}")
                logger.debug(f"Deleted {delete_result.delete_count if hasattr(delete_result, 'delete_count') else 'unknown'} document records for file {file_id}")
                
        except Exception as e:
            logger.warning(f"Error deleting existing records for file {file_id}: {e}")
            # Continue processing even if deletion fails
    
    def _is_file_already_processed(self, file_id: int) -> bool:
        """
        DEPRECATED: Use _try_claim_file_for_processing() instead for thread safety.
        This method is kept for compatibility but should not be used in multi-threaded code.
        """
        # Quick non-locking check (may have race conditions)
        if file_id in self.processed_files:
            return True
        return False
    def _process_single_file(self, file_info: Dict, opportunity_id: str, posted_date: str, replace_existing_records: bool = False):
        """Process a single file for both embeddings and entities"""
        file_id = file_info.get('file_id')
        file_size_bytes = file_info.get('file_size_bytes')
        
        if file_id is None:
            logger.warning(f"Skipping processing and mapping for file with NULL file_id: {file_info.get('file_location')}")
            return []
            
        # Create context dict for performance logging with file size
        context = {'file_id': file_id}
        if file_size_bytes is not None:
            context['file_size_bytes'] = file_size_bytes
            context['file_size_mb'] = round(file_size_bytes / (1024 * 1024), 2) if file_size_bytes > 0 else 0
            
        with time_operation('process_single_file', context):
            # THREAD-SAFE ATOMIC CLAIM: Try to claim file for processing
            # Note: No longer checking for duplicates since same file can be processed for different opportunities
            if file_id and not self._try_claim_file_for_processing(file_id, replace_existing_records):
                logger.debug(f"File {file_id} already claimed/processed by another thread or previous session")
                return []  # Return empty entities list
            file_path = self.replace_document_path(file_info['file_location'])
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return []

            # File is now CLAIMED by this thread - proceed with expensive processing
            try:
                with time_operation('file_read', {'file_path': file_path, 'file_size_bytes': file_size_bytes}):
                    text_content = extract_text_from_file(file_path)
                    if not text_content:
                        logger.warning(f"No text extracted from file: {file_path}")
                        return []
                
                # Process embeddings
                with time_operation('text_chunk', {'text_length': len(text_content), 'file_size_bytes': file_size_bytes}):
                    chunks = self.chunker.chunk_text(text_content)
                    self._update_stats('total_chunks_generated', len(chunks))
                
                # Filter boilerplate - NOW VECTORIZED FOR 20x SPEEDUP
                with time_operation('boilerplate_filter', {'chunk_count': len(chunks), 'file_size_bytes': file_size_bytes}):
                    non_boilerplate_chunks = self.boilerplate_manager.filter_non_boilerplate_chunks(chunks)
                    boilerplate_filtered = len(chunks) - len(non_boilerplate_chunks)
                    self._update_stats('boilerplate_chunks_filtered', boilerplate_filtered)
                
                if non_boilerplate_chunks:
                    # Batch process embeddings
                    with time_operation('embedding_generate', {'chunk_count': len(non_boilerplate_chunks), 'file_size_bytes': file_size_bytes}):
                        self._process_embedding_chunks_batch(non_boilerplate_chunks, file_info, opportunity_id, posted_date)
                        self._update_stats('documents_embedded', 1)
                    
                    # File is already marked as processed in session cache by _try_claim_file_for_processing()
                else:
                    # Even if no embeddings, file is already marked as processed
                    self._update_stats('documents_skipped', 1)                # Extract entities asynchronously
                if self.enable_entity_extraction and self.entity_queue and len(text_content.strip()) > 50:
                    self.entity_queue.submit_task(
                        text_content, 
                        opportunity_id, 
                        'document', 
                        file_info['file_id']
                    )
                
                return []  # No longer returning entities since processing is async
                
            except Exception as e:
                logger.error(f"Error processing file {file_info['file_location']}: {e}")
                return []
    
    def _process_embedding_chunks_batch(self, chunks_with_embeddings: List[Tuple[str, np.ndarray]], file_info: Dict, opportunity_id: str, posted_date: str):
        """Process embedding chunks in optimal batches with PHASE 1 OPTIMIZATIONS
        
        Args:
            chunks_with_embeddings: List of (chunk_text, embedding) tuples from boilerplate filtering
            file_info: File information dictionary
            opportunity_id: Opportunity ID
            posted_date: Posted date string
        """
        with time_operation('process_embedding_chunks_batch', {'total_chunks': len(chunks_with_embeddings)}):
            batch_size = self.batch_sizes['embedding_batch_size']
            
            for i in range(0, len(chunks_with_embeddings), batch_size):
                batch_chunks_with_embeddings = chunks_with_embeddings[i:i + batch_size]
                
                # Extract embeddings that were already computed by boilerplate filtering
                embeddings_batch = [embedding for _, embedding in batch_chunks_with_embeddings]
                
                logger.debug(f"Using pre-computed embeddings from boilerplate filtering: {len(embeddings_batch)} embeddings")
                
                # PHASE 1 OPTIMIZATION: True batch vector insertion
                with time_operation('vector_store', {'vector_count': len(batch_chunks_with_embeddings)}):
                    try:
                        # Check if batch vector inserts are enabled
                        from config import ENABLE_BATCH_VECTOR_INSERTS
                        enable_batch_inserts = ENABLE_BATCH_VECTOR_INSERTS
                    except ImportError:
                        enable_batch_inserts = True  # Default to optimized approach
                    
                    if enable_batch_inserts:
                        # OPTIMIZED: Batch all vectors into single insert operation
                        vectors_to_insert = []
                        for j, ((chunk, embedding), computed_embedding) in enumerate(zip(batch_chunks_with_embeddings, embeddings_batch)):
                            vector_data = [
                                file_info['file_id'],              # file_id
                                opportunity_id,                    # opportunity_id
                                computed_embedding.tolist(),      # embedding (use pre-computed)
                                posted_date or datetime.now().isoformat(),  # posted_date
                                1.0,                               # base_importance
                                i + j,                             # chunk_index
                                len(chunks_with_embeddings),       # total_chunks
                                chunk[:2000],                      # text_content
                                file_info.get('file_location', ''), # file_location
                                ''                                 # section_type (empty for now)
                            ]
                            vectors_to_insert.append(vector_data)
                        
                        # Single batch insert (MUCH faster than individual inserts)
                        if vectors_to_insert:
                            # Transpose data for Milvus batch format
                            transposed_data = [
                                [item[0] for item in vectors_to_insert],  # file_ids
                                [item[1] for item in vectors_to_insert],  # opportunity_ids
                                [item[2] for item in vectors_to_insert],  # embeddings
                                [item[3] for item in vectors_to_insert],  # posted_dates
                                [item[4] for item in vectors_to_insert],  # base_importance
                                [item[5] for item in vectors_to_insert],  # chunk_indices
                                [item[6] for item in vectors_to_insert],  # total_chunks
                                [item[7] for item in vectors_to_insert],  # text_content
                                [item[8] for item in vectors_to_insert],  # file_location
                                [item[9] for item in vectors_to_insert],  # section_type
                            ]
                            result = self.collections['opportunity_documents'].insert(transposed_data)
                            logger.debug(f"Batch inserted {len(vectors_to_insert)} vectors: {result}")
                    else:
                        # FALLBACK: Original individual inserts (for comparison/debugging)
                        for j, ((chunk, embedding), computed_embedding) in enumerate(zip(batch_chunks_with_embeddings, embeddings_batch)):
                            data = [
                                [file_info['file_id']],
                                [opportunity_id],
                                [computed_embedding.tolist()],
                                [posted_date or datetime.now().isoformat()],
                                [1.0],  # base_importance
                                [i + j],  # chunk_index
                                [len(chunks_with_embeddings)],  # total_chunks
                                [chunk[:2000]],  # text_content
                                [file_info.get('file_location', '')],  # file_location
                                ['']  # section_type
                            ]
                            
                            self.collections['opportunity_documents'].insert(data)
    
    def _process_opportunity_entities(self, opportunity_id: str, entities_to_store: List):
        """Process and store entities for an opportunity"""
        with time_operation('process_opportunity_entities', {'entity_count': len(entities_to_store)}):
            # Filter by confidence first
            with time_operation('entity_filter_confidence'):
                filtered_entities = self._filter_entities_by_confidence(entities_to_store)
            
            # Then consolidate to remove duplicates
            with time_operation('entity_consolidation'):
                consolidated_entities = self._consolidate_entities_per_opportunity(filtered_entities)
            
            if consolidated_entities:
                with time_operation('entity_storage_db', {'entity_count': len(consolidated_entities)}):
                    stored_count = self.entity_manager.store_entities(consolidated_entities)
                    self._update_stats('entities_extracted', stored_count)
                    logger.debug(f"Stored {stored_count} consolidated entities for opportunity {opportunity_id}")
    
    def _log_final_stats(self, processing_time: float):
        """Log comprehensive final statistics"""
        logger.info("="*80)
        logger.info("📊 SCALABLE PROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"⏱️  Total Processing Time: {processing_time:.2f} seconds")
        logger.info(f"🏢 Opportunities Processed: {self.stats['opportunities_processed']}")
        logger.info(f"📝 Titles Embedded: {self.stats['titles_embedded']}")
        logger.info(f"📄 Descriptions Embedded: {self.stats['descriptions_embedded']}")
        logger.info(f"📁 Documents Embedded: {self.stats['documents_embedded']}")
        logger.info(f"⏭️  Documents Skipped: {self.stats['documents_skipped']}")
        
        # Deduplication statistics
        if self.stats.get('documents_already_processed', 0) > 0:
            logger.info(f"🔄 Files Already Processed (Skipped): {self.stats['documents_already_processed']}")
        if self.stats.get('documents_race_condition_skipped', 0) > 0:
            logger.info(f"🏁 Files Skipped (Thread Race): {self.stats['documents_race_condition_skipped']}")
        
        logger.info(f"🧩 Total Chunks Generated: {self.stats['total_chunks_generated']}")
        logger.info(f"🧹 Boilerplate Chunks Filtered: {self.stats['boilerplate_chunks_filtered']}")
        logger.info(f"👥 Entities Extracted: {self.stats['entities_extracted']}")
        logger.info(f"❌ Errors: {self.stats['errors']}")
        logger.info(f"💾 Peak Memory Usage: {self.stats['peak_memory_mb']:.1f} MB")
        logger.info(f"🖥️  Average CPU Usage: {self.stats['avg_cpu_percent']:.1f}%")
        
        # Performance calculations
        if self.stats['opportunities_processed'] > 0:
            avg_time_per_opp = processing_time / self.stats['opportunities_processed']
            logger.info(f"⚡ Average Time per Opportunity: {avg_time_per_opp:.2f} seconds")
        
        if self.stats['total_chunks_generated'] > 0:
            chunks_per_second = self.stats['total_chunks_generated'] / processing_time
            logger.info(f"🔄 Chunks Processed per Second: {chunks_per_second:.1f}")
        
        # Queue monitoring and file I/O optimization statistics
        if self.stats.get('queue_samples', 0) > 0:
            logger.info("="*80)
            logger.info("📈 PRODUCER/CONSUMER QUEUE ANALYSIS")
            logger.info("="*80)
            logger.info(f"📊 Queue Max Size: {self.stats['queue_max_size']}")
            logger.info(f"📊 Queue Average Size: {self.stats['queue_avg_size']:.1f}")
            logger.info(f"📊 Queue Samples: {self.stats['queue_samples']}")
            logger.info(f"🚀 Producer Throughput: {self.stats['producer_throughput_mb_per_sec']:.2f} MB/sec")
            logger.info(f"❌ File Load Errors: {self.stats['file_load_errors']}")
            
            # Queue efficiency analysis
            if self.stats['queue_avg_size'] < 2.0:
                logger.warning("⚠️  Low queue size suggests producer may be bottleneck")
            elif self.stats['queue_avg_size'] > 50:
                logger.warning("⚠️  High queue size suggests consumer may be bottleneck")
            else:
                logger.info("✅ Queue size indicates good producer/consumer balance")
    
    # Include essential supporting methods from the base processor
    def get_opportunities_data(self, start_row_id: int, end_row_id: int) -> List[Dict]:
        """Get opportunities data from SQL Server"""
        if not self.sql_conn:
            logger.warning("SQL Server connection not available")
            return []
            
        cursor = None
        try:
            cursor = self.sql_conn.cursor()
            
            selection_query = """
            SELECT A.OpportunityId, A.Title, B.description as Description, 
                   D.fileId, D.fileLocation, A.postedDate, D.fileSizeBytes
            FROM FBOInternalAPI.Opportunities A 
            LEFT OUTER JOIN FBOInternalAPI.Descriptions B ON A.opportunityId=B.opportunityId 
            LEFT OUTER JOIN FBOInternalAPI.OpportunityAttachments C ON A.opportunityId=C.OpportunityId AND C.deletedFlag=0 
            LEFT OUTER JOIN FBOInternalAPI.OpportunityExtractedFiles D ON C.ExtractedFileId=D.fileId
            WHERE A.rowID >= ? and A.rowID <= ?
            ORDER BY A.rowID
            """
            
            cursor.execute(selection_query, (start_row_id, end_row_id))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'opportunity_id': row[0],
                    'title': row[1] if row[1] else '',
                    'description': row[2] if row[2] else '',
                    'file_id': row[3] if row[3] else None,
                    'file_location': row[4] if row[4] else None,
                    'posted_date': row[5].isoformat() if row[5] else None,
                    'file_size_bytes': row[6] if row[6] else None
                })
            
            logger.info(f"Retrieved {len(results)} records from SQL Server")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving opportunities data: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def _group_opportunities_data(self, opportunities_data: List[Dict]) -> Dict:
        """Group opportunities data by opportunity_id"""
        grouped_data = {}
        for record in opportunities_data:
            opp_id = record['opportunity_id']
            if opp_id not in grouped_data:
                grouped_data[opp_id] = {
                    'title': record['title'],
                    'description': record['description'],
                    'posted_date': record['posted_date'],
                    'files': []
                }
            
            if record['file_id'] and record['file_location']:
                grouped_data[opp_id]['files'].append({
                    'file_id': record['file_id'],
                    'file_location': record['file_location'],
                    'file_size_bytes': record['file_size_bytes']
                })
        
        return grouped_data
    
    def replace_document_path(self, original_path: str) -> str:
        """Replace document path prefix for accessing files"""
        if not DOCUMENT_PATH_TO_REPLACE or not DOCUMENT_PATH_REPLACEMENT_VALUE:
            return original_path
            
        if original_path.startswith(DOCUMENT_PATH_TO_REPLACE):
            new_path = original_path.replace(DOCUMENT_PATH_TO_REPLACE, DOCUMENT_PATH_REPLACEMENT_VALUE, 1)
            new_path = new_path.replace('\\', '/')
            return new_path
        else:
            return original_path
    
    # Copy essential filter and consolidation methods
    def _filter_entities_by_confidence(self, entities) -> List:
        """Filter entities based on confidence thresholds"""
        filtered_entities = []
        
        logger.info(f"🔍 Filtering {len(entities)} entities by confidence threshold {ENTITY_CONF_THRESHOLD}")
        
        for entity in entities:
            confidence = entity.confidence_score
            
            logger.debug(f"Entity: {entity.name or 'No Name'} / {entity.email or 'No Email'} - Confidence: {confidence}")
            
            if confidence < ENTITY_CONF_THRESHOLD:
                logger.debug(f"❌ Entity filtered out: confidence {confidence} < {ENTITY_CONF_THRESHOLD}")
                continue
            
            if entity.email or entity.name:
                filtered_entities.append(entity)
                logger.debug(f"✅ Entity passed filter: {entity.name or entity.email}")
            else:
                logger.debug(f"❌ Entity filtered out: no email or name")
        
        logger.info(f"📊 Confidence filtering result: {len(filtered_entities)} entities passed out of {len(entities)}")
        return filtered_entities
    
    def _consolidate_entities_per_opportunity(self, entities) -> List:
        """Consolidate entities per opportunity with absolute uniqueness"""
        if not entities:
            return []
        
        logger.info(f"🔄 Consolidating {len(entities)} entities per opportunity")
        
        opportunity_groups = {}
        for entity in entities:
            opp_id = entity.opportunity_id
            if opp_id not in opportunity_groups:
                opportunity_groups[opp_id] = []
            opportunity_groups[opp_id].append(entity)
        
        logger.debug(f"Entity groups by opportunity: {[(opp_id, len(ents)) for opp_id, ents in opportunity_groups.items()]}")
        
        consolidated = []
        
        for opp_id, opp_entities in opportunity_groups.items():
            seen_emails = set()
            seen_names = set()
            opportunity_entities = []
            
            logger.debug(f"Processing {len(opp_entities)} entities for opportunity {opp_id}")
            
            for entity in opp_entities:
                if entity.email and entity.email.strip():
                    email_key = entity.email.lower().strip()
                    if email_key not in seen_emails:
                        seen_emails.add(email_key)
                        opportunity_entities.append(entity)
                        logger.debug(f"✅ Added entity with email: {entity.email}")
                    else:
                        logger.debug(f"❌ Duplicate email filtered: {entity.email}")
                elif entity.name and entity.name.strip():
                    name_key = entity.name.lower().strip()
                    if name_key not in seen_names:
                        seen_names.add(name_key)
                        opportunity_entities.append(entity)
                        logger.debug(f"✅ Added entity with name: {entity.name}")
                    else:
                        logger.debug(f"❌ Duplicate name filtered: {entity.name}")
                else:
                    logger.debug(f"❌ Entity has no email or name")
            
            logger.debug(f"Opportunity {opp_id}: {len(opportunity_entities)} unique entities")
            consolidated.extend(opportunity_entities)
        
        logger.info(f"📊 Consolidation result: {len(consolidated)} unique entities across all opportunities")
        return consolidated
    
    def process_title_embedding(self, opportunity_id: str, title: str, posted_date: str = None, replace_existing: bool = False) -> bool:
        """Process title embedding and store in vector database"""
        try:
            # Titles are typically short and don't need chunking
            # Generate embedding for title
            with time_operation('title_embedding_generation'):
                title_embedding = self.encode_with_pool([title], normalize_embeddings=True)[0]
            
            # Insert into titles collection (single chunk)
            if 'titles' in self.collections:
                collection = self.collections['titles']
                with time_operation('title_vector_insert'):
                    # Schema: opportunity_id, embedding, posted_date, importance_score, chunk_index, total_chunks, text_content
                    collection.insert([
                        [opportunity_id],           # opportunity_id field
                        [title_embedding],          # embedding field
                        [posted_date or ''],        # posted_date field
                        [1.0],                      # importance_score (default)
                        [0],                        # chunk_index (0 for single chunk)
                        [1],                        # total_chunks (1 for single chunk)
                        [title]                     # text_content field
                    ])
                    # Note: Removed immediate flush() for better performance
                
                logger.debug(f"Inserted title embedding for opportunity {opportunity_id}")
                return True
            else:
                logger.error("Titles collection not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process title embedding for {opportunity_id}: {e}")
            return False

    def process_description_embedding(self, opportunity_id: str, description: str, posted_date: str = None, replace_existing: bool = False) -> bool:
        """Process description embedding with chunking and store in vector database"""
        try:
            # Descriptions can be long and need chunking
            chunks = self.chunker.chunk_text(description)
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                logger.warning(f"No chunks generated for description of opportunity {opportunity_id}")
                return False
            
            # Generate embeddings for all chunks
            with time_operation('description_embedding_generation'):
                chunk_embeddings = self.encode_with_pool(chunks, normalize_embeddings=True)
            
            # Insert each chunk into descriptions collection
            if 'descriptions' in self.collections:
                collection = self.collections['descriptions']
                
                # Prepare batch data for all chunks
                opportunity_ids = [opportunity_id] * total_chunks
                embeddings = chunk_embeddings
                posted_dates = [posted_date or ''] * total_chunks
                importance_scores = [1.0] * total_chunks  # Default importance
                chunk_indices = list(range(total_chunks))
                total_chunks_list = [total_chunks] * total_chunks
                text_contents = chunks
                
                with time_operation('description_vector_insert'):
                    # Schema: opportunity_id, embedding, posted_date, importance_score, chunk_index, total_chunks, text_content
                    collection.insert([
                        opportunity_ids,            # opportunity_id field
                        embeddings,                 # embedding field
                        posted_dates,               # posted_date field
                        importance_scores,          # importance_score
                        chunk_indices,              # chunk_index
                        total_chunks_list,          # total_chunks
                        text_contents               # text_content field
                    ])
                    # Note: Removed immediate flush() for better performance
                
                logger.debug(f"Inserted {total_chunks} description embedding chunks for opportunity {opportunity_id}")
                return True
            else:
                logger.error("Descriptions collection not available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process description embedding for {opportunity_id}: {e}")
            return False

    # ==================== SEARCH FUNCTIONALITY ====================
    
    def search_similar_documents(self, query: str, limit: int = 10, boost_factor: float = 1.0, 
                               include_entities: bool = False, 
                               title_similarity_threshold: float = 0.5,
                               description_similarity_threshold: float = 0.5,
                               document_similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform aggregated similarity search using a text query string
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            boost_factor: Score boosting multiplier (legacy parameter)
            include_entities: Whether to include entity information (legacy parameter)
            title_similarity_threshold: Minimum similarity score for title matches
            description_similarity_threshold: Minimum similarity score for description matches
            document_similarity_threshold: Minimum similarity score for document matches
            
        Returns:
            List of aggregated search results with opportunity-level scores
        """
        try:
            logger.info(f"Starting aggregated similarity search for query: '{query}' (limit: {limit})")
            
            # Generate embedding for the query
            query_embedding = self.encode_with_pool([query], normalize_embeddings=True)[0]
            
            # Collect all similar opportunities across all collections
            similar_opportunities = {}
            
            # Search in parallel across collections
            def search_collection(collection_key, threshold):
                collection_name_map = {
                    'titles': 'opportunity_titles',
                    'descriptions': 'opportunity_descriptions', 
                    'opportunity_documents': 'opportunity_documents'
                }
                collection_name = collection_name_map[collection_key]
                
                try:
                    if collection_key not in self.collections:
                        logger.warning(f"Collection {collection_key} not found")
                        return
                        
                    collection = self.collections[collection_key]
                    collection.load()
                    
                    # Enhanced search parameters for better recall
                    search_params = {
                        "metric_type": "COSINE",
                        "params": {"nprobe": 20}  # Increased from 10 for better recall
                    }
                    
                    # Get output fields based on collection type
                    if collection_key in ['titles', 'descriptions']:
                        output_fields = ["opportunity_id", "posted_date", "importance_score", 
                                       "chunk_index", "total_chunks"]
                    else:  # opportunity_documents
                        output_fields = ["opportunity_id", "posted_date", "base_importance", 
                                       "chunk_index", "total_chunks", "file_location", "section_type"]
                    
                    # Perform vector search
                    results = collection.search(
                        data=[query_embedding],
                        anns_field="embedding",
                        param=search_params,
                        limit=limit * 3,  # Get more results for aggregation
                        output_fields=output_fields
                    )
                    
                    # Process results and collect scores
                    if collection_key == 'titles':
                        score_key = 'title_score'
                    elif collection_key == 'descriptions':
                        score_key = 'description_score'
                    else:  # opportunity_documents
                        score_key = 'document_score'
                        
                    self._process_search_results(results[0], similar_opportunities, score_key, threshold)
                    
                    logger.info(f"Processed {len(results[0])} results from {collection_name}")
                    
                except Exception as e:
                    logger.warning(f"Search failed for collection {collection_name}: {e}")
            
            # Execute searches in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(search_collection, 'titles', title_similarity_threshold),
                    executor.submit(search_collection, 'descriptions', description_similarity_threshold),
                    executor.submit(search_collection, 'opportunity_documents', document_similarity_threshold)
                ]
                
                # Wait for all searches to complete
                for future in futures:
                    future.result()
            
            # Aggregate scores across chunks for each opportunity
            aggregated_opportunities = self._aggregate_opportunity_scores(similar_opportunities)
            
            # Format final results
            final_results = []
            for opp_id, scores in aggregated_opportunities.items():
                result = {
                    'opportunity_id': opp_id,
                    'title_score': scores['title_score'],
                    'description_score': scores['description_score'], 
                    'document_score': scores['document_score'],
                    'combined_score': scores['title_score'] + scores['description_score'] + scores['document_score'],
                    'document_match_count': scores['document_match_count'],
                    'title_match_count': scores['title_match_count'],
                    'description_match_count': scores['description_match_count']
                }
                final_results.append(result)
            
            # Sort by highest combined score
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            final_results = final_results[:limit]
            
            logger.info(f"Aggregated similarity search completed: {len(final_results)} results found")
            return final_results
            
        except Exception as e:
            logger.error(f"Aggregated similarity search failed: {e}")
            return []
    
    def search_similar_opportunities(self, opportunity_ids: List[str], 
                                   title_similarity_threshold: float = 0.5,
                                   description_similarity_threshold: float = 0.5,
                                   document_similarity_threshold: float = 0.5,
                                   start_posted_date: Optional[str] = None,
                                   end_posted_date: Optional[str] = None,
                                   document_sow_boost_multiplier: float = 0.0,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Perform opportunity-based similarity search using opportunity GUIDs
        
        Args:
            opportunity_ids: List of opportunity GUIDs to search from
            title_similarity_threshold: Minimum similarity score for title matches
            description_similarity_threshold: Minimum similarity score for description matches  
            document_similarity_threshold: Minimum similarity score for document matches
            start_posted_date: Optional start date filter (YYYY-MM-DD format)
            end_posted_date: Optional end date filter (YYYY-MM-DD format)
            document_sow_boost_multiplier: Boost multiplier for document SOW scores
            limit: Maximum number of results to return
            
        Returns:
            List of similar opportunities with scores for each content type
        """
        try:
            logger.info(f"Starting opportunity similarity search for {len(opportunity_ids)} opportunities")
            
            # Get embeddings for input opportunities
            input_embeddings = self._get_opportunity_embeddings(opportunity_ids)
            
            # Build date filter
            date_filter = self._build_date_filter(start_posted_date, end_posted_date)
            
            # Collect all similar opportunities
            similar_opportunities = {}
            
            # For each input opportunity, find similar ones
            for input_opp_id, embeddings in input_embeddings.items():
                if not any([embeddings.get('title'), embeddings.get('description'), embeddings.get('documents')]):
                    logger.warning(f"No embeddings found for opportunity {input_opp_id}")
                    continue
                
                # Prepare search tasks for parallel execution
                search_tasks = []
                
                # Add title search task
                if embeddings.get('title'):
                    search_tasks.append({
                        'type': 'title',
                        'collection': 'opportunity_titles',
                        'embedding': embeddings['title']['embedding'],
                        'threshold': title_similarity_threshold,
                        'score_key': 'title_score'
                    })
                
                # Add description search task
                if embeddings.get('description'):
                    search_tasks.append({
                        'type': 'description', 
                        'collection': 'opportunity_descriptions',
                        'embedding': embeddings['description']['embedding'],
                        'threshold': description_similarity_threshold,
                        'score_key': 'description_score'
                    })
                
                # Add document search tasks
                if embeddings.get('documents'):
                    for doc_embedding in embeddings['documents']:
                        search_tasks.append({
                            'type': 'document',
                            'collection': 'opportunity_documents',
                            'embedding': doc_embedding['embedding'],
                            'threshold': document_similarity_threshold,
                            'score_key': 'document_score',
                            'boost_factor': doc_embedding.get('boost_factor', 1.0),
                            'use_boost': document_sow_boost_multiplier > 0.0,
                            'boost_multiplier': document_sow_boost_multiplier
                        })
                
                # Execute searches in parallel
                with ThreadPoolExecutor(max_workers=min(len(search_tasks), 4)) as executor:
                    future_to_task = {
                        executor.submit(self._execute_search_task, task, date_filter, limit): task 
                        for task in search_tasks
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            results = future.result()
                            if results:
                                self._process_search_results(results, similar_opportunities,
                                                           task['score_key'], task['threshold'],
                                                           use_boosted=task.get('use_boost', False))
                        except Exception as e:
                            logger.warning(f"Search task failed for {task['type']}: {e}")
            
            # Aggregate scores using multi-level strategy
            aggregated_opportunities = self._aggregate_opportunity_scores(similar_opportunities)
            
            # Format final results
            final_results = []
            for opp_id, scores in aggregated_opportunities.items():
                # Skip if it's one of the input opportunities
                if opp_id in opportunity_ids:
                    continue
                    
                result = {
                    'opportunity_id': opp_id,
                    'title_score': scores['title_score'],
                    'description_score': scores['description_score'], 
                    'document_score': scores['document_score'],
                    'document_match_count': scores['document_match_count'],
                    'title_match_count': scores['title_match_count'],
                    'description_match_count': scores['description_match_count']
                }
                final_results.append(result)
            
            # Sort by highest combined score
            final_results.sort(key=lambda x: (x['title_score'] + x['description_score'] + x['document_score']), reverse=True)
            final_results = final_results[:limit]
            
            logger.info(f"Opportunity similarity search completed: {len(final_results)} results found")
            return final_results
            
        except Exception as e:
            logger.error(f"Opportunity similarity search failed: {e}")
            return []
    
    def _get_opportunity_embeddings(self, opportunity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get embeddings for given opportunity GUIDs"""
        embeddings_data = {}
        
        try:
            # Get title embeddings
            if 'opportunity_titles' in self.collections:
                collection = self.collections['opportunity_titles']
                collection.load()
                
                # Query for each opportunity ID
                for opp_id in opportunity_ids:
                    try:
                        results = collection.query(
                            expr=f'opportunity_id == "{opp_id}"',
                            output_fields=["embedding", "text_content"]
                        )
                        
                        if results:
                            if opp_id not in embeddings_data:
                                embeddings_data[opp_id] = {}
                            embeddings_data[opp_id]['title'] = {
                                'embedding': results[0]['embedding'],
                                'text': results[0]['text_content']
                            }
                    except Exception as e:
                        logger.warning(f"Failed to get title embedding for {opp_id}: {e}")
            
            # Get description embeddings
            if 'opportunity_descriptions' in self.collections:
                collection = self.collections['opportunity_descriptions']
                collection.load()
                
                for opp_id in opportunity_ids:
                    try:
                        results = collection.query(
                            expr=f'opportunity_id == "{opp_id}"',
                            output_fields=["embedding", "text_content"]
                        )
                        
                        if results:
                            if opp_id not in embeddings_data:
                                embeddings_data[opp_id] = {}
                            embeddings_data[opp_id]['description'] = {
                                'embedding': results[0]['embedding'],
                                'text': results[0]['text_content']
                            }
                    except Exception as e:
                        logger.warning(f"Failed to get description embedding for {opp_id}: {e}")
            
            # Get document embeddings
            if 'opportunity_documents' in self.collections:
                collection = self.collections['opportunity_documents']
                collection.load()
                
                for opp_id in opportunity_ids:
                    try:
                        results = collection.query(
                            expr=f'opportunity_id == "{opp_id}"',
                            output_fields=["embedding", "file_id", "section_type", "base_importance"]
                        )
                        
                        if results:
                            if opp_id not in embeddings_data:
                                embeddings_data[opp_id] = {}
                            
                            doc_embeddings = []
                            for result in results:
                                doc_embeddings.append({
                                    'embedding': result['embedding'],
                                    'file_id': result['file_id'],
                                    'section_type': result.get('section_type'),
                                    'boost_factor': result.get('base_importance', 1.0)
                                })
                            
                            embeddings_data[opp_id]['documents'] = doc_embeddings
                            
                    except Exception as e:
                        logger.warning(f"Failed to get document embeddings for {opp_id}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to get opportunity embeddings: {e}")
        
        return embeddings_data
    
    def _search_collection_by_embedding(self, collection_name: str, embedding: List[float],
                                      date_filter: Optional[str], limit: int) -> List[Any]:
        """Search a specific collection using an embedding"""
        try:
            if collection_name not in self.collections:
                return []
                
            collection = self.collections[collection_name]
            collection.load()
            
            search_params = {
                "metric_type": "COSINE", 
                "params": {"nprobe": 10}
            }
            
            results = collection.search(
                data=[embedding],
                anns_field="embedding", 
                param=search_params,
                limit=limit,
                expr=date_filter,
                output_fields=["opportunity_id", "file_id", "posted_date", "section_type", "base_importance"]
            )
            
            return results[0] if results else []
            
        except Exception as e:
            logger.warning(f"Search failed for collection {collection_name}: {e}")
            return []
    
    def _execute_search_task(self, task: Dict, date_filter: Optional[str], limit: int) -> List[Any]:
        """Execute a single search task (used for parallel execution)"""
        try:
            # Perform the search
            results = self._search_collection_by_embedding(
                task['collection'], task['embedding'], date_filter, limit
            )
            
            # Apply boost multiplier for document results if needed
            if task['type'] == 'document' and task.get('use_boost', False):
                boost_multiplier = task.get('boost_multiplier', 0.0)
                boost_factor = task.get('boost_factor', 1.0)
                
                for hit in results:
                    # For COSINE metric: distance 1.0 = identical, distance 0.0 = opposite
                    base_score = hit.distance  # COSINE distance IS the similarity score
                    boosted_score = self._calculate_boosted_score(
                        base_score, boost_factor, boost_multiplier
                    )
                    hit.boosted_score = boosted_score
            
            return results
            
        except Exception as e:
            logger.warning(f"Search task execution failed for {task['type']}: {e}")
            return []
    
    def _process_search_results(self, search_results: List[Any], similar_opportunities: Dict,
                              score_key: str, threshold: float, use_boosted: bool = False):
        """Process search results and collect all scores for aggregation"""
        for hit in search_results:
            opp_id = hit.entity.get('opportunity_id')
            if not opp_id:
                continue
                
            score = getattr(hit, 'boosted_score', None) if use_boosted else None
            if score is None:
                # For COSINE metric in Milvus: distance 1.0 = identical vectors, distance 0.0 = opposite vectors
                # Convert to similarity score where 1.0 = perfect match, 0.0 = no similarity
                if hasattr(hit, 'distance'):
                    score = hit.distance  # For COSINE, distance IS the similarity score
                else:
                    score = 1.0 - hit.distance  # Fallback for other metrics
                
            if score < threshold:
                continue
                
            if opp_id not in similar_opportunities:
                similar_opportunities[opp_id] = {
                    'title_scores': [],
                    'description_scores': [],
                    'document_scores': []
                }
                
            # Collect all scores instead of just keeping the max
            if score_key == 'title_score':
                similar_opportunities[opp_id]['title_scores'].append(score)
            elif score_key == 'description_score':
                similar_opportunities[opp_id]['description_scores'].append(score)
            elif score_key == 'document_score':
                similar_opportunities[opp_id]['document_scores'].append(score)
    
    def _aggregate_opportunity_scores(self, similar_opportunities: Dict) -> Dict:
        """
        Aggregate multiple chunk scores per opportunity using multi-level strategy:
        - Title: Use max score (usually single chunk)
        - Description: Use top-3 average 
        - Document: Use top-5 average with count bonus
        """
        import math
        
        aggregated = {}
        
        for opp_id, score_data in similar_opportunities.items():
            title_scores = score_data.get('title_scores', [])
            description_scores = score_data.get('description_scores', [])
            document_scores = score_data.get('document_scores', [])
            
            # Title: Usually 1 chunk, use max
            title_score = max(title_scores) if title_scores else 0.0
            
            # Description: Few chunks, use top-3 average
            desc_scores_sorted = sorted(description_scores, reverse=True)[:3]
            description_score = sum(desc_scores_sorted) / len(desc_scores_sorted) if desc_scores_sorted else 0.0
            
            # Documents: Many chunks, use top-5 average with count bonus
            doc_scores_sorted = sorted(document_scores, reverse=True)[:5]
            base_doc_score = sum(doc_scores_sorted) / len(doc_scores_sorted) if doc_scores_sorted else 0.0
            
            # Bonus for having multiple strong matches (diminishing returns)
            # Use log(1 + count) to avoid issues with count=0 and provide diminishing returns
            count_bonus = 1.0 + 0.1 * math.log(1 + len(document_scores))
            count_bonus = min(1.2, count_bonus)  # Cap at 20% bonus
            document_score = base_doc_score * count_bonus
            
            aggregated[opp_id] = {
                'title_score': title_score,
                'description_score': description_score,
                'document_score': document_score,
                'document_match_count': len(document_scores),
                'title_match_count': len(title_scores),
                'description_match_count': len(description_scores)
            }
        
        return aggregated
    
    def _build_date_filter(self, start_date: Optional[str], end_date: Optional[str]) -> Optional[str]:
        """Build date filter expression for Milvus queries"""
        if not start_date and not end_date:
            return None
            
        filters = []
        if start_date:
            filters.append(f"posted_date >= '{start_date}'")
        if end_date:
            filters.append(f"posted_date <= '{end_date}'")
            
        return " and ".join(filters)
    
    def _calculate_boosted_score(self, unboosted_score: float, boost_factor: float, 
                               boost_multiplier: float) -> float:
        """Calculate boosted score using the formula: UBS + (UBS * (BF - 1) * DocumentSOWBoostMultiplier)"""
        if boost_multiplier == 0.0:
            return unboosted_score
            
        boosted_score = unboosted_score + (unboosted_score * (boost_factor - 1) * boost_multiplier)
        return min(boosted_score, 1.0)  # Cap at 1.0


def main():
    """Main function to test the scalable processor with new configuration"""
    try:
        print("🚀 Starting Scalable Document Processing Test")
        print("="*60)
        print(f"Configuration:")
        print(f"  Opportunity Workers: {MAX_OPPORTUNITY_WORKERS}")
        print(f"  File Workers per Opportunity: {MAX_FILE_WORKERS_PER_OPPORTUNITY}")
        print(f"  Total Concurrent Workers: {MAX_OPPORTUNITY_WORKERS * MAX_FILE_WORKERS_PER_OPPORTUNITY}")
        print(f"  Parallel Processing: {ENABLE_PARALLEL_PROCESSING}")
        print("="*60)
        
        # Initialize the processor
        processor = ScalableEnhancedProcessor()
        
        # Test with first 5 rows
        start_row = 1
        end_row = 5
        
        print(f"Processing opportunities from row {start_row} to {end_row}...")
        processor.process_scalable_batch(start_row, end_row, replace_existing_records=False)
        
        print("✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
