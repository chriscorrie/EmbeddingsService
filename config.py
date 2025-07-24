import os

# HuggingFace Offline Mode - prevent API calls
os.environ['HF_HUB_OFFLINE'] = '1'  # Force offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Transformers offline mode

# Paths
SAMBA_SHARE_PATH = '/mnt/HomerShare/FBO Attachments'
DOCUMENTS_PATH = '/mnt/HomerShare/FBO Attachments'  # Path to documents to process
VECTOR_DB_PATH = '/mnt/NVME_1/vector_db'

# Boilerplate Documents Configuration
BOILERPLATE_DOCS_PATH = '/mnt/HomerShare/BoilerplateDocuments'  # Directory containing boilerplate contract documents

# Document Path Replacement Configuration
# Used when documents are stored on a different path than where they're accessed for processing
DOCUMENT_PATH_TO_REPLACE = 'D:\\'  # Path prefix in database that needs to be replaced
DOCUMENT_PATH_REPLACEMENT_VALUE = '/mnt/HomerShare/'  # Replacement path prefix

# SQL Server connection
SQL_SERVER_CONNECTION_STRING = 'DRIVER={FreeTDS};SERVER=HOMER.THE-CORRIES.COM;PORT=1433;DATABASE=FedProcurementData;UID=sa;PWD=Bu11d@g94;TDS_Version=8.0;'
SQL_CONNECTION_STRING = 'DRIVER={FreeTDS};SERVER=HOMER.THE-CORRIES.COM;PORT=1433;DATABASE=FedProcurementData;UID=sa;PWD=Bu11d@g94;TDS_Version=8.0;'  # Alias for consistency

# Embedding model
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# Entity Extraction Configuration
ENABLE_ENTITY_EXTRACTION = True  # System-wide setting for entity extraction

# Entity Extraction Confidence Thresholds
ENTITY_PERSON_CONF_THRESHOLD = 0.8   # Person confidence threshold  
ENTITY_TITLE_CONF_THRESHOLD = 0.8    # Title confidence threshold
ENTITY_CONF_THRESHOLD = 0.8          # General entity confidence threshold for storage

# Parallel Processing Configuration - SINGLE WORKER OPTIMAL (PROVEN)
MAX_OPPORTUNITY_WORKERS = 1           # Single worker architecture optimal for GPU
MAX_FILE_WORKERS_PER_OPPORTUNITY = 1  # Sequential file processing (prevent model overload)
ENABLE_PARALLEL_PROCESSING = False    # Single worker architecture for maximum GPU performance

# Dynamic File Worker Scaling (for opportunities with many files) - CONTROLLED SCALING
ENABLE_DYNAMIC_FILE_WORKERS = False  # Disable dynamic scaling to prevent resource contention (was True)
MIN_FILES_FOR_SCALING = 8            # Minimum files to trigger additional file workers (increased from 4 - much more conservative)
MAX_DYNAMIC_FILE_WORKERS = 2         # Maximum file workers for high-file-count opportunities (reduced from 4 - minimal threading)

# Performance Optimization Settings - OPTIMAL FOR REALISTIC CONTENT
EMBEDDING_BATCH_SIZE = 256               # OPTIMAL for realistic federal procurement documents (7,879 sentences/sec)
ENTITY_BATCH_SIZE = 1000                 # Doubled entity processing batches
VECTOR_INSERT_BATCH_SIZE = 1600          # Doubled vector database batch size

# Resource Management - GPU OPTIMIZED (15.5GB VRAM available)
MAX_MEMORY_USAGE_MB = 8192           # Increased for GPU workloads (was 4096)
ENABLE_MEMORY_MONITORING = True     # Monitor and log memory usage during processing
CPU_CORE_MULTIPLIER = 1.5           # Reduced CPU workers since GPU handles embedding load

# Boilerplate Removal Configuration
BOILERPLATE_SIMILARITY_THRESHOLD = 0.9  # Cosine similarity threshold for identifying boilerplate chunks

# Aggressive Performance Overrides (bypass conservative resource manager)
BYPASS_RESOURCE_MANAGER = True              # Skip conservative resource calculations
FORCE_AGGRESSIVE_CONFIG = True              # Use aggressive settings regardless of resource manager
AGGRESSIVE_I_O_OPTIMIZATION = True          # Optimize for I/O bound workloads (23x speedup potential)

# Phase 1 Optimization Flags - POST BOILERPLATE FIX
ENABLE_VECTORIZED_SIMILARITIES = True       # Use vectorized boilerplate filtering (20x speedup)
ENABLE_BATCH_VECTOR_INSERTS = True          # Batch database vector operations (3x speedup potential)
ENABLE_PARALLEL_ENTITY_EXTRACTION = True    # Run entities parallel with embeddings (Phase 2)
ENABLE_CHUNK_EMBEDDING_CACHE = True         # Cache duplicate chunk embeddings (Phase 2)

# Phase 2 Optimization Settings - MAXIMUM GPU CACHE
CHUNK_CACHE_SIZE = 50000                     # Doubled cache for maximum GPU performance
CHUNK_CACHE_SIMILARITY_THRESHOLD = 0.98     # Similarity threshold for considering chunks identical
ENTITY_WORKER_POOL_SIZE = 2                 # Further reduced since GPU dominates

# Large Document Handling - MAXIMUM GPU PERFORMANCE
LARGE_DOC_CHUNK_THRESHOLD = 1000            # Much higher threshold for GPU processing
LARGE_DOC_SEQUENTIAL_PROCESSING = False     # GPU can handle parallel processing better
LARGE_DOC_EMBEDDING_BATCH_SIZE = 1024       # MAXIMUM GPU batches for large documents
PROCESSING_TIMEOUT_MINUTES = 60             # Longer timeout for massive GPU processing

# Real-Time Progress Monitoring
ENABLE_REAL_TIME_PROGRESS = True            # Enable real-time progress updates via callbacks
PROGRESS_UPDATE_INTERVAL = 10               # Background progress monitor update interval (seconds)

# Embedding Model Optimization - SINGLE WORKER ARCHITECTURE (PROVEN OPTIMAL)
EMBEDDING_MODEL_POOL_SIZE = 1               # Single model instance (multi-worker fails with PyTorch 2.9)
EMBEDDING_QUEUE_MAX_SIZE = 1000             # Maximum queued embedding requests
EMBEDDING_TIMEOUT_SECONDS = 30              # Timeout for embedding operations
ENABLE_EMBEDDING_MODEL_POOL = False         # Single worker - no need for pool

# GPU Configuration - OPTIMAL SINGLE WORKER PERFORMANCE (15.5GB VRAM)
ENABLE_GPU_ACCELERATION = True              # Enable GPU acceleration with PyTorch 2.9+ Blackwell support
GPU_DEVICE = 'cuda'                         # CUDA device ('cuda' or 'cuda:0', 'cuda:1', etc.)
GPU_BATCH_SIZE_MULTIPLIER = 1               # Direct batch size control (32,768 = optimal)
FALLBACK_TO_CPU = True                      # Fallback to CPU if GPU not available
