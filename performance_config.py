# Performance Configuration for High-Performance Processing
"""
Performance optimization settings for parallel and async processing
"""

# Performance Settings
MAX_WORKERS = 4  # Number of parallel worker threads
EMBEDDING_BATCH_SIZE = 32  # Optimal batch size for embedding generation
FILE_PROCESSING_BATCH_SIZE = 4  # Number of files to process in parallel per opportunity
ENTITY_BATCH_SIZE = 50  # Batch size for entity database operations
VECTOR_INSERT_BATCH_SIZE = 100  # Batch size for vector database inserts

# Async Processing Settings
ENABLE_ASYNC_DB_OPERATIONS = True  # Enable asynchronous database operations
ENABLE_PARALLEL_FILE_PROCESSING = True  # Enable parallel file processing
ENABLE_CONCURRENT_EMBEDDING_ENTITY = True  # Enable concurrent embedding and entity extraction

# Memory Management
MAX_MEMORY_USAGE_MB = 2048  # Maximum memory usage in MB
CHUNK_MEMORY_LIMIT_MB = 512  # Memory limit for chunk processing
EMBEDDING_MEMORY_LIMIT_MB = 1024  # Memory limit for embedding operations

# Database Connection Pooling
SQL_CONNECTION_POOL_SIZE = 5  # SQL Server connection pool size
MILVUS_CONNECTION_TIMEOUT = 30  # Milvus connection timeout in seconds

# Performance Monitoring
ENABLE_PERFORMANCE_LOGGING = True  # Enable detailed performance logging
LOG_PROCESSING_TIMES = True  # Log individual processing times
ENABLE_MEMORY_MONITORING = True  # Monitor memory usage during processing
