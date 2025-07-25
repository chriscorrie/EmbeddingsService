# Document Embedding API Service

## Overview
Production-ready document embedding API service with advanced semantic search capabilities.

## Current Architecture
- **Main API Service**: `production_rest_api_service.py` 
- **Core Processor**: `scalable_processor.py` (Phase 2 optimized)
- **Production Config**: `production_config.py`
- **System Service**: `document-embedding-api.service`
- **Management**: `manage_service.sh`

## Service Management

### System Service
```bash
# Check service status
sudo systemctl status document-embedding-api

# Service management
sudo systemctl start|stop|restart document-embedding-api

# View logs
sudo journalctl -u document-embedding-api -f

# Management script
./manage_service.sh status|logs|restart
```

### Quick Status Check
```bash
# Run comprehensive status check
./status_check.sh

# Manual checks
sudo docker ps  # Check Milvus containers
curl http://localhost:5000/health  # Test API
curl http://localhost:5000/docs/   # Access documentation
```

### Troubleshooting
- **Milvus not responding**: Check `sudo docker logs milvus-standalone`
- **API errors**: Check `sudo journalctl -u document-embedding-api -f`
- **Performance issues**: Verify GPU/CUDA availability and Milvus resource allocation
- **Database reset**: Use `python3 ResetDatabase.py` for clean restart

## API Endpoints

### Base Configuration
- **Base URL**: `http://localhost:5000/api/v1`
- **Interactive Documentation**: `http://localhost:5000/docs/` (Swagger UI)
- **OpenAPI Spec**: `http://localhost:5000/swagger.json`

### Health & Documentation Endpoints
- **Health Check (Legacy)**: `GET /health`
- **Health Check (Namespaced)**: `GET /api/v1/health`
- **API Documentation**: `GET /docs/`

### Embedding Processing Endpoints
- **Process Embeddings**: `POST /api/v1/embeddings/process-embeddings`
  - Purpose: Process document embeddings for a range of rows
  - Parameters: `start_row_id`, `end_row_id`, `reprocess`
  - Returns: `task_id` for tracking processing status

### Status Monitoring Endpoints
- **Get Processing Status**: `GET /api/v1/status/processing-status/<task_id>`
  - Purpose: Check status of a specific processing task
  - Returns: Task status, progress, and completion details

- **Get All Processing Status**: `GET /api/v1/status/processing-status`
  - Purpose: List all current processing tasks
  - Returns: Array of all active/recent tasks

### Search Endpoints
- **Similarity Search**: `POST /api/v1/search/similarity-search`
  - Purpose: Semantic search using natural language queries
  - Parameters: `query`, `limit`, `title_similarity_threshold`, `description_similarity_threshold`, `document_similarity_threshold`
  - Returns: Ranked results with similarity scores across all content types

- **Opportunity Search**: `POST /api/v1/search/opportunity-search`
  - Purpose: Find similar opportunities using specific opportunity GUIDs
  - Parameters: `opportunity_ids[]`, similarity thresholds, date filters, `document_sow_boost_multiplier`
  - Returns: Similar opportunities ranked by combined similarity scores

## High-Performance Infrastructure

### Milvus Vector Database Optimization
- **CPU Allocation**: 16 cores dedicated to Milvus
- **Memory Allocation**: 32GB RAM for high-throughput operations
- **GPU Compatibility**: Optimized for 5,269 sentences/second GPU processing
- **NVME Storage**: Fast persistent storage for vector data
- **Connection Limits**: Scaled for concurrent high-volume operations

### Performance Specifications
- **GPU Processing**: 5,269 sentences/second with PyTorch 2.9.dev + CUDA 12.9
- **Parallel Processing**: Phase 2 optimizations with multi-threaded execution
- **Vector Database**: Milvus v2.3.0 with aggressive resource allocation
- **System Resources**: 24 CPU cores, 61GB total RAM, dedicated server hardware

### Docker Configuration
High-performance `docker-compose.yml` includes:
- Optimized memory management settings
- Increased connection limits and timeouts
- Performance-tuned environment variables
- Resource constraints matching hardware capabilities

## Quick Test Examples

### Health Check
```bash
# Legacy health endpoint
curl http://localhost:5000/health

# Namespaced health endpoint
curl http://localhost:5000/api/v1/health
```

### Process Embeddings
```bash
curl -X POST http://localhost:5000/api/v1/embeddings/process-embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "start_row_id": 1, 
    "end_row_id": 100, 
    "reprocess": false
  }'
```

### Similarity Search
```bash
curl -X POST http://localhost:5000/api/v1/search/similarity-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence machine learning", 
    "limit": 5,
    "title_similarity_threshold": 0.7,
    "description_similarity_threshold": 0.6,
    "document_similarity_threshold": 0.5
  }'
```

### Opportunity Search
```bash
curl -X POST http://localhost:5000/api/v1/search/opportunity-search \
  -H "Content-Type: application/json" \
  -d '{
    "opportunity_ids": ["12345678-1234-1234-1234-123456789abc"], 
    "title_similarity_threshold": 0.7,
    "description_similarity_threshold": 0.6,
    "document_similarity_threshold": 0.5,
    "limit": 10
  }'
```

### Check Processing Status
```bash
# Specific task
curl http://localhost:5000/api/v1/status/processing-status/<task_id>

# All tasks
curl http://localhost:5000/api/v1/status/processing-status
```

### Interactive Documentation
Visit `http://localhost:5000/docs/` for full interactive API documentation with request/response examples.

## Core Files
```
├── production_rest_api_service.py      # Main API service
├── scalable_processor.py               # Document processor  
├── production_config.py                # Production settings
├── document_section_analyzer.py        # Section analysis
├── entity_extractor.py                 # Entity extraction
├── semantic_chunker.py                 # Text chunking
├── semantic_boilerplate_manager.py     # Boilerplate handling
├── resource_manager.py                 # Resource management
├── performance_timer.py                # Performance monitoring
├── config.py                          # Base configuration
├── requirements.txt                    # Dependencies
├── manage_service.sh                   # Service management
├── document-embedding-api.service      # Systemd service
├── venv/                              # Virtual environment
├── logs/                              # Application logs
└── entities.db                       # Entity database
```

## Archive
Legacy files moved to: `/home/chris/document_embedding_archive/`

## Client Integration
See `/home/chris/EmbeddingsClient/` for C# client implementation.
