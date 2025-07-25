# Document Embedding API Service

## Overview
Production-ready document embedding API service with advanced semantic search capabilities.

## Current Architecture
- **Main API Service**: `production_rest_api_service.py` 
- **Core Processor**: `scalable_processor.py` (Phase 2 optimized)
- **Production Config**: `production_config.py`
- **System Service**: `document-embedding-api.service`
- **Management**: `scripts/manage_service.sh`

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
./scripts/manage_service.sh status|logs|restart
```

### Quick Status Check
```bash
# Run comprehensive status check
./scripts/status_check.sh

# Manual checks
sudo docker ps  # Check Milvus containers
curl http://localhost:5000/health  # Test API
curl http://localhost:5000/docs/   # Access documentation
```

### Troubleshooting
- **Milvus not responding**: Check `sudo docker logs milvus-standalone`
- **API errors**: Check `sudo journalctl -u document-embedding-api -f`
- **Performance issues**: Verify GPU/CUDA availability and Milvus resource allocation
- **Database reset**: Use `scripts/reset_database.sh` or `python3 ResetDatabase.py` for clean restart

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

## Performance Testing Standards

### Standard Test Parameters
**CRITICAL**: For consistent performance testing and benchmarking, always use these parameters:

```bash
# Standard Performance Test - ALWAYS use these exact parameters
curl -X POST http://localhost:5000/api/v1/embeddings/process-embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "start_row_id": 1, 
    "end_row_id": 35, 
    "reprocess": false
  }'
```

**Testing Guidelines:**
- **Row Range**: Always test rows 1-35 (35 opportunities for consistent sample size)
- **Reprocess**: Always set to `false` for testing (avoids vector DB deletion overhead)
- **Consistency**: Use identical parameters across all performance comparisons
- **Baseline**: This gives ~35 opportunities with mixed document loads for realistic testing

### Performance Comparison Workflow
1. **Restart Service**: `sudo systemctl restart document-embedding-api`
2. **Run Standard Test**: Use the exact curl command above
3. **Monitor Logs**: `sudo journalctl -u document-embedding-api -f`
4. **Check Performance Reports**: `logs/performance_report_*.json`
5. **Document Results**: Record processing time and key metrics

### Configuration Testing
When testing different configurations (entity extraction, parallel processing, etc.):
1. Update `config.py` settings
2. Restart service to pick up changes
3. Run standard test (rows 1-35, reprocess=false)
4. Compare results using identical parameters

## Project Structure
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
├── ResetDatabase.py                    # Database reset utility
├── document-embedding-api.service      # Systemd service
├── debug/                             # Debug, test & performance scripts
│   ├── debug_*.py                     # Debugging scripts
│   ├── test_*.py                      # Feature testing scripts
│   ├── *_performance_test.py          # Performance testing scripts
│   └── *_validation_test.py           # System validation scripts
├── scripts/                           # Utility scripts
│   ├── reset_database.sh              # Database reset wrapper
│   ├── manage_service.sh              # Service management
│   ├── status_check.sh                # System status checker
│   └── startup.sh                     # Service startup script
├── venv/                              # Virtual environment
├── logs/                              # Application logs
└── entities.db                       # Entity database
```

## Scripts Directory
All utility scripts are organized in the `scripts/` directory:

### Database Management
- **`reset_database.sh`** - Wrapper for ResetDatabase.py with automatic venv activation

### Service Management  
- **`manage_service.sh`** - Service control and management
- **`manage_service_enhanced.sh`** - Enhanced service management with installation and monitoring
- **`service_manager.sh`** - Alternative service manager
- **`status_check.sh`** - Comprehensive system status checker
- **`startup.sh`** - Service startup script (used by systemd)

### System Setup & Deployment
- **`setup_production_service.sh`** - Production environment setup
- **`cleanup_old_minio_data.sh`** - MinIO data cleanup utility

### Infrastructure Management
- **`restore_pcie_drives.sh`** - PCIE drive restoration after hardware changes
- **`migrate_to_nvme.sh`** - NVME migration utilities

### Usage Examples
```bash
# Check system status
./scripts/status_check.sh

# Reset database
./scripts/reset_database.sh

# Manage service
./scripts/manage_service.sh status|logs|restart

# Restore PCIE drives
./scripts/restore_pcie_drives.sh
```

**For AI Assistants**: Always check the `scripts/` directory for utility scripts, especially database and system management tools.

## Archive
Legacy files moved to: `/home/chris/document_embedding_archive/`

## Development and Debugging
All debug, diagnostic, testing, and performance scripts are organized in the `debug/` directory:
- **Location**: `debug/` folder contains all non-production development scripts
- **Naming**: Use `debug_[component]_[issue].py`, `test_[feature].py`, or `[component]_performance_test.py` formats
- **Usage**: Run from project root: `python debug/script_name.py`
- **Documentation**: See `debug/README.md` for complete guidelines and script inventory

**For AI Assistants**: Always place new debug, test, performance, and validation scripts in the `debug/` directory, never in the project root.

## Client Integration
See `/home/chris/EmbeddingsClient/` for C# client implementation.
