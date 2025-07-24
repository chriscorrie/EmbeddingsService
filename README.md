# Document Embedding API Service v3

## Overview
Production-ready document embedding API service with advanced semantic search capabilities using the v3 API architecture.

## Current Architecture
- **Main API Service**: `production_rest_api_service_v3.py` 
- **Core Processor**: `scalable_processor.py` (Phase 2 optimized)
- **Production Config**: `production_config.py`
- **System Service**: `document-embedding-api-v3.service`
- **Management**: `manage_service_v3.sh`

## Service Status
```bash
# Check service status
sudo systemctl status document-embedding-api-v3

# Service management
sudo systemctl start|stop|restart document-embedding-api-v3

# View logs
sudo journalctl -u document-embedding-api-v3 -f

# Management script
./manage_service_v3.sh status|logs|restart
```

## API Endpoints
**Base URL**: `http://192.168.15.206:5000/api/v1`

- **Health Check**: `GET /health`
- **Process Embeddings**: `POST /embeddings/process-embeddings`
- **Processing Status**: `GET /status/processing-status/<task_id>`
- **All Processing Status**: `GET /status/processing-status`
- **Similarity Search**: `POST /search/similarity-search`
- **Opportunity Search**: `POST /search/opportunity-search`

## Quick Test
```bash
# Health check
curl http://192.168.15.206:5000/api/v1/health

# Process embeddings
curl -X POST http://192.168.15.206:5000/api/v1/embeddings/process-embeddings \
  -H "Content-Type: application/json" \
  -d '{"start_row_id": 1, "end_row_id": 10, "reprocess": false}'

# Similarity search
curl -X POST http://192.168.15.206:5000/api/v1/search/similarity-search \
  -H "Content-Type: application/json" \
  -d '{"query": "software development services", "limit": 5, "boost_factor": 1.5}'

# Opportunity search
curl -X POST http://192.168.15.206:5000/api/v1/search/opportunity-search \
  -H "Content-Type: application/json" \
  -d '{"opportunity_ids": ["12345678-1234-1234-1234-123456789abc"], "title_similarity_threshold": 0.7}'

# Check processing status
curl http://192.168.15.206:5000/api/v1/status/processing-status/<task_id>
```

## Core Files
```
├── production_rest_api_service_v3.py    # Main API service
├── scalable_processor.py                # Document processor  
├── production_config.py                 # Production settings
├── document_section_analyzer.py         # Section analysis
├── entity_extractor.py                  # Entity extraction
├── semantic_chunker.py                  # Text chunking
├── semantic_boilerplate_manager.py      # Boilerplate handling
├── resource_manager.py                  # Resource management
├── performance_timer.py                 # Performance monitoring
├── config.py                           # Base configuration
├── requirements.txt                     # Dependencies
├── manage_service_v3.sh                # Service management
├── document-embedding-api-v3.service   # Systemd service
├── venv/                               # Virtual environment
├── logs/                               # Application logs
└── entities.db                        # Entity database
```

## Archive
Legacy files moved to: `/home/chris/document_embedding_archive/`

## Client Integration
See `/home/chris/EmbeddingsClient/` for C# client implementation.
