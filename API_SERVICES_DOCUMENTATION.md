# API Services Documentation

## 🌐 Document Embedding API Endpoints

### **Production API Service**
- **URL**: `http://192.168.15.206:5000`
- **Service**: `production_rest_api_service_v2.py`
- **Status**: ✅ Active (Auto-starting systemd service)
- **Features**: Phase 2 optimized processing with full functionality

### **OpenAPI Documentation Service**  
- **URL**: `http://192.168.15.206:8080`
- **Service**: `openapi_rest_service.py`
- **Status**: ✅ Active (Auto-starting systemd service)
- **Features**: Interactive Swagger UI with comprehensive API documentation

## 🔥 Firewall Configuration
Both services are properly configured with firewall access.

## 📋 API Endpoints

**Base URL**: `http://192.168.15.206:5000/api/v1`

### Document Processing
- `POST /api/v1/embeddings/process-embeddings` - Process documents with embeddings
  - **Request Body**: 
    ```json
    {
      "start_row_id": 1,
      "end_row_id": 35,
      "reprocess": false
    }
    ```
  - **Response**: Returns task_id for tracking processing status

### Status Monitoring  
- `GET /api/v1/status/processing-status/{task_id}` - Check processing status for a specific task
- `GET /api/v1/status/processing-status` - Get all active processing tasks

### Search & Query
- `POST /api/v1/search/similarity-search` - Search documents with semantic similarity
- `POST /api/v1/search/opportunity-search` - Search for similar opportunities by date range
- `POST /api/v1/search/find-similar-opportunities` - Find similar opportunities

### Analytics & Metrics
- `GET /metrics/performance` - Get performance metrics
- `GET /health` - Service health check
- `GET /stats` - Processing statistics

### Entity Management
- `POST /extract_entities` - Extract entities from text
- `DELETE /opportunities/{opportunity_id}` - Delete opportunity data

## 📊 Performance Logging

### Task-Based Performance Reports
The system now generates task-specific performance reports with the following filename structure:
- **Format**: `logs/performance_report_{task_id}.json`
- **Example**: `logs/performance_report_embed_2025-07-25_16-47-23_abc123.json`
- **Benefits**: 
  - Unique reports for each processing task
  - Easy comparison between different optimization runs
  - No overwriting of previous performance data
  - Task ID includes timestamp and unique identifier

### Legacy Fallback
If no task_id is provided (direct script execution), the system falls back to:
- **Format**: `logs/performance_report_{start_row_id}_{end_row_id}.json`
- **Example**: `logs/performance_report_1_35.json`

Both services expose identical functionality with the documentation service providing interactive testing capabilities through Swagger UI.
