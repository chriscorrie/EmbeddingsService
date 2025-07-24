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
Both services are properly Host: 192.168.15.206
User: chris
Port: 22Process documents with embeddings
- `GET /processing_status/{job_id}` - Check processing status

### Search & Query
- `POST /search` - Search documents with semantic similarity
- `POST /search_opportunities_by_date` - Date-filtered opportunity search
- `POST /find_similar_opportunities` - Find similar opportunities

### Analytics & Metrics
- `GET /metrics/performance` - Get performance metrics
- `GET /health` - Service health check
- `GET /stats` - Processing statistics

### Entity Management
- `POST /extract_entities` - Extract entities from text
- `DELETE /opportunities/{opportunity_id}` - Delete opportunity data

Both services expose identical functionality with the documentation service providing interactive testing capabilities through Swagger UI.
