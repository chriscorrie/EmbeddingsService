#!/bin/bash
# EmbeddingsService Status Checker
# Quick status check for all components

echo "🔍 EmbeddingsService Status Check"
echo "=================================="

# Check Docker containers
echo "🐳 Docker Containers:"
if /usr/bin/docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(milvus|etcd|minio)" | head -3; then
    echo "✅ Milvus containers are running"
else
    echo "❌ Milvus containers not found"
fi
echo ""

# Check systemd service
echo "🚀 Embedding API Service:"
SERVICE_STATUS=$(sudo systemctl is-active document-embedding-api.service)
if [ "$SERVICE_STATUS" = "active" ]; then
    echo "✅ document-embedding-api.service: $SERVICE_STATUS"
else
    echo "❌ document-embedding-api.service: $SERVICE_STATUS"
fi
echo ""

# Check API health
echo "🏥 API Health Check:"
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✅ API is responding at http://localhost:5000"
    echo "📊 API Status: $(curl -s http://localhost:5000/health | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")"
    echo "🔧 API Version: $(curl -s http://localhost:5000/health | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])")"
else
    echo "❌ API is not responding"
fi
echo ""

# Check key endpoints
echo "📋 Available Endpoints:"
echo "  🏥 Health (Legacy): http://localhost:5000/health"
echo "  🏥 Health (API):    http://localhost:5000/api/v1/health"
echo "  📖 Docs:           http://localhost:5000/docs/"
echo "  🔄 Embeddings:     http://localhost:5000/api/v1/embeddings/"
echo "  📊 Status:         http://localhost:5000/api/v1/status/"
echo "  🔍 Search:         http://localhost:5000/api/v1/search/"
echo ""

# Service management commands
echo "🛠️  Service Management:"
echo "  Status:  sudo systemctl status document-embedding-api.service"
echo "  Restart: sudo systemctl restart document-embedding-api.service"
echo "  Logs:    sudo journalctl -u document-embedding-api.service -f"
