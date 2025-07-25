#!/bin/bash
# Startup script for EmbeddingsService
# Ensures Milvus containers are running before starting the embedding service

set -e

echo "🚀 Starting EmbeddingsService startup sequence..."

# Change to project directory
cd /home/chris/Projects/EmbeddingsService

# Function to check if Milvus is healthy
check_milvus_health() {
    local max_attempts=30
    local attempt=1
    
    echo "🔍 Checking Milvus health..."
    
    while [ $attempt -le $max_attempts ]; do
        if /usr/bin/docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(milvus-standalone|milvus-minio|milvus-etcd)" | grep -v "unhealthy" > /dev/null; then
            echo "✅ Milvus containers are running"
            
            # Test connection
            if python3 -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('Connection successful')" 2>/dev/null; then
                echo "✅ Milvus connection verified"
                return 0
            fi
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts: Waiting for Milvus to be healthy..."
        sleep 10
        ((attempt++))
    done
    
    echo "❌ Milvus failed to become healthy after $max_attempts attempts"
    return 1
}

# Start Milvus containers if not running
echo "🐳 Starting Milvus containers..."
/usr/bin/docker-compose up -d

# Wait for Milvus to be healthy
if check_milvus_health; then
    echo "✅ Milvus is ready"
else
    echo "❌ Milvus startup failed"
    exit 1
fi

echo "🎉 EmbeddingsService startup complete!"
echo "📊 Service status:"
echo "  - Milvus: $(/usr/bin/docker ps --format "table {{.Names}}\t{{.Status}}" | grep milvus | wc -l) containers running"
echo "  - Embedding API: $(sudo systemctl is-active document-embedding-api.service)"
echo "  - Health check: http://localhost:5000/health"
