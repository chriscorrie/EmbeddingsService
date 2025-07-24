#!/bin/bash
# Production startup script for Document Embedding REST API Service

set -e

# Configuration
PROJECT_DIR="/home/chris/document_embedding_project_clean"
SERVICE_NAME="document-embedding-api"
PYTHON_SERVICE="production_rest_api_service.py"
VENV_DIR="$PROJECT_DIR/venv"
USER="chris"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Document Embedding REST API Service Setup${NC}"
echo "================================================="

# Check if running as correct user
if [ "$USER" != "$(whoami)" ]; then
    echo -e "${RED}❌ Please run this script as user: $USER${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}📦 Creating Python virtual environment...${NC}"
    cd "$PROJECT_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
    source "$VENV_DIR/bin/activate"
fi

# Install/update dependencies
echo -e "${YELLOW}📦 Installing/updating dependencies...${NC}"
cd "$PROJECT_DIR"
pip install -q flask flask-cors sentence-transformers pymilvus pyodbc python-dotenv
pip install -q PyPDF2 python-docx openpyxl python-pptx

# Test the service
echo -e "${YELLOW}🧪 Testing service configuration...${NC}"
python3 -c "
import sys
sys.path.append('$PROJECT_DIR')
try:
    from enhanced_processor import EnhancedSectionedProcessor
    from document_section_analyzer import DocumentSectionAnalyzer
    from production_config import get_server_info
    print('✅ Configuration test passed')
except Exception as e:
    print(f'❌ Configuration test failed: {e}')
    sys.exit(1)
"

# Create systemd service file
echo -e "${YELLOW}⚙️  Creating systemd service...${NC}"
sudo cp "$PROJECT_DIR/document-embedding-api.service" "/etc/systemd/system/"
sudo systemctl daemon-reload

# Enable and start the service
echo -e "${YELLOW}🔧 Enabling and starting service...${NC}"
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

# Wait a moment for service to start
sleep 3

# Check service status
echo -e "${YELLOW}🔍 Checking service status...${NC}"
if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${GREEN}✅ Service is running!${NC}"
    
    # Get service information
    python3 -c "
import sys
sys.path.append('$PROJECT_DIR')
try:
    from production_config import get_server_info
    info = get_server_info()
    print(f'🌐 Service URL: {info[\"base_url\"]}')
    print(f'💚 Health Check: {info[\"health_url\"]}')
    print(f'📡 Hostname: {info[\"hostname\"]}')
    print(f'🔌 Port: {info[\"port\"]}')
except Exception as e:
    print(f'Could not get server info: {e}')
"
    
    echo ""
    echo -e "${BLUE}🧪 Test the service:${NC}"
    echo "curl -X GET http://$(hostname -I | awk '{print $1}'):5000/api/v1/health"
    echo ""
    echo -e "${BLUE}📊 Check service status:${NC}"
    echo "sudo systemctl status $SERVICE_NAME"
    echo ""
    echo -e "${BLUE}📜 View service logs:${NC}"
    echo "sudo journalctl -u $SERVICE_NAME -f"
    echo ""
    echo -e "${BLUE}🛑 Stop service:${NC}"
    echo "sudo systemctl stop $SERVICE_NAME"
    echo ""
    echo -e "${BLUE}🔄 Restart service:${NC}"
    echo "sudo systemctl restart $SERVICE_NAME"
    
else
    echo -e "${RED}❌ Service failed to start!${NC}"
    echo "Check logs with: sudo journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Setup complete! Service is running and will auto-start on boot.${NC}"
