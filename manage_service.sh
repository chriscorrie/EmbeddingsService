#!/bin/bash
# Service management script for Document Embedding REST API

SERVICE_NAME="document-embedding-api"
PROJECT_DIR="/home/chris/document_embedding_project_clean"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_usage() {
    echo -e "${BLUE}Document Embedding REST API Service Manager${NC}"
    echo "Usage: $0 {start|stop|restart|status|logs|url|test}"
    echo ""
    echo "Commands:"
    echo "  start    - Start the service"
    echo "  stop     - Stop the service"
    echo "  restart  - Restart the service"
    echo "  status   - Show service status"
    echo "  logs     - Show service logs (live)"
    echo "  url      - Show service URLs"
    echo "  test     - Test service health"
}

get_service_urls() {
    cd "$PROJECT_DIR"
    python3 -c "
import sys
sys.path.append('$PROJECT_DIR')
try:
    from production_config import get_server_info
    info = get_server_info()
    print(f'Base URL: {info[\"base_url\"]}')
    print(f'Health Check: {info[\"health_url\"]}')
    print(f'Hostname: {info[\"hostname\"]}')
    print(f'Local IP: {info[\"local_ip\"]}')
    print(f'Port: {info[\"port\"]}')
except Exception as e:
    print(f'Error getting server info: {e}')
"
}

test_service() {
    cd "$PROJECT_DIR"
    HEALTH_URL=$(python3 -c "
import sys
sys.path.append('$PROJECT_DIR')
try:
    from production_config import get_server_info
    info = get_server_info()
    print(info['health_url'])
except:
    print('http://localhost:5000/api/v1/health')
")
    
    echo -e "${YELLOW}Testing service health...${NC}"
    if curl -s -f "$HEALTH_URL" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is healthy!${NC}"
        curl -s "$HEALTH_URL" | python3 -m json.tool
    else
        echo -e "${RED}❌ Service is not responding${NC}"
        return 1
    fi
}

case "$1" in
    start)
        echo -e "${YELLOW}Starting $SERVICE_NAME...${NC}"
        sudo systemctl start "$SERVICE_NAME"
        sleep 2
        sudo systemctl status "$SERVICE_NAME" --no-pager
        ;;
    stop)
        echo -e "${YELLOW}Stopping $SERVICE_NAME...${NC}"
        sudo systemctl stop "$SERVICE_NAME"
        sudo systemctl status "$SERVICE_NAME" --no-pager
        ;;
    restart)
        echo -e "${YELLOW}Restarting $SERVICE_NAME...${NC}"
        sudo systemctl restart "$SERVICE_NAME"
        sleep 2
        sudo systemctl status "$SERVICE_NAME" --no-pager
        ;;
    status)
        echo -e "${BLUE}Service Status:${NC}"
        sudo systemctl status "$SERVICE_NAME" --no-pager
        echo ""
        if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
            echo -e "${GREEN}✅ Service is running${NC}"
            get_service_urls
        else
            echo -e "${RED}❌ Service is not running${NC}"
        fi
        ;;
    logs)
        echo -e "${BLUE}Service Logs (Press Ctrl+C to exit):${NC}"
        sudo journalctl -u "$SERVICE_NAME" -f
        ;;
    url)
        echo -e "${BLUE}Service URLs:${NC}"
        get_service_urls
        ;;
    test)
        test_service
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
