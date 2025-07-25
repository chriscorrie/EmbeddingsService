#!/bin/bash
# Service Configuration Script - Switch between standard and OpenAPI service

SERVICE_NAME="document-embedding-api"
STANDARD_SERVICE="/home/chris/document_embedding_project_clean/enhanced_rest_api_service.py"
OPENAPI_SERVICE="/home/chris/document_embedding_project_clean/openapi_rest_service.py"
SYSTEMD_SERVICE="/etc/systemd/system/${SERVICE_NAME}.service"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_usage() {
    echo -e "${BLUE}Document Embedding API Service Manager${NC}"
    echo "Usage: $0 {status|standard|openapi|restart|stop|start}"
    echo ""
    echo "Commands:"
    echo "  status    - Show current service status"
    echo "  standard  - Switch to standard REST API service"
    echo "  openapi   - Switch to OpenAPI/Swagger documented service"
    echo "  restart   - Restart the service"
    echo "  stop      - Stop the service"
    echo "  start     - Start the service"
}

get_current_service() {
    if [ -f "$SYSTEMD_SERVICE" ]; then
        if grep -q "openapi_rest_service.py" "$SYSTEMD_SERVICE"; then
            echo "openapi"
        else
            echo "standard"
        fi
    else
        echo "unknown"
    fi
}

switch_to_standard() {
    echo -e "${YELLOW}Switching to standard REST API service...${NC}"
    
    # Stop current service
    sudo systemctl stop "$SERVICE_NAME"
    
    # Update systemd service file
    sudo sed -i "s|openapi_rest_service.py|enhanced_rest_api_service.py|g" "$SYSTEMD_SERVICE"
    
    # Reload systemd and start service
    sudo systemctl daemon-reload
    sudo systemctl start "$SERVICE_NAME"
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Switched to standard REST API service${NC}"
        echo -e "${BLUE}Service available at: http://$(hostname -I | awk '{print $1}'):5000/api/v1/${NC}"
    else
        echo -e "${RED}❌ Failed to start standard service${NC}"
    fi
}

switch_to_openapi() {
    echo -e "${YELLOW}Switching to OpenAPI/Swagger documented service...${NC}"
    
    # Stop current service
    sudo systemctl stop "$SERVICE_NAME"
    
    # Update systemd service file
    sudo sed -i "s|enhanced_rest_api_service.py|openapi_rest_service.py|g" "$SYSTEMD_SERVICE"
    
    # Reload systemd and start service
    sudo systemctl daemon-reload
    sudo systemctl start "$SERVICE_NAME"
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}✅ Switched to OpenAPI/Swagger documented service${NC}"
        echo -e "${BLUE}Service available at: http://$(hostname -I | awk '{print $1}'):5000/api/v1/${NC}"
        echo -e "${BLUE}Swagger UI available at: http://$(hostname -I | awk '{print $1}'):5000/docs/${NC}"
        echo -e "${BLUE}OpenAPI spec available at: http://$(hostname -I | awk '{print $1}'):5000/swagger.json${NC}"
    else
        echo -e "${RED}❌ Failed to start OpenAPI service${NC}"
    fi
}

show_status() {
    echo -e "${BLUE}Service Status:${NC}"
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "  ${GREEN}✅ Service is running${NC}"
        current_service=$(get_current_service)
        echo -e "  ${YELLOW}Current service type: ${current_service}${NC}"
        
        # Show service info
        local_ip=$(hostname -I | awk '{print $1}')
        echo -e "  ${BLUE}Base URL: http://${local_ip}:5000/api/v1/${NC}"
        
        if [ "$current_service" = "openapi" ]; then
            echo -e "  ${BLUE}Swagger UI: http://${local_ip}:5000/docs/${NC}"
            echo -e "  ${BLUE}OpenAPI Spec: http://${local_ip}:5000/swagger.json${NC}"
        fi
        
        # Test health endpoint
        if [ "$current_service" = "openapi" ]; then
            health_url="http://${local_ip}:5000/api/v1/health/health"
        else
            health_url="http://${local_ip}:5000/api/v1/health"
        fi
        
        if curl -s -f "$health_url" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ Health check: PASS${NC}"
        else
            echo -e "  ${RED}❌ Health check: FAIL${NC}"
        fi
    else
        echo -e "  ${RED}❌ Service is not running${NC}"
    fi
}

restart_service() {
    echo -e "${YELLOW}Restarting service...${NC}"
    sudo systemctl restart "$SERVICE_NAME"
    sleep 2
    show_status
}

stop_service() {
    echo -e "${YELLOW}Stopping service...${NC}"
    sudo systemctl stop "$SERVICE_NAME"
    echo -e "${GREEN}✅ Service stopped${NC}"
}

start_service() {
    echo -e "${YELLOW}Starting service...${NC}"
    sudo systemctl start "$SERVICE_NAME"
    sleep 2
    show_status
}

case "$1" in
    status)
        show_status
        ;;
    standard)
        switch_to_standard
        ;;
    openapi)
        switch_to_openapi
        ;;
    restart)
        restart_service
        ;;
    stop)
        stop_service
        ;;
    start)
        start_service
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
