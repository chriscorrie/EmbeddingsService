#!/bin/bash
# Management script for Document Embedding API v3 Service

SERVICE_NAME="document-embedding-api-v3"
SERVICE_FILE="/home/chris/document_embedding_project_clean/${SERVICE_NAME}.service"
SYSTEMD_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

case "$1" in
    install)
        echo "Installing Document Embedding API v3 service..."
        
        # Copy service file to systemd directory
        sudo cp "$SERVICE_FILE" "$SYSTEMD_PATH"
        
        # Reload systemd daemon
        sudo systemctl daemon-reload
        
        # Enable the service
        sudo systemctl enable "$SERVICE_NAME"
        
        echo "✓ Service installed and enabled"
        echo "Use 'sudo systemctl start $SERVICE_NAME' to start the service"
        ;;
        
    start)
        echo "Starting Document Embedding API v3 service..."
        sudo systemctl start "$SERVICE_NAME"
        echo "✓ Service started"
        ;;
        
    stop)
        echo "Stopping Document Embedding API v3 service..."
        sudo systemctl stop "$SERVICE_NAME"
        echo "✓ Service stopped"
        ;;
        
    restart)
        echo "Restarting Document Embedding API v3 service..."
        sudo systemctl restart "$SERVICE_NAME"
        echo "✓ Service restarted"
        ;;
        
    status)
        echo "Document Embedding API v3 service status:"
        sudo systemctl status "$SERVICE_NAME"
        ;;
        
    logs)
        echo "Document Embedding API v3 service logs:"
        sudo journalctl -u "$SERVICE_NAME" -f
        ;;
        
    logs-tail)
        echo "Last 50 lines of Document Embedding API v3 service logs:"
        sudo journalctl -u "$SERVICE_NAME" -n 50
        ;;
        
    uninstall)
        echo "Uninstalling Document Embedding API v3 service..."
        
        # Stop and disable the service
        sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
        sudo systemctl disable "$SERVICE_NAME" 2>/dev/null || true
        
        # Remove service file
        sudo rm -f "$SYSTEMD_PATH"
        
        # Reload systemd daemon
        sudo systemctl daemon-reload
        
        echo "✓ Service uninstalled"
        ;;
        
    *)
        echo "Usage: $0 {install|start|stop|restart|status|logs|logs-tail|uninstall}"
        echo ""
        echo "Commands:"
        echo "  install    - Install and enable the service"
        echo "  start      - Start the service"
        echo "  stop       - Stop the service"
        echo "  restart    - Restart the service"
        echo "  status     - Show service status"
        echo "  logs       - Follow service logs in real-time"
        echo "  logs-tail  - Show last 50 lines of logs"
        echo "  uninstall  - Remove the service"
        exit 1
        ;;
esac
