#!/bin/bash

# OpenAPI Documentation Service Manager
# Usage: ./manage_openapi_service.sh [start|stop|restart|status|logs]

SERVICE_NAME="openapi-documentation.service"

case "$1" in
    start)
        echo "Starting OpenAPI Documentation Service..."
        sudo systemctl start $SERVICE_NAME
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    stop)
        echo "Stopping OpenAPI Documentation Service..."
        sudo systemctl stop $SERVICE_NAME
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    restart)
        echo "Restarting OpenAPI Documentation Service..."
        sudo systemctl restart $SERVICE_NAME
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    status)
        echo "OpenAPI Documentation Service Status:"
        sudo systemctl status $SERVICE_NAME --no-pager
        echo ""
        echo "Network Status:"
        ss -tlpn | grep :8080
        ;;
    logs)
        echo "OpenAPI Documentation Service Logs (last 50 lines):"
        sudo journalctl -u $SERVICE_NAME -n 50 --no-pager
        ;;
    enable)
        echo "Enabling OpenAPI Documentation Service for auto-start..."
        sudo systemctl enable $SERVICE_NAME
        ;;
    disable)
        echo "Disabling OpenAPI Documentation Service auto-start..."
        sudo systemctl disable $SERVICE_NAME
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the service"
        echo "  stop     - Stop the service"
        echo "  restart  - Restart the service"
        echo "  status   - Show service and network status"
        echo "  logs     - Show recent service logs"
        echo "  enable   - Enable auto-start on boot"
        echo "  disable  - Disable auto-start on boot"
        exit 1
        ;;
esac
