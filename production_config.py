#!/usr/bin/env python3
"""
Production configuration for Document Embedding REST API Service
"""

import os
import logging
from pathlib import Path

# Server Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5000
DEBUG = False  # Production mode

# Project paths
PROJECT_ROOT = Path(__file__).parent
VENV_PATH = PROJECT_ROOT / 'venv'
LOGS_PATH = PROJECT_ROOT / 'logs'

# Create logs directory if it doesn't exist
LOGS_PATH.mkdir(exist_ok=True)

# Logging Configuration
LOG_LEVEL = logging.INFO
LOG_FILE = LOGS_PATH / 'document_embedding_api.log'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# API Configuration
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'

# Request limits
MAX_BATCH_SIZE = 1000
DEFAULT_BATCH_SIZE = 100
MAX_SEARCH_LIMIT = 100
DEFAULT_SEARCH_LIMIT = 10

# Processing Configuration
MAX_CONCURRENT_REQUESTS = 1  # Only one embedding generation at a time
PROCESSING_TIMEOUT = 3600  # 1 hour timeout for embedding generation
STATUS_CHECK_INTERVAL = 5  # seconds

# Security Configuration
CORS_ORIGINS = ['*']  # Configure specific origins in production
RATE_LIMIT_ENABLED = False  # Enable if needed

# Health Check Configuration
HEALTH_CHECK_INTERVAL = 30  # seconds

def setup_logging():
    """Setup logging configuration for production"""
    from logging.handlers import RotatingFileHandler
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    
    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def get_server_info():
    """Get server information for endpoint display"""
    import socket
    hostname = socket.gethostname()
    
    # Get local IP address
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"
    
    return {
        'hostname': hostname,
        'local_ip': local_ip,
        'port': PORT,
        'base_url': f'http://{local_ip}:{PORT}{API_PREFIX}',
        'health_url': f'http://{local_ip}:{PORT}{API_PREFIX}/health'
    }

# Environment-specific overrides
if os.getenv('FLASK_ENV') == 'development':
    DEBUG = True
    LOG_LEVEL = logging.DEBUG
