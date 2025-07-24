#!/usr/bin/env python3
"""
Environment-based configuration for Document Embedding REST API
This allows changing parameters without editing code files
"""

import os
from pathlib import Path

# Load environment variables from .env file if it exists
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)

# Paths (with environment variable overrides)
SAMBA_SHARE_PATH = os.getenv('SAMBA_SHARE_PATH', '/mnt/HomerShare/FBO Attachments')
DOCUMENTS_PATH = os.getenv('DOCUMENTS_PATH', '/mnt/HomerShare/FBO Attachments')
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', '/mnt/NVME_1/vector_db')

# Document Path Replacement Configuration
DOCUMENT_PATH_TO_REPLACE = os.getenv('DOCUMENT_PATH_TO_REPLACE', 'D:\\')
DOCUMENT_PATH_REPLACEMENT_VALUE = os.getenv('DOCUMENT_PATH_REPLACEMENT_VALUE', '\\\\NetworkShare\\DDrive\\')

# SQL Server connection
SQL_SERVER_CONNECTION_STRING = os.getenv('SQL_SERVER_CONNECTION_STRING', 'your_connection_string_here')
SQL_CONNECTION_STRING = os.getenv('SQL_CONNECTION_STRING', SQL_SERVER_CONNECTION_STRING)

# Embedding model
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

# Server Configuration
HOST = os.getenv('API_HOST', '0.0.0.0')
PORT = int(os.getenv('API_PORT', '5000'))
DEBUG = os.getenv('API_DEBUG', 'false').lower() == 'true'

# Processing Configuration
MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '1000'))
DEFAULT_BATCH_SIZE = int(os.getenv('DEFAULT_BATCH_SIZE', '100'))
MAX_SEARCH_LIMIT = int(os.getenv('MAX_SEARCH_LIMIT', '100'))
DEFAULT_SEARCH_LIMIT = int(os.getenv('DEFAULT_SEARCH_LIMIT', '10'))

# Security Configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'false').lower() == 'true'

def print_config():
    """Print current configuration for debugging"""
    print("Current Configuration:")
    print(f"  Document Path To Replace: {DOCUMENT_PATH_TO_REPLACE}")
    print(f"  Document Path Replacement: {DOCUMENT_PATH_REPLACEMENT_VALUE}")
    print(f"  API Host: {HOST}")
    print(f"  API Port: {PORT}")
    print(f"  Max Batch Size: {MAX_BATCH_SIZE}")
    print(f"  Default Batch Size: {DEFAULT_BATCH_SIZE}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  SQL Connection: {SQL_CONNECTION_STRING}")

if __name__ == "__main__":
    print_config()
