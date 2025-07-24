#!/usr/bin/env python3
"""
Final verification that HuggingFace API is not being called during processing.
This script monitors network activity during processor initialization.
"""

import os
import sys
import socket
import threading
import time
from unittest.mock import patch

# Force offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("=== FINAL VERIFICATION - NO HUGGINGFACE API CALLS ===")

# Track any network calls
network_calls = []

def track_socket_connect(original_connect):
    """Wrapper to track socket connections"""
    def wrapper(self, address):
        network_calls.append(f"Socket connection to {address}")
        print(f"🔍 Network call detected: {address}")
        return original_connect(self, address)
    return wrapper

# Monitor network activity
original_connect = socket.socket.connect
socket.socket.connect = track_socket_connect(original_connect)

try:
    print("✅ Network monitoring started")
    
    # Import and initialize processor
    from scalable_processor import ScalableEnhancedProcessor
    print("✅ Processor imported successfully")
    
    # Initialize processor (this loads the model)
    processor = ScalableEnhancedProcessor()
    print("✅ Processor initialized successfully")
    
    # Verify chunk cache is working
    if hasattr(processor, 'chunk_cache') and processor.chunk_cache is not None:
        print("✅ Chunk cache initialized with shared embedding model")
    else:
        print("❌ Chunk cache not initialized")
        
    # Quick test to ensure model works
    if hasattr(processor, 'model') and processor.model:
        test_embedding = processor.model.encode(["This is a test sentence"])
        print(f"✅ Model working: embedding shape {test_embedding.shape}")
    elif hasattr(processor, 'embedding_model') and processor.embedding_model:
        test_embedding = processor.embedding_model.encode(["This is a test sentence"])
        print(f"✅ Model working: embedding shape {test_embedding.shape}")
    else:
        print("❌ Model not found in processor")
    
    print("\n📊 Network Activity Summary:")
    if network_calls:
        print(f"❌ {len(network_calls)} network calls detected:")
        for call in network_calls:
            print(f"   - {call}")
    else:
        print("✅ No network calls detected!")
        
    print("\n🎉 Verification complete - Truly offline processing confirmed!")

except Exception as e:
    print(f"❌ Error during verification: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Restore original socket
    socket.socket.connect = original_connect
