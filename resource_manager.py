#!/usr/bin/env python3
"""
Intelligent Resource Manager for Dynamic Scaling
"""

import os
import psutil
import logging
from typing import Tuple, Dict, Any
from config import (
    MAX_OPPORTUNITY_WORKERS,
    MAX_MEMORY_USAGE_MB,
    CPU_CORE_MULTIPLIER,
    ENABLE_MEMORY_MONITORING
)

logger = logging.getLogger(__name__)

class ResourceManager:
    """
    Intelligent resource management for optimal parallel processing
    """
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.optimal_workers = self._calculate_optimal_workers()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            return {
                'cpu_cores': psutil.cpu_count(logical=False),  # Physical cores
                'cpu_threads': psutil.cpu_count(logical=True),  # Logical cores
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else 'Unknown',
                'cpu_usage_percent': psutil.cpu_percent(interval=1)
            }
        except Exception as e:
            logger.warning(f"Could not get complete system info: {e}")
            return {
                'cpu_cores': 4,  # Conservative default
                'cpu_threads': 4,
                'total_memory_gb': 8,
                'available_memory_gb': 4,
                'memory_percent': 50,
                'cpu_freq_max': 'Unknown',
                'cpu_usage_percent': 20
            }
    
    def _calculate_optimal_workers(self) -> Dict[str, int]:
        """Calculate optimal worker counts based on system resources for producer/consumer architecture"""
        # Base calculations on physical CPU cores
        physical_cores = self.system_info['cpu_cores']
        available_memory_gb = self.system_info['available_memory_gb']
        
        # Calculate opportunity workers (consumer threads in producer/consumer architecture)
        base_opportunity_workers = max(1, int(physical_cores * CPU_CORE_MULTIPLIER))
        
        # Limit by configured maximum
        opportunity_workers = min(base_opportunity_workers, MAX_OPPORTUNITY_WORKERS)
        
        # Memory-based limits
        memory_limit_workers = max(1, int(available_memory_gb / 2))  # 2GB per worker
        
        # Apply memory constraints
        if opportunity_workers > memory_limit_workers:
            logger.warning(f"Reducing opportunity workers from {opportunity_workers} to {memory_limit_workers} due to memory constraints")
            opportunity_workers = memory_limit_workers
        
        # For producer/consumer architecture, total workers = consumer threads
        total_max_workers = opportunity_workers
        
        return {
            'opportunity_workers': opportunity_workers,
            'total_max_workers': total_max_workers
        }
    
    def get_resource_config(self) -> Dict[str, Any]:
        """Get complete resource configuration for the processor"""
        return {
            'system_info': self.system_info,
            'optimal_workers': self.optimal_workers,
            'memory_monitoring': ENABLE_MEMORY_MONITORING,
            'max_memory_mb': MAX_MEMORY_USAGE_MB
        }
    
    def log_resource_info(self):
        """Log comprehensive resource information"""
        logger.info("🖥️  System Resource Analysis:")
        logger.info(f"   CPU Cores (Physical): {self.system_info['cpu_cores']}")
        logger.info(f"   CPU Threads (Logical): {self.system_info['cpu_threads']}")
        logger.info(f"   Total Memory: {self.system_info['total_memory_gb']:.1f} GB")
        logger.info(f"   Available Memory: {self.system_info['available_memory_gb']:.1f} GB")
        logger.info(f"   Memory Usage: {self.system_info['memory_percent']:.1f}%")
        logger.info(f"   CPU Usage: {self.system_info['cpu_usage_percent']:.1f}%")
        
        logger.info("⚡ Optimal Configuration:")
        logger.info(f"   Consumer Workers: {self.optimal_workers['opportunity_workers']}")
        logger.info(f"   Total Max Workers: {self.optimal_workers['total_max_workers']}")
        logger.info("   Architecture: Producer/Consumer")
    
    def check_resource_health(self) -> Tuple[bool, str]:
        """Check if system has sufficient resources for processing"""
        current_memory = psutil.virtual_memory()
        current_cpu = psutil.cpu_percent(interval=1)
        
        # Memory check
        if current_memory.percent > 90:
            return False, f"Memory usage too high: {current_memory.percent:.1f}%"
        
        if current_memory.available / (1024**3) < 1:  # Less than 1GB available
            return False, f"Insufficient available memory: {current_memory.available / (1024**3):.1f} GB"
        
        # CPU check
        if current_cpu > 95:
            return False, f"CPU usage too high: {current_cpu:.1f}%"
        
        return True, "System resources healthy"
    
    def get_dynamic_batch_sizes(self) -> Dict[str, int]:
        """Calculate optimal batch sizes based on available memory"""
        available_memory_gb = self.system_info['available_memory_gb']
        
        # Scale batch sizes based on available memory
        if available_memory_gb >= 8:
            embedding_batch_size = 64
            entity_batch_size = 100
            vector_batch_size = 200
        elif available_memory_gb >= 4:
            embedding_batch_size = 32
            entity_batch_size = 50
            vector_batch_size = 100
        else:
            embedding_batch_size = 16
            entity_batch_size = 25
            vector_batch_size = 50
        
        return {
            'embedding_batch_size': embedding_batch_size,
            'entity_batch_size': entity_batch_size,
            'vector_batch_size': vector_batch_size
        }

def get_optimal_configuration() -> Dict[str, Any]:
    """
    Get optimal configuration for the current system
    """
    manager = ResourceManager()
    manager.log_resource_info()
    
    # Check system health
    healthy, message = manager.check_resource_health()
    if not healthy:
        logger.warning(f"⚠️  System resource warning: {message}")
    
    config = manager.get_resource_config()
    config['batch_sizes'] = manager.get_dynamic_batch_sizes()
    config['system_healthy'] = healthy
    config['health_message'] = message
    
    return config

if __name__ == "__main__":
    # Test the resource manager
    logging.basicConfig(level=logging.INFO)
    config = get_optimal_configuration()
    
    print("\n🔧 Recommended Configuration:")
    print(f"Consumer Workers: {config['optimal_workers']['opportunity_workers']}")
    print(f"Total Max Workers: {config['optimal_workers']['total_max_workers']}")
    print(f"Embedding Batch Size: {config['batch_sizes']['embedding_batch_size']}")
    print(f"System Health: {'✅ Good' if config['system_healthy'] else '⚠️ ' + config['health_message']}")
