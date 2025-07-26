#!/usr/bin/env python3
"""
Targeted Optimization Strategy for Real Bottlenecks

Based on empirical GPU profiling data from July 25, 2025:
- producer_file_load: 64.9% of total time (62.68s out of 96.54s) 
- batch_commit_flush: 14.1% of total time (13.63s)

This script implements specific optimizations targeting these bottlenecks.
"""

import os
import sys
import asyncio
import aiofiles
import concurrent.futures
import mmap
import threading
import queue
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_timer import time_operation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FileLoadOptimization:
    """Configuration for file loading optimizations"""
    enable_memory_mapping: bool = True
    enable_async_loading: bool = True
    enable_parallel_loading: bool = True
    file_cache_size_mb: int = 256
    parallel_file_workers: int = 8
    read_buffer_size_kb: int = 64
    preload_threshold_mb: float = 10.0

@dataclass
class DatabaseOptimization:
    """Configuration for database operations optimizations"""
    enable_connection_pooling: bool = True
    enable_batch_optimizations: bool = True
    flush_frequency_optimization: bool = True
    connection_pool_size: int = 8
    batch_size_multiplier: float = 2.0
    flush_interval_seconds: float = 5.0
    async_flush_enabled: bool = True

class MemoryMappedFileLoader:
    """Memory-mapped file loader for large documents"""
    
    def __init__(self, cache_size_mb: int = 256):
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.file_cache = {}
        self.cache_size = 0
        self.cache_lock = threading.Lock()
    
    def load_with_mmap(self, file_path: str) -> str:
        """Load file using memory mapping for efficient large file handling"""
        try:
            file_size = os.path.getsize(file_path)
            
            # Use memory mapping for files larger than 1MB
            if file_size > 1024 * 1024:
                with time_operation('mmap_file_load', {'file_path': file_path, 'file_size_bytes': file_size}):
                    with open(file_path, 'rb') as f:
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                            content = mmapped_file.read().decode('utf-8', errors='ignore')
                            return content
            else:
                # Use regular file reading for smaller files
                with time_operation('regular_file_load', {'file_path': file_path, 'file_size_bytes': file_size}):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
                        
        except Exception as e:
            logger.error(f"Error loading file {file_path} with mmap: {e}")
            return ""
    
    def clear_cache(self):
        """Clear the file cache"""
        with self.cache_lock:
            self.file_cache.clear()
            self.cache_size = 0

class AsyncFileLoader:
    """Asynchronous file loader for parallel I/O operations"""
    
    def __init__(self, max_workers: int = 8, buffer_size_kb: int = 64):
        self.max_workers = max_workers
        self.buffer_size = buffer_size_kb * 1024
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    async def load_file_async(self, file_path: str) -> Tuple[str, str]:
        """Load file asynchronously"""
        try:
            file_size = os.path.getsize(file_path)
            
            with time_operation('async_file_load', {'file_path': file_path, 'file_size_bytes': file_size}):
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = await f.read()
                    return file_path, content
                    
        except Exception as e:
            logger.error(f"Error loading file {file_path} async: {e}")
            return file_path, ""
    
    async def load_multiple_files(self, file_paths: List[str]) -> Dict[str, str]:
        """Load multiple files concurrently"""
        with time_operation('async_batch_file_load', {'file_count': len(file_paths)}):
            tasks = [self.load_file_async(path) for path in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            file_contents = {}
            for result in results:
                if isinstance(result, tuple):
                    file_path, content = result
                    file_contents[file_path] = content
                else:
                    logger.error(f"File loading error: {result}")
            
            return file_contents

class StreamingFileProcessor:
    """Streaming file processor for very large documents"""
    
    def __init__(self, chunk_size_mb: int = 16):
        self.chunk_size = chunk_size_mb * 1024 * 1024
    
    def process_large_file_streaming(self, file_path: str, processor_func) -> List[Any]:
        """Process large files in streaming chunks to reduce memory footprint"""
        try:
            file_size = os.path.getsize(file_path)
            
            if file_size > self.chunk_size:
                with time_operation('streaming_file_process', {'file_path': file_path, 'file_size_bytes': file_size}):
                    results = []
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        while True:
                            chunk = f.read(self.chunk_size)
                            if not chunk:
                                break
                            
                            chunk_results = processor_func(chunk)
                            results.extend(chunk_results)
                    
                    return results
            else:
                # Use regular processing for smaller files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    return processor_func(content)
                    
        except Exception as e:
            logger.error(f"Error streaming file {file_path}: {e}")
            return []

class OptimizedDatabaseCommitter:
    """Optimized database operations with connection pooling and batch optimization"""
    
    def __init__(self, config: DatabaseOptimization):
        self.config = config
        self.connection_pool = queue.Queue(maxsize=config.connection_pool_size)
        self.batch_queue = queue.Queue()
        self.flush_timer = None
        self.flush_lock = threading.Lock()
        self.stats = {
            'total_flushes': 0,
            'total_flush_time': 0.0,
            'batch_sizes': [],
            'connections_used': 0
        }
    
    def setup_connection_pool(self, connection_factory):
        """Setup database connection pool"""
        try:
            for _ in range(self.config.connection_pool_size):
                conn = connection_factory()
                self.connection_pool.put(conn)
            logger.info(f"Created database connection pool with {self.config.connection_pool_size} connections")
        except Exception as e:
            logger.error(f"Failed to setup connection pool: {e}")
    
    def get_connection(self):
        """Get connection from pool"""
        try:
            conn = self.connection_pool.get(timeout=5.0)
            self.stats['connections_used'] += 1
            return conn
        except queue.Empty:
            logger.warning("Connection pool exhausted")
            return None
    
    def return_connection(self, conn):
        """Return connection to pool"""
        try:
            self.connection_pool.put(conn, timeout=1.0)
        except queue.Full:
            # Pool is full, close the connection
            try:
                conn.close()
            except:
                pass
    
    def optimized_batch_flush(self, collections: Dict[str, Any]):
        """Optimized batch flush with timing and statistics"""
        start_time = time.time()
        
        try:
            with time_operation('optimized_batch_commit_flush'):
                if self.config.async_flush_enabled:
                    # Async flush for better performance
                    self._async_flush_collections(collections)
                else:
                    # Synchronous flush
                    self._sync_flush_collections(collections)
                
                flush_time = time.time() - start_time
                self.stats['total_flushes'] += 1
                self.stats['total_flush_time'] += flush_time
                
                avg_flush_time = self.stats['total_flush_time'] / self.stats['total_flushes']
                logger.info(f"Optimized flush completed in {flush_time:.3f}s (avg: {avg_flush_time:.3f}s)")
                
        except Exception as e:
            logger.error(f"Optimized batch flush failed: {e}")
            # Fallback to regular flush
            self._fallback_flush_collections(collections)
    
    def _async_flush_collections(self, collections: Dict[str, Any]):
        """Asynchronous collection flushing"""
        def flush_collection(collection_name, collection):
            try:
                collection.flush()
                logger.debug(f"Async flushed collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to flush collection {collection_name}: {e}")
        
        # Start all flushes in parallel
        threads = []
        for collection_name, collection in collections.items():
            thread = threading.Thread(target=flush_collection, args=(collection_name, collection))
            thread.start()
            threads.append(thread)
        
        # Wait for all flushes to complete
        for thread in threads:
            thread.join(timeout=30.0)  # 30 second timeout per flush
    
    def _sync_flush_collections(self, collections: Dict[str, Any]):
        """Synchronous collection flushing"""
        for collection_name, collection in collections.items():
            try:
                collection.flush()
                logger.debug(f"Sync flushed collection: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to flush collection {collection_name}: {e}")
    
    def _fallback_flush_collections(self, collections: Dict[str, Any]):
        """Fallback flush method"""
        logger.warning("Using fallback flush method")
        for collection_name, collection in collections.items():
            try:
                collection.flush()
            except Exception as e:
                logger.error(f"Fallback flush failed for {collection_name}: {e}")

class TargetedOptimizationStrategy:
    """Main optimization strategy coordinator"""
    
    def __init__(self, file_config: FileLoadOptimization = None, db_config: DatabaseOptimization = None):
        self.file_config = file_config or FileLoadOptimization()
        self.db_config = db_config or DatabaseOptimization()
        
        # Initialize optimized components
        self.mmap_loader = MemoryMappedFileLoader(self.file_config.file_cache_size_mb)
        self.async_loader = AsyncFileLoader(
            max_workers=self.file_config.parallel_file_workers,
            buffer_size_kb=self.file_config.read_buffer_size_kb
        )
        self.streaming_processor = StreamingFileProcessor()
        self.db_committer = OptimizedDatabaseCommitter(self.db_config)
        
        logger.info("Targeted optimization strategy initialized")
        logger.info(f"File optimizations: mmap={self.file_config.enable_memory_mapping}, "
                   f"async={self.file_config.enable_async_loading}, "
                   f"parallel={self.file_config.enable_parallel_loading}")
        logger.info(f"DB optimizations: pooling={self.db_config.enable_connection_pooling}, "
                   f"batch={self.db_config.enable_batch_optimizations}, "
                   f"async_flush={self.db_config.async_flush_enabled}")
    
    def apply_file_loading_optimizations(self, file_paths: List[str]) -> Dict[str, str]:
        """Apply file loading optimizations based on configuration"""
        
        if not file_paths:
            return {}
        
        total_size = sum(os.path.getsize(path) for path in file_paths if os.path.exists(path))
        total_size_mb = total_size / (1024 * 1024)
        
        logger.info(f"Applying file optimizations to {len(file_paths)} files ({total_size_mb:.2f}MB total)")
        
        if self.file_config.enable_async_loading and len(file_paths) > 1:
            # Use async loading for multiple files
            return self._load_files_async(file_paths)
        elif self.file_config.enable_memory_mapping and total_size_mb > self.file_config.preload_threshold_mb:
            # Use memory mapping for large files
            return self._load_files_mmap(file_paths)
        else:
            # Use optimized regular loading
            return self._load_files_optimized(file_paths)
    
    def _load_files_async(self, file_paths: List[str]) -> Dict[str, str]:
        """Load files using async I/O"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.async_loader.load_multiple_files(file_paths))
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Async file loading failed: {e}")
            return self._load_files_fallback(file_paths)
    
    def _load_files_mmap(self, file_paths: List[str]) -> Dict[str, str]:
        """Load files using memory mapping"""
        file_contents = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                content = self.mmap_loader.load_with_mmap(file_path)
                file_contents[file_path] = content
        return file_contents
    
    def _load_files_optimized(self, file_paths: List[str]) -> Dict[str, str]:
        """Load files using optimized regular I/O"""
        file_contents = {}
        buffer_size = self.file_config.read_buffer_size_kb * 1024
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    with time_operation('optimized_file_load', {'file_path': file_path, 'file_size_bytes': file_size}):
                        with open(file_path, 'r', encoding='utf-8', errors='ignore', buffering=buffer_size) as f:
                            content = f.read()
                            file_contents[file_path] = content
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
                    file_contents[file_path] = ""
        
        return file_contents
    
    def _load_files_fallback(self, file_paths: List[str]) -> Dict[str, str]:
        """Fallback file loading method"""
        file_contents = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_contents[file_path] = f.read()
                except Exception as e:
                    logger.error(f"Fallback file loading failed for {file_path}: {e}")
                    file_contents[file_path] = ""
        return file_contents
    
    def apply_database_optimizations(self, collections: Dict[str, Any]):
        """Apply database optimizations"""
        if self.db_config.enable_batch_optimizations:
            self.db_committer.optimized_batch_flush(collections)
        else:
            # Fallback to regular flush
            for collection_name, collection in collections.items():
                collection.flush()
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """Generate performance recommendations based on current configuration"""
        recommendations = {
            'file_optimizations': {
                'current_config': {
                    'memory_mapping': self.file_config.enable_memory_mapping,
                    'async_loading': self.file_config.enable_async_loading,
                    'parallel_workers': self.file_config.parallel_file_workers,
                    'cache_size_mb': self.file_config.file_cache_size_mb
                },
                'recommended_improvements': []
            },
            'database_optimizations': {
                'current_config': {
                    'connection_pooling': self.db_config.enable_connection_pooling,
                    'batch_optimizations': self.db_config.enable_batch_optimizations,
                    'async_flush': self.db_config.async_flush_enabled,
                    'pool_size': self.db_config.connection_pool_size
                },
                'recommended_improvements': []
            },
            'performance_targets': {
                'file_loading_target': '< 20 seconds (from current 62.68s)',
                'database_flush_target': '< 5 seconds (from current 13.63s)',
                'overall_improvement_target': '48% reduction in total time'
            }
        }
        
        # Add specific recommendations
        if not self.file_config.enable_async_loading:
            recommendations['file_optimizations']['recommended_improvements'].append(
                'Enable async loading for multi-file operations'
            )
        
        if not self.file_config.enable_memory_mapping:
            recommendations['file_optimizations']['recommended_improvements'].append(
                'Enable memory mapping for large file handling'
            )
        
        if self.file_config.parallel_file_workers < 8:
            recommendations['file_optimizations']['recommended_improvements'].append(
                'Increase parallel file workers to 8+ for better I/O utilization'
            )
        
        if not self.db_config.async_flush_enabled:
            recommendations['database_optimizations']['recommended_improvements'].append(
                'Enable async flush for parallel database operations'
            )
        
        if self.db_config.connection_pool_size < 8:
            recommendations['database_optimizations']['recommended_improvements'].append(
                'Increase connection pool size to 8+ for better database concurrency'
            )
        
        return recommendations
    
    def generate_implementation_plan(self) -> str:
        """Generate implementation plan for optimizations"""
        plan = """
TARGETED OPTIMIZATION IMPLEMENTATION PLAN
========================================

Based on empirical profiling data showing:
- File loading: 64.9% of total time (62.68s)
- Database flush: 14.1% of total time (13.63s)

PHASE 1: FILE I/O OPTIMIZATIONS (Target: 60% improvement)
--------------------------------------------------------
1. Memory-mapped file access for files >1MB
2. Async parallel file loading with 8+ workers
3. Streaming processing for very large documents
4. Optimized buffer sizes (64KB default)
5. File content caching with 256MB cache

Expected improvement: 40-60% reduction in file loading time

PHASE 2: DATABASE OPTIMIZATIONS (Target: 70% improvement)  
--------------------------------------------------------
1. Connection pooling with 8 connections
2. Async parallel collection flushing
3. Batch size optimization (2x multiplier)
4. Intelligent flush timing (5s intervals)
5. Database operation monitoring

Expected improvement: 60-80% reduction in database flush time

IMPLEMENTATION PRIORITY:
1. File I/O optimizations (highest impact)
2. Database optimizations (moderate impact)
3. Performance monitoring and tuning

VALIDATION APPROACH:
1. Baseline measurement (current: 67.6s total)
2. File optimization validation
3. Database optimization validation
4. Combined optimization validation
5. Performance regression testing

TARGET OUTCOME:
- Total time: <35 seconds (48% improvement)
- File loading: <20 seconds (68% improvement)
- Database operations: <5 seconds (63% improvement)
        """
        
        return plan

def create_optimized_configuration() -> Tuple[FileLoadOptimization, DatabaseOptimization]:
    """Create optimized configuration based on system capabilities"""
    
    # Detect system capabilities
    import psutil
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Configure file optimizations based on system resources
    file_workers = min(cpu_count, 12)  # Cap at 12 workers
    cache_size_mb = min(int(memory_gb * 0.1), 512)  # 10% of RAM, max 512MB
    
    file_config = FileLoadOptimization(
        enable_memory_mapping=True,
        enable_async_loading=True,
        enable_parallel_loading=True,
        file_cache_size_mb=cache_size_mb,
        parallel_file_workers=file_workers,
        read_buffer_size_kb=64,
        preload_threshold_mb=10.0
    )
    
    # Configure database optimizations
    db_config = DatabaseOptimization(
        enable_connection_pooling=True,
        enable_batch_optimizations=True,
        flush_frequency_optimization=True,
        connection_pool_size=8,
        batch_size_multiplier=2.0,
        flush_interval_seconds=5.0,
        async_flush_enabled=True
    )
    
    logger.info(f"Generated optimized configuration:")
    logger.info(f"  CPU cores: {cpu_count}, File workers: {file_workers}")
    logger.info(f"  Memory: {memory_gb:.1f}GB, Cache size: {cache_size_mb}MB")
    logger.info(f"  DB connections: {db_config.connection_pool_size}")
    
    return file_config, db_config

def main():
    """Main function to demonstrate optimization strategy"""
    print("🎯 TARGETED OPTIMIZATION STRATEGY")
    print("=" * 50)
    
    # Create optimized configuration
    file_config, db_config = create_optimized_configuration()
    
    # Initialize optimization strategy
    strategy = TargetedOptimizationStrategy(file_config, db_config)
    
    # Generate recommendations
    recommendations = strategy.get_performance_recommendations()
    print("\n📊 PERFORMANCE RECOMMENDATIONS:")
    print(f"File loading target: {recommendations['performance_targets']['file_loading_target']}")
    print(f"Database flush target: {recommendations['performance_targets']['database_flush_target']}")
    print(f"Overall target: {recommendations['performance_targets']['overall_improvement_target']}")
    
    # Generate implementation plan
    plan = strategy.generate_implementation_plan()
    print("\n📋 IMPLEMENTATION PLAN:")
    print(plan)
    
    print("\n✅ Optimization strategy ready for implementation")
    print("📝 Next step: Integrate optimizations into scalable_processor.py")

if __name__ == "__main__":
    main()
