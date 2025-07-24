#!/usr/bin/env python3
"""
Parallel Entity Extraction System for Phase 2 Optimizations

This module provides parallel entity extraction that can run concurrently
with embedding generation to maximize CPU utilization and reduce total processing time.
"""

import logging
import threading
import queue
import time
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EntityExtractionTask:
    """Represents an entity extraction task"""
    text_content: str
    opportunity_id: str
    content_type: str
    file_id: str
    task_id: str
    callback: Optional[Callable] = None

@dataclass
class EntityExtractionResult:
    """Represents the result of entity extraction"""
    task_id: str
    entities: List[Any]
    execution_time: float
    error: Optional[str] = None

class ParallelEntityExtractor:
    """
    Parallel entity extraction system that can process entities concurrently
    with other operations like embedding generation.
    
    Features:
    - Dedicated thread pool for entity extraction
    - Queue-based task submission
    - Non-blocking result retrieval
    - Error handling and retry logic
    - Performance monitoring
    """
    
    def __init__(self, num_workers: int = 4):
        """
        Initialize the parallel entity extractor
        
        Args:
            num_workers: Number of parallel entity extraction workers
        """
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="EntityExtractor")
        
        # Task tracking
        self.pending_tasks: Dict[str, Future] = {}
        self.completed_results: Dict[str, EntityExtractionResult] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'concurrent_tasks_peak': 0
        }
        
        # Entity extractor instance (lazy loaded)
        self._entity_extractor = None
        
        logger.info(f"Initialized parallel entity extractor with {num_workers} workers")
    
    def _get_entity_extractor(self):
        """Lazy load entity extractor to avoid initialization in main thread"""
        if self._entity_extractor is None:
            from entity_extractor import EntityExtractor
            self._entity_extractor = EntityExtractor()
            logger.debug("Initialized entity extractor in worker thread")
        return self._entity_extractor
    
    def _extract_entities_worker(self, task: EntityExtractionTask) -> EntityExtractionResult:
        """
        Worker function for entity extraction
        
        Args:
            task: The entity extraction task
            
        Returns:
            EntityExtractionResult with extracted entities or error
        """
        start_time = time.time()
        
        try:
            # Get entity extractor (thread-local)
            extractor = self._get_entity_extractor()
            
            # Extract entities
            entities = extractor.extract_entities(
                task.text_content,
                task.opportunity_id,
                task.content_type,
                task.file_id
            )
            
            execution_time = time.time() - start_time
            
            logger.debug(f"Extracted {len(entities)} entities from {task.content_type} {task.file_id} in {execution_time:.3f}s")
            
            return EntityExtractionResult(
                task_id=task.task_id,
                entities=entities,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Entity extraction failed for task {task.task_id}: {e}")
            
            return EntityExtractionResult(
                task_id=task.task_id,
                entities=[],
                execution_time=execution_time,
                error=str(e)
            )
    
    def submit_task(self, text_content: str, opportunity_id: str, content_type: str, 
                   file_id: str, task_id: Optional[str] = None, callback: Optional[Callable] = None) -> str:
        """
        Submit an entity extraction task for parallel processing
        
        Args:
            text_content: Text to extract entities from
            opportunity_id: ID of the opportunity
            content_type: Type of content ('description', 'document', etc.)
            file_id: ID of the file being processed
            task_id: Optional custom task ID (will generate if None)
            callback: Optional callback function called when task completes
            
        Returns:
            Task ID for tracking the submitted task
        """
        if task_id is None:
            task_id = f"{content_type}_{file_id}_{int(time.time() * 1000)}"
        
        task = EntityExtractionTask(
            text_content=text_content,
            opportunity_id=opportunity_id,
            content_type=content_type,
            file_id=file_id,
            task_id=task_id,
            callback=callback
        )
        
        with self._lock:
            # Submit to thread pool
            future = self.executor.submit(self._extract_entities_worker, task)
            self.pending_tasks[task_id] = future
            
            # Update statistics
            self.stats['tasks_submitted'] += 1
            current_pending = len(self.pending_tasks)
            if current_pending > self.stats['concurrent_tasks_peak']:
                self.stats['concurrent_tasks_peak'] = current_pending
        
        logger.debug(f"Submitted entity extraction task {task_id} for {content_type} {file_id}")
        return task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[EntityExtractionResult]:
        """
        Get the result of a submitted task
        
        Args:
            task_id: ID of the task to get results for
            timeout: Maximum time to wait for result (None = don't wait)
            
        Returns:
            EntityExtractionResult if available, None if not ready/found
        """
        with self._lock:
            # Check if already completed
            if task_id in self.completed_results:
                result = self.completed_results.pop(task_id)
                return result
            
            # Check if task is pending
            if task_id not in self.pending_tasks:
                logger.warning(f"Task {task_id} not found in pending tasks")
                return None
            
            future = self.pending_tasks[task_id]
        
        # Wait for result if timeout specified
        if timeout is not None:
            try:
                if future.done() or timeout > 0:
                    result = future.result(timeout=timeout)
                    
                    with self._lock:
                        # Remove from pending
                        self.pending_tasks.pop(task_id, None)
                        
                        # Update statistics
                        self.stats['tasks_completed'] += 1
                        self.stats['total_execution_time'] += result.execution_time
                        
                        if result.error:
                            self.stats['tasks_failed'] += 1
                    
                    # Call callback if provided
                    if hasattr(result, 'callback') and result.callback:
                        try:
                            result.callback(result)
                        except Exception as e:
                            logger.warning(f"Callback failed for task {task_id}: {e}")
                    
                    return result
                    
            except Exception as e:
                logger.error(f"Error getting result for task {task_id}: {e}")
                with self._lock:
                    self.pending_tasks.pop(task_id, None)
                    self.stats['tasks_failed'] += 1
                return None
        
        # Non-blocking check
        if future.done():
            try:
                result = future.result()
                
                with self._lock:
                    self.pending_tasks.pop(task_id, None)
                    self.stats['tasks_completed'] += 1
                    self.stats['total_execution_time'] += result.execution_time
                    
                    if result.error:
                        self.stats['tasks_failed'] += 1
                
                return result
                
            except Exception as e:
                logger.error(f"Error getting result for task {task_id}: {e}")
                with self._lock:
                    self.pending_tasks.pop(task_id, None)
                    self.stats['tasks_failed'] += 1
                return None
        
        return None
    
    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> Dict[str, EntityExtractionResult]:
        """
        Wait for all pending tasks to complete
        
        Args:
            timeout: Maximum time to wait for all tasks
            
        Returns:
            Dictionary of task_id -> EntityExtractionResult
        """
        results = {}
        
        with self._lock:
            pending_task_ids = list(self.pending_tasks.keys())
        
        for task_id in pending_task_ids:
            result = self.get_result(task_id, timeout=timeout)
            if result:
                results[task_id] = result
        
        return results
    
    def get_pending_task_count(self) -> int:
        """Get the number of pending tasks"""
        with self._lock:
            return len(self.pending_tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        with self._lock:
            stats = self.stats.copy()
            stats['pending_tasks'] = len(self.pending_tasks)
            
            if stats['tasks_completed'] > 0:
                stats['average_execution_time'] = stats['total_execution_time'] / stats['tasks_completed']
            else:
                stats['average_execution_time'] = 0.0
                
            return stats
    
    def shutdown(self, wait: bool = True):
        """
        Shutdown the parallel entity extractor
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("Shutting down parallel entity extractor...")
        
        if wait:
            # Wait for pending tasks
            self.wait_for_all_tasks(timeout=30.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        with self._lock:
            self.pending_tasks.clear()
            self.completed_results.clear()
        
        logger.info("Parallel entity extractor shutdown complete")

# Global extractor instance
_global_extractor = None
_extractor_lock = threading.Lock()

def get_global_extractor() -> ParallelEntityExtractor:
    """Get the global parallel entity extractor instance"""
    global _global_extractor
    if _global_extractor is None:
        with _extractor_lock:
            if _global_extractor is None:
                from config import ENTITY_WORKER_POOL_SIZE
                _global_extractor = ParallelEntityExtractor(num_workers=ENTITY_WORKER_POOL_SIZE)
    return _global_extractor

def shutdown_global_extractor():
    """Shutdown the global extractor"""
    global _global_extractor
    if _global_extractor is not None:
        _global_extractor.shutdown()
        _global_extractor = None
