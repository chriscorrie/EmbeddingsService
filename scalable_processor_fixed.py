# Here's the corrected process_scalable_batch_producer_consumer function
    def process_scalable_batch_producer_consumer(self, start_row_id: int, end_row_id: int, replace_existing_records: bool = False, task_id: str = None):
        """
        Process batch using producer/consumer architecture for optimal performance
        """
        # Initialize task-specific stats if task_id is provided
        if task_id:
            # Create completely isolated stats for this task
            self.task_specific_stats = {
                'opportunities_processed': 0,
                'documents_processed': 0,
                'documents_skipped': 0,
                'total_chunks_generated': 0,
                'entities_extracted': 0,
                'errors': 0
            }
            self.current_task_id = task_id
            logger.info(f"🔧 Initialized task-specific stats for task {task_id}")
        else:
            self.task_specific_stats = None
            self.current_task_id = None
        
        # Reset shared statistics for this processing run (for legacy compatibility)
        self._reset_stats()
        
        with time_operation('process_scalable_batch_producer_consumer', {'start_row': start_row_id, 'end_row': end_row_id}):
            start_time = time.time()
            logger.info(f"🚀 Starting producer/consumer processing for rows {start_row_id} to {end_row_id}")
            
            # Create bounded queue for opportunities
            opportunity_queue = queue.Queue(maxsize=100)
            
            # Consumer thread control
            consumer_count = getattr(self, 'opportunity_workers', 2)
            consumers_active = threading.Event()
            consumers_active.set()
            
            # Error handling
            processing_errors = []
            error_lock = threading.Lock()
            
            def add_error(error_msg: str):
                with error_lock:
                    processing_errors.append(error_msg)
                    logger.error(error_msg)
            
            # Producer thread function with file deduplication
            def producer_thread():
                """Producer thread with file deduplication support using stored procedures"""
                producer_start_time = time.time()
                try:
                    if not self.sql_conn:
                        add_error("SQL Server connection not available")
                        return
                    
                    # Import deduplication manager
                    from file_deduplication import FileDeduplicationManager
                    dedup_manager = FileDeduplicationManager(self.sql_conn)
                    
                    # Execute stored procedure to get embedding content with ExistingFile flag
                    logger.info(f"Executing GetEmbeddingContent stored procedure for rows {start_row_id}-{end_row_id}")
                    rows = dedup_manager.get_embedding_content(start_row_id, end_row_id)
                    
                    # Process results with deduplication logic
                    dedup_results = dedup_manager.process_deduplicated_documents(rows)
                    
                    opportunities = dedup_results['opportunities']
                    deduplicated_documents = dedup_results['deduplicated_documents']
                    date_updates = dedup_results['date_updates']
                    dedup_stats = dedup_results['stats']
                    
                    logger.info(f"Deduplication results: {dedup_stats}")
                    
                    # Process unique files that need text extraction
                    total_text_size_mb = 0.0
                    files_processed = 0
                    files_skipped = 0
                    
                    for file_id, dedup_doc in deduplicated_documents.items():
                        if not dedup_doc.existing_file:
                            # Load file content for new files
                            file_path = self.replace_document_path(dedup_doc.file_location)
                            
                            if os.path.exists(file_path):
                                try:
                                    with time_operation('producer_file_load', {'file_id': file_id, 'file_size_bytes': dedup_doc.file_size_bytes}):
                                        from process_documents import extract_text_from_file
                                        text_content = extract_text_from_file(file_path)
                                        if text_content:
                                            dedup_doc.text_content = text_content
                                            text_size_mb = len(text_content.encode('utf-8')) / (1024 * 1024)
                                            total_text_size_mb += text_size_mb
                                            files_processed += 1
                                            logger.debug(f"Loaded deduplicated file {file_id}: {text_size_mb:.2f}MB")
                                        else:
                                            dedup_doc.load_error = f"No text extracted from file: {file_path}"
                                            logger.warning(f"No text extracted from deduplicated file: {file_path}")
                                except Exception as e:
                                    dedup_doc.load_error = str(e)
                                    logger.error(f"Error loading deduplicated file {file_path}: {e}")
                                    with self.stats_lock:
                                        self.stats['file_load_errors'] += 1
                            else:
                                dedup_doc.load_error = f"File not found: {file_path}"
                                logger.warning(f"Deduplicated file not found: {file_path}")
                                with self.stats_lock:
                                    self.stats['file_load_errors'] += 1
                        else:
                            files_skipped += 1
                    
                    # Update opportunities with loaded content
                    for opportunity in opportunities.values():
                        for document in opportunity.documents:
                            if hasattr(document, 'existing_file') and not document.existing_file:
                                if document.file_id in deduplicated_documents:
                                    dedup_doc = deduplicated_documents[document.file_id]
                                    document.text_content = dedup_doc.text_content
                                    document.load_error = dedup_doc.load_error
                    
                    # Handle date range updates for existing files in Milvus
                    if date_updates:
                        self._handle_milvus_date_updates(date_updates)
                    
                    # Queue opportunities for processing
                    opportunities_produced = 0
                    for opportunity in opportunities.values():
                        opportunity_queue.put(opportunity)
                        opportunities_produced += 1
                        
                        if opportunities_produced % 5 == 0:
                            queue_size = opportunity_queue.qsize()
                            logger.info(f"Producer: {opportunities_produced} opportunities queued, queue size: {queue_size}")
                    
                    producer_end_time = time.time()
                    producer_elapsed = producer_end_time - producer_start_time
                    
                    logger.info(f"✅ Producer completed in {producer_elapsed:.2f}s")
                    logger.info(f"   📊 Deduplication savings: {files_skipped} files skipped, {files_processed} files processed")
                    logger.info(f"   📊 Total text loaded: {total_text_size_mb:.2f}MB from {files_processed} unique files")
                    logger.info(f"   📊 Opportunities produced: {opportunities_produced}")
                    
                    # Update stats with deduplication metrics
                    with self.stats_lock:
                        self.stats.update({
                            'files_processed': files_processed,
                            'files_skipped': files_skipped,
                            'deduplication_ratio': files_skipped / max(files_processed + files_skipped, 1),
                            'unique_files_processed': len(deduplicated_documents),
                            'date_updates_applied': len(date_updates)
                        })
                    
                except Exception as e:
                    add_error(f"Producer thread failed: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Signal consumer that production is complete
                    for _ in range(consumer_count):
                        opportunity_queue.put(None)

            # Consumer worker function
            def consumer_worker(worker_id: int):
                """Consumer worker that processes opportunities from the queue"""
                try:
                    while consumers_active.is_set():
                        try:
                            # Get opportunity from queue with timeout
                            opportunity = opportunity_queue.get(timeout=1.0)
                            
                            # Check for sentinel value (None means producer is done)
                            if opportunity is None:
                                logger.debug(f"Consumer {worker_id} received shutdown signal")
                                break
                            
                            # Process the opportunity (this handles the stats update)
                            self._process_opportunity_simplified(opportunity, replace_existing_records)
                            
                        except queue.Empty:
                            # Timeout occurred, check if we should continue
                            continue
                        except Exception as e:
                            add_error(f"Consumer {worker_id} error processing opportunity: {e}")
                        finally:
                            try:
                                opportunity_queue.task_done()
                            except ValueError:
                                # task_done() called more times than items in queue
                                pass
                
                except Exception as e:
                    add_error(f"Consumer {worker_id} fatal error: {e}")
                finally:
                    # Final flush for this consumer
                    try:
                        logger.debug(f"Consumer {worker_id} performing final flush...")
                        self._flush_all_vector_collections()
                        logger.info(f"Consumer {worker_id} finished processing")
                    except Exception as e:
                        add_error(f"Consumer {worker_id} final flush error: {e}")

            # Start producer thread
            producer = threading.Thread(target=producer_thread, name="ProducerThread")
            producer.start()

            # Start consumer workers
            consumers = []
            for i in range(consumer_count):
                consumer = threading.Thread(target=consumer_worker, args=(i,), name=f"ConsumerWorker-{i}")
                consumer.start()
                consumers.append(consumer)

            # Wait for producer to finish
            producer.join()
            logger.info("Producer thread completed")

            # Wait for all consumers to finish
            for consumer in consumers:
                consumer.join()
            logger.info("All consumer workers completed")

            # Stop memory monitoring and cleanup
            self.stop_memory_monitoring()

            # Entity extraction cleanup
            if self.enable_entity_extraction and self.entity_queue:
                logger.info("Shutting down entity extraction queue...")
                self.entity_queue.shutdown(timeout=30.0)

            # Final processing time
            processing_time = time.time() - start_time
            self.stats['processing_time'] = processing_time

            # Log any errors that occurred
            if processing_errors:
                logger.error(f"Processing completed with {len(processing_errors)} errors:")
                for error in processing_errors:
                    logger.error(f"  - {error}")

            # Print performance summary
            logger.info("🔍 PRODUCER/CONSUMER ANALYSIS COMPLETE - See detailed timing below:")
            print_summary()
            # Ensure logs directory exists for performance report
            os.makedirs("logs", exist_ok=True)
            # Use task_id for unique performance reports, fallback to row range if no task_id
            logger.info(f"🔍 DEBUG: Performance report task_id='{task_id}', start_row_id={start_row_id}, end_row_id={end_row_id}")
            if task_id:
                report_filename = f"logs/performance_report_{task_id}.json"
                logger.info(f"🔍 Task-specific performance report: {report_filename}")
            else:
                report_filename = f"logs/performance_report_{start_row_id}_{end_row_id}.json"
                logger.info(f"🔍 Legacy performance report: {report_filename}")
            save_report(report_filename)

            self._log_final_stats(processing_time)

            # Return processing statistics (use task-specific stats when available)
            if hasattr(self, 'task_specific_stats') and self.task_specific_stats is not None:
                stats_to_return = self.task_specific_stats.copy()
                stats_to_return.update({
                    'processing_time': processing_time,
                    'errors': self.task_specific_stats['errors'] + len(processing_errors)
                })
                logger.info(f"🔍 Returning task-specific stats: {self.task_specific_stats['opportunities_processed']} opportunities")
            else:
                stats_to_return = {
                    'opportunities_processed': self.stats['opportunities_processed'],
                    'documents_processed': self.stats['documents_processed'],
                    'documents_skipped': self.stats['documents_skipped'],
                    'total_chunks_generated': self.stats['total_chunks_generated'],
                    'processing_time': processing_time,
                    'errors': self.stats['errors'] + len(processing_errors)
                }
                logger.info(f"🔍 Returning shared stats: {self.stats['opportunities_processed']} opportunities")

            return stats_to_return
