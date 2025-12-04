"""
Pinecone upload manager for background vector uploads
"""
import queue
import threading
from time import sleep
from typing import List, Optional, Callable


class PineconeUploadManager:
    """Manages background uploads to Pinecone using a queue and worker threads."""
    
    def __init__(self, index, num_workers: int = 4, batch_size: int = 100, log_callback: Optional[Callable[[str], None]] = None):
        self.index = index
        self.batch_size = batch_size
        self.upload_queue = queue.Queue()
        self.workers = []
        self.is_running = True
        self.uploaded_count = 0
        self.errors = []
        self.lock = threading.Lock()
        self.log_callback = log_callback or print
        self.index_name = getattr(index, 'name', 'unknown')
        
        for i in range(num_workers):
            worker = threading.Thread(target=self._upload_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        
        self.log_callback(f"  → Started {num_workers} upload workers for {self.index_name} index")

    def _upload_worker(self, worker_id: int):
        while self.is_running or not self.upload_queue.empty():
            try:
                vectors_batch = self.upload_queue.get(timeout=1)
                try:
                    if vectors_batch:
                        batch_size = len(vectors_batch)
                        max_retries = 3
                        retry_count = 0
                        success = False
                        
                        while retry_count < max_retries and not success:
                            try:
                                self.index.upsert(vectors=vectors_batch, namespace="__default__")
                                with self.lock:
                                    self.uploaded_count += len(vectors_batch)
                                    total_uploaded = self.uploaded_count
                                self.log_callback(f"  ✓ [{self.index_name}] Worker {worker_id}: Uploaded batch with {batch_size} embeddings (Total uploaded: {total_uploaded})")
                                success = True
                            except Exception as e:
                                error_str = str(e).lower()
                                if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'axios', 'http', 'request']) and retry_count < max_retries - 1:
                                    retry_count += 1
                                    self.log_callback(f"  ⚠ [{self.index_name}] Worker {worker_id}: Retry {retry_count}/{max_retries} after error: {str(e)[:100]}")
                                    sleep(2.0 * retry_count)
                                    continue
                                with self.lock:
                                    self.errors.append(f"Worker {worker_id}: {e}")
                                self.log_callback(f"  ✗ [{self.index_name}] Worker {worker_id}: Upload failed after {retry_count + 1} attempts: {str(e)[:100]}")
                                break
                finally:
                    self.upload_queue.task_done()
            except queue.Empty:
                continue

    def add_to_queue(self, vectors: List):
        """Add vectors to upload queue. Expects a batch of vectors (up to 100)."""
        if vectors:
            batch_size = len(vectors)
            queue_size = self.upload_queue.qsize()
            self.upload_queue.put(vectors)
            # Log every batch to show progress
            self.log_callback(f"  📤 [{self.index_name}] Queued batch with {batch_size} embeddings (Queue: {queue_size + 1} batches waiting)")

    def wait_for_completion(self):
        """Wait for all queued uploads to complete."""
        queue_size = self.upload_queue.qsize()
        if queue_size > 0:
            self.log_callback(f"  ⏳ [{self.index_name}] Waiting for {queue_size} batches to complete...")
        self.upload_queue.join()
        self.log_callback(f"  ✓ [{self.index_name}] All uploads completed. Total uploaded: {self.uploaded_count}")

    def stop(self):
        """Stop the upload workers."""
        self.is_running = False
        for worker in self.workers:
            worker.join(timeout=2)
    
    def get_stats(self):
        """Get upload statistics."""
        with self.lock:
            return {"uploaded": self.uploaded_count, "errors": len(self.errors)}

