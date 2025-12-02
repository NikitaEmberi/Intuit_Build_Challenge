"""
Consumer class that reads items from a shared queue and stores them in a destination container.
"""

import threading
import time
import logging
from typing import List, Any, Optional, Dict

# Configure logging for the consumer
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Consumer(threading.Thread):
    """
    Consumer thread that reads items from a shared queue
    and stores them in a destination container.
    """
    def __init__(self, name: str, shared_queue, destination_container: List[Any],delay: float = 0.0, max_items: Optional[int] = None):
        """
        Initializes a consumer thread.

        Args:
            name (str): Name of the consumer thread.
            shared_queue (SharedQueue): Shared queue to read items from.
            destination_container (List[Any]): List to store consumed items.
            delay (float): Delay in seconds between each item.
            max_items (Optional[int]): Maximum number of items to consume. None means consume all items.
        """
        super().__init__(name = name)

        if destination_container is None:
            raise ValueError("Destination container cannot be None")
        
        self.shared_queue = shared_queue
        self.destination_container = destination_container
        self.delay = delay
        self.max_items = max_items
        
        # Thread control
        self._stop_event = threading.Event()
        
        # Statistics
        self._items_consumed = 0
        self._start_time = None
        self._end_time = None
        
        # Set daemon to False to ensure proper shutdown
        self.daemon = False
        
        max_str = f"{max_items}" if max_items else "unlimited"
        logger.info(f"Consumer '{name}' initialized (max_items={max_str})")
    
    def run(self):
        """
        Main consumer loop. Reads items from queue and stores in destination.
        """
        self._start_time = time.time()
        logger.info(f"Consumer '{self.name}' starting consumption...")
        
        try:
            while not self._stop_event.is_set():
                # Check if we've reached the maximum items
                if self.max_items is not None and self._items_consumed >= self.max_items:
                    logger.info(
                        f"Consumer '{self.name}' reached max_items limit: "
                        f"{self._items_consumed}/{self.max_items}"
                    )
                    break
                
                # Attempt to get item from queue (will block if empty)
                try:
                    logger.debug(f"Consumer '{self.name}' attempting to consume item...")
                    
                    # Get item with timeout to allow checking stop event periodically
                    item = self.shared_queue.get(timeout=1.0)
                    
                    if item is not None:
                        # Store item in destination container
                        self.destination_container.append(item)
                        self._items_consumed += 1
                        
                        logger.info(
                            f"Consumer '{self.name}' consumed item {self._items_consumed}: {item}"
                        )
                        
                        # Simulate processing work
                        if self.delay > 0:
                            time.sleep(self.delay)
                    else:
                        # Timeout occurred, check stop event and continue
                        logger.debug(f"Consumer '{self.name}' get() timeout, checking stop event...")
                        continue
                        
                except Exception as e:
                    # Handle timeout or other exceptions
                    if self._stop_event.is_set():
                        logger.info(f"Consumer '{self.name}' received stop signal during get()")
                        break
                    
                    # Log error but continue consuming
                    logger.debug(f"Consumer '{self.name}' get() exception: {e}")
                    continue
            
            # Log completion status
            if self._stop_event.is_set():
                logger.info(
                    f"Consumer '{self.name}' stopped by signal: "
                    f"{self._items_consumed} items consumed"
                )
            elif self.max_items and self._items_consumed >= self.max_items:
                logger.info(
                    f"Consumer '{self.name}' completed successfully: "
                    f"{self._items_consumed}/{self.max_items} items consumed"
                )
            else:
                logger.info(
                    f"Consumer '{self.name}' completed: "
                    f"{self._items_consumed} items consumed"
                )
                
        except Exception as e:
            logger.error(f"Consumer '{self.name}' encountered unexpected error: {e}", exc_info=True)
        
        finally:
            self._end_time = time.time()
            logger.info(f"Consumer '{self.name}' shutting down")
    
    def stop(self, timeout: Optional[float] = None) -> bool:
        """
        Stops the consumer thread.

        Args:
            timeout (float): Maximum time to wait for thread to stop.
        
        Returns:
            bool: True if stopped successfully, False if timeout occurred.
        """
        logger.info(f"Stopping consumer '{self.name}'...")
        self._stop_event.set()
        
        # Wait for thread to finish
        self.join(timeout=timeout)
        
        if self.is_alive():
            logger.warning(f"Consumer '{self.name}' did not stop within timeout")
            return False
        else:
            logger.info(f"Consumer '{self.name}' stopped successfully")
            return True
    
    def is_stopped(self) -> bool:
        """
        Check if the stop event has been set.
        
        Returns:
            bool: True if stop has been requested, False otherwise
        """
        return self._stop_event.is_set()

    def get_stats(self) -> Dict[str, Any]:
        """
        Gets statistics about the consumer's performance.
        
        Returns:
            dict: Dictionary containing:
                - 'name': Consumer thread name
                - 'items_consumed': Number of items successfully consumed
                - 'max_items': Maximum items to consume (or None)
                - 'completion_rate': Percentage of max_items consumed (0-100, or N/A)
                - 'elapsed_time': Time taken for consumption (seconds)
                - 'items_per_second': Consumption rate
                - 'is_alive': Whether thread is still running
                - 'is_stopped': Whether stop was requested
                - 'destination_size': Current size of destination container
        """
        elapsed_time = 0
        if self._start_time:
            end = self._end_time if self._end_time else time.time()
            elapsed_time = end - self._start_time
        
        # Calculate completion rate if max_items is set
        if self.max_items is not None:
            completion_rate = (self._items_consumed / self.max_items * 100) if self.max_items > 0 else 0
        else:
            completion_rate = None  # N/A for unlimited consumption
        
        items_per_second = (self._items_consumed / elapsed_time) if elapsed_time > 0 else 0
        
        return {
            'name': self.name,
            'items_consumed': self._items_consumed,
            'max_items': self.max_items,
            'completion_rate': round(completion_rate, 2) if completion_rate is not None else 'N/A',
            'elapsed_time': round(elapsed_time, 2),
            'items_per_second': round(items_per_second, 2),
            'is_alive': self.is_alive(),
            'is_stopped': self.is_stopped(),
            'destination_size': len(self.destination_container)
        }
    