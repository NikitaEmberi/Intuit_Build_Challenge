"""
Producer class that reads items from a source and adds them to a shared queue.
"""

import threading
import time
from typing import Any, Optional, List, Dict
import logging
from shared_queue import SharedQueue

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class Producer(threading.Thread):
    def __init__(self, name: str, shared_queue: SharedQueue, source_container: List[Any], delay: float = 0.0):
        """
        Initializes a producer thread.

        Args:
            name (str): Name of the producer thread.
            shared_queue (SharedQueue): Shared queue to add items to.
            source_container (List[Any]): List of items to read from.
            delay (float): Delay in seconds between each item.
        
        Raises:
            ValueError: if delay is less than 0.
        """

        super().__init__(name = name)

        if source_container is None or len(source_container) == 0:
            raise ValueError("Source container cannot be None or empty")

        self.shared_queue = shared_queue
        self.source_container = source_container
        self.delay = delay

        # Thread control variable
        self._stop_event = threading.Event()

        # statistics
        self._items_produced = 0
        self._start_time = None
        self._end_time = None

        # set daemon to False to ensure proper shutdown
        self.daemon = False
        logger.info(f"Producer '{name}' initialized with {len(source_container)} items to produce")
    

    def run(self):
        """
        Main producer loop. Reads items from source and places into queue.
        """
        self.start_time = time.time()
        logger.info(f"Producer '{self.name}' starting production at time {self.start_time}...")

        try:
            for idx, item in enumerate(self.source_container):
                # Check if we've been asked to stop
                if self._stop_event.is_set():
                    logger.info(f"Producer '{self.name}' received stop signal, shutting down...")
                    break

                # attempting to put item into shared queue
                try:
                    logger.debug(f"Producer '{self.name}' attempting to produce item {idx}: {item}")

                    # Put item with timeout to allow checking stop event periodically
                    success = self.shared_queue.put(item, timeout = 1.0)

                    if success:
                        self._items_produced += 1
                        logger.info(
                            f"Producer '{self.name}' produced item {self._items_produced}/{len(self.source_container)}: {item}"
                        )
                    else:
                        # Timeout occurred, retry if not stopped
                        if not self._stop_event.is_set():
                            self.shared_queue.put(item, timeout = 1.0)
                            self._items_produced += 1
                            logger.info(
                                f"Producer '{self.name}' produced item {self._items_produced}/{len(self.source_container)}: {item}"
                            )
                    
                    if self.delay > 0:
                        time.sleep(self.delay)
                except Exception as e:
                    logger.error(f"Producer '{self.name}' error producing item: {e}")
                    continue
            
            # checking if we have produced all items
            if self._items_produced == len(self.source_container):
                logger.info(
                    f"Producer '{self.name}' completed successfully: "
                    f"{self._items_produced}/{len(self.source_container)} items produced"
                )
            else:
                logger.warning(
                    f"Producer '{self.name}' stopped early: "
                    f"{self._items_produced}/{len(self.source_container)} items produced"
                )
        except Exception as e:
            logger.error(f"Producer '{self.name}' encountered unexpected error: {e}", exc_info=True)
        finally:
            self._end_time = time.time()
            logger.info(f"Producer '{self.name}' shutting down at time {self._end_time}...")


    def stop(self, timeout: Optional[float] = None):
        """
        Stops the producer thread.
        """
        logger.info(f"Stopping producer '{self.name}'...")
        self._stop_event.set()

        # wiating for thread to finish before returning
        self.join(timeout = timeout)

        if self.is_alive():
            logger.warning(f"Producer '{self.name}' did not stop within timeout")
            return False
        else:
            logger.info(f"Producer '{self.name}' stopped successfully")
            return True
    
    def is_stopped(self) -> bool:
        """
        checks if the stop event has been set
        Returns:
            bool: True if stop has been requested, False otherwise
        """
        return self._stop_event.is_set()

        
    def get_stats(self) -> Dict[str, str]:
        """
        Gets statistics about the producer.
        Returns:
            dict: Dictionary containing:
                - 'name': Producer thread name
                - 'items_produced': Number of items successfully produced
                - 'total_items': Total items in source container
                - 'completion_rate': Percentage of items produced (0-100)
                - 'elapsed_time': Time taken for production (seconds)
                - 'items_per_second': Production rate
                - 'is_alive': Whether thread is still running
                - 'is_stopped': Whether stop was requested
        """

        elapsed_time = 0
        if self._start_time:
            end = self._end_time if self._end_time else time.time()
            elapsed_time = end - self._start_time

        total_items = len(self.source_container)
        completion_rate = (self._items_produced / total_items * 100) if total_items > 0 else 0
        items_per_second = (self._items_produced / elapsed_time) if elapsed_time > 0 else 0
        
        return {
            'name': self.name,
            'items_produced': self._items_produced,
            'total_items': total_items,
            'completion_rate': round(completion_rate, 2),
            'elapsed_time': round(elapsed_time, 2),
            'items_per_second': round(items_per_second, 2),
            'is_alive': self.is_alive(),
            'is_stopped': self.is_stopped()
        }










                








