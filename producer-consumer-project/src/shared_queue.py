"""
Thread-safe queue with blocking operations.
"""

import threading
from typing import Any, Optional, Dict
from collections import deque
import time

class SharedQueue:
    """
    Thread-safe queue that blocks producers when full and consumers when empty
    """
    def __init__(self, capacity: int = 5):
        """
        Initializes a shared queue with a given capacity.

        Args:
            capacity (int): Maximim number of items the queue can hold.
                            Must always be greater than 0.
        Raises:
            ValueError: if capacity is less than or equal to 0.
        """

        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")
        
        self.capacity = capacity
        self._queue = deque()

        # single lock for synchronization
        self._lock = threading.Lock()

        # condition variables for wait/notify operations
        self._not_full = threading.Condition(self._lock)
        self._not_empty = threading.Condition(self._lock)

        # For Logging
        self._total_put = 0
        self._total_get = 0

    def put(self, item: Any, timeout: Optional[float] = None) -> bool:

        """
        Adds an item to the queue. Blocks if queue is full.

        Args:
            item (Any): The item to add to the queue.
            timeout (Optional[float]): Maximum time to wait for space to be available. None means wait indefinitely.
        Returns:
            bool: True if item was added, False if timed out.
        """

        end_time = time.time() + timeout if timeout is not None else None

        with self._not_full:
            # wait while queue is full
            while self.is_full():
                if end_time is not None:
                    remaining_time = end_time - time.time()
                    if remaining_time <= 0:
                        return False # timeout occured
                    self._not_full.wait(timeout = remaining_time)
                else:
                    self._not_full.wait() # wait indefinitely
            
 
            self._queue.append(item)
            self._total_put += 1

            # Notify one waiting consumer that queue is not empty
            self._not_empty.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """
        Removes and returns an item from the queue. Blocks if queue is empty.

        Args:
            timeout (Optional[float]): Maximum time to wait for an item to be available. None means wait indefinitely.
        Returns:
            Optional[Any]: The item removed from the queue, or None if timed out.
        """

        end_time = time.time() + timeout if timeout is not None else None

        with self._not_empty:
            # wait while queue is empty
            while self.is_empty():
                if end_time is not None:
                    remaining_time = end_time - time.time()

                    if remaining_time <= 0:
                        return None # timeout occured
                    self._not_empty.wait(timeout = remaining_time)
                else:
                    self._not_empty.wait() # wait indefinitely
                
            
            item = self._queue.popleft()
            self._total_get += 1

            # Notify one waiting producer that queue is not full
            self._not_full.notify()
            return item

    
    def size(self) -> int:
        """
        Gets the current number of items in the queue.
        
        Returns:
            int: Current queue size
        """
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        """
        Checks if the queue is empty.

        Returns:
            bool: True if queue is empty, False otherwise.
        This method assumes the lock is already held by the caller.
        """

        return len(self._queue) == 0
    

    def is_full(self) -> bool:
        """
        Checks if the queue is full.

        Returns:
            bool: True if queue is full, False otherwise.
        This method assumes the lock is already held by the caller.
        """
        return len(self._queue) >= self.capacity
    
    def get_stats(self) -> Dict[str, str]:
        """
        Gets statistics about the queue.

        Returns:
            Dict[str, str]: A dictionary containing:
            - 'current_size': Current number of items in queue
            - 'capacity': Maximum queue capacity
            - 'total_put': Total items added since creation
            - 'total_get': Total items removed since creation
            - 'items_in_transit': Difference between put and get 
        """

        with self._lock:
            return {
                "current_size": str(len(self._queue)),
                "capacity": str(self.capacity),
                "total_put": str(self._total_put),
                "total_get": str(self._total_get),
                "items_in_transit": str(self._total_put - self._total_get)
            }
    
               

                   
                   

        