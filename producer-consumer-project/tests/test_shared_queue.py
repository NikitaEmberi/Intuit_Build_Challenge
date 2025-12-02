"""
Unit tests for the SharedQueue class.
Tests thread synchronization, blocking operations, and wait/notify mechanism.
"""
import unittest
import threading
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared_queue import SharedQueue


class TestSharedQueueInitialization(unittest.TestCase):
    """Tests for SharedQueue initialization."""

    def test_init_with_default_capacity(self):
        """Queue should initialize with default capacity of 5."""
        queue = SharedQueue()
        self.assertEqual(queue.capacity, 5)

    def test_init_with_custom_capacity(self):
        """Queue should initialize with custom capacity."""
        queue = SharedQueue(capacity=10)
        self.assertEqual(queue.capacity, 10)

    def test_init_with_zero_capacity_raises_error(self):
        """Queue should raise ValueError for zero capacity."""
        with self.assertRaises(ValueError) as context:
            SharedQueue(capacity=0)
        self.assertIn("greater than 0", str(context.exception))

    def test_init_with_negative_capacity_raises_error(self):
        """Queue should raise ValueError for negative capacity."""
        with self.assertRaises(ValueError) as context:
            SharedQueue(capacity=-5)
        self.assertIn("greater than 0", str(context.exception))

    def test_queue_starts_empty(self):
        """Queue should be empty after initialization."""
        queue = SharedQueue(capacity=5)
        self.assertTrue(queue.is_empty())
        self.assertEqual(queue.size(), 0)


class TestSharedQueuePut(unittest.TestCase):
    """Tests for put() operation."""

    def test_put_single_item(self):
        """Should add a single item to the queue."""
        queue = SharedQueue(capacity=5)
        result = queue.put("item1")
        
        self.assertTrue(result)
        self.assertEqual(queue.size(), 1)

    def test_put_multiple_items(self):
        """Should add multiple items to the queue."""
        queue = SharedQueue(capacity=5)
        
        for i in range(5):
            result = queue.put(f"item{i}")
            self.assertTrue(result)
        
        self.assertEqual(queue.size(), 5)
        self.assertTrue(queue.is_full())

    def test_put_various_types(self):
        """Should accept various data types."""
        queue = SharedQueue(capacity=10)
        
        queue.put("string")
        queue.put(123)
        queue.put(3.14)
        queue.put([1, 2, 3])
        queue.put({"key": "value"})
        queue.put(None)
        
        self.assertEqual(queue.size(), 6)

    def test_put_timeout_on_full_queue(self):
        """Should return False when queue is full and timeout expires."""
        queue = SharedQueue(capacity=2)
        queue.put("item1")
        queue.put("item2")  # Queue is now full
        
        start_time = time.time()
        result = queue.put("item3", timeout=0.5)
        elapsed = time.time() - start_time
        
        self.assertFalse(result)
        self.assertGreaterEqual(elapsed, 0.4)  # Should wait ~0.5s
        self.assertEqual(queue.size(), 2)  # Item not added

    def test_put_blocks_when_full(self):
        """Put should block when queue is full until space available."""
        queue = SharedQueue(capacity=2)
        queue.put("item1")
        queue.put("item2")
        
        result_holder = {"success": False}
        
        def delayed_get():
            time.sleep(0.3)
            queue.get()
        
        def blocked_put():
            result_holder["success"] = queue.put("item3", timeout=2.0)
        
        getter = threading.Thread(target=delayed_get)
        putter = threading.Thread(target=blocked_put)
        
        putter.start()
        getter.start()
        
        putter.join(timeout=3)
        getter.join(timeout=3)
        
        self.assertTrue(result_holder["success"])
        self.assertEqual(queue.size(), 2)


class TestSharedQueueGet(unittest.TestCase):
    """Tests for get() operation."""

    def test_get_single_item(self):
        """Should retrieve a single item from the queue."""
        queue = SharedQueue(capacity=5)
        queue.put("item1")
        
        item = queue.get()
        
        self.assertEqual(item, "item1")
        self.assertEqual(queue.size(), 0)

    def test_get_maintains_fifo_order(self):
        """Should maintain FIFO order."""
        queue = SharedQueue(capacity=5)
        items = ["first", "second", "third"]
        
        for item in items:
            queue.put(item)
        
        for expected in items:
            actual = queue.get()
            self.assertEqual(actual, expected)

    def test_get_timeout_on_empty_queue(self):
        """Should return None when queue is empty and timeout expires."""
        queue = SharedQueue(capacity=5)
        
        start_time = time.time()
        result = queue.get(timeout=0.5)
        elapsed = time.time() - start_time
        
        self.assertIsNone(result)
        self.assertGreaterEqual(elapsed, 0.4)

    def test_get_blocks_when_empty(self):
        """Get should block when queue is empty until item available."""
        queue = SharedQueue(capacity=5)
        result_holder = {"item": None}
        
        def delayed_put():
            time.sleep(0.3)
            queue.put("delayed_item")
        
        def blocked_get():
            result_holder["item"] = queue.get(timeout=2.0)
        
        putter = threading.Thread(target=delayed_put)
        getter = threading.Thread(target=blocked_get)
        
        getter.start()
        putter.start()
        
        getter.join(timeout=3)
        putter.join(timeout=3)
        
        self.assertEqual(result_holder["item"], "delayed_item")


class TestSharedQueueUtilityMethods(unittest.TestCase):
    """Tests for utility methods: size(), is_empty(), is_full()."""

    def test_size_reflects_queue_state(self):
        """Size should accurately reflect queue contents."""
        queue = SharedQueue(capacity=5)
        
        self.assertEqual(queue.size(), 0)
        
        queue.put("item1")
        self.assertEqual(queue.size(), 1)
        
        queue.put("item2")
        self.assertEqual(queue.size(), 2)
        
        queue.get()
        self.assertEqual(queue.size(), 1)

    def test_is_empty_when_empty(self):
        """is_empty should return True for empty queue."""
        queue = SharedQueue(capacity=5)
        self.assertTrue(queue.is_empty())

    def test_is_empty_when_not_empty(self):
        """is_empty should return False when queue has items."""
        queue = SharedQueue(capacity=5)
        queue.put("item")
        self.assertFalse(queue.is_empty())

    def test_is_full_when_full(self):
        """is_full should return True when queue reaches capacity."""
        queue = SharedQueue(capacity=2)
        queue.put("item1")
        queue.put("item2")
        self.assertTrue(queue.is_full())

    def test_is_full_when_not_full(self):
        """is_full should return False when queue has space."""
        queue = SharedQueue(capacity=5)
        queue.put("item1")
        self.assertFalse(queue.is_full())


class TestSharedQueueStatistics(unittest.TestCase):
    """Tests for get_stats() method."""

    def test_stats_initial_state(self):
        """Stats should show initial state correctly."""
        queue = SharedQueue(capacity=10)
        stats = queue.get_stats()
        
        self.assertEqual(stats["current_size"], "0")
        self.assertEqual(stats["capacity"], "10")
        self.assertEqual(stats["total_put"], "0")
        self.assertEqual(stats["total_get"], "0")
        self.assertEqual(stats["items_in_transit"], "0")

    def test_stats_after_operations(self):
        """Stats should track put and get operations."""
        queue = SharedQueue(capacity=10)
        
        # Put 5 items
        for i in range(5):
            queue.put(i)
        
        # Get 2 items
        queue.get()
        queue.get()
        
        stats = queue.get_stats()
        
        self.assertEqual(stats["current_size"], "3")
        self.assertEqual(stats["total_put"], "5")
        self.assertEqual(stats["total_get"], "2")
        self.assertEqual(stats["items_in_transit"], "3")


class TestSharedQueueThreadSafety(unittest.TestCase):
    """Tests for thread safety and concurrent access."""

    def test_concurrent_puts(self):
        """Multiple threads should safely put items concurrently."""
        queue = SharedQueue(capacity=100)
        num_threads = 10
        items_per_thread = 10
        
        def put_items(thread_id):
            for i in range(items_per_thread):
                queue.put(f"thread{thread_id}_item{i}")
        
        threads = [
            threading.Thread(target=put_items, args=(i,))
            for i in range(num_threads)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        
        self.assertEqual(queue.size(), num_threads * items_per_thread)

    def test_concurrent_gets(self):
        """Multiple threads should safely get items concurrently."""
        queue = SharedQueue(capacity=100)
        num_items = 100
        
        # Pre-populate queue
        for i in range(num_items):
            queue.put(i)
        
        results = []
        results_lock = threading.Lock()
        
        def get_items(count):
            for _ in range(count):
                item = queue.get(timeout=1.0)
                if item is not None:
                    with results_lock:
                        results.append(item)
        
        threads = [
            threading.Thread(target=get_items, args=(20,))
            for _ in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        
        # All items should be consumed exactly once
        self.assertEqual(len(results), num_items)
        self.assertEqual(set(results), set(range(num_items)))

    def test_concurrent_put_and_get(self):
        """Concurrent producers and consumers should work correctly."""
        queue = SharedQueue(capacity=10)
        produced = []
        consumed = []
        produced_lock = threading.Lock()
        consumed_lock = threading.Lock()
        
        def producer(items):
            for item in items:
                queue.put(item)
                with produced_lock:
                    produced.append(item)
        
        def consumer(count):
            for _ in range(count):
                item = queue.get(timeout=2.0)
                if item is not None:
                    with consumed_lock:
                        consumed.append(item)
        
        items_to_produce = list(range(50))
        
        prod_thread = threading.Thread(target=producer, args=(items_to_produce,))
        cons_thread = threading.Thread(target=consumer, args=(50,))
        
        prod_thread.start()
        cons_thread.start()
        
        prod_thread.join(timeout=10)
        cons_thread.join(timeout=10)
        
        self.assertEqual(set(produced), set(consumed))
        self.assertEqual(len(consumed), 50)


class TestSharedQueueWaitNotify(unittest.TestCase):
    """Tests for wait/notify mechanism."""

    def test_notify_wakes_waiting_consumer(self):
        """Put should notify and wake a waiting consumer."""
        queue = SharedQueue(capacity=5)
        wake_time = {"time": None}
        
        def waiting_consumer():
            queue.get(timeout=5.0)  # Will block
            wake_time["time"] = time.time()
        
        consumer = threading.Thread(target=waiting_consumer)
        consumer.start()
        
        time.sleep(0.3)  # Let consumer start waiting
        put_time = time.time()
        queue.put("wake up!")
        
        consumer.join(timeout=3)
        
        self.assertIsNotNone(wake_time["time"])
        # Consumer should wake up shortly after put
        self.assertLess(wake_time["time"] - put_time, 0.5)

    def test_notify_wakes_waiting_producer(self):
        """Get should notify and wake a waiting producer."""
        queue = SharedQueue(capacity=1)
        queue.put("fill")  # Queue is now full
        
        wake_time = {"time": None}
        
        def waiting_producer():
            queue.put("blocked", timeout=5.0)
            wake_time["time"] = time.time()
        
        producer = threading.Thread(target=waiting_producer)
        producer.start()
        
        time.sleep(0.3)  # Let producer start waiting
        get_time = time.time()
        queue.get()  # Make space
        
        producer.join(timeout=3)
        
        self.assertIsNotNone(wake_time["time"])
        # Producer should wake up shortly after get
        self.assertLess(wake_time["time"] - get_time, 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
