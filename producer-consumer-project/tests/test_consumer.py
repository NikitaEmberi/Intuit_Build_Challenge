"""
Unit tests for the Consumer class.
"""
import unittest
import threading
import time
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared_queue import SharedQueue
from consumer import Consumer


class TestConsumerInitialization(unittest.TestCase):
    """Tests for Consumer initialization."""

    def test_init_with_valid_params(self):
        """Consumer should initialize with valid parameters."""
        queue = SharedQueue(capacity=5)
        destination = []
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination
        )
        
        self.assertEqual(consumer.name, "TestConsumer")
        self.assertEqual(consumer.destination_container, destination)
        self.assertEqual(consumer.shared_queue, queue)
        self.assertEqual(consumer.delay, 0.0)
        self.assertIsNone(consumer.max_items)

    def test_init_with_none_destination_raises_error(self):
        """Consumer should raise ValueError if destination is None."""
        queue = SharedQueue(capacity=5)
        
        with self.assertRaises(ValueError) as context:
            Consumer(
                name="TestConsumer",
                shared_queue=queue,
                destination_container=None
            )
        
        self.assertIn("cannot be None", str(context.exception))

    def test_init_with_custom_delay(self):
        """Consumer should accept custom delay."""
        queue = SharedQueue(capacity=5)
        destination = []
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination,
            delay=0.5
        )
        
        self.assertEqual(consumer.delay, 0.5)

    def test_init_with_max_items(self):
        """Consumer should accept max_items limit."""
        queue = SharedQueue(capacity=5)
        destination = []
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination,
            max_items=10
        )
        
        self.assertEqual(consumer.max_items, 10)


class TestConsumerConsumption(unittest.TestCase):
    """Tests for Consumer item consumption."""

    def test_consumes_items_from_queue(self):
        """Consumer should consume items and store in destination."""
        queue = SharedQueue(capacity=10)
        destination = []
        items = [1, 2, 3, 4, 5]
        
        # Pre-populate the queue
        for item in items:
            queue.put(item)
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination,
            max_items=5  # Stop after consuming all items
        )
        
        consumer.start()
        consumer.join(timeout=5)
        
        self.assertEqual(destination, items)
        self.assertEqual(consumer._items_consumed, 5)

    def test_consumes_with_max_items_limit(self):
        """Consumer should stop after reaching max_items."""
        queue = SharedQueue(capacity=10)
        destination = []
        
        # Put more items than max_items
        for i in range(10):
            queue.put(i)
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination,
            max_items=5  # Only consume 5
        )
        
        consumer.start()
        consumer.join(timeout=5)
        
        self.assertEqual(len(destination), 5)
        self.assertEqual(consumer._items_consumed, 5)
        # Queue should still have remaining items
        self.assertEqual(queue.size(), 5)

    def test_blocks_on_empty_queue(self):
        """Consumer should block when queue is empty."""
        queue = SharedQueue(capacity=5)
        destination = []
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination
        )
        
        consumer.start()
        
        # Consumer should be alive, waiting for items
        time.sleep(0.5)
        self.assertTrue(consumer.is_alive())
        
        # Add items
        queue.put("item1")
        queue.put("item2")
        
        time.sleep(0.5)
        
        # Stop consumer
        consumer.stop(timeout=3)
        
        # Should have consumed the items
        self.assertIn("item1", destination)
        self.assertIn("item2", destination)

    def test_consumption_with_delay(self):
        """Consumer should respect delay between items."""
        queue = SharedQueue(capacity=10)
        destination = []
        delay = 0.1
        
        for i in range(3):
            queue.put(i)
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination,
            delay=delay,
            max_items=3
        )
        
        start_time = time.time()
        consumer.start()
        consumer.join(timeout=5)
        elapsed = time.time() - start_time
        
        # Should take at least (count - 1) * delay seconds
        expected_min_time = 2 * delay
        self.assertGreaterEqual(elapsed, expected_min_time * 0.9)


class TestConsumerStopMechanism(unittest.TestCase):
    """Tests for Consumer stop functionality."""

    def test_is_stopped_initially_false(self):
        """Consumer should not be stopped initially."""
        queue = SharedQueue(capacity=5)
        destination = []
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination
        )
        
        self.assertFalse(consumer.is_stopped())

    def test_stop_terminates_consumer(self):
        """Consumer should stop when stop() is called."""
        queue = SharedQueue(capacity=5)
        destination = []
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination
        )
        
        consumer.start()
        time.sleep(0.2)  # Let it start
        
        result = consumer.stop(timeout=3)
        
        self.assertTrue(result)
        self.assertTrue(consumer.is_stopped())
        self.assertFalse(consumer.is_alive())

    def test_stop_during_consumption(self):
        """Consumer should stop gracefully during consumption."""
        queue = SharedQueue(capacity=10)
        destination = []
        
        # Add many items with delay so we can stop mid-consumption
        for i in range(20):
            queue.put(i)
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination,
            delay=0.2  # Slow consumption
        )
        
        consumer.start()
        time.sleep(0.5)  # Let it consume a few
        
        consumer.stop(timeout=3)
        
        # Should have consumed some but not all
        self.assertGreater(len(destination), 0)
        self.assertLess(len(destination), 20)


class TestConsumerStatistics(unittest.TestCase):
    """Tests for Consumer statistics."""

    def test_get_stats_before_start(self):
        """Stats should be available before consumer starts."""
        queue = SharedQueue(capacity=5)
        destination = []
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination,
            max_items=10
        )
        
        stats = consumer.get_stats()
        
        self.assertEqual(stats['name'], "TestConsumer")
        self.assertEqual(stats['items_consumed'], 0)
        self.assertEqual(stats['max_items'], 10)
        self.assertFalse(stats['is_alive'])
        self.assertFalse(stats['is_stopped'])
        self.assertEqual(stats['destination_size'], 0)

    def test_get_stats_after_consumption(self):
        """Stats should reflect consumed items."""
        queue = SharedQueue(capacity=10)
        destination = []
        
        for i in range(5):
            queue.put(i)
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination,
            max_items=5
        )
        
        consumer.start()
        consumer.join(timeout=5)
        
        stats = consumer.get_stats()
        
        self.assertEqual(stats['items_consumed'], 5)
        self.assertEqual(stats['destination_size'], 5)
        self.assertEqual(stats['completion_rate'], 100.0)
        self.assertFalse(stats['is_alive'])
        self.assertGreater(stats['elapsed_time'], 0)

    def test_get_stats_with_unlimited_consumption(self):
        """Stats should show N/A for completion_rate when max_items is None."""
        queue = SharedQueue(capacity=5)
        destination = []
        
        consumer = Consumer(
            name="TestConsumer",
            shared_queue=queue,
            destination_container=destination
            # max_items defaults to None
        )
        
        stats = consumer.get_stats()
        
        self.assertIsNone(stats['max_items'])
        self.assertEqual(stats['completion_rate'], 'N/A')


class TestConsumerThreadSafety(unittest.TestCase):
    """Tests for thread safety."""

    def test_multiple_consumers_single_queue(self):
        """Multiple consumers should safely share a queue."""
        queue = SharedQueue(capacity=20)
        destination1 = []
        destination2 = []
        
        # Pre-populate queue
        for i in range(10):
            queue.put(i)
        
        consumer1 = Consumer(
            name="Consumer1",
            shared_queue=queue,
            destination_container=destination1,
            max_items=5
        )
        consumer2 = Consumer(
            name="Consumer2",
            shared_queue=queue,
            destination_container=destination2,
            max_items=5
        )
        
        consumer1.start()
        consumer2.start()
        
        consumer1.join(timeout=5)
        consumer2.join(timeout=5)
        
        # Both should complete
        self.assertEqual(consumer1._items_consumed, 5)
        self.assertEqual(consumer2._items_consumed, 5)
        
        # Combined should have all items (no duplicates)
        all_consumed = destination1 + destination2
        self.assertEqual(len(all_consumed), 10)
        self.assertEqual(set(all_consumed), set(range(10)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
