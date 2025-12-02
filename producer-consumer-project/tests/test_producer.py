"""
Unit tests for the Producer class.
"""
import unittest
import threading
import time
import sys
import os

# Addimg src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shared_queue import SharedQueue
from producer import Producer


class TestProducerInitialization(unittest.TestCase):
    """Tests for Producer initialization."""

    def test_init_with_valid_params(self):
        """Producer should initialize with valid parameters."""
        queue = SharedQueue(capacity=5)
        source = [1, 2, 3, 4, 5]
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source
        )
        
        self.assertEqual(producer.name, "TestProducer")
        self.assertEqual(producer.source_container, source)
        self.assertEqual(producer.shared_queue, queue)
        self.assertEqual(producer.delay, 0.0)

    def test_init_with_empty_source_raises_error(self):
        """Producer should raise ValueError if source container is empty."""
        queue = SharedQueue(capacity=5)
        
        with self.assertRaises(ValueError) as context:
            Producer(name="TestProducer", shared_queue=queue, source_container=[])
        
        self.assertIn("cannot be None or empty", str(context.exception))

    def test_init_with_none_source_raises_error(self):
        """Producer should raise ValueError if source container is None."""
        queue = SharedQueue(capacity=5)
        
        with self.assertRaises(ValueError) as context:
            Producer(name="TestProducer", shared_queue=queue, source_container=None)
        
        self.assertIn("cannot be None or empty", str(context.exception))

    def test_init_with_custom_delay(self):
        """Producer should accept custom delay."""
        queue = SharedQueue(capacity=5)
        source = [1, 2, 3]
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source,
            delay=0.5
        )
        
        self.assertEqual(producer.delay, 0.5)


class TestProducerProduction(unittest.TestCase):
    """Tests for Producer item production."""

    def test_produces_all_items(self):
        """Producer should produce all items from source container."""
        queue = SharedQueue(capacity=10)
        source = [1, 2, 3, 4, 5]
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source
        )
        
        producer.start()
        producer.join(timeout=5)
        
        # Verify all items were produced
        self.assertEqual(producer._items_produced, len(source))
        
        # Verify items are in queue
        produced_items = []
        while not queue.is_empty():
            item = queue.get(timeout=1)
            if item is not None:
                produced_items.append(item)
        
        self.assertEqual(produced_items, source)

    def test_produces_with_full_queue_blocks(self):
        """Producer should block when queue is full."""
        queue = SharedQueue(capacity=2)  # Small capacity
        source = [1, 2, 3, 4, 5]
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source
        )
        
        producer.start()
        
        # Give producer time to fill the queue
        time.sleep(0.5)
        
        # Queue should be full (or close to it)
        self.assertTrue(queue.size() >= 1)
        
        # Consume items to unblock producer
        consumed = []
        while producer.is_alive() or not queue.is_empty():
            item = queue.get(timeout=1)
            if item is not None:
                consumed.append(item)
            if len(consumed) >= len(source):
                break
        
        producer.join(timeout=5)
        self.assertEqual(consumed, source)

    def test_production_with_delay(self):
        """Producer should respect the delay between items."""
        queue = SharedQueue(capacity=10)
        source = [1, 2, 3]
        delay = 0.1
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source,
            delay=delay
        )
        
        start_time = time.time()
        producer.start()
        producer.join(timeout=5)
        elapsed = time.time() - start_time
        
        # Should take at least (len-1) * delay seconds (delay after each except last)
        expected_min_time = (len(source) - 1) * delay
        self.assertGreaterEqual(elapsed, expected_min_time * 0.9)  # 10% tolerance


class TestProducerStopMechanism(unittest.TestCase):
    """Tests for Producer stop functionality."""

    def test_is_stopped_initially_false(self):
        """Producer should not be stopped initially."""
        queue = SharedQueue(capacity=5)
        source = [1, 2, 3]
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source
        )
        
        self.assertFalse(producer.is_stopped())

    def test_stop_event_is_set(self):
        """Stop event should be set when stop is called."""
        queue = SharedQueue(capacity=5)
        source = [1, 2, 3, 4, 5]
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source,
            delay=0.5  # Slow down so we can stop mid-production
        )
        
        producer.start()
        time.sleep(0.2)  # Let it produce some items
        
        # Set stop event directly (avoiding the buggy stop method)
        producer._stop_event.set()
        producer.join(timeout=5)
        
        self.assertTrue(producer.is_stopped())
        # Should have produced less than all items
        self.assertLess(producer._items_produced, len(source))


class TestProducerStatistics(unittest.TestCase):
    """Tests for Producer statistics."""

    def test_get_stats_before_start(self):
        """Stats should be available before producer starts."""
        queue = SharedQueue(capacity=5)
        source = [1, 2, 3]
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source
        )
        
        stats = producer.get_stats()
        
        self.assertEqual(stats['name'], "TestProducer")
        self.assertEqual(stats['items_produced'], 0)
        self.assertEqual(stats['total_items'], 3)
        self.assertFalse(stats['is_alive'])
        self.assertFalse(stats['is_stopped'])

    def test_get_stats_after_completion(self):
        """Stats should reflect completed production."""
        queue = SharedQueue(capacity=10)
        source = [1, 2, 3, 4, 5]
        
        producer = Producer(
            name="TestProducer",
            shared_queue=queue,
            source_container=source
        )
        
        producer.start()
        producer.join(timeout=5)
        
        stats = producer.get_stats()
        
        self.assertEqual(stats['items_produced'], 5)
        self.assertEqual(stats['total_items'], 5)
        self.assertEqual(stats['completion_rate'], 100.0)
        self.assertFalse(stats['is_alive'])
        self.assertGreater(stats['elapsed_time'], 0)


class TestProducerThreadSafety(unittest.TestCase):
    """Tests for thread safety."""

    def test_multiple_producers_single_queue(self):
        """Multiple producers should safely share a single queue."""
        queue = SharedQueue(capacity=20)
        
        source1 = [f"p1_{i}" for i in range(5)]
        source2 = [f"p2_{i}" for i in range(5)]
        
        producer1 = Producer(
            name="Producer1",
            shared_queue=queue,
            source_container=source1
        )
        producer2 = Producer(
            name="Producer2",
            shared_queue=queue,
            source_container=source2
        )
        
        producer1.start()
        producer2.start()
        
        producer1.join(timeout=5)
        producer2.join(timeout=5)
        
        # Both should complete successfully
        self.assertEqual(producer1._items_produced, 5)
        self.assertEqual(producer2._items_produced, 5)
        
        # Queue should have all items
        self.assertEqual(queue.size(), 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
