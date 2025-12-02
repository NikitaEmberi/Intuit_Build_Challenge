"""
Producer-Consumer Pattern Demonstration

This program demonstrates thread synchronization and communication using:
- SharedQueue: Thread-safe blocking queue with wait/notify mechanism
- Producer: Thread that reads from source and places items into queue
- Consumer: Thread that reads from queue and stores in destination

"""

import time
import logging
from typing import List, Any

from shared_queue import SharedQueue
from producer import Producer
from consumer import Consumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_separator(title: str = "") -> None:
    """Print a visual separator with optional title."""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def print_stats(producer: Producer, consumer: Consumer, queue: SharedQueue) -> None:
    """Print statistics for producer, consumer, and queue."""
    print("\n--- Statistics ---")
    
    producer_stats = producer.get_stats()
    print(f"\nProducer '{producer_stats['name']}':")
    print(f"  Items Produced: {producer_stats['items_produced']}/{producer_stats['total_items']}")
    print(f"  Completion Rate: {producer_stats['completion_rate']}%")
    print(f"  Elapsed Time: {producer_stats['elapsed_time']}s")
    print(f"  Rate: {producer_stats['items_per_second']} items/sec")
    
    consumer_stats = consumer.get_stats()
    print(f"\nConsumer '{consumer_stats['name']}':")
    print(f"  Items Consumed: {consumer_stats['items_consumed']}")
    print(f"  Destination Size: {consumer_stats['destination_size']}")
    print(f"  Elapsed Time: {consumer_stats['elapsed_time']}s")
    print(f"  Rate: {consumer_stats['items_per_second']} items/sec")
    
    queue_stats = queue.get_stats()
    print(f"\nSharedQueue:")
    print(f"  Current Size: {queue_stats['current_size']}/{queue_stats['capacity']}")
    print(f"  Total Put: {queue_stats['total_put']}")
    print(f"  Total Get: {queue_stats['total_get']}")


def demo_basic_producer_consumer() -> None:
    """
    Demo 1: Basic single producer, single consumer.
    Demonstrates the fundamental producer-consumer pattern.
    """
    print_separator("DEMO 1: Basic Producer-Consumer")
    
    # Source data (simulating data to be transferred)
    source_data: List[Any] = [f"Item_{i}" for i in range(1, 11)]
    destination_data: List[Any] = []
    
    print(f"\nSource Container: {source_data}")
    print(f"Destination Container: {destination_data}")
    print(f"Items to transfer: {len(source_data)}")
    
    # Create shared queue with limited capacity (demonstrates blocking)
    queue = SharedQueue(capacity=3)
    print(f"Queue Capacity: {queue.capacity} (small to demonstrate blocking)")
    
    # Create producer and consumer
    producer = Producer(
        name="Producer-1",
        shared_queue=queue,
        source_container=source_data,
        delay=0.1  # Small delay to observe behavior
    )
    
    consumer = Consumer(
        name="Consumer-1",
        shared_queue=queue,
        destination_container=destination_data,
        delay=0.15,  # Slightly slower than producer
        max_items=len(source_data)  # Stop after consuming all items
    )
    
    print("\n--- Starting Threads ---")
    start_time = time.time()
    
    # Start both threads
    producer.start()
    consumer.start()
    
    # Wait for completion
    producer.join(timeout=30)
    consumer.join(timeout=30)
    
    elapsed = time.time() - start_time
    
    print("\n--- Transfer Complete ---")
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Destination Container: {destination_data}")
    
    # Verify transfer
    if source_data == destination_data:
        print("✓ SUCCESS: All items transferred correctly!")
    else:
        print("✗ ERROR: Data mismatch!")
    
    print_stats(producer, consumer, queue)


def demo_fast_producer_slow_consumer() -> None:
    """
    Demo 2: Fast producer, slow consumer.
    Demonstrates queue blocking when full.
    """
    print_separator("DEMO 2: Fast Producer, Slow Consumer")
    
    source_data = list(range(1, 16))  # 15 numbers
    destination_data: List[Any] = []
    
    print(f"\nProducing: {len(source_data)} items")
    print("Producer: Fast (no delay)")
    print("Consumer: Slow (0.2s delay)")
    
    queue = SharedQueue(capacity=5)
    print(f"Queue Capacity: {queue.capacity}")
    
    producer = Producer(
        name="FastProducer",
        shared_queue=queue,
        source_container=source_data,
        delay=0.0  # No delay - produces as fast as possible
    )
    
    consumer = Consumer(
        name="SlowConsumer",
        shared_queue=queue,
        destination_container=destination_data,
        delay=0.2,  # Slow consumer
        max_items=len(source_data)
    )
    
    print("\n--- Starting (Producer will block when queue is full) ---")
    start_time = time.time()
    
    producer.start()
    consumer.start()
    
    producer.join(timeout=30)
    consumer.join(timeout=30)
    
    elapsed = time.time() - start_time
    
    print(f"\n--- Complete in {elapsed:.2f}s ---")
    print(f"Items transferred: {len(destination_data)}")
    
    if set(source_data) == set(destination_data):
        print("✓ SUCCESS: All items transferred!")
    
    print_stats(producer, consumer, queue)


def demo_multiple_producers_consumers() -> None:
    """
    Demo 3: Multiple producers and consumers.
    Demonstrates thread-safe concurrent access.
    """
    print_separator("DEMO 3: Multiple Producers & Consumers")
    
    # Two source containers
    source_1 = [f"A{i}" for i in range(1, 6)]  # A1-A5
    source_2 = [f"B{i}" for i in range(1, 6)]  # B1-B5
    
    # Two destination containers
    dest_1: List[Any] = []
    dest_2: List[Any] = []
    
    print(f"\nProducer-1 source: {source_1}")
    print(f"Producer-2 source: {source_2}")
    
    # Shared queue for all
    queue = SharedQueue(capacity=4)
    
    # Create producers
    producer_1 = Producer(
        name="Producer-A",
        shared_queue=queue,
        source_container=source_1,
        delay=0.1
    )
    producer_2 = Producer(
        name="Producer-B",
        shared_queue=queue,
        source_container=source_2,
        delay=0.1
    )
    
    # Create consumers (each takes 5 items)
    consumer_1 = Consumer(
        name="Consumer-1",
        shared_queue=queue,
        destination_container=dest_1,
        delay=0.1,
        max_items=5
    )
    consumer_2 = Consumer(
        name="Consumer-2",
        shared_queue=queue,
        destination_container=dest_2,
        delay=0.1,
        max_items=5
    )
    
    print("\n--- Starting 2 Producers + 2 Consumers ---")
    start_time = time.time()
    
    # Start all threads
    producer_1.start()
    producer_2.start()
    consumer_1.start()
    consumer_2.start()
    
    # Wait for all to complete
    producer_1.join(timeout=30)
    producer_2.join(timeout=30)
    consumer_1.join(timeout=30)
    consumer_2.join(timeout=30)
    
    elapsed = time.time() - start_time
    
    print(f"\n--- Complete in {elapsed:.2f}s ---")
    print(f"Consumer-1 received: {dest_1}")
    print(f"Consumer-2 received: {dest_2}")
    
    # Verify all items consumed
    all_produced = set(source_1 + source_2)
    all_consumed = set(dest_1 + dest_2)
    
    if all_produced == all_consumed:
        print("✓ SUCCESS: All items consumed (distributed between consumers)!")
    else:
        print(f"Missing: {all_produced - all_consumed}")
    
    print(f"\nQueue final stats: {queue.get_stats()}")


def demo_graceful_shutdown() -> None:
    """
    Demo 4: Graceful shutdown demonstration.
    Shows how to stop threads cleanly.
    """
    print_separator("DEMO 4: Graceful Shutdown")
    
    source_data = [f"Data_{i}" for i in range(1, 101)]  # 100 items
    destination_data: List[Any] = []
    
    print(f"\nSource has {len(source_data)} items")
    print("Will stop after ~1 second to demonstrate graceful shutdown")
    
    queue = SharedQueue(capacity=10)
    
    producer = Producer(
        name="StoppableProducer",
        shared_queue=queue,
        source_container=source_data,
        delay=0.05
    )
    
    consumer = Consumer(
        name="StoppableConsumer",
        shared_queue=queue,
        destination_container=destination_data,
        delay=0.05
    )
    
    print("\n--- Starting threads ---")
    producer.start()
    consumer.start()
    
    # Let them run for a bit
    time.sleep(1.0)
    
    print("\n--- Requesting graceful shutdown ---")
    producer.stop(timeout=2.0)
    consumer.stop(timeout=2.0)
    
    print(f"\nItems produced: {producer._items_produced}")
    print(f"Items consumed: {len(destination_data)}")
    print(f"Items still in queue: {queue.size()}")
    
    if not producer.is_alive() and not consumer.is_alive():
        print("✓ SUCCESS: Both threads stopped gracefully!")


def main() -> None:
    """Main entry point - runs all demonstrations."""
    print("\n" + "=" * 60)
    print("  PRODUCER-CONSUMER PATTERN DEMONSTRATION")
    print("  Thread Synchronization & Communication")
    print("=" * 60)
    # Run demonstrations
    demo_basic_producer_consumer()
    demo_fast_producer_slow_consumer()
    demo_multiple_producers_consumers()
    demo_graceful_shutdown()
    
    print_separator("ALL DEMONSTRATIONS COMPLETE")
    print("\nKey Concepts Demonstrated:")
    print("  ✓ Thread synchronization with threading.Condition")
    print("  ✓ Blocking queue with put()/get() operations")
    print("  ✓ Wait/notify mechanism for thread communication")
    print("  ✓ Concurrent programming with multiple threads")
    print("  ✓ Graceful shutdown using threading.Event")
    print()


if __name__ == "__main__":
    main()
