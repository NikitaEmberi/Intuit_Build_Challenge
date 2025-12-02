# Producer-Consumer Pattern Implementation

A Python implementation of the classic **Producer-Consumer pattern** demonstrating thread synchronization and communication using blocking queues and the wait/notify mechanism.

## 📋 Overview

This project implements a concurrent data transfer system where:
- **Producer threads** read items from a source container and place them into a shared queue
- **Consumer threads** read items from the queue and store them in a destination container
- **SharedQueue** provides thread-safe blocking operations with wait/notify synchronization

## ✨ Features

- **Thread Synchronization**: Uses `threading.Condition` for coordinating access
- **Blocking Queue**: Thread-safe queue that blocks when full (for producers) or empty (for consumers)
- **Wait/Notify Mechanism**: Efficient thread communication without busy-waiting
- **Graceful Shutdown**: Clean thread termination using `threading.Event`
- **Statistics Tracking**: Monitor production/consumption rates and queue state
- **Timeout Support**: Configurable timeouts for all blocking operations

## 📁 Project Structure

```
producer-consumer-project/
├── src/
│   ├── main.py              # Demo application
│   ├── shared_queue.py      # Thread-safe blocking queue
│   ├── producer.py          # Producer thread class
│   └── consumer.py          # Consumer thread class
├── tests/
│   ├── test_shared_queue.py # Queue unit tests
│   ├── test_producer.py     # Producer unit tests
│   └── test_consumer.py     # Consumer unit tests
├── requirements.txt
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nikitaemberi/producer-consumer-project.git
   cd producer-consumer-project
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Running the Application

### Run the Demo

```bash
cd src
python main.py
```

### Run with Python module syntax

```bash
python -m src.main
```

## 📊 Sample Output

```
============================================================
  PRODUCER-CONSUMER PATTERN DEMONSTRATION
  Thread Synchronization & Communication
============================================================

============================================================
  DEMO 1: Basic Producer-Consumer
============================================================

Source Container: ['Item_1', 'Item_2', 'Item_3', 'Item_4', 'Item_5', ...]
Destination Container: []
Items to transfer: 10
Queue Capacity: 3 (small to demonstrate blocking)

--- Starting Threads ---
12:30:45 - Producer-1 - INFO - Producer 'Producer-1' starting production...
12:30:45 - Consumer-1 - INFO - Consumer 'Consumer-1' starting consumption...
12:30:45 - Producer-1 - INFO - Producer 'Producer-1' produced item 1/10: Item_1
12:30:45 - Consumer-1 - INFO - Consumer 'Consumer-1' consumed item 1: Item_1
12:30:45 - Producer-1 - INFO - Producer 'Producer-1' produced item 2/10: Item_2
...

--- Transfer Complete ---
Total Time: 2.53s
Destination Container: ['Item_1', 'Item_2', 'Item_3', ..., 'Item_10']
✓ SUCCESS: All items transferred correctly!

--- Statistics ---

Producer 'Producer-1':
  Items Produced: 10/10
  Completion Rate: 100.0%
  Elapsed Time: 1.05s
  Rate: 9.52 items/sec

Consumer 'Consumer-1':
  Items Consumed: 10
  Destination Size: 10
  Elapsed Time: 2.52s
  Rate: 3.97 items/sec

SharedQueue:
  Current Size: 0/3
  Total Put: 10
  Total Get: 10
```

## 🧪 Running Tests

### Run all tests
```bash
# Using pytest
pytest tests/ -v

# Using unittest
python -m unittest discover tests/ -v
```

### Run specific test file
```bash
pytest tests/test_shared_queue.py -v
pytest tests/test_producer.py -v
pytest tests/test_consumer.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## 🔧 Key Components

### SharedQueue
Thread-safe blocking queue implementation.

```python
from shared_queue import SharedQueue

queue = SharedQueue(capacity=10)
queue.put(item)          # Blocks if full
item = queue.get()       # Blocks if empty
queue.put(item, timeout=5.0)  # With timeout
```

### Producer
Thread that produces items from a source container.

```python
from producer import Producer

producer = Producer(
    name="Producer-1",
    shared_queue=queue,
    source_container=[1, 2, 3, 4, 5],
    delay=0.1  # Optional delay between items
)
producer.start()
producer.join()
```

### Consumer
Thread that consumes items into a destination container.

```python
from consumer import Consumer

destination = []
consumer = Consumer(
    name="Consumer-1",
    shared_queue=queue,
    destination_container=destination,
    delay=0.1,
    max_items=5  # Optional limit
)
consumer.start()
consumer.join()
```