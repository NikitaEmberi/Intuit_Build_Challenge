# Intuit Build Challenge - Coding Solutions

This repository contains two independent projects demonstrating proficiency in Python programming, functional programming paradigms, and concurrent programming patterns.

---

## 📦 Projects Overview

| Project | Description | Key Concepts |
|---------|-------------|--------------|
| [Data-Analysis](#1-data-analysis) | Sales data analysis using functional programming | Streams, Lambda, map/filter/reduce, Aggregations |
| [Producer-Consumer](#2-producer-consumer-pattern) | Thread synchronization pattern implementation | Threading, Blocking Queues, Wait/Notify |

---

## 📂 Repository Structure

```
Intuit/
├── README.md                              # ← You are here (Common README)
│
├── Data-Analysis/                         # Project 1: Functional Programming
│   ├── README.md                          # Detailed setup & documentation
│   ├── requirements.txt                   # Python dependencies
│   ├── data/
│   │   └── sales_data.csv                 # Kaggle sales dataset (2,823 records)
│   ├── src/
│   │   ├── main.py                        # Entry point - runs all analyses
│   │   ├── data_loader.py                 # CSV reading & validation
│   │   └── data_analyzer.py               # Core analysis functions
│   └── output/
│       └── analysis_results.txt           # Complete analysis output
│
└── producer-consumer-project/             # Project 2: Concurrent Programming
    ├── README.md                          # Detailed setup & documentation
    ├── requirements.txt                   # Python dependencies
    ├── src/
    │   ├── main.py                        # Demo application
    │   ├── shared_queue.py                # Thread-safe blocking queue
    │   ├── producer.py                    # Producer thread class
    │   ├── consumer.py                    # Consumer thread class
    │   └── item.py                        # Item data class
    └── tests/
        ├── test_shared_queue.py           # Queue unit tests
        ├── test_producer.py               # Producer unit tests
        └── test_consumer.py               # Consumer unit tests
```

---

## 📍 Source Code Locations

### Project 1: Data-Analysis
| File | Purpose | Path |
|------|---------|------|
| `main.py` | Entry point - orchestrates all 15 analytical queries | `Data-Analysis/src/main.py` |
| `data_loader.py` | CSV reading, validation, type conversion | `Data-Analysis/src/data_loader.py` |
| `data_analyzer.py` | Core analysis with functional programming | `Data-Analysis/src/data_analyzer.py` |

### Project 2: Producer-Consumer
| File | Purpose | Path |
|------|---------|------|
| `main.py` | Demo application with multiple scenarios | `producer-consumer-project/src/main.py` |
| `shared_queue.py` | Thread-safe blocking queue with wait/notify | `producer-consumer-project/src/shared_queue.py` |
| `producer.py` | Producer thread implementation | `producer-consumer-project/src/producer.py` |
| `consumer.py` | Consumer thread implementation | `producer-consumer-project/src/consumer.py` |

---

## 🧪 Unit Tests Locations

### Producer-Consumer Project Tests
All unit tests are located in `producer-consumer-project/tests/`:

| Test File | Tests For | Path |
|-----------|-----------|------|
| `test_shared_queue.py` | SharedQueue blocking operations, capacity limits, thread safety | `producer-consumer-project/tests/test_shared_queue.py` |
| `test_producer.py` | Producer thread creation, item production, statistics | `producer-consumer-project/tests/test_producer.py` |
| `test_consumer.py` | Consumer thread creation, item consumption, graceful shutdown | `producer-consumer-project/tests/test_consumer.py` |

### Data-Analysis Tests
The Data-Analysis project includes inline validation and verification within the main execution flow in `Data-Analysis/src/main.py`, which validates all analytical methods during runtime.

---

## 📊 Sample Output Locations

| Project | Output Type | Location |
|---------|-------------|----------|
| Data-Analysis | Complete analysis results | `Data-Analysis/output/analysis_results.txt` |
| Data-Analysis | Console output | Run `python main.py` in `Data-Analysis/src/` |
| Producer-Consumer | Console output | Run `python main.py` in `producer-consumer-project/src/` |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup & Run Both Projects

```bash
# Clone the repository
git clone [https://github.com/nikitaemberi/Intuit.git](https://github.com/NikitaEmberi/Intuit_Build_Challenge.git)
cd Intuit_Build_Challenge

# ============================================
# Project 1: Data-Analysis
# ============================================
cd Data-Analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run all analyses (results printed to console)
cd src
python main.py

# Deactivate virtual environment
deactivate
cd ../..

# ============================================
# Project 2: Producer-Consumer
# ============================================
cd producer-consumer-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the demo (results printed to console)
cd src
python main.py
cd ..

# Run unit tests
pytest tests/ -v

# Deactivate virtual environment
deactivate
```

---

## 🔬 Running Unit Tests

### Producer-Consumer Project

```bash
cd producer-consumer-project

# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_shared_queue.py -v
pytest tests/test_producer.py -v
pytest tests/test_consumer.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Using unittest
python -m unittest discover tests/ -v
```

---

## 📋 Project Summaries

### 1. Data-Analysis

**Purpose**: Demonstrates proficiency with functional programming paradigms by performing aggregation and grouping operations on sales data.

**Key Features**:
- Stream operations via pandas DataFrame operations
- Lambda expressions for inline transformations and filtering
- Higher-order functions for composable operations
- Method chaining for fluent data pipelines
- `map()`, `filter()`, `reduce()` implementations
- 15 comprehensive analytical queries

**Dataset**: Kaggle Sales Data Sample (2,823 records, 2003-2005)

**Detailed Documentation**: [`Data-Analysis/README.md`](Data-Analysis/README.md)

---

### 2. Producer-Consumer Pattern

**Purpose**: Implements the classic Producer-Consumer pattern demonstrating thread synchronization and communication.

**Key Features**:
- Thread synchronization using `threading.Condition`
- Blocking queue implementation (blocks when full/empty)
- Wait/Notify mechanism for efficient thread communication
- Graceful shutdown using `threading.Event`
- Statistics tracking for production/consumption rates
- Configurable timeouts for blocking operations

**Detailed Documentation**: [`producer-consumer-project/README.md`](producer-consumer-project/README.md)

---
