# Sales Data Analysis Application

A Python application demonstrating proficiency with functional programming paradigms by performing various aggregation and grouping operations on sales data in CSV format.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Functional Programming Concepts](#functional-programming-concepts)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Analytical Queries](#analytical-queries)
- [Assumptions & Design Choices](#assumptions--design-choices)
- [Sample Output](#sample-output)

---

## Overview

This application reads sales data from a CSV file and executes multiple analytical queries using functional programming paradigms including:

- **Stream operations** via pandas DataFrame operations
- **Data aggregation** using `groupby()`, `agg()`, and `reduce()`
- **Lambda expressions** for inline transformations and filtering
- **Higher-order functions** for composable operations
- **Method chaining** for fluent data pipelines

---

## Dataset Information

### Source
- **Dataset**: Sales Data Sample from Kaggle
- **File**: `data/sales_data.csv`
- **Records**: 2,823 sales transactions
- **Time Period**: 2003-2005

### Columns Used

| Column | Type | Description |
|--------|------|-------------|
| `ORDERNUMBER` | int | Unique order identifier |
| `QUANTITYORDERED` | int | Number of items ordered |
| `PRICEEACH` | float | Price per unit |
| `SALES` | float | Total sale amount |
| `ORDERDATE` | datetime | Date of order |
| `STATUS` | category | Order status (Shipped, Cancelled, etc.) |
| `QTR_ID` | int | Quarter (1-4) |
| `MONTH_ID` | int | Month (1-12) |
| `YEAR_ID` | int | Year |
| `PRODUCTLINE` | category | Product category |
| `CUSTOMERNAME` | str | Customer name |
| `COUNTRY` | category | Customer country |
| `TERRITORY` | category | Sales territory (NA, EMEA, APAC, Japan) |
| `DEALSIZE` | category | Deal size category (Small, Medium, Large) |

### Why This Dataset?

1. **Rich categorical data** - Multiple grouping dimensions (product line, country, territory, deal size)
2. **Time-series data** - Year, quarter, and month for trend analysis
3. **Numerical data** - Sales, quantity, and price for aggregations
4. **Real-world structure** - Authentic sales data with various statuses and customer information
5. **Sufficient volume** - 2,823 records provides meaningful analysis without excessive processing time

---

## Functional Programming Concepts

### 1. Lambda Expressions

```python
# Filtering with lambda
high_value_sales = df[df.apply(lambda row: row['SALES'] > 5000, axis=1)]

# Sorting with lambda key
top_sales = df.sort_values(by='SALES', key=lambda x: x, ascending=False)

# Custom aggregation with lambda
cv = df['SALES'].agg(lambda x: x.std() / x.mean())
```

### 2. Higher-Order Functions

```python
# Function that accepts functions as arguments
def filter_and_aggregate(self, filter_func, agg_column, agg_func):
    filtered_df = self.df[filter_func(self.df)]
    return filtered_df[agg_column].agg(agg_func)

# Usage
result = analyzer.filter_and_aggregate(
    filter_func=lambda df: df['SALES'] > 5000,
    agg_column='SALES',
    agg_func='sum'
)
```

### 3. Method Chaining (Fluent Interface)

```python
# Pipeline-style data transformation
result = (df
    .query("YEAR_ID == 2004")
    .groupby('COUNTRY')['SALES']
    .agg(['sum', 'mean', 'count'])
    .sort_values('sum', ascending=False)
    .head(10))
```

### 4. map(), filter(), reduce()

```python
from functools import reduce

# map() - Transform each value
sales_in_thousands = df['SALES'].map(lambda x: x / 1000)

# filter() equivalent with pandas
shipped_orders = df[df['STATUS'].map(lambda x: x == 'Shipped')]

# reduce() - Aggregate values
total = reduce(lambda acc, x: acc + x, df['SALES'], 0)
```

### 5. Groupby with Aggregation

```python
# Single grouping with multiple aggregations
revenue_by_product = df.groupby('PRODUCTLINE')['SALES'].agg(['sum', 'mean', 'count'])

# Multi-level grouping
territory_product = df.groupby(['TERRITORY', 'PRODUCTLINE'])['SALES'].sum()
```

---

## Project Structure

```
Data-Analysis/
├── data/
│   └── sales_data.csv              # Kaggle sales dataset
├── src/
│   ├── data_loader.py              # CSV reading, validation, type conversion
│   ├── data_analyzer.py            # Core analysis functions
│   └── main.py                     # Entry point, runs all queries
├── output/
│   └── analysis_results.txt        # Analysis results output
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `data_loader.py` | DataLoader class for CSV reading, validation, type conversion, data cleaning |
| `data_analyzer.py` | DataAnalyzer and SalesAnalyzer classes with grouping, aggregation, statistical functions |
| `main.py` | Orchestrates data loading, validation, and executes 15 analytical queries |

---

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup

```bash
# Navigate to project directory
cd Data-Analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run All Analyses

```bash
cd Data-Analysis/src
python main.py
```

### Run Individual Modules

```bash
# Test data loader
python -m src.data_loader

# Test analyzer functions
python -m src.data_analyzer
```

### Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## Analytical Queries

The application performs 15 comprehensive analyses, sample output is shown in `output/analysis_results.txt`:

| Query | Description | Functional Concept |
|-------|-------------|-------------------|
| 1 | Basic Aggregation Metrics | `sum()`, `mean()`, `count()` |
| 2 | Revenue by Product Line | `groupby()` + aggregation |
| 3 | Sales by Territory | `groupby()` + sorting |
| 4 | Top 10 Customers | Multi-aggregation + ranking |
| 5 | Monthly Sales Trend | Time-series resampling |
| 6 | Deal Size Distribution | Categorical grouping |
| 7 | High-Value Transactions | Lambda filtering + percentile |
| 8 | Statistical Distribution | `mean`, `std`, `skewness`, `kurtosis` |
| 9 | Percentile Analysis | Quantile calculations |
| 10 | Product Performance | Multi-column aggregation |
| 11 | Country Analysis | `groupby()` + `head()` |
| 12 | Multi-Level Grouping | Hierarchical groupby |
| 13 | Lambda Filtering | Higher-order functions |
| 14 | Correlation Analysis | Statistical correlation |
| 15 | Custom Aggregation | Coefficient of variation with lambda |

---

## Assumptions & Design Choices

### Data Assumptions

1. **Sales values are in USD** - No currency conversion needed
2. **Date format is M/D/YYYY H:MM** - Parsed using pandas datetime
3. **Missing values handled gracefully** - Empty fields use defaults or are excluded
4. **Records with valid ORDERNUMBER and SALES are processed** - Core fields required

### Design Choices

1. **Pandas-based implementation** - Efficient vectorized operations for data analysis

2. **Class-based architecture** - `DataLoader` for loading/validation, `SalesAnalyzer` for analysis

3. **Immutability principle** - DataFrame copied on initialization to avoid side effects

4. **Method chaining support** - Results stored and returned for fluent API

5. **Comprehensive validation** - Required columns, data types, duplicates, negative values checked

6. **Results persistence** - Output saved to `output/analysis_results.txt`

### Why Pandas for Functional Programming?

| Benefit | Implementation |
|---------|----------------|
| **Lambda expressions** | Used in `apply()`, `agg()`, `map()`, `filter()` |
| **Higher-order functions** | Functions accept other functions as parameters |
| **Method chaining** | Fluent pipeline-style transformations |
| **Immutability** | Original DataFrame preserved, copies used |
| **Declarative style** | Express *what* to compute, not *how* |