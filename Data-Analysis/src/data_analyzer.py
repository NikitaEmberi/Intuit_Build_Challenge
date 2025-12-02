"""
Data Analyzer Module
Performs comprehensive data analysis using functional programming paradigms.
Demonstrates pandas operations: groupby, aggregation, filtering, mapping, and lambda expressions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
from functools import reduce


class DataAnalyzer:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize analyzer with a pandas DataFrame.
        
        Args:
            dataframe (pd.DataFrame): The dataset to analyze
        """
        self.df = dataframe.copy()  # Immutability - avoiding modifying original
        self.results = {}  # Store analysis results
    
    def _store_result(self, query_name: str, result: Any) -> Any:
        """
        Store query result and return it (for method chaining).
        
        Args:
            query_name (str): Name identifier for the query
            result: The query result to store
            
        Returns:
            The result (allows chaining)
        """
        self.results[query_name] = result
        return result
    
    def _print_section_header(self, title: str):
        """Print formatted section header."""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    # ========================================================================
    # BASIC AGGREGATION QUERIES
    # ========================================================================
    
    def total_aggregation(self, column: str, agg_func: str = 'sum') -> float:
        """
        Perform basic aggregation on a numeric column.
        
        Args:
            column (str): Column name to aggregate
            agg_func (str): Aggregation function ('sum', 'mean', 'count', etc.)
            
        Returns:
            float: Aggregated value
        """
        result = self.df[column].agg(agg_func)
        return self._store_result(f'{agg_func}_{column}', result)
    
    def multiple_aggregations(self, column: str, 
                            agg_funcs: List[str] = None) -> pd.Series:
        """
        Perform multiple aggregations on a single column.
        
        Args:
            column (str): Column name to aggregate
            agg_funcs (list): List of aggregation functions
            
        Returns:
            pd.Series: Series with multiple aggregation results
        """
        if agg_funcs is None:
            agg_funcs = ['sum', 'mean', 'median', 'std', 'min', 'max', 'count']
        
        result = self.df[column].agg(agg_funcs)
        return self._store_result(f'multi_agg_{column}', result)
    
    def custom_aggregation(self, column: str, 
                          custom_func: Callable) -> Any:
        """
        Apply custom aggregation function using lambda or user-defined function.
        
        Args:
            column (str): Column name
            custom_func (callable): Custom aggregation function
            
        Returns:
            Aggregation result
        """
        result = self.df[column].agg(custom_func)
        return self._store_result(f'custom_agg_{column}', result)
    
    # ========================================================================
    # GROUPING AND AGGREGATION QUERIES
    # ========================================================================
    
    def group_by_single(self, group_column: str, 
                       agg_column: str, 
                       agg_func: str = 'sum') -> pd.Series:
        """
        Group by single column and aggregate.
        
        Args:
            group_column (str): Column to group by
            agg_column (str): Column to aggregate
            agg_func (str): Aggregation function
            
        Returns:
            pd.Series: Grouped and aggregated results
        """
        result = (self.df
                 .groupby(group_column)[agg_column]
                 .agg(agg_func)
                 .sort_values(ascending=False))
        
        return self._store_result(f'group_{group_column}_by_{agg_func}', result)
    
    def group_by_multiple_agg(self, group_column: str,
                             agg_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Group by column with multiple aggregations on different columns.
        
        Args:
            group_column (str): Column to group by
            agg_dict (dict): Dictionary mapping columns to aggregation functions
                            e.g., {'SALES': ['sum', 'mean'], 'QUANTITY': ['sum']}
            
        Returns:
            pd.DataFrame: Grouped results with multiple aggregations
        """
        result = (self.df
                 .groupby(group_column)
                 .agg(agg_dict)
                 .sort_values(by=(list(agg_dict.keys())[0], agg_dict[list(agg_dict.keys())[0]][0]), 
                            ascending=False))
        
        return self._store_result(f'multi_agg_group_{group_column}', result)
    
    def multi_level_grouping(self, group_columns: List[str],
                            agg_column: str,
                            agg_func: str = 'sum') -> pd.Series:
        """
        Group by multiple columns (hierarchical grouping).
        
        Args:
            group_columns (list): List of columns to group by
            agg_column (str): Column to aggregate
            agg_func (str): Aggregation function
            
        Returns:
            pd.Series: Multi-indexed grouped results
        """
        result = (self.df
                 .groupby(group_columns)[agg_column]
                 .agg(agg_func)
                 .sort_values(ascending=False))
        
        return self._store_result(f'multi_group_{"_".join(group_columns)}', result)
    
    # ========================================================================
    # FILTERING AND TRANSFORMATION QUERIES
    # ========================================================================
    
    def filter_and_aggregate(self, filter_func: Callable,
                            agg_column: str,
                            agg_func: str = 'sum') -> float:
        """
        Filter data using lambda/function, then aggregate.
        
        Args:
            filter_func (callable): Function that returns boolean mask
            agg_column (str): Column to aggregate after filtering
            agg_func (str): Aggregation function
            
        Returns:
            float: Aggregated value from filtered data
        """
        result = (self.df
                 .loc[filter_func(self.df)]
                 [agg_column]
                 .agg(agg_func))
        
        return self._store_result('filtered_aggregation', result)
    
    def top_n_by_column(self, n: int, column: str, 
                       group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Get top N records by column value.
        
        Args:
            n (int): Number of top records
            column (str): Column to sort by
            group_by (str, optional): If provided, get top N per group
            
        Returns:
            pd.DataFrame: Top N records
        """
        if group_by:
            result = (self.df
                     .sort_values(column, ascending=False)
                     .groupby(group_by)
                     .head(n))
        else:
            result = (self.df
                     .nlargest(n, column))
        
        return self._store_result(f'top_{n}_by_{column}', result)
    
    def apply_transformation(self, column: str, 
                           transform_func: Callable,
                           new_column_name: str) -> pd.DataFrame:
        """
        Apply transformation using lambda/function to create new column.
        
        Args:
            column (str): Source column
            transform_func (callable): Transformation function
            new_column_name (str): Name for new column
            
        Returns:
            pd.DataFrame: DataFrame with new column
        """
        result_df = self.df.copy()
        result_df[new_column_name] = result_df[column].apply(transform_func)
        
        return self._store_result(f'transformed_{new_column_name}', result_df)
    
    def apply_row_wise_operation(self, operation_func: Callable,
                                new_column_name: str) -> pd.DataFrame:
        """
        Apply function across rows (multiple columns).
        
        Args:
            operation_func (callable): Function that takes a row and returns value
            new_column_name (str): Name for new column
            
        Returns:
            pd.DataFrame: DataFrame with computed column
        """
        result_df = self.df.copy()
        result_df[new_column_name] = result_df.apply(operation_func, axis=1)
        
        return self._store_result(f'row_operation_{new_column_name}', result_df)
    
    # ========================================================================
    # STATISTICAL ANALYSIS QUERIES
    # ========================================================================
    
    def percentile_analysis(self, column: str, 
                          percentiles: List[float] = None) -> pd.Series:
        """
        Calculate percentiles for a column.
        
        Args:
            column (str): Column to analyze
            percentiles (list): List of percentiles (0-1), defaults to quartiles
            
        Returns:
            pd.Series: Percentile values
        """
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
        
        result = self.df[column].quantile(percentiles)
        return self._store_result(f'percentiles_{column}', result)
    
    def correlation_analysis(self, col1: str, col2: str) -> float:
        """
        Calculate correlation between two numeric columns.
        
        Args:
            col1 (str): First column
            col2 (str): Second column
            
        Returns:
            float: Correlation coefficient
        """
        result = self.df[col1].corr(self.df[col2])
        return self._store_result(f'correlation_{col1}_{col2}', result)
    
    def distribution_analysis(self, column: str) -> Dict[str, float]:
        """
        Analyze distribution statistics for a column.
        
        Args:
            column (str): Column to analyze
            
        Returns:
            dict: Distribution statistics
        """
        result = {
            'mean': self.df[column].mean(),
            'median': self.df[column].median(),
            'mode': self.df[column].mode()[0] if len(self.df[column].mode()) > 0 else None,
            'std': self.df[column].std(),
            'variance': self.df[column].var(),
            'skewness': self.df[column].skew(),
            'kurtosis': self.df[column].kurtosis(),
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            'range': self.df[column].max() - self.df[column].min()
        }
        
        return self._store_result(f'distribution_{column}', result)
    
    # ========================================================================
    # ADVANCED FUNCTIONAL PROGRAMMING QUERIES
    # ========================================================================
    
    def chain_operations(self, operations: List[Tuple[str, Dict]]) -> pd.DataFrame:
        """
        Chain multiple operations using pipe() - demonstrates functional pipeline.
        
        Args:
            operations (list): List of (method_name, kwargs) tuples
            
        Returns:
            pd.DataFrame: Result of chained operations
        """
        result = self.df.copy()
        
        for method_name, kwargs in operations:
            if hasattr(result, method_name):
                result = getattr(result, method_name)(**kwargs)
        
        return self._store_result('chained_operations', result)
    
    def aggregation_with_filter(self, group_column: str,
                               agg_column: str,
                               filter_func: Callable,
                               agg_func: str = 'sum') -> pd.Series:
        """
        Combined filtering and grouping operation.
        
        Args:
            group_column (str): Column to group by
            agg_column (str): Column to aggregate
            filter_func (callable): Filter function
            agg_func (str): Aggregation function
            
        Returns:
            pd.Series: Filtered, grouped, and aggregated results
        """
        result = (self.df
                 .loc[filter_func(self.df)]
                 .groupby(group_column)[agg_column]
                 .agg(agg_func)
                 .sort_values(ascending=False))
        
        return self._store_result('filter_group_agg', result)
    
    def map_reduce_operation(self, map_func: Callable,
                            reduce_func: Callable,
                            column: str) -> Any:
        """
        Demonstrate map-reduce pattern.
        
        Args:
            map_func (callable): Map function applied to each element
            reduce_func (callable): Reduce function to combine results
            column (str): Column to operate on
            
        Returns:
            Reduced result
        """
        # Map phase
        mapped = self.df[column].map(map_func)
        
        # Reduce phase
        result = reduce(reduce_func, mapped)
        
        return self._store_result('map_reduce', result)
    
    def window_function_analysis(self, partition_by: str,
                                order_by: str,
                                value_column: str,
                                operation: str = 'cumsum') -> pd.DataFrame:
        """
        Apply window functions (cumulative operations within groups).
        
        Args:
            partition_by (str): Column to partition by
            order_by (str): Column to order by
            value_column (str): Column to apply operation on
            operation (str): Window operation ('cumsum', 'cummax', 'rank', etc.)
            
        Returns:
            pd.DataFrame: DataFrame with window function result
        """
        result_df = self.df.copy().sort_values([partition_by, order_by])
        
        if operation in ['cumsum', 'cummax', 'cummin', 'cumprod']:
            result_df[f'{operation}_{value_column}'] = (
                result_df.groupby(partition_by)[value_column]
                .transform(operation)
            )
        elif operation == 'rank':
            result_df[f'rank_{value_column}'] = (
                result_df.groupby(partition_by)[value_column]
                .rank(method='dense', ascending=False)
            )
        
        return self._store_result('window_function', result_df)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_result(self, query_name: str) -> Any:
        """
        Retrieve stored query result.
        
        Args:
            query_name (str): Name of the query
            
        Returns:
            Stored result or None
        """
        return self.results.get(query_name)
    
    def get_all_results(self) -> Dict[str, Any]:
        """
        Get all stored results.
        
        Returns:
            dict: All query results
        """
        return self.results.copy()
    
    def clear_results(self):
        """Clear all stored results."""
        self.results = {}
    
    def print_result(self, query_name: str, max_rows: int = 10):
        """
        Pretty print a stored result.
        
        Args:
            query_name (str): Name of query to print
            max_rows (int): Maximum rows to display for DataFrames/Series
        """
        if query_name not in self.results:
            print(f"No result found for: {query_name}")
            return
        
        result = self.results[query_name]
        
        print(f"\n{'='*70}")
        print(f"Result: {query_name}")
        print(f"{'='*70}")
        
        if isinstance(result, (pd.DataFrame, pd.Series)):
            print(result.head(max_rows))
            if len(result) > max_rows:
                print(f"... ({len(result) - max_rows} more rows)")
        elif isinstance(result, dict):
            for key, value in result.items():
                print(f"{key}: {value}")
        else:
            print(result)
        
        print()


# ============================================================================
# SALES-SPECIFIC ANALYZER (Example Implementation)
# ============================================================================

class SalesAnalyzer(DataAnalyzer):
    """
    Sales-specific analyzer extending base DataAnalyzer.
    Contains domain-specific queries for sales data.
    """
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with sales DataFrame.
        
        Args:
            dataframe: Sales data with standard columns
        """
        super().__init__(dataframe)
        self._validate_sales_columns()
    
    def _validate_sales_columns(self):
        """Check if required sales columns exist."""
        recommended_cols = ['SALES', 'QUANTITYORDERED', 'ORDERDATE']
        missing = [col for col in recommended_cols if col not in self.df.columns]
        if missing:
            print(f"Warning: Recommended columns missing: {missing}")
    
    def total_revenue(self) -> float:
        """Calculate total revenue across all sales."""
        return self.total_aggregation('SALES', 'sum')
    
    def average_order_value(self) -> float:
        """Calculate average order value."""
        return self.total_aggregation('SALES', 'mean')
    
    def revenue_by_category(self, category_column: str = 'PRODUCTLINE') -> pd.Series:
        """
        Calculate revenue by product category.
        
        Args:
            category_column: Column containing categories
            
        Returns:
            pd.Series: Revenue per category, sorted descending
        """
        return self.group_by_single(category_column, 'SALES', 'sum')
    
    def sales_by_region(self, region_column: str = 'TERRITORY') -> pd.Series:
        """Calculate sales by geographic region."""
        return self.group_by_single(region_column, 'SALES', 'sum')
    
    def top_customers(self, n: int = 10, customer_column: str = 'CUSTOMERNAME') -> pd.DataFrame:
        """
        Get top N customers by total revenue.
        
        Args:
            n: Number of top customers
            customer_column: Column containing customer names
            
        Returns:
            pd.DataFrame: Top customers with revenue
        """
        result = (self.df
                 .groupby(customer_column)['SALES']
                 .agg(['sum', 'count', 'mean'])
                 .rename(columns={'sum': 'total_revenue', 
                                'count': 'order_count',
                                'mean': 'avg_order_value'})
                 .sort_values('total_revenue', ascending=False)
                 .head(n))
        
        return self._store_result('top_customers', result)
    
    def high_value_transactions(self, threshold_percentile: float = 0.9) -> pd.DataFrame:
        """
        Get high-value transactions above a percentile threshold.
        
        Args:
            threshold_percentile: Percentile threshold (0-1)
            
        Returns:
            pd.DataFrame: High-value transactions
        """
        threshold = self.df['SALES'].quantile(threshold_percentile)
        result = self.df[self.df['SALES'] >= threshold].copy()
        
        return self._store_result('high_value_transactions', result)
    
    def monthly_sales_trend(self, date_column: str = 'ORDERDATE') -> pd.Series:
        """
        Calculate monthly sales trends.
        
        Args:
            date_column: Column containing dates
            
        Returns:
            pd.Series: Monthly sales totals
        """
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            self.df[date_column] = pd.to_datetime(self.df[date_column])
        
        result = (self.df
                 .set_index(date_column)
                 .resample('M')['SALES']
                 .sum()
                 .sort_index())
        
        return self._store_result('monthly_sales', result)
    
    def product_performance(self, product_column: str = 'PRODUCTCODE',
                          top_n: int = 10) -> pd.DataFrame:
        """
        Analyze product performance metrics.
        
        Args:
            product_column: Column containing product identifiers
            top_n: Number of top products to return
            
        Returns:
            pd.DataFrame: Product performance metrics
        """
        result = (self.df
                 .groupby(product_column)
                 .agg({
                     'SALES': ['sum', 'mean', 'count'],
                     'QUANTITYORDERED': 'sum'
                 })
                 .round(2))
        
        # Flatten column names
        result.columns = ['_'.join(col).strip() for col in result.columns.values]
        result = result.sort_values('SALES_sum', ascending=False).head(top_n)
        
        return self._store_result('product_performance', result)
    
    def deal_size_distribution(self, deal_column: str = 'DEALSIZE') -> pd.DataFrame:
        """
        Analyze distribution of deal sizes.
        
        Args:
            deal_column: Column containing deal size categories
            
        Returns:
            pd.DataFrame: Deal size statistics
        """
        result = (self.df
                 .groupby(deal_column)
                 .agg({
                     'SALES': ['sum', 'mean', 'count'],
                     'ORDERNUMBER': 'nunique'
                 })
                 .round(2))
        
        result.columns = ['total_revenue', 'avg_sale', 'transaction_count', 'unique_orders']
        
        return self._store_result('deal_size_distribution', result)


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demonstrate_functional_programming(df: pd.DataFrame):
    """
    Demonstrate various functional programming concepts using the analyzer.
    
    Args:
        df: Input DataFrame
    """
    analyzer = DataAnalyzer(df)
    
    print("\n" + "="*70)
    print("FUNCTIONAL PROGRAMMING DEMONSTRATIONS")
    print("="*70)
    
    # 1. Lambda expressions
    print("\n1. LAMBDA EXPRESSIONS")
    print("-" * 70)
    
    # Filter using lambda
    high_sales = analyzer.filter_and_aggregate(
        filter_func=lambda x: x['SALES'] > x['SALES'].mean(),
        agg_column='SALES',
        agg_func='sum'
    )
    print(f"Total sales above average: ${high_sales:,.2f}")
    
    # 2. Map operation
    print("\n2. MAP OPERATIONS")
    print("-" * 70)
    
    if 'PRICEEACH' in df.columns and 'MSRP' in df.columns:
        # Calculate discount percentage
        result = analyzer.apply_transformation(
            column='PRICEEACH',
            transform_func=lambda x: x * 1.1,  # 10% markup example
            new_column_name='PRICE_WITH_MARKUP'
        )
        print("Applied 10% markup transformation using lambda")
        print(result[['PRICEEACH', 'PRICE_WITH_MARKUP']].head())
    
    # 3. Reduce operation (map-reduce)
    print("\n3. MAP-REDUCE PATTERN")
    print("-" * 70)
    
    # Example: Square all sales values then sum
    total_squared = analyzer.map_reduce_operation(
        map_func=lambda x: x ** 2,
        reduce_func=lambda a, b: a + b,
        column='SALES'
    )
    print(f"Sum of squared sales: {total_squared:,.2f}")
    
    # 4. Method chaining
    print("\n4. METHOD CHAINING")
    print("-" * 70)
    
    if 'PRODUCTLINE' in df.columns:
        chained_result = (df
                         .groupby('PRODUCTLINE')['SALES']
                         .sum()
                         .sort_values(ascending=False)
                         .head(3))
        print("Top 3 product lines by revenue:")
        print(chained_result)
    
    # 5. Higher-order functions
    print("\n5. HIGHER-ORDER FUNCTIONS (Custom Aggregation)")
    print("-" * 70)
    
    # Custom aggregation: coefficient of variation
    cv_result = analyzer.custom_aggregation(
        column='SALES',
        custom_func=lambda x: x.std() / x.mean() if x.mean() != 0 else 0
    )
    print(f"Coefficient of Variation for SALES: {cv_result:.4f}")


def run_comprehensive_analysis(df: pd.DataFrame):
    """
    Run comprehensive analysis on sales data.
    
    Args:
        df: Sales DataFrame
    """
    analyzer = SalesAnalyzer(df)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE SALES ANALYSIS")
    print("="*70)
    
    # Basic aggregations
    print("\n BASIC METRICS")
    print("-" * 70)
    total_rev = analyzer.total_revenue()
    avg_order = analyzer.average_order_value()
    total_quantity = analyzer.total_aggregation('QUANTITYORDERED', 'sum')
    
    print(f"Total Revenue: ${total_rev:,.2f}")
    print(f"Average Order Value: ${avg_order:,.2f}")
    print(f"Total Quantity Sold: {total_quantity:,.0f} units")
    
    # Revenue by category
    if 'PRODUCTLINE' in df.columns:
        print("\ REVENUE BY PRODUCT LINE")
        print("-" * 70)
        category_revenue = analyzer.revenue_by_category()
        print(category_revenue.head(10))
    
    # Geographic analysis
    if 'TERRITORY' in df.columns:
        print("\n SALES BY TERRITORY")
        print("-" * 70)
        territory_sales = analyzer.sales_by_region()
        print(territory_sales)
    
    # Top customers
    if 'CUSTOMERNAME' in df.columns:
        print("\n TOP 10 CUSTOMERS")
        print("-" * 70)
        top_cust = analyzer.top_customers(n=10)
        print(top_cust)
    
    # Deal size analysis
    if 'DEALSIZE' in df.columns:
        print("\n DEAL SIZE DISTRIBUTION")
        print("-" * 70)
        deal_dist = analyzer.deal_size_distribution()
        print(deal_dist)
    
    # Statistical analysis
    print("\n STATISTICAL ANALYSIS")
    print("-" * 70)
    distribution = analyzer.distribution_analysis('SALES')
    for key, value in distribution.items():
        print(f"{key.capitalize():15s}: {value:,.2f}")
    
    # Percentiles
    print("\n SALES PERCENTILES")
    print("-" * 70)
    percentiles = analyzer.percentile_analysis('SALES')
    print(percentiles)
    
    # High-value transactions
    print("\n HIGH-VALUE TRANSACTIONS (Top 10%)")
    print("-" * 70)
    high_value = analyzer.high_value_transactions(threshold_percentile=0.90)
    print(f"Count: {len(high_value)}")
    print(f"Total Value: ${high_value['SALES'].sum():,.2f}")
    print(f"Average Value: ${high_value['SALES'].mean():,.2f}")
    
    # Monthly trends
    if 'ORDERDATE' in df.columns:
        print("\n MONTHLY SALES TREND")
        print("-" * 70)
        monthly = analyzer.monthly_sales_trend()
        print(monthly)
