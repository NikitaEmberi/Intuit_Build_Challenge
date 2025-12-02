"""
Main Application - Sales Data Analysis
Demonstrates functional programming with pandas stream operations.
Performs comprehensive analysis on CSV sales data.
"""

import sys
from pathlib import Path
from datetime import datetime
import io

from data_loader import DataLoader
from data_analyzer import SalesAnalyzer, DataAnalyzer, demonstrate_functional_programming


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your CSV file
CSV_FILE_PATH = "../data/sales_data.csv"

# Configuration for sales dataset
SALES_DATA_CONFIG = {
    'required_columns': [
        'ORDERNUMBER', 'SALES', 'ORDERDATE', 'CUSTOMERNAME'
    ],
    'date_columns': ['ORDERDATE'],
    'numeric_columns': [
        'QUANTITYORDERED', 'PRICEEACH', 'SALES', 'MSRP',
        'QTR_ID', 'MONTH_ID', 'YEAR_ID', 'ORDERLINENUMBER', 'ORDERNUMBER'
    ],
    'categorical_columns': [
        'STATUS', 'PRODUCTLINE', 'COUNTRY', 'TERRITORY', 'DEALSIZE'
    ],
    'unique_key': ['ORDERNUMBER', 'ORDERLINENUMBER'],
    'non_negative_columns': ['QUANTITYORDERED', 'PRICEEACH', 'SALES'],
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title, char='='):
    """Print a formatted section header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}\n")


def print_subheader(title, char='-'):
    """Print a formatted subsection header."""
    width = 80
    print(f"\n{title}")
    print(f"{char * width}")


def format_currency(value):
    """Format number as currency."""
    return f"${value:,.2f}"


def format_number(value):
    """Format number with thousands separator."""
    return f"{value:,.0f}"


def format_percentage(value):
    """Format number as percentage."""
    return f"{value:.2%}"


def save_results_to_file(content, filename="analysis_results.txt"):
    """
    Save analysis results to a text file.
    
    Args:
        content (str): Content to save
        filename (str): Output filename
    """
    output_dir = Path("../output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n Results saved to: {output_path}")
    except Exception as e:
        print(f"\n Error saving results: {e}")


# ============================================================================
# ANALYSIS QUERIES
# ============================================================================

def query_1_basic_metrics(analyzer):
    """
    Query 1: Basic Aggregation Metrics
    Demonstrates: Basic aggregation operations (sum, mean, count)
    """
    print_subheader("QUERY 1: Basic Aggregation Metrics")
    
    total_revenue = analyzer.total_revenue()
    avg_order_value = analyzer.average_order_value()
    total_quantity = analyzer.total_aggregation('QUANTITYORDERED', 'sum')
    total_orders = analyzer.total_aggregation('ORDERNUMBER', 'nunique')
    
    print(f"{'Total Revenue:':<30} {format_currency(total_revenue)}")
    print(f"{'Average Order Value:':<30} {format_currency(avg_order_value)}")
    print(f"{'Total Quantity Sold:':<30} {format_number(total_quantity)} units")
    print(f"{'Total Unique Orders:':<30} {format_number(total_orders)}")
    
    print(f"\n Insight: Average order value of {format_currency(avg_order_value)} "
          f"across {format_number(total_orders)} orders")


def query_2_revenue_by_product_line(analyzer):
    """
    Query 2: Revenue by Product Line
    Demonstrates: Single-column grouping with aggregation
    """
    print_subheader("QUERY 2: Revenue by Product Line")
    
    revenue_by_product = analyzer.revenue_by_category('PRODUCTLINE')
    
    print(revenue_by_product.to_string())
    
    # Calculate percentages
    total = revenue_by_product.sum()
    top_product = revenue_by_product.index[0]
    top_revenue = revenue_by_product.iloc[0]
    top_pct = (top_revenue / total) * 100
    
    print(f"\n Insight: '{top_product}' leads with {format_currency(top_revenue)} "
          f"({top_pct:.1f}% of total revenue)")


def query_3_sales_by_territory(analyzer):
    """
    Query 3: Sales by Geographic Territory
    Demonstrates: Grouping with sorting
    """
    print_subheader("QUERY 3: Sales by Territory")
    
    territory_sales = analyzer.sales_by_region('TERRITORY')
    
    for territory, sales in territory_sales.items():
        print(f"{territory:<15} {format_currency(sales)}")
    
    top_territory = territory_sales.index[0]
    print(f"\n💡 Insight: {top_territory} is the top-performing territory")


def query_4_top_customers(analyzer):
    """
    Query 4: Top Customers Analysis
    Demonstrates: Grouping with multiple aggregations
    """
    print_subheader("QUERY 4: Top 10 Customers")
    
    top_customers = analyzer.top_customers(n=10)
    
    print(top_customers.to_string())
    
    top_customer = top_customers.index[0]
    top_customer_revenue = top_customers.iloc[0]['total_revenue']
    
    print(f"\n💡 Insight: Top customer '{top_customer}' generated "
          f"{format_currency(top_customer_revenue)} in revenue")


def query_5_monthly_sales_trend(analyzer):
    """
    Query 5: Monthly Sales Trends
    Demonstrates: Time-series aggregation, resampling
    """
    print_subheader("QUERY 5: Monthly Sales Trend")
    
    monthly_sales = analyzer.monthly_sales_trend()
    
    # Display monthly data
    for date, sales in monthly_sales.items():
        print(f"{date.strftime('%Y-%m'):<15} {format_currency(sales)}")
    
    # Find best and worst months
    best_month = monthly_sales.idxmax()
    best_sales = monthly_sales.max()
    worst_month = monthly_sales.idxmin()
    worst_sales = monthly_sales.min()
    
    print(f"\n Insight: Best month: {best_month.strftime('%B %Y')} "
          f"({format_currency(best_sales)})")
    print(f"Worst month: {worst_month.strftime('%B %Y')} "
          f"({format_currency(worst_sales)})")


def query_6_deal_size_distribution(analyzer):
    """
    Query 6: Deal Size Distribution
    Demonstrates: Categorical grouping with multiple metrics
    """
    print_subheader("QUERY 6: Deal Size Distribution")
    
    deal_distribution = analyzer.deal_size_distribution('DEALSIZE')
    
    print(deal_distribution.to_string())
    
    # Calculate percentages
    total_revenue = deal_distribution['total_revenue'].sum()
    for deal_size in deal_distribution.index:
        revenue = deal_distribution.loc[deal_size, 'total_revenue']
        pct = (revenue / total_revenue) * 100
        count = deal_distribution.loc[deal_size, 'transaction_count']
        print(f"\n{deal_size}: {format_percentage(pct/100)} of revenue "
              f"({format_number(count)} transactions)")


def query_7_high_value_transactions(analyzer):
    """
    Query 7: High-Value Transactions (Top 10%)
    Demonstrates: Filtering with lambda, percentile analysis
    """
    print_subheader("QUERY 7: High-Value Transactions (Top 10%)")
    
    high_value = analyzer.high_value_transactions(threshold_percentile=0.90)
    
    print(f"{'Number of transactions:':<30} {format_number(len(high_value))}")
    print(f"{'Total value:':<30} {format_currency(high_value['SALES'].sum())}")
    print(f"{'Average value:':<30} {format_currency(high_value['SALES'].mean())}")
    print(f"{'Minimum value:':<30} {format_currency(high_value['SALES'].min())}")
    print(f"{'Maximum value:':<30} {format_currency(high_value['SALES'].max())}")
    
    print(f"\n💡 Insight: Top 10% of transactions account for "
          f"{format_currency(high_value['SALES'].sum())} in sales")


def query_8_statistical_analysis(analyzer):
    """
    Query 8: Statistical Distribution Analysis
    Demonstrates: Statistical functions, comprehensive metrics
    """
    print_subheader("QUERY 8: Statistical Analysis of Sales")
    
    distribution = analyzer.distribution_analysis('SALES')
    
    print(f"{'Mean:':<20} {format_currency(distribution['mean'])}")
    print(f"{'Median:':<20} {format_currency(distribution['median'])}")
    print(f"{'Mode:':<20} {format_currency(distribution['mode']) if distribution['mode'] else 'N/A'}")
    print(f"{'Std Deviation:':<20} {format_currency(distribution['std'])}")
    print(f"{'Variance:':<20} {format_currency(distribution['variance'])}")
    print(f"{'Skewness:':<20} {distribution['skewness']:.4f}")
    print(f"{'Kurtosis:':<20} {distribution['kurtosis']:.4f}")
    print(f"{'Min:':<20} {format_currency(distribution['min'])}")
    print(f"{'Max:':<20} {format_currency(distribution['max'])}")
    print(f"{'Range:':<20} {format_currency(distribution['range'])}")
    
    # Interpret skewness
    skew = distribution['skewness']
    if skew > 0.5:
        skew_interpretation = "positively skewed (right tail)"
    elif skew < -0.5:
        skew_interpretation = "negatively skewed (left tail)"
    else:
        skew_interpretation = "approximately symmetric"
    
    print(f"\nInsight: Sales distribution is {skew_interpretation}")


def query_9_percentile_analysis(analyzer):
    """
    Query 9: Percentile Analysis
    Demonstrates: Quantile calculations
    """
    print_subheader("QUERY 9: Sales Percentile Analysis")
    
    percentiles = analyzer.percentile_analysis('SALES', 
                                               [0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    
    print(f"{'25th Percentile (Q1):':<25} {format_currency(percentiles[0.25])}")
    print(f"{'50th Percentile (Median):':<25} {format_currency(percentiles[0.50])}")
    print(f"{'75th Percentile (Q3):':<25} {format_currency(percentiles[0.75])}")
    print(f"{'90th Percentile:':<25} {format_currency(percentiles[0.90])}")
    print(f"{'95th Percentile:':<25} {format_currency(percentiles[0.95])}")
    print(f"{'99th Percentile:':<25} {format_currency(percentiles[0.99])}")
    
    iqr = percentiles[0.75] - percentiles[0.25]
    print(f"\n{'Interquartile Range:':<25} {format_currency(iqr)}")
    
    print(f"\nInsight: 50% of transactions fall between "
          f"{format_currency(percentiles[0.25])} and {format_currency(percentiles[0.75])}")


def query_10_product_performance(analyzer):
    """
    Query 10: Top Product Performance
    Demonstrates: Multi-column aggregation, ranking
    """
    print_subheader("QUERY 10: Top 10 Products by Performance")
    
    product_perf = analyzer.product_performance('PRODUCTCODE', top_n=10)
    
    print(product_perf.to_string())
    
    top_product = product_perf.index[0]
    top_revenue = product_perf.iloc[0]['SALES_sum']
    
    print(f"\n💡 Insight: Product '{top_product}' is the top performer with "
          f"{format_currency(top_revenue)} in total sales")


def query_11_country_analysis(analyzer):
    """
    Query 11: Sales by Country
    Demonstrates: Grouping with sorting, top N
    """
    print_subheader("QUERY 11: Top 10 Countries by Sales")
    
    country_sales = analyzer.group_by_single('COUNTRY', 'SALES', 'sum')
    top_countries = country_sales.head(10)
    
    print(top_countries.to_string())
    
    top_country = top_countries.index[0]
    top_country_sales = top_countries.iloc[0]
    
    print(f"\n💡 Insight: {top_country} leads with {format_currency(top_country_sales)}")


def query_12_multi_level_grouping(analyzer):
    """
    Query 12: Multi-Level Hierarchical Grouping
    Demonstrates: Complex grouping (Territory → Product Line)
    """
    print_subheader("QUERY 12: Sales by Territory and Product Line")
    
    multi_group = analyzer.multi_level_grouping(
        ['TERRITORY', 'PRODUCTLINE'], 
        'SALES', 
        'sum'
    )
    
    print(multi_group.head(15).to_string())
    
    print(f"\nInsight: Multi-dimensional analysis reveals performance patterns "
          f"across {len(multi_group)} territory-product combinations")


def query_13_lambda_filtering(analyzer):
    """
    Query 13: Advanced Filtering with Lambda
    Demonstrates: Lambda expressions, functional filtering
    """
    print_subheader("QUERY 13: Advanced Filtering (Large Deals > $5000)")
    
    # Filter for high-value sales using lambda
    high_value_sales = analyzer.filter_and_aggregate(
        filter_func=lambda df: df['SALES'] > 5000,
        agg_column='SALES',
        agg_func='sum'
    )
    
    # Count of high-value transactions
    df = analyzer.df
    high_value_count = len(df[df['SALES'] > 5000])
    total_count = len(df)
    percentage = (high_value_count / total_count) * 100
    
    print(f"{'Total high-value sales:':<30} {format_currency(high_value_sales)}")
    print(f"{'Number of transactions:':<30} {format_number(high_value_count)}")
    print(f"{'Percentage of all orders:':<30} {format_percentage(percentage/100)}")
    
    print(f"\nInsight: {format_percentage(percentage/100)} of orders are high-value (>${format_currency(5000)})")


def query_14_correlation_analysis(analyzer):
    """
    Query 14: Correlation Analysis
    Demonstrates: Statistical correlation between variables
    """
    print_subheader("🔗 QUERY 14: Correlation Analysis")
    
    # Correlation between quantity and price
    corr_qty_price = analyzer.correlation_analysis('QUANTITYORDERED', 'PRICEEACH')
    print(f"Quantity vs Price correlation: {corr_qty_price:.4f}")
    
    # Correlation between price and sales
    corr_price_sales = analyzer.correlation_analysis('PRICEEACH', 'SALES')
    print(f"Price vs Sales correlation:    {corr_price_sales:.4f}")
    
    # Interpret correlation
    if abs(corr_qty_price) > 0.7:
        strength = "strong"
    elif abs(corr_qty_price) > 0.4:
        strength = "moderate"
    else:
        strength = "weak"
    
    direction = "positive" if corr_qty_price > 0 else "negative"
    
    print(f"\nInsight: {strength.capitalize()} {direction} correlation between "
          f"quantity ordered and price")


def query_15_custom_aggregation(analyzer):
    """
    Query 15: Custom Aggregation Functions
    Demonstrates: Higher-order functions, custom lambda aggregations
    """
    print_subheader("QUERY 15: Custom Aggregation - Coefficient of Variation")
    
    # Coefficient of Variation (CV) = std / mean
    cv_sales = analyzer.custom_aggregation(
        'SALES',
        lambda x: (x.std() / x.mean()) if x.mean() != 0 else 0
    )
    
    cv_quantity = analyzer.custom_aggregation(
        'QUANTITYORDERED',
        lambda x: (x.std() / x.mean()) if x.mean() != 0 else 0
    )
    
    print(f"Coefficient of Variation (Sales):    {cv_sales:.4f}")
    print(f"Coefficient of Variation (Quantity): {cv_quantity:.4f}")
    
    print(f"\n Insight: CV of {cv_sales:.4f} indicates "
          f"{'high' if cv_sales > 0.5 else 'moderate'} variability in sales values")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_queries(analyzer):
    """
    Execute all analytical queries.
    
    Args:
        analyzer: SalesAnalyzer instance
    """
    print_header("COMPREHENSIVE SALES DATA ANALYSIS", '=')
    
    queries = [
        query_1_basic_metrics,
        query_2_revenue_by_product_line,
        query_3_sales_by_territory,
        query_4_top_customers,
        query_5_monthly_sales_trend,
        query_6_deal_size_distribution,
        query_7_high_value_transactions,
        query_8_statistical_analysis,
        query_9_percentile_analysis,
        query_10_product_performance,
        query_11_country_analysis,
        query_12_multi_level_grouping,
        query_13_lambda_filtering,
        query_14_correlation_analysis,
        query_15_custom_aggregation,
    ]
    
    for i, query_func in enumerate(queries, 1):
        try:
            query_func(analyzer)
        except Exception as e:
            print(f"\n✗ Error in {query_func.__name__}: {e}")
            continue
    
    print_header("ANALYSIS COMPLETE", '=')


def main():
    """
    Main application entry point.
    """
    # Capture output to save later
    output_buffer = io.StringIO()
    original_stdout = sys.stdout
    
    # Tee output to both console and buffer
    class TeeOutput:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, text):
            for stream in self.streams:
                stream.write(text)
        def flush(self):
            for stream in self.streams:
                stream.flush()
    
    sys.stdout = TeeOutput(original_stdout, output_buffer)
    
    # Print application header
    print_header("SALES DATA ANALYSIS APPLICATION")
    print(f"Assignment: Functional Programming with Data Streams")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {CSV_FILE_PATH}")
    
    # Step 1: Load Data
    print_header("STEP 1: DATA LOADING", '-')
    
    loader = DataLoader(CSV_FILE_PATH, config=SALES_DATA_CONFIG)
    
    try:
        # Try different encodings
        df = loader.load_data(encoding='latin-1')
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        sys.exit(1)
    
    # Step 2: Validate Data
    print_header("STEP 2: DATA VALIDATION", '-')
    
    is_valid = loader.validate_data()
    
    if not is_valid:
        print("\n Warning: Data validation failed. Proceeding with analysis anyway...")
    
    # Step 3: Display Data Summary
    print_header("STEP 3: DATA SUMMARY", '-')
    
    loader.display_summary(n_rows=5)
    
    # Step 4: Get Clean Data
    print_header("STEP 4: DATA CLEANING", '-')
    
    clean_df = loader.get_clean_data(drop_na_subset=['ORDERNUMBER', 'SALES'])
    print(f"Clean dataset prepared: {len(clean_df):,} rows ready for analysis")
    
    # Step 5: Create Analyzer
    print_header("STEP 5: INITIALIZE ANALYZER", '-')
    
    analyzer = SalesAnalyzer(clean_df)
    print("SalesAnalyzer initialized successfully")
    
    # Step 6: Functional Programming Demonstrations
    print_header("STEP 6: FUNCTIONAL PROGRAMMING DEMONSTRATIONS", '-')
    
    demonstrate_functional_programming(clean_df)
    
    # Step 7: Run All Analytical Queries
    print_header("STEP 7: ANALYTICAL QUERIES", '=')
    
    run_all_queries(analyzer)
    
    # Step 8: Summary
    print_header("EXECUTION SUMMARY", '=')
    
    print(f"Total records analyzed: {len(clean_df):,}")
    print(f"Total queries executed: 15")
    print(f"Analysis completed successfully")

    sys.stdout = original_stdout
    save_results_to_file(output_buffer.getvalue(), "analysis_results.txt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
