"""
Data Loader Module
Handles CSV reading, validation, and data cleaning.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any


class DataLoader:
    """
    Generic CSV data loader with configurable validation rules.
    """
    
    def __init__(self, filepath: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data loader with file path and optional configuration.
        
        Args:
            filepath (str): Path to the CSV file
            config (dict, optional): Configuration dictionary with validation rules
                - required_columns: List of columns that must exist
                - date_columns: List of columns to parse as dates
                - numeric_columns: List of columns to convert to numeric
                - categorical_columns: List of columns to convert to category type
                - unique_key: Column(s) to check for duplicates
                - non_negative_columns: Columns that shouldn't have negative values
        """
        self.filepath = filepath
        self.df = None
        self.config = config or {}
        self.data_quality_report = {}
    
    def load_data(self, encoding: str = 'latin-1', **kwargs) -> pd.DataFrame:
        """
        Load CSV file into pandas DataFrame.
        
        Args:
            encoding (str): File encoding
            **kwargs: Additional arguments to pass to pd.read_csv()
        
        Returns:
            pandas.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            pd.errors.EmptyDataError: If CSV is empty
        """
        try:
            # Checks if file exists
            if not Path(self.filepath).exists():
                raise FileNotFoundError(f"CSV file not found: {self.filepath}")
            
            # Loads CSV
            self.df = pd.read_csv(
                self.filepath,
                encoding=encoding,
                low_memory=False,
                **kwargs
            )
            
            print(f"Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Perform data type conversions if configured
            if self.config:
                self._convert_data_types()
            
            return self.df
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: CSV file is empty")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error loading data: {e}")
            sys.exit(1)
    
    def _convert_data_types(self):
        """
        Converts columns to appropriate data types based on configuration.
        """
        if self.df is None:
            return
        
        # Converting date columns
        date_columns = self.config.get('date_columns', [])
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"Converted {col} to datetime")
        
        # Converting numeric columns
        numeric_columns = self.config.get('numeric_columns', [])
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                print(f"Converted {col} to numeric")
        
        # Converting categorical columns
        categorical_columns = self.config.get('categorical_columns', [])
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
                print(f"Converted {col} to category")
    
    def validate_data(self) -> bool:
        """
        Performs data quality validation checks based on configuration.
        
        Returns:
            bool: True if data passes validation, False otherwise
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return False
        
        print("\n" + "="*70)
        print("DATA VALIDATION REPORT")
        print("="*70)
        
        is_valid = True
        
        # Check 1: Required columns exist (if configured)
        required_columns = self.config.get('required_columns', [])
        if required_columns:
            print("\n--- Required Columns Check ---")
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                is_valid = False
            else:
                print(f"All {len(required_columns)} required columns present")
        
        # Check 2: Missing values analysis
        print("\n--- Missing Values Analysis ---")
        missing_counts = self.df.isnull().sum()
        missing_with_nulls = missing_counts[missing_counts > 0]
        
        if len(missing_with_nulls) > 0:
            print(f"Found missing values in {len(missing_with_nulls)} columns:")
            for col, count in missing_with_nulls.items():
                pct = (count / len(self.df)) * 100
                print(f"  • {col}: {count} ({pct:.2f}%)")
            self.data_quality_report['missing_values'] = missing_with_nulls.to_dict()
        else:
            print("No missing values found in any column")
        
        # Check 3: Duplicate rows check (if unique key configured)
        unique_key = self.config.get('unique_key')
        if unique_key:
            print("\n--- Duplicate Records Check ---")
            if isinstance(unique_key, str):
                unique_key = [unique_key]
            
            # Check if all key columns exist
            missing_keys = [col for col in unique_key if col not in self.df.columns]
            if missing_keys:
                print(f"Cannot check duplicates - missing key columns: {missing_keys}")
            else:
                duplicates = self.df.duplicated(subset=unique_key).sum()
                print(f"Duplicate records based on {unique_key}: {duplicates}")
                
                if duplicates > 0:
                    print(f"Warning: Found {duplicates} duplicate records")
                    self.data_quality_report['duplicates'] = duplicates
                else:
                    print("No duplicate records found")
        
        # Check 4: Negative values check (if configured)
        non_negative_columns = self.config.get('non_negative_columns', [])
        if non_negative_columns:
            print("\n--- Negative Values Check ---")
            found_negatives = False
            
            for col in non_negative_columns:
                if col in self.df.columns:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        negative_count = (self.df[col] < 0).sum()
                        if negative_count > 0:
                            print(f"{col}: {negative_count} negative values found")
                            found_negatives = True
                            is_valid = False
                        else:
                            print(f"{col}: No negative values")
            
            if not found_negatives:
                print("All specified columns have non-negative values")
        
        # Check 5: Date range (for date columns)
        date_columns = self.config.get('date_columns', [])
        if date_columns:
            print("\n--- Date Range Information ---")
            for col in date_columns:
                if col in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    min_date = self.df[col].min()
                    max_date = self.df[col].max()
                    print(f"{col}: {min_date.date()} to {max_date.date()}")
                    self.data_quality_report[f'{col}_range'] = (str(min_date.date()), str(max_date.date()))
        
        # Check 6: Custom validation function (if provided)
        custom_validator = self.config.get('custom_validator')
        if custom_validator and callable(custom_validator):
            print("\n--- Custom Validation ---")
            try:
                custom_result = custom_validator(self.df)
                if not custom_result:
                    print("Custom validation failed")
                    is_valid = False
                else:
                    print("Custom validation passed")
            except Exception as e:
                print(f"Custom validation error: {e}")
                is_valid = False
        
        print("\n" + "="*70)
        
        if is_valid:
            print("DATA VALIDATION PASSED")
        else:
            print("DATA VALIDATION FAILED - Please review issues above")
        
        print("="*70 + "\n")
        
        return is_valid
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Gets comprehensive information about the loaded data.
        
        Returns:
            dict: Dictionary containing various data statistics
        """
        if self.df is None:
            return {}
        
        info = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.df.isnull().sum().to_dict(),
        }
        
        return info
    
    def display_summary(self, n_rows: int = 5):
        """
        Display comprehensive summary of the data.
        
        Args:
            n_rows (int): Number of sample rows to display (default: 5)
        """
        if self.df is None:
            print("✗ No data loaded. Call load_data() first.")
            return
        
        print("\n" + "="*70)
        print("DATA SUMMARY")
        print("="*70)
        
        # Basic info
        print(f"\nDataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        # Column info
        print(f"\n--- Columns ({len(self.df.columns)}) ---")
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            non_null = self.df[col].count()
            print(f"{i:2d}. {col:30s} | Type: {str(dtype):15s} | Non-Null: {non_null:,}")
        
        # Sample data
        print(f"\n--- First {n_rows} Rows ---")
        print(self.df.head(n_rows).to_string())
        
        # Numeric statistics
        numeric_df = self.df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            print("\n--- Numeric Column Statistics ---")
            print(numeric_df.describe().to_string())
        
        # Categorical columns value counts (top 5 categories)
        categorical_df = self.df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            print("\n--- Categorical Columns (Top 5 Values Each) ---")
            for col in categorical_df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts().head(5).to_string())
        
        # Memory usage
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"\n--- Memory Usage ---")
        print(f"Total: {memory_mb:.2f} MB")
        
        print("\n" + "="*70 + "\n")
    
    def get_clean_data(self, drop_na_subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Return cleaned DataFrame ready for analysis.
        
        Args:
            drop_na_subset (list, optional): Columns to check for NaN before dropping rows
        
        Returns:
            pandas.DataFrame: Cleaned data
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return None
        
        # Create a copy to avoid modifying original
        clean_df = self.df.copy()
        
        # Remove rows with missing values in specified columns
        if drop_na_subset:
            initial_rows = len(clean_df)
            clean_df = clean_df.dropna(subset=drop_na_subset)
            rows_removed = initial_rows - len(clean_df)
            
            if rows_removed > 0:
                print(f"ℹ Removed {rows_removed} rows with missing values in {drop_na_subset}")
        
        return clean_df
    
    def save_data(self, output_path: str, index: bool = False):
        """
        Save DataFrame to CSV file.
        
        Args:
            output_path (str): Path where CSV should be saved
            index (bool): Whether to write row indices (default: False)
        """
        if self.df is None:
            print("✗ No data loaded. Call load_data() first.")
            return
        
        try:
            self.df.to_csv(output_path, index=index)
            print(f"Data saved to {output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")


# Convenience function for quick loading
def load_csv(filepath: str, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Quick function to load and validate CSV data.
    
    Args:
        filepath (str): Path to CSV file
        config (dict, optional): Configuration for validation
        
    Returns:
        pandas.DataFrame: Loaded and validated data
    """
    loader = DataLoader(filepath, config)
    df = loader.load_data()
    loader.validate_data()
    return df

