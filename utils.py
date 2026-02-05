"""
Utility functions for loading and preprocessing the movies dataset.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def load_movies_dataset(filepath: str = "data_movies_clean.xlsx") -> pd.DataFrame:
    """
    Load the movies dataset from Excel or CSV file.
    
    Args:
        filepath: Path to the dataset file
        
    Returns:
        DataFrame containing the movies dataset
    """
    try:
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath, engine='openpyxl')
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        print(f"Loaded {len(df)} movies from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def preprocess_data(df: pd.DataFrame, 
                    dimensions: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Preprocess the movies dataset for tree structures.
    
    Args:
        df: Raw movies DataFrame
        dimensions: List of numerical columns to use (default: all numerical dimensions)
        
    Returns:
        Tuple of (processed_data_array, cleaned_dataframe)
    """
    if dimensions is None:
        dimensions = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle missing values - fill with median for numerical columns
    for dim in dimensions:
        if dim in df_clean.columns:
            median_val = df_clean[dim].median()
            df_clean[dim] = df_clean[dim].fillna(median_val)
            # Replace any infinity values
            df_clean[dim] = df_clean[dim].replace([np.inf, -np.inf], median_val)
    
    # Filter out rows with all zeros in dimensions
    mask = (df_clean[dimensions] != 0).any(axis=1)
    df_clean = df_clean[mask].reset_index(drop=True)
    
    # Extract numerical data as numpy array
    data_array = df_clean[dimensions].values.astype(np.float64)
    
    print(f"Preprocessed {len(df_clean)} valid movies")
    print(f"Dimensions used: {dimensions}")
    
    return data_array, df_clean


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data to [0, 1] range for each dimension.
    
    Args:
        data: Raw data array
        
    Returns:
        Tuple of (normalized_data, min_values, max_values)
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    
    normalized = (data - min_vals) / range_vals
    
    return normalized, min_vals, max_vals


def denormalize_data(normalized_data: np.ndarray, 
                    min_vals: np.ndarray, 
                    max_vals: np.ndarray) -> np.ndarray:
    """
    Convert normalized data back to original scale.
    
    Args:
        normalized_data: Normalized data array
        min_vals: Minimum values for each dimension
        max_vals: Maximum values for each dimension
        
    Returns:
        Denormalized data array
    """
    range_vals = max_vals - min_vals
    return normalized_data * range_vals + min_vals


def get_sample_queries() -> dict:
    """
    Return example query configurations for demonstration.
    
    Returns:
        Dictionary of example queries
    """
    queries = {
        'budget_range': {
            'description': 'Movies with budget $5M-$20M and rating 7-9',
            'budget': (5_000_000, 20_000_000),
            'vote_average': (7.0, 9.0)
        },
        'popular_movies': {
            'description': 'Popular movies with high revenue',
            'popularity': (50, 500),
            'revenue': (100_000_000, 1_000_000_000)
        },
        'runtime_range': {
            'description': 'Movies between 90-120 minutes with good ratings',
            'runtime': (90, 120),
            'vote_average': (6.5, 10.0)
        }
    }
    return queries
