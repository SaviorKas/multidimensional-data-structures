"""
Implementation of the specific project query from specification:
Find N-top most similar Production-Company-Names or Genre-Names where:
- release_date: 2000-2020
- popularity: 3-6
- vote_average: 3-5
- runtime: 30-60 minutes
- origin_country: 'US' or 'GB'
- original_language: 'en'
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import time

from combined_queries import (
    query_kdtree_lsh,
    query_quadtree_lsh,
    query_rangetree_lsh,
    query_rtree_lsh
)


def run_project_query(trees: Dict, data: np.ndarray, df: pd.DataFrame,
                     query_text: str = "Warner Bros",
                     text_attribute: str = "production_company_names",
                     n_top: int = 3) -> Dict:
    """
    Run the specific project query on all four tree structures and compare results.
    
    Args:
        trees: Dictionary of built tree structures
        data: Numerical data array
        df: Original DataFrame
        query_text: Text to search for (e.g., company name or genre)
        text_attribute: Which attribute to search ('production_company_names' or 'genre_names')
        n_top: Number of top results to return
        
    Returns:
        Dictionary containing results for each tree type
    """
    print("\n" + "=" * 80)
    print("  PROJECT SPECIFIC QUERY")
    print("=" * 80)
    print(f"\nQuery: Find {n_top} most similar {text_attribute} to '{query_text}'")
    print("\nFilters:")
    print("  - Release date: 2000-2020")
    print("  - Popularity: 3-6")
    print("  - Vote average: 3-5")
    print("  - Runtime: 30-60 minutes")
    print("  - Origin country: US or GB")
    print("  - Original language: en")
    
    # Define filters
    spatial_filters = {
        'popularity': (3, 6),
        'vote_average': (3, 5),
        'runtime': (30, 60)
    }
    
    metadata_filters = {
        'release_date': ('2000-01-01', '2020-12-31'),
        'origin_country': ['US', 'GB'],
        'original_language': 'en'
    }
    
    results = {}
    
    # Query with K-D Tree + LSH
    print("\n" + "-" * 80)
    print("Method 1: K-D Tree + LSH")
    print("-" * 80)
    try:
        indices, result_df, query_time = query_kdtree_lsh(
            trees['kdtree'], data, df,
            spatial_filters, text_attribute, query_text,
            metadata_filters, top_k=n_top
        )
        
        print(f"Query time: {query_time:.4f}s")
        print(f"Results found: {len(result_df)}")
        
        if len(result_df) > 0:
            print("\nTop results:")
            for i, (idx, row) in enumerate(result_df.iterrows(), 1):
                print(f"  {i}. {row['title']}")
                print(f"     {text_attribute}: {row[text_attribute]}")
                print(f"     Release: {row['release_date']}, Rating: {row['vote_average']:.1f}, "
                      f"Popularity: {row['popularity']:.1f}, Runtime: {row['runtime']:.0f}min")
        
        results['kdtree'] = {
            'indices': indices,
            'df': result_df,
            'time': query_time,
            'count': len(result_df)
        }
    except Exception as e:
        print(f"Error: {e}")
        results['kdtree'] = {'error': str(e)}
    
    # Query with Quadtree + LSH
    print("\n" + "-" * 80)
    print("Method 2: Quadtree + LSH")
    print("-" * 80)
    try:
        indices, result_df, query_time = query_quadtree_lsh(
            trees['quadtree'], data, df,
            spatial_filters, text_attribute, query_text,
            metadata_filters, top_k=n_top
        )
        
        print(f"Query time: {query_time:.4f}s")
        print(f"Results found: {len(result_df)}")
        
        if len(result_df) > 0:
            print("\nTop results:")
            for i, (idx, row) in enumerate(result_df.iterrows(), 1):
                print(f"  {i}. {row['title']}")
                print(f"     {text_attribute}: {row[text_attribute]}")
                print(f"     Release: {row['release_date']}, Rating: {row['vote_average']:.1f}, "
                      f"Popularity: {row['popularity']:.1f}, Runtime: {row['runtime']:.0f}min")
        
        results['quadtree'] = {
            'indices': indices,
            'df': result_df,
            'time': query_time,
            'count': len(result_df)
        }
    except Exception as e:
        print(f"Error: {e}")
        results['quadtree'] = {'error': str(e)}
    
    # Query with Range Tree + LSH
    print("\n" + "-" * 80)
    print("Method 3: Range Tree + LSH")
    print("-" * 80)
    try:
        indices, result_df, query_time = query_rangetree_lsh(
            trees['range_tree'], data, df,
            spatial_filters, text_attribute, query_text,
            metadata_filters, top_k=n_top
        )
        
        print(f"Query time: {query_time:.4f}s")
        print(f"Results found: {len(result_df)}")
        
        if len(result_df) > 0:
            print("\nTop results:")
            for i, (idx, row) in enumerate(result_df.iterrows(), 1):
                print(f"  {i}. {row['title']}")
                print(f"     {text_attribute}: {row[text_attribute]}")
                print(f"     Release: {row['release_date']}, Rating: {row['vote_average']:.1f}, "
                      f"Popularity: {row['popularity']:.1f}, Runtime: {row['runtime']:.0f}min")
        
        results['range_tree'] = {
            'indices': indices,
            'df': result_df,
            'time': query_time,
            'count': len(result_df)
        }
    except Exception as e:
        print(f"Error: {e}")
        results['range_tree'] = {'error': str(e)}
    
    # Query with R-Tree + LSH
    print("\n" + "-" * 80)
    print("Method 4: R-Tree + LSH")
    print("-" * 80)
    try:
        indices, result_df, query_time = query_rtree_lsh(
            trees['rtree'], data, df,
            spatial_filters, text_attribute, query_text,
            metadata_filters, top_k=n_top
        )
        
        print(f"Query time: {query_time:.4f}s")
        print(f"Results found: {len(result_df)}")
        
        if len(result_df) > 0:
            print("\nTop results:")
            for i, (idx, row) in enumerate(result_df.iterrows(), 1):
                print(f"  {i}. {row['title']}")
                print(f"     {text_attribute}: {row[text_attribute]}")
                print(f"     Release: {row['release_date']}, Rating: {row['vote_average']:.1f}, "
                      f"Popularity: {row['popularity']:.1f}, Runtime: {row['runtime']:.0f}min")
        
        results['rtree'] = {
            'indices': indices,
            'df': result_df,
            'time': query_time,
            'count': len(result_df)
        }
    except Exception as e:
        print(f"Error: {e}")
        results['rtree'] = {'error': str(e)}
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("  QUERY RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<25} {'Results':<15} {'Query Time':<15}")
    print("-" * 55)
    
    for method in ['kdtree', 'quadtree', 'range_tree', 'rtree']:
        if method in results and 'error' not in results[method]:
            count = results[method]['count']
            qtime = results[method]['time']
            method_name = f"{method.replace('_', ' ').title()} + LSH"
            print(f"{method_name:<25} {count:<15} {qtime:<15.4f}s")
    
    return results
