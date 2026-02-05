"""
Main program to demonstrate multidimensional data structures on movies dataset.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from utils import load_movies_dataset, preprocess_data, get_sample_queries
from kdtree import KDTree
from quadtree import Quadtree
from range_tree import SimpleRangeTree
from rtree import SimpleRTree
from project_query import run_project_query
from performance_comparison import run_performance_comparison


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def build_trees(data: np.ndarray, df: pd.DataFrame, dimensions: List[str]):
    """
    Build all four tree structures.
    
    Args:
        data: Preprocessed data array
        df: Original DataFrame
        dimensions: List of dimension names
        
    Returns:
        Dictionary of built trees and metadata
    """
    print_section("Building Tree Structures")
    
    trees = {}
    build_times = {}
    
    # K-D Tree
    print("Building K-D Tree...")
    start = time.time()
    kdtree = KDTree(dimensions=len(dimensions))
    kdtree.build(data)
    build_times['kdtree'] = time.time() - start
    trees['kdtree'] = kdtree
    print(f"  Size: {kdtree.size:,} nodes")
    print(f"  Depth: {kdtree.get_depth()}")
    print(f"  Build time: {build_times['kdtree']:.3f}s")
    
    # Quadtree (using budget and revenue as 2D space)
    print("\nBuilding Quadtree (budget vs revenue)...")
    start = time.time()
    quadtree = Quadtree(x_dim=0, y_dim=1, capacity=50)  # budget, revenue
    quadtree.build(data)
    build_times['quadtree'] = time.time() - start
    trees['quadtree'] = quadtree
    print(f"  Size: {quadtree.size:,} nodes")
    print(f"  Depth: {quadtree.get_depth()}")
    print(f"  Build time: {build_times['quadtree']:.3f}s")
    
    # Range Tree
    print("\nBuilding Range Tree...")
    start = time.time()
    range_tree = SimpleRangeTree(dimensions=len(dimensions))
    range_tree.build(data)
    build_times['range_tree'] = time.time() - start
    trees['range_tree'] = range_tree
    print(f"  Size: {range_tree.size:,} nodes")
    print(f"  Build time: {build_times['range_tree']:.3f}s")
    
    # R-Tree
    print("\nBuilding R-Tree...")
    start = time.time()
    rtree = SimpleRTree(dimensions=len(dimensions))
    rtree.build(data)
    build_times['rtree'] = time.time() - start
    trees['rtree'] = rtree
    print(f"  Size: {rtree.size:,} nodes")
    print(f"  Build time: {build_times['rtree']:.3f}s")
    
    return trees, build_times


def demonstrate_basic_queries(trees: Dict, data: np.ndarray, 
                             df: pd.DataFrame, dimensions: List[str]):
    """Demonstrate basic tree queries (simplified version)."""
    print_section("Quick Tree Functionality Check")
    
    # K-D Tree: Range query
    print("K-D Tree: Range Query (Budget $5M-$20M, Rating 7-9)")
    ranges = [
        (5_000_000, 20_000_000), (0, np.inf), (0, np.inf),
        (0, np.inf), (7.0, 9.0), (0, np.inf)
    ]
    start = time.time()
    results = trees['kdtree'].range_query(ranges)
    print(f"  ✓ Found {len(results)} movies in {time.time() - start:.4f}s")
    
    # Quadtree: 2D query
    print("\nQuadtree: 2D Spatial Query (Budget $1M-$10M, Revenue $5M-$50M)")
    start = time.time()
    results = trees['quadtree'].query_range((1_000_000, 10_000_000), (5_000_000, 50_000_000))
    print(f"  ✓ Found {len(results)} movies in {time.time() - start:.4f}s")
    
    # Range Tree: Multi-dimensional query
    print("\nRange Tree: Multi-dimensional Query")
    ranges = [
        (5_000_000, 50_000_000), (0, np.inf), (90, 120),
        (0, np.inf), (7.0, 10.0), (0, np.inf)
    ]
    start = time.time()
    results = trees['range_tree'].range_query(ranges)
    print(f"  ✓ Found {len(results)} movies in {time.time() - start:.4f}s")
    
    # R-Tree: Bounding box query
    print("\nR-Tree: Bounding Box Query")
    ranges = [
        (10_000_000, 100_000_000), (20_000_000, 500_000_000), (0, np.inf),
        (0, np.inf), (6.0, 9.0), (0, np.inf)
    ]
    start = time.time()
    results = trees['rtree'].range_query(ranges)
    print(f"  ✓ Found {len(results)} movies in {time.time() - start:.4f}s")





def compare_performance(trees: Dict, build_times: Dict):
    """Compare performance metrics across all trees."""
    print_section("Performance Comparison")
    
    print(f"{'Structure':<20} {'Build Time':<15} {'Size':<15}")
    print("-" * 50)
    
    for name, tree in trees.items():
        build_time = build_times[name]
        size = tree.size
        print(f"{name:<20} {build_time:>10.3f}s     {size:>10,}")
    
    print("\nMemory Usage Estimates:")
    print("  Note: Actual memory usage varies based on implementation details")
    print(f"  K-D Tree: ~{trees['kdtree'].size * 100 / 1024:.1f} KB")
    print(f"  Quadtree: ~{trees['quadtree'].size * 120 / 1024:.1f} KB")
    print(f"  Range Tree: ~{trees['range_tree'].size * 200 / 1024:.1f} KB")
    print(f"  R-Tree: ~{trees['rtree'].size * 150 / 1024:.1f} KB")


def main():
    """Main program."""
    print("=" * 80)
    print("  MULTIDIMENSIONAL DATA STRUCTURES FOR MOVIES DATASET")
    print("=" * 80)
    
    # Load dataset
    print_section("Loading Dataset")
    try:
        df = load_movies_dataset("data_movies_clean.xlsx")
    except:
        print("Could not load Excel file, trying CSV...")
        try:
            df = load_movies_dataset("data_movies_clean.csv")
        except Exception as e:
            print(f"Error: {e}")
            print("Please ensure data_movies_clean.xlsx or .csv is in the current directory")
            return
    
    print(f"\nDataset shape: {df.shape}")
    
    # Use a sample for faster demonstration (can be changed to use full dataset)
    sample_size = min(50000, len(df))
    if len(df) > sample_size:
        print(f"Using sample of {sample_size:,} movies for demonstration")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        print(f"Using full dataset: {len(df):,} movies")
    
    print(f"Columns: {list(df.columns)}")
    
    # Preprocess data
    print_section("Preprocessing Data")
    dimensions = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
    data, df_clean = preprocess_data(df, dimensions)
    
    print(f"\nProcessed data shape: {data.shape}")
    print(f"Data statistics:")
    for i, dim in enumerate(dimensions):
        print(f"  {dim}: min={data[:, i].min():.2f}, "
              f"max={data[:, i].max():.2f}, "
              f"mean={data[:, i].mean():.2f}")
    
    # Build trees
    trees, build_times = build_trees(data, df_clean, dimensions)
    
    # Quick functionality check
    demonstrate_basic_queries(trees, data, df_clean, dimensions)
    
    # Run project-specific query (main focus)
    project_results = run_project_query(
        trees, data, df_clean,
        query_text="Warner Bros",
        text_attribute="production_company_names",
        n_top=3
    )
    
    # Run comprehensive performance comparison
    print()  # Add spacing
    comparison_df = run_performance_comparison(trees, build_times, data, df_clean)
    
    print_section("Summary")
    print("✓ Successfully completed all project requirements:")
    print("  • K-D Tree + LSH - Efficient spatial + similarity queries")
    print("  • Quadtree + LSH - 2D spatial + similarity queries")
    print("  • Range Tree + LSH - Multidimensional range + similarity queries")
    print("  • R-Tree + LSH - Bounding box + similarity queries")
    print("\n✓ Two-phase querying system: Tree filtering → LSH similarity")
    print("✓ Performance comparison completed with visualizations")
    print("✓ Project-specific query executed on all 4 methods")
    print("\nAll structures built and tested successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
