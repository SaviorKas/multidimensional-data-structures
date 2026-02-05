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


def demonstrate_kdtree_queries(kdtree: KDTree, data: np.ndarray, 
                               df: pd.DataFrame, dimensions: List[str]):
    """Demonstrate K-D Tree queries."""
    print_section("K-D Tree Queries")
    
    # Range query
    print("Query 1: Range Query")
    print("  Find movies with budget $5M-$20M and rating 7-9")
    
    # Create range based on dimension indices
    ranges = [
        (5_000_000, 20_000_000),     # budget
        (0, np.inf),                  # revenue (any)
        (0, np.inf),                  # runtime (any)
        (0, np.inf),                  # popularity (any)
        (7.0, 9.0),                   # vote_average
        (0, np.inf)                   # vote_count (any)
    ]
    
    start = time.time()
    results = kdtree.range_query(ranges)
    query_time = time.time() - start
    
    print(f"  Found {len(results)} movies in {query_time:.4f}s")
    
    # Show sample results
    if results:
        sample_indices = results[:5]
        print("\n  Sample results:")
        for idx in sample_indices:
            movie = df.iloc[idx]
            print(f"    - {movie['title']}: "
                  f"Budget=${movie['budget']:,.0f}, "
                  f"Rating={movie['vote_average']:.1f}")
    
    # Nearest neighbor query
    print("\n\nQuery 2: Nearest Neighbor Search")
    print("  Find 5 movies similar to a random movie")
    
    random_idx = np.random.randint(0, len(data))
    target_movie = df.iloc[random_idx]
    target_point = data[random_idx]
    
    print(f"  Target: {target_movie['title']}")
    print(f"    Budget=${target_movie['budget']:,.0f}, "
          f"Revenue=${target_movie['revenue']:,.0f}, "
          f"Rating={target_movie['vote_average']:.1f}")
    
    start = time.time()
    neighbors = kdtree.nearest_neighbors(target_point, k=6)  # +1 to exclude target itself
    query_time = time.time() - start
    
    print(f"\n  Found {len(neighbors)} neighbors in {query_time:.4f}s")
    print("  Similar movies:")
    
    for i, (idx, dist) in enumerate(neighbors[1:6]):  # Skip first (target itself)
        movie = df.iloc[idx]
        print(f"    {i+1}. {movie['title']} (distance: {dist:.2f})")
        print(f"       Budget=${movie['budget']:,.0f}, "
              f"Revenue=${movie['revenue']:,.0f}, "
              f"Rating={movie['vote_average']:.1f}")


def demonstrate_quadtree_queries(quadtree: Quadtree, data: np.ndarray, 
                                 df: pd.DataFrame, dimensions: List[str]):
    """Demonstrate Quadtree queries."""
    print_section("Quadtree Queries (Budget vs Revenue)")
    
    print("Query: Spatial Range Query")
    print("  Find movies with budget $1M-$10M and revenue $5M-$50M")
    
    start = time.time()
    results = quadtree.query_range(
        x_range=(1_000_000, 10_000_000),      # budget
        y_range=(5_000_000, 50_000_000)       # revenue
    )
    query_time = time.time() - start
    
    print(f"  Found {len(results)} movies in {query_time:.4f}s")
    
    # Show sample results
    if results:
        sample_indices = results[:5]
        print("\n  Sample results:")
        for idx in sample_indices:
            movie = df.iloc[idx]
            print(f"    - {movie['title']}: "
                  f"Budget=${movie['budget']:,.0f}, "
                  f"Revenue=${movie['revenue']:,.0f}")


def demonstrate_rangetree_queries(range_tree: SimpleRangeTree, data: np.ndarray,
                                   df: pd.DataFrame, dimensions: List[str]):
    """Demonstrate Range Tree queries."""
    print_section("Range Tree Queries")
    
    print("Query: Multidimensional Range Query")
    print("  Find movies with:")
    print("    - Budget: $5M-$50M")
    print("    - Runtime: 90-120 minutes")
    print("    - Rating: 7-10")
    
    ranges = [
        (5_000_000, 50_000_000),     # budget
        (0, np.inf),                  # revenue (any)
        (90, 120),                    # runtime
        (0, np.inf),                  # popularity (any)
        (7.0, 10.0),                  # vote_average
        (0, np.inf)                   # vote_count (any)
    ]
    
    start = time.time()
    results = range_tree.range_query(ranges)
    query_time = time.time() - start
    
    print(f"  Found {len(results)} movies in {query_time:.4f}s")
    
    # Show sample results
    if results:
        sample_indices = results[:5]
        print("\n  Sample results:")
        for idx in sample_indices:
            movie = df.iloc[idx]
            print(f"    - {movie['title']}: "
                  f"Budget=${movie['budget']:,.0f}, "
                  f"Runtime={movie['runtime']:.0f}min, "
                  f"Rating={movie['vote_average']:.1f}")


def demonstrate_rtree_queries(rtree: SimpleRTree, data: np.ndarray,
                              df: pd.DataFrame, dimensions: List[str]):
    """Demonstrate R-Tree queries."""
    print_section("R-Tree Queries")
    
    print("Query: Bounding Box Range Query")
    print("  Find movies with:")
    print("    - Budget: $10M-$100M")
    print("    - Revenue: $20M-$500M")
    print("    - Rating: 6-9")
    
    ranges = [
        (10_000_000, 100_000_000),   # budget
        (20_000_000, 500_000_000),   # revenue
        (0, np.inf),                  # runtime (any)
        (0, np.inf),                  # popularity (any)
        (6.0, 9.0),                   # vote_average
        (0, np.inf)                   # vote_count (any)
    ]
    
    start = time.time()
    results = rtree.range_query(ranges)
    query_time = time.time() - start
    
    print(f"  Found {len(results)} movies in {query_time:.4f}s")
    
    # Show sample results
    if results:
        sample_indices = results[:5]
        print("\n  Sample results:")
        for idx in sample_indices:
            movie = df.iloc[idx]
            print(f"    - {movie['title']}: "
                  f"Budget=${movie['budget']:,.0f}, "
                  f"Revenue=${movie['revenue']:,.0f}, "
                  f"Rating={movie['vote_average']:.1f}")


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
    
    # Demonstrate queries
    demonstrate_kdtree_queries(trees['kdtree'], data, df_clean, dimensions)
    demonstrate_quadtree_queries(trees['quadtree'], data, df_clean, dimensions)
    demonstrate_rangetree_queries(trees['range_tree'], data, df_clean, dimensions)
    demonstrate_rtree_queries(trees['rtree'], data, df_clean, dimensions)
    
    # Performance comparison
    compare_performance(trees, build_times)
    
    print_section("Summary")
    print("Successfully demonstrated all four data structures:")
    print("  ✓ K-D Tree - Efficient nearest neighbor and range queries")
    print("  ✓ Quadtree - 2D spatial indexing")
    print("  ✓ Range Tree - Multidimensional range queries")
    print("  ✓ R-Tree - Bounding box queries")
    print("\nAll structures built and tested successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
