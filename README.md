# Multidimensional Data Structures for Movies Dataset

This project implements four multidimensional data structures (K-D Tree, Quadtree, Range Tree, and R-Tree) for efficiently indexing and querying a large movies dataset with 946K+ movies from 1900-2025.

## 🌟 Features

- **K-D Tree**: Efficient multidimensional point queries and nearest neighbor searches
- **Quadtree**: 2D spatial indexing for fast region queries
- **Range Tree**: Optimized multidimensional orthogonal range queries
- **R-Tree**: Bounding box indexing for spatial data
- **Comprehensive Testing**: Full test suite included
- **Performance Metrics**: Built-in benchmarking and comparison

## 📊 Dataset

The movies dataset includes **946,460 movies** (1900-2025) with the following numerical dimensions:
- `budget` - Movie budget in USD
- `revenue` - Total revenue in USD
- `runtime` - Movie duration in minutes
- `popularity` - Popularity score (TMDB-like metrics)
- `vote_average` - Average user rating
- `vote_count` - Number of votes received

See [README_DATASET.md](README_DATASET.md) for detailed dataset information.

## 🚀 Quick Start

### Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the dataset file `data_movies_clean.xlsx` is in the project directory.

### Running the Demo

Run the main demonstration program:
```bash
python main.py
```

This will:
- Load and preprocess the dataset (uses 50K sample by default for speed)
- Build all four tree structures
- Demonstrate example queries for each tree type
- Display performance metrics and comparisons

### Running Tests

Run the comprehensive test suite with a dataset sample:
```bash
python test_structures.py
```

Run unit tests (fast, no dataset required):
```bash
python test_unit.py
```

## 📁 Project Structure

```
/
├── kdtree.py           # K-D Tree implementation
├── quadtree.py         # Quadtree implementation  
├── range_tree.py       # Range Tree implementation
├── rtree.py            # R-Tree implementation
├── main.py             # Main demonstration program
├── test_structures.py  # Integration test suite
├── test_unit.py        # Unit test suite
├── examples.py         # Practical usage examples
├── utils.py            # Helper functions (data loading, preprocessing)
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── README_DATASET.md   # Dataset documentation
├── analysis_notebook.ipynb  # EDA notebook
└── data_movies_clean.xlsx   # Dataset file (not in repo)
```

## 🔧 Usage Examples

### K-D Tree: Nearest Neighbor Search
```python
from kdtree import KDTree
import numpy as np

# Build tree
kdtree = KDTree(dimensions=6)
kdtree.build(data)

# Find 5 nearest movies to a target movie
target_point = data[100]
neighbors = kdtree.nearest_neighbors(target_point, k=5)

# Range query
ranges = [
    (5_000_000, 20_000_000),  # budget: $5M-$20M
    (0, np.inf),               # revenue: any
    (0, np.inf),               # runtime: any
    (0, np.inf),               # popularity: any
    (7.0, 9.0),                # rating: 7-9
    (0, np.inf)                # vote_count: any
]
results = kdtree.range_query(ranges)
```

### Quadtree: 2D Spatial Queries
```python
from quadtree import Quadtree

# Build tree (using budget and revenue dimensions)
quadtree = Quadtree(x_dim=0, y_dim=1, capacity=50)
quadtree.build(data)

# Find movies in a 2D region
results = quadtree.query_range(
    x_range=(1_000_000, 10_000_000),    # budget: $1M-$10M
    y_range=(5_000_000, 50_000_000)     # revenue: $5M-$50M
)
```

### Range Tree: Multidimensional Range Queries
```python
from range_tree import SimpleRangeTree

# Build tree
range_tree = SimpleRangeTree(dimensions=6)
range_tree.build(data)

# Multi-dimensional range query
ranges = [
    (5_000_000, 50_000_000),  # budget: $5M-$50M
    (0, np.inf),               # revenue: any
    (90, 120),                 # runtime: 90-120 min
    (0, np.inf),               # popularity: any
    (7.0, 10.0),               # rating: 7-10
    (0, np.inf)                # vote_count: any
]
results = range_tree.range_query(ranges)
```

### R-Tree: Bounding Box Queries
```python
from rtree import SimpleRTree

# Build tree
rtree = SimpleRTree(dimensions=6)
rtree.build(data)

# Bounding box query
ranges = [
    (10_000_000, 100_000_000),  # budget: $10M-$100M
    (20_000_000, 500_000_000),  # revenue: $20M-$500M
    (0, np.inf),                 # runtime: any
    (0, np.inf),                 # popularity: any
    (6.0, 9.0),                  # rating: 6-9
    (0, np.inf)                  # vote_count: any
]
results = rtree.range_query(ranges)
```

## 📈 Performance

Typical performance on a 10,000 movie sample:

| Structure   | Build Time | Depth | Query Time |
|-------------|-----------|-------|------------|
| K-D Tree    | ~0.21s    | 14    | ~0.001s    |
| Quadtree    | ~0.06s    | 23    | ~0.001s    |
| Range Tree  | ~0.03s    | N/A   | ~0.002s    |
| R-Tree      | ~0.002s   | 1     | ~0.003s    |

*Note: Performance varies based on query complexity and dataset size*

## 🔍 Implementation Details

### K-D Tree
- Recursive construction with median-based splitting
- Efficient pruning during range queries
- Distance-based nearest neighbor search with early termination

### Quadtree
- Dynamic node splitting based on capacity threshold
- Spatial subdivision into four quadrants
- Fast region intersection tests

### Range Tree
- Simplified implementation using sorted arrays per dimension
- Set intersection for multi-dimensional queries
- Binary search for efficient range finding

### R-Tree
- Minimum Bounding Rectangle (MBR) based indexing
- Area-based insertion strategy
- Hierarchical spatial organization

## 🛠️ Technical Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- openpyxl >= 3.1.0

## 📝 Notes

- The dataset is large (946K rows). The main program uses a 50K sample by default for demonstration purposes.
- To use the full dataset, modify the `sample_size` variable in `main.py`.
- All implementations include proper error handling and edge case management.
- Code includes type hints and comprehensive docstrings.

## 🎯 Example Queries

The implementation demonstrates several real-world query scenarios:

1. **Budget and Rating Filter**: Find movies with budget $5M-$20M and rating 7-9
2. **Similar Movie Search**: Find 5 nearest movies to a given movie in feature space
3. **2D Regional Search**: Find movies in a budget vs revenue region
4. **Multi-constraint Search**: Movies with specific budget, runtime, and rating ranges
5. **Bounding Box Search**: Region-based queries across multiple dimensions

## 🧪 Testing

The project includes comprehensive test coverage:

**Integration Tests** (`test_structures.py`):
- Tests all structures with real dataset samples (10,000 movies)
- Validates tree construction and size
- Tests all query types
- Measures performance

**Unit Tests** (`test_unit.py`):
- Fast tests with synthetic data
- Tests core functionality without dataset
- Validates edge cases
- Tests utility functions

**Examples** (`examples.py`):
- Practical usage demonstrations
- Real-world query scenarios
- Shows best practices

Run all tests:
```bash
python test_unit.py && python test_structures.py
```

## 📚 References

- K-D Tree: Bentley, J. L. (1975). "Multidimensional binary search trees"
- Quadtree: Finkel, R. A., & Bentley, J. L. (1974). "Quad trees"
- Range Tree: Bentley, J. L. (1980). "Multidimensional divide-and-conquer"
- R-Tree: Guttman, A. (1984). "R-trees: A dynamic index structure"

## 👨‍💻 Author

Implementation by GitHub Copilot for the multidimensional-data-structures repository.

Dataset by **Mustafa Sayed Said**.
