#!/usr/bin/env python3
"""Random clustering baseline module for omnibenchmark.

This module assigns random cluster labels to observations as a baseline method.
It generates predictions for k-2, k-1, k, k+1, k+2 clusters where k is the
true number of clusters.
"""

import argparse
import gzip
import os
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Random clustering baseline - assigns random cluster labels"
    )
    parser.add_argument("--data.matrix", dest="data_matrix", required=True,
                        help="Path to the data matrix to cluster (gzipped)")
    parser.add_argument("--data.true_labels", dest="data_true_labels", required=True,
                        help="Path to the true labels/partitioning (gzipped)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    parser.add_argument("--name", required=True,
                        help="Name of the dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def random_clustering(n_samples, k, rng):
    """Assign random cluster labels to samples.

    Args:
        n_samples: Number of samples to assign labels to
        k: Number of clusters
        rng: numpy random generator

    Returns:
        Array of random cluster labels (1-indexed)
    """
    return rng.integers(1, k + 1, size=n_samples)


def main():
    args = parse_args()

    # Set random seed for reproducibility
    rng = np.random.default_rng(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data matrix to get number of samples
    data = np.loadtxt(args.data_matrix, ndmin=2)
    n_samples = data.shape[0]

    # Load true labels to determine k (true number of clusters)
    true_labels = np.loadtxt(args.data_true_labels, ndmin=1)
    k_true = int(max(true_labels))

    # Generate k range: k-2, k-1, k, k+1, k+2 (minimum k=2)
    k_range = [max(2, k_true + offset) for offset in range(-2, 3)]

    # Generate random cluster assignments for each k value
    results = {}
    for k in k_range:
        labels = random_clustering(n_samples, k, rng)
        results[k] = labels

    # Write output in the expected format
    output_file = os.path.join(args.output_dir, f"{args.name}_ks_range.labels.gz")

    with gzip.open(output_file, 'wt') as f:
        # Write header
        header = ",".join([f"k={k}" for k in k_range])
        f.write(header + "\n")

        # Write labels for each sample
        for i in range(n_samples):
            row = ",".join([str(results[k][i]) for k in k_range])
            f.write(row + "\n")

    print(f"Random clustering completed:")
    print(f"  Samples: {n_samples}")
    print(f"  True k: {k_true}")
    print(f"  K range: {k_range}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
