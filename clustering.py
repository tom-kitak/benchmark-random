#!/usr/bin/env python3
"""Clustering module for benchmarking pipeline."""

import argparse
import os
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Clustering module")
    parser.add_argument("--data.matrix", dest="data_matrix", required=True,
                        help="Path to the data matrix to cluster")
    parser.add_argument("--data.true_labels", dest="data_true_labels", required=True,
                        help="Path to the true labels/partitioning")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    parser.add_argument("--name", required=True,
                        help="Name of the module")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Module: {args.name}")
    print(f"Data matrix: {args.data_matrix}")
    print(f"True labels: {args.data_true_labels}")
    print(f"Output dir: {args.output_dir}")
    print(f"Seed: {args.seed}")

    # TODO: Add clustering logic here


if __name__ == "__main__":
    main()
