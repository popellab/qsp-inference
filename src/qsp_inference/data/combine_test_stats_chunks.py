#!/usr/bin/env python3
"""
Combine test statistics chunks on HPC using pandas.

This script runs on HPC to properly combine chunk CSV files, handling
duplicate headers in params files.

Usage:
    python combine_test_stats_chunks.py <chunks_dir>
"""

import sys
from pathlib import Path
import pandas as pd


def combine_chunks(chunks_dir: str) -> None:
    """
    Combine test stats and params chunks using pandas.

    Args:
        chunks_dir: Directory containing chunk_*_test_stats.csv and chunk_*_params.csv
    """
    chunks_path = Path(chunks_dir)

    if not chunks_path.exists():
        print(f"Error: Directory not found: {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all test stats chunks (sorted)
    test_stats_chunks = sorted(chunks_path.glob("chunk_*_test_stats.csv"))

    if not test_stats_chunks:
        print(f"Error: No test stats chunks found in {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(test_stats_chunks)} test stats chunks")

    # Combine test stats chunks (no headers, just numeric data)
    test_stats_dfs = []
    for chunk_file in test_stats_chunks:
        df = pd.read_csv(chunk_file, header=None)
        test_stats_dfs.append(df)

    test_stats_combined = pd.concat(test_stats_dfs, ignore_index=True)

    # Save combined test stats
    combined_test_stats_file = chunks_path / "combined_test_stats.csv"
    test_stats_combined.to_csv(combined_test_stats_file, header=False, index=False)
    print(f"✓ Combined test stats: {test_stats_combined.shape[0]} rows, {test_stats_combined.shape[1]} cols")
    print(f"  Saved to: {combined_test_stats_file}")

    # Find params chunks
    params_chunks = sorted(chunks_path.glob("chunk_*_params.csv"))

    if params_chunks:
        print(f"Found {len(params_chunks)} params chunks")

        # Combine params chunks (have headers - pandas handles automatically)
        params_dfs = []
        for chunk_file in params_chunks:
            df = pd.read_csv(chunk_file)
            params_dfs.append(df)

        params_combined = pd.concat(params_dfs, ignore_index=True)

        # Save combined params
        combined_params_file = chunks_path / "combined_params.csv"
        params_combined.to_csv(combined_params_file, index=False)
        print(f"✓ Combined params: {params_combined.shape[0]} rows, {params_combined.shape[1]} cols")
        print(f"  Saved to: {combined_params_file}")
    else:
        print("No params chunks found (older format)")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python combine_test_stats_chunks.py <chunks_dir>")
        sys.exit(1)

    chunks_dir = sys.argv[1]
    combine_chunks(chunks_dir)
