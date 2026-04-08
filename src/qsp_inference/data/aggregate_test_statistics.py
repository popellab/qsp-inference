#!/usr/bin/env python3
"""
Aggregate test statistics from LLM batch results.

Reads a test statistic input CSV to identify test statistics and their context hashes,
finds all test statistic YAML files for each test_statistic_id/hash combination,
pools the expected distributions, and writes to a CSV file for validation.

Usage:
    python metadata/aggregate_test_statistics.py test_statistic_input.csv test_statistics_dir output_dir

Example:
    python metadata/aggregate_test_statistics.py \
        ../qsp-llm-workflows/batch_jobs/input_data/test_statistic_input_gvax_entinostat_abc123.csv \
        ../qsp-metadata-storage/test_statistics \
        ../qsp-metadata-storage/scratch/
"""

import sys
import csv
import yaml
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _load_yaml_safe(yaml_file: Path) -> Optional[Dict]:
    """
    Safely load YAML file with error handling.

    Args:
        yaml_file: Path to YAML file

    Returns:
        Parsed YAML dict or None if loading fails
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load {yaml_file.name}: {e}")
        return None


def extract_distribution_from_yaml(yaml_file: Path) -> Optional[Dict]:
    """
    Extract expected distribution from test statistic YAML file.

    Args:
        yaml_file: Path to YAML file

    Returns:
        Dict with 'mean', 'variance', 'ci95', 'units', or None if extraction fails
    """
    data = _load_yaml_safe(yaml_file)
    if data is None:
        return None

    # Test statistics have expected_distribution field
    if 'expected_distribution' not in data:
        return None

    dist = data['expected_distribution']

    # Extract distribution parameters
    mean = dist.get('mean')
    variance = dist.get('variance')
    ci95 = dist.get('ci95')
    units = dist.get('units', '')

    if mean is None or variance is None:
        return None

    # Parse values (might be strings)
    if isinstance(mean, str):
        try:
            mean = float(mean)
        except ValueError:
            return None

    if isinstance(variance, str):
        try:
            variance = float(variance)
        except ValueError:
            return None

    # Parse CI95 (should be a list [lower, upper])
    ci95_tuple = None
    if ci95 and isinstance(ci95, list) and len(ci95) == 2:
        try:
            ci95_tuple = (float(ci95[0]), float(ci95[1]))
        except (ValueError, TypeError):
            pass

    # Extract author_year from data_sources for reporting
    author_year = None
    if 'data_sources' in data and isinstance(data['data_sources'], dict):
        # Get first data source key as author_year
        author_years = list(data['data_sources'].keys())
        if author_years:
            author_year = author_years[0]

    return {
        'mean': mean,
        'variance': variance,
        'ci95': ci95_tuple,
        'units': units,
        'author_year': author_year
    }


def extract_model_output_code(yaml_file: Path) -> Optional[str]:
    """
    Extract model_output.code from test statistic YAML file.

    Args:
        yaml_file: Path to YAML file

    Returns:
        MATLAB code string or None if extraction fails
    """
    data = _load_yaml_safe(yaml_file)
    if data is None:
        return None

    if 'model_output' not in data:
        return None

    if 'code' not in data['model_output']:
        return None

    return data['model_output']['code']


def extract_required_species(yaml_file: Path) -> Optional[str]:
    """
    Extract required_species from test statistic YAML file.

    Args:
        yaml_file: Path to YAML file

    Returns:
        Comma-separated species list or None if extraction fails
    """
    data = _load_yaml_safe(yaml_file)
    if data is None:
        return None

    if 'required_species' not in data:
        return None

    required_species = data['required_species']

    # Handle different formats
    if isinstance(required_species, str):
        # If it's a string like '[V_T.mAPC,V_T.APC]', parse it
        required_species = required_species.strip('[]')
        # Split by comma and clean up
        species_list = [s.strip() for s in required_species.split(',')]
    elif isinstance(required_species, list):
        # Already a list
        species_list = required_species
    else:
        return None

    return ','.join(species_list)


def find_test_statistics(test_statistic_id: str, context_hash: str,
                         stats_dir: Path) -> List[Path]:
    """
    Find all test statistic YAML files for a given test_statistic_id/hash.

    Pattern: {test_statistic_id}_{cancer_type}_{context_hash}.yaml

    Args:
        test_statistic_id: Test statistic identifier
        context_hash: Context hash
        stats_dir: Directory containing YAML files

    Returns:
        List of matching YAML file paths
    """
    pattern = f"{test_statistic_id}_*_{context_hash}.yaml"
    return list(stats_dir.glob(pattern))


def pool_distributions(distributions: List[Dict]) -> Dict:
    """
    Pool multiple test statistic distributions.

    Uses inverse-variance weighting to combine distributions from multiple sources.
    Assumes all distributions are for the same underlying quantity.

    Args:
        distributions: List of distribution dicts with 'mean', 'variance', 'units'

    Returns:
        Dict with pooled_mean, pooled_variance, pooled_ci95, units
    """
    if not distributions:
        return None

    # Extract means and variances
    means = np.array([d['mean'] for d in distributions])
    variances = np.array([d['variance'] for d in distributions])
    units = distributions[0]['units']  # Use first units

    # Check for valid variances (positive)
    valid_idx = variances > 0
    if not np.any(valid_idx):
        # No valid variances, just average means
        pooled_mean = np.mean(means)
        pooled_variance = np.var(means, ddof=1) if len(means) > 1 else 0.0
        pooled_ci95 = None
    else:
        # Use inverse-variance weighting
        valid_means = means[valid_idx]
        valid_variances = variances[valid_idx]

        # Weights are inverse variances
        weights = 1.0 / valid_variances
        weight_sum = np.sum(weights)

        # Pooled mean
        pooled_mean = np.sum(weights * valid_means) / weight_sum

        # Pooled variance
        pooled_variance = 1.0 / weight_sum

        # Pooled 95% CI (using normal approximation)
        se = np.sqrt(pooled_variance)
        z_critical = 1.96  # 95% CI
        pooled_ci95 = (pooled_mean - z_critical * se, pooled_mean + z_critical * se)

    # Collect CI95 ranges for union
    ci95_ranges = [d['ci95'] for d in distributions if d['ci95'] is not None]

    # Also compute union of all CI95 intervals
    union_ci95 = None
    if ci95_ranges:
        all_lowers = [r[0] for r in ci95_ranges]
        all_uppers = [r[1] for r in ci95_ranges]
        union_ci95 = (min(all_lowers), max(all_uppers))

    return {
        'pooled_mean': pooled_mean,
        'pooled_variance': pooled_variance,
        'pooled_ci95': pooled_ci95,
        'union_ci95': union_ci95,
        'units': units,
        'raw_means': means.tolist(),
        'raw_variances': variances.tolist()
    }


def format_value(value: float) -> str:
    """Format value for CSV output."""
    if abs(value) < 1e-3 or abs(value) > 1e3:
        return f"{value:.2e}"
    else:
        return f"{value:.4g}"


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate test statistics from LLM batch results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python metadata/data/aggregate_test_statistics.py \\
    ../qsp-llm-workflows/batch_jobs/input_data/test_statistic_input_gvax_entinostat_abc123.csv \\
    ../qsp-metadata-storage/test_statistics \\
    ../qsp-metadata-storage/scratch/
        """
    )

    parser.add_argument('input_csv', type=Path,
                        help='Input CSV with test_statistic_id and context_hash')
    parser.add_argument('stats_dir', type=Path,
                        help='Directory containing test statistic YAML files')
    parser.add_argument('output_dir', type=Path,
                        help='Directory for output CSV')

    args = parser.parse_args()

    input_csv = args.input_csv
    stats_dir = args.stats_dir
    output_dir = args.output_dir

    # Validate inputs
    if not input_csv.exists():
        print(f"Error: Input CSV not found: {input_csv}")
        sys.exit(1)

    if not stats_dir.exists():
        print(f"Error: Test statistics directory not found: {stats_dir}")
        sys.exit(1)

    # Extract scenario_id and hash from input CSV filename
    # Pattern: test_statistic_input_{scenario_id}_{hash}.csv
    match = re.search(r'test_statistic_input_(.+?)_([a-f0-9]+)\.csv', input_csv.name)
    if not match:
        print(f"Error: Could not extract scenario_id and hash from filename: {input_csv.name}")
        sys.exit(1)

    scenario_id = match.group(1)
    context_hash = match.group(2)

    # Read input CSV to get cancer_type and process test statistics
    pooled_results = []
    total_stats = 0
    found_stats = 0
    cancer_type = None

    print(f"Reading input CSV: {input_csv.name}")
    print(f"Looking for test statistics in: {stats_dir}")
    print()

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Get cancer_type from first row
            if cancer_type is None:
                cancer_type = row.get('cancer_type', 'unknown')
            total_stats += 1
            test_statistic_id = row['test_statistic_id']
            row_context_hash = row.get('context_hash', context_hash)

            # Find all test statistic files for this id/hash
            yaml_files = find_test_statistics(test_statistic_id, row_context_hash, stats_dir)

            if not yaml_files:
                print(f"  ⚠ No test statistics found for {test_statistic_id} (hash: {row_context_hash})")
                continue

            # Extract distributions from all files
            distributions = []
            for yaml_file in yaml_files:
                dist = extract_distribution_from_yaml(yaml_file)
                if dist:
                    distributions.append(dist)

            # Extract code and required species from first valid file
            model_output_code = None
            required_species = None
            for yaml_file in yaml_files:
                if model_output_code is None:
                    model_output_code = extract_model_output_code(yaml_file)
                if required_species is None:
                    required_species = extract_required_species(yaml_file)
                # Break once we have both
                if model_output_code and required_species:
                    break

            if not distributions:
                print(f"  ⚠ Could not parse distributions for {test_statistic_id}")
                continue

            if not model_output_code:
                print(f"  ⚠ No model_output_code found for {test_statistic_id}")
                continue

            if not required_species:
                print(f"  ⚠ No required_species found for {test_statistic_id}")
                continue

            # Pool distributions
            pooled_result = pool_distributions(distributions)

            if pooled_result is None:
                continue

            found_stats += 1
            print(f"  ✓ {test_statistic_id}: {len(distributions)} distribution(s) → μ={format_value(pooled_result['pooled_mean'])}, σ²={format_value(pooled_result['pooled_variance'])} {pooled_result['units']}")

            # List sources
            author_years = [d.get('author_year', 'unknown') for d in distributions]
            print(f"      Sources: {', '.join(author_years)}")

            # Add to results
            pooled_results.append({
                'test_statistic_id': test_statistic_id,
                'pooled_mean': pooled_result['pooled_mean'],
                'pooled_variance': pooled_result['pooled_variance'],
                'pooled_ci95': pooled_result['pooled_ci95'],
                'union_ci95': pooled_result['union_ci95'],
                'units': pooled_result['units'],
                'num_sources': len(distributions),
                'raw_means': pooled_result['raw_means'],
                'raw_variances': pooled_result['raw_variances'],
                'required_species': required_species,
                'model_output_code': model_output_code
            })

    # Create output directory and filename
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename with cancer_type and scenario_id
    output_csv = output_dir / f"{cancer_type}_{scenario_id}_test_statistics_{context_hash}.csv"

    print(f"Output will be written to: {output_csv}")

    # Write output CSV
    if pooled_results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['test_statistic_id', 'mean', 'variance', 'ci95_lower', 'ci95_upper',
                           'union_ci95_lower', 'union_ci95_upper', 'units', 'n_sources',
                           'raw_means', 'raw_variances', 'required_species', 'model_output_code'])

            for result in pooled_results:
                ci_lower = format_value(result['pooled_ci95'][0]) if result['pooled_ci95'] else ''
                ci_upper = format_value(result['pooled_ci95'][1]) if result['pooled_ci95'] else ''

                union_lower = format_value(result['union_ci95'][0]) if result['union_ci95'] else ''
                union_upper = format_value(result['union_ci95'][1]) if result['union_ci95'] else ''

                # Format raw values as comma-separated lists
                raw_means_str = ', '.join([format_value(v) for v in result['raw_means']])
                raw_vars_str = ', '.join([format_value(v) for v in result['raw_variances']])

                writer.writerow([
                    result['test_statistic_id'],
                    format_value(result['pooled_mean']),
                    format_value(result['pooled_variance']),
                    ci_lower,
                    ci_upper,
                    union_lower,
                    union_upper,
                    result['units'],
                    result['num_sources'],
                    raw_means_str,
                    raw_vars_str,
                    result['required_species'],
                    result['model_output_code']
                ])

        print()
        print("=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Output file: {output_csv}")
        print(f"Test statistics with data: {found_stats}/{total_stats}")
        print(f"Total distributions pooled: {sum(r['num_sources'] for r in pooled_results)}")
    else:
        print()
        print("No test statistics found to aggregate")
        sys.exit(1)


if __name__ == "__main__":
    main()
