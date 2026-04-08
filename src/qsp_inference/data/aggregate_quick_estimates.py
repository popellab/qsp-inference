#!/usr/bin/env python3
"""
Aggregate quick parameter estimates from LLM batch results.

Reads an extraction input CSV to identify parameters and their definition hashes,
finds all quick estimate YAML files for each parameter/hash combination,
computes distribution parameters using the ranges from each estimate (not sample std),
and writes to a CSV file with distribution info.

Output CSV format:
    name, baseline_value, expected_value, units, distribution, dist_param1, dist_param2, n

Distribution calculation:
    - Uses ranges from each LLM estimate (assuming ±2σ coverage)
    - For lognormal: μ = mean(ln(values)), σ = mean((ln(b) - ln(a))/4) across ranges
    - For normal: mean = mean(values), std = mean((b - a)/4) across ranges
    - Falls back to sample std only if ranges are missing

Distribution types:
    - lognormal: dist_param1=μ (log mean), dist_param2=σ (log std), expected_value=exp(μ + σ²/2)
    - normal: dist_param1=mean, dist_param2=std, expected_value=mean
    - fixed: dist_param1=value, dist_param2=0, expected_value=value (single estimate without range)

Usage:
    python metadata/aggregate_quick_estimates.py extraction_input.csv quick_estimates_dir output_dir [model_name]

Example:
    python metadata/aggregate_quick_estimates.py \
        ../qsp-llm-workflows/batch_jobs/input_data/pdac_extraction_input_cda00473.csv \
        ../qsp-metadata-storage/quick-estimates \
        projects/pdac_2025/cache/ \
        immune_oncology_model_PDAC
"""

import sys
import csv
import yaml
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats


def extract_estimate_from_yaml(yaml_file: Path) -> Optional[Dict]:
    """
    Extract parameter estimate from quick estimate YAML file.

    Args:
        yaml_file: Path to YAML file

    Returns:
        Dict with 'value', 'range', 'units', or None if extraction fails
    """
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Quick estimates have parameter_estimate field (singular)
        if 'parameter_estimate' not in data:
            return None

        estimate = data['parameter_estimate']

        # Extract value and range
        value = estimate.get('value')
        range_list = estimate.get('range')
        units = estimate.get('units', '') or data.get('parameter_units', '')

        if value is None:
            return None

        # Parse value (might be string with scientific notation)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return None

        # Parse range (should be a list [min, max])
        range_tuple = None
        if range_list and isinstance(range_list, list) and len(range_list) == 2:
            try:
                range_tuple = (float(range_list[0]), float(range_list[1]))
            except (ValueError, TypeError):
                pass

        return {
            'value': value,
            'range': range_tuple,
            'units': units
        }
    except Exception as e:
        print(f"Warning: Could not parse {yaml_file.name}: {e}")
        return None


def find_quick_estimates(param_name: str, cancer_type: str, definition_hash: str,
                         estimates_dir: Path) -> List[Path]:
    """
    Find all quick estimate YAML files for a given parameter/hash.

    Pattern: {param_name}_{cancer_type}_{definition_hash}_deriv*.yaml

    Args:
        param_name: Parameter name
        cancer_type: Cancer type
        definition_hash: Definition/context hash
        estimates_dir: Directory containing YAML files

    Returns:
        List of matching YAML file paths
    """
    pattern = f"{param_name}_{cancer_type}_{definition_hash}_deriv*.yaml"
    return list(estimates_dir.glob(pattern))


def aggregate_estimates(estimates: List[Dict]) -> Dict:
    """
    Aggregate multiple parameter estimates by averaging.

    Uses ranges from LLM estimates (rather than sample std) to determine uncertainty,
    assuming each range represents approximately ±2σ (95% coverage).

    Args:
        estimates: List of estimate dicts with 'value', 'range', 'units'

    Returns:
        Dict with mean_value, distribution info based on ranges
    """
    if not estimates:
        return None

    # Extract values and ranges
    values = np.array([e['value'] for e in estimates])
    ranges = [e['range'] for e in estimates if e['range'] is not None]
    units = estimates[0]['units']  # Use first non-empty units
    for e in estimates:
        if e['units']:
            units = e['units']
            break

    # Check if all values are positive (use lognormal approach)
    use_lognormal = np.all(values > 0)

    # Single estimate case
    if len(values) == 1:
        mean_value = values[0]

        # If we have a range for the single estimate, use it to infer sigma
        if len(ranges) == 1:
            if use_lognormal:
                # Lognormal: range represents [μ - 2σ, μ + 2σ] in log space
                log_lower = np.log(ranges[0][0])
                log_upper = np.log(ranges[0][1])
                log_mean = (log_lower + log_upper) / 2
                log_std = (log_upper - log_lower) / 4  # Assuming ±2σ coverage

                distribution = 'lognormal'
                dist_param1 = log_mean
                dist_param2 = log_std
            else:
                # Normal: range represents [mean - 2σ, mean + 2σ]
                mean_val = (ranges[0][0] + ranges[0][1]) / 2
                std_val = (ranges[0][1] - ranges[0][0]) / 4

                distribution = 'normal'
                dist_param1 = mean_val
                dist_param2 = std_val
        else:
            # No range - use fixed
            distribution = 'fixed'
            dist_param1 = mean_value
            dist_param2 = 0.0

    # Multiple estimates - use ranges to estimate sigma
    elif len(ranges) > 0:
        if use_lognormal:
            # Lognormal approach
            # 1. Compute geometric mean of point estimates
            log_values = np.log(values)
            log_mean = np.mean(log_values)
            mean_value = np.exp(log_mean)

            # 2. Estimate sigma from ranges
            # Each range [a, b] implies: ln(a) ≈ μ - 2σ, ln(b) ≈ μ + 2σ
            # So: σ ≈ (ln(b) - ln(a)) / 4
            log_sigmas = []
            for r in ranges:
                if r[0] > 0 and r[1] > 0:
                    log_sigma = (np.log(r[1]) - np.log(r[0])) / 4
                    log_sigmas.append(log_sigma)

            # Average the sigma estimates
            log_std = np.mean(log_sigmas) if log_sigmas else 0.3  # Default if no valid ranges

            distribution = 'lognormal'
            dist_param1 = log_mean
            dist_param2 = log_std
        else:
            # Normal approach
            # 1. Compute mean of point estimates
            mean_value = np.mean(values)

            # 2. Estimate sigma from ranges
            # Each range [a, b] implies: a ≈ mean - 2σ, b ≈ mean + 2σ
            # So: σ ≈ (b - a) / 4
            sigmas = [(r[1] - r[0]) / 4 for r in ranges]
            std_val = np.mean(sigmas)

            distribution = 'normal'
            dist_param1 = mean_value
            dist_param2 = std_val

    # Multiple estimates but no ranges - fall back to sample std
    else:
        if use_lognormal:
            log_values = np.log(values)
            log_mean = np.mean(log_values)
            log_std = np.std(log_values, ddof=1)
            mean_value = np.exp(log_mean)

            distribution = 'lognormal'
            dist_param1 = log_mean
            dist_param2 = log_std
        else:
            mean_value = np.mean(values)
            std_val = np.std(values, ddof=1)

            distribution = 'normal'
            dist_param1 = mean_value
            dist_param2 = std_val

    # Calculate expected value from distribution parameters
    if distribution == 'lognormal':
        # For lognormal: E[X] = exp(μ + σ²/2)
        expected_value = np.exp(dist_param1 + dist_param2**2 / 2)
    elif distribution == 'normal':
        # For normal: E[X] = mean
        expected_value = dist_param1
    else:  # fixed
        expected_value = dist_param1

    return {
        'mean_value': mean_value,
        'expected_value': expected_value,
        'units': units,
        'is_lognormal': use_lognormal,
        'raw_values': values.tolist(),
        'distribution': distribution,
        'dist_param1': dist_param1,
        'dist_param2': dist_param2
    }


def format_scientific(value: float) -> str:
    """Format value in scientific notation."""
    return f"{value:.2e}"


def generate_lognormal_from_cv(baseline_value: float, cv: float) -> Tuple[float, float]:
    """
    Generate lognormal distribution parameters from baseline value and coefficient of variation.

    For a lognormal distribution with target expected value E[X] = baseline_value and CV:
        CV = sqrt(exp(σ²) - 1)
        => σ = sqrt(ln(1 + CV²))
        => μ = ln(E[X]) - σ²/2

    Args:
        baseline_value: Target expected value (must be positive)
        cv: Coefficient of variation (e.g., 0.3 for 30%)

    Returns:
        Tuple of (mu, sigma) for lognormal distribution
    """
    if baseline_value <= 0:
        raise ValueError(f"Baseline value must be positive for lognormal: {baseline_value}")

    # Calculate sigma from CV
    sigma = np.sqrt(np.log(1 + cv**2))

    # Calculate mu to achieve target expected value
    mu = np.log(baseline_value) - sigma**2 / 2

    return (mu, sigma)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate quick parameter estimates from LLM batch results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard aggregation from LLM estimates:
  python aggregate_quick_estimates.py \\
    extraction_input.csv quick_estimates_dir output_dir immune_oncology_model_PDAC

  # Override all estimates with CV-based distributions around baseline values:
  python aggregate_quick_estimates.py \\
    extraction_input.csv quick_estimates_dir output_dir immune_oncology_model_PDAC \\
    --cv-override 0.3
        """
    )

    parser.add_argument('extraction_csv', type=Path,
                        help='Input CSV with parameter names and definition hashes')
    parser.add_argument('estimates_dir', type=Path,
                        help='Directory containing quick estimate YAML files')
    parser.add_argument('output_dir', type=Path,
                        help='Directory for output CSV')
    parser.add_argument('model_name', nargs='?',
                        help='(Optional) Model function name for baseline values')
    parser.add_argument('--cv-override', type=float, metavar='CV',
                        help='Override all estimates with lognormal distributions around baseline values using specified coefficient of variation (e.g., 0.3 for 30%%)')

    args = parser.parse_args()

    extraction_csv = args.extraction_csv
    estimates_dir = args.estimates_dir
    output_dir = args.output_dir
    model_name = args.model_name
    cv_override = args.cv_override

    # Validate inputs
    if not extraction_csv.exists():
        print(f"Error: Extraction CSV not found: {extraction_csv}")
        sys.exit(1)

    if not estimates_dir.exists():
        print(f"Error: Estimates directory not found: {estimates_dir}")
        sys.exit(1)

    # CV override requires model_name for baseline values
    if cv_override is not None:
        if not model_name:
            print("Error: --cv-override requires model_name argument to get baseline values")
            sys.exit(1)
        if cv_override <= 0:
            print(f"Error: CV must be positive, got {cv_override}")
            sys.exit(1)

    # Extract hash from extraction CSV filename
    # Pattern: {cancer_type}_extraction_input_{hash}.csv
    match = re.search(r'extraction_input_([a-f0-9]+)\.csv', extraction_csv.name)
    if not match:
        print(f"Error: Could not extract hash from filename: {extraction_csv.name}")
        sys.exit(1)

    extraction_hash = match.group(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output CSV filename
    output_csv = output_dir / f"pdac_quick_estimates_{extraction_hash}.csv"

    print(f"Reading extraction CSV: {extraction_csv.name}")
    if cv_override is not None:
        print(f"CV-override mode: generating distributions with CV={cv_override}")
    else:
        print(f"Looking for estimates in: {estimates_dir}")
    print(f"Output will be written to: {output_csv}")
    print()

    # Export baseline parameter values if model_name is provided (needed for cv_override)
    baseline_values = {}
    if model_name:
        print()
        print("Exporting baseline parameter values from model...")

        import tempfile
        import subprocess

        # Create temp file for baseline values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            baseline_csv = tmp.name

        # Call MATLAB to export baseline values
        matlab_cmd = (
            f"export_baseline_parameters('{model_name}', '{baseline_csv}'); "
            f"exit;"
        )

        result = subprocess.run(
            ['matlab', '-batch', matlab_cmd],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"Warning: Failed to export baseline values from MATLAB")
            print(f"  {result.stderr}")
            if cv_override is not None:
                print("Error: Cannot use --cv-override without baseline values")
                sys.exit(1)
        else:
            # Read baseline values
            try:
                with open(baseline_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        baseline_values[row['parameter_id']] = float(row['baseline_value'])

                print(f"  ✓ Loaded {len(baseline_values)} baseline parameter values")
            except Exception as e:
                print(f"Warning: Could not read baseline values: {e}")
                if cv_override is not None:
                    print("Error: Cannot use --cv-override without baseline values")
                    sys.exit(1)
            finally:
                # Clean up temp file
                Path(baseline_csv).unlink(missing_ok=True)

        print()

    # Read extraction CSV and process parameters
    aggregated_results = []
    total_params = 0
    found_params = 0

    with open(extraction_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_params += 1
            param_name = row['parameter_name']
            cancer_type = row['cancer_type']
            definition_hash = row['definition_hash']

            # CV override mode: generate distributions from baseline values
            if cv_override is not None:
                if param_name not in baseline_values:
                    print(f"  ⚠ No baseline value for {param_name} (skipping)")
                    continue

                baseline_val = baseline_values[param_name]

                if baseline_val <= 0:
                    print(f"  ⚠ Non-positive baseline for {param_name}: {baseline_val} (skipping)")
                    continue

                try:
                    mu, sigma = generate_lognormal_from_cv(baseline_val, cv_override)

                    # Calculate expected value
                    expected_val = np.exp(mu + sigma**2 / 2)

                    found_params += 1
                    print(f"  ✓ {param_name}: CV={cv_override} → lognormal(μ={format_scientific(mu)}, σ={format_scientific(sigma)})")

                    aggregated_results.append({
                        'name': param_name,
                        'expected_value': expected_val,
                        'units': '',  # Units not available in extraction CSV
                        'distribution': 'lognormal',
                        'dist_param1': mu,
                        'dist_param2': sigma,
                        'num_estimates': 1
                    })
                except ValueError as e:
                    print(f"  ⚠ Error generating distribution for {param_name}: {e}")
                    continue

            # Standard LLM estimate aggregation mode
            else:
                # Find all quick estimate files for this parameter
                yaml_files = find_quick_estimates(param_name, cancer_type, definition_hash, estimates_dir)

                if not yaml_files:
                    print(f"  ⚠ No estimates found for {param_name} (hash: {definition_hash})")
                    continue

                # Extract estimates from each file
                estimates = []
                for yaml_file in yaml_files:
                    estimate = extract_estimate_from_yaml(yaml_file)
                    if estimate:
                        estimates.append(estimate)

                if not estimates:
                    print(f"  ⚠ Could not parse estimates for {param_name}")
                    continue

                # Aggregate estimates
                agg_result = aggregate_estimates(estimates)

                if agg_result is None:
                    continue

                found_params += 1
                dist_type = agg_result['distribution']
                if dist_type == 'lognormal':
                    # For lognormal, show μ ± σ in log-space
                    print(f"  ✓ {param_name}: {len(estimates)} estimate(s) → {dist_type}(μ={format_scientific(agg_result['dist_param1'])}, σ={format_scientific(agg_result['dist_param2'])}) {agg_result['units']}")
                elif dist_type == 'normal':
                    print(f"  ✓ {param_name}: {len(estimates)} estimate(s) → {dist_type}(mean={format_scientific(agg_result['dist_param1'])}, std={format_scientific(agg_result['dist_param2'])}) {agg_result['units']}")
                else:  # fixed
                    print(f"  ✓ {param_name}: {len(estimates)} estimate(s) → {dist_type}(value={format_scientific(agg_result['dist_param1'])}) {agg_result['units']}")

                # Add to results
                aggregated_results.append({
                    'name': param_name,
                    'expected_value': agg_result['expected_value'],
                    'units': agg_result['units'],
                    'distribution': agg_result['distribution'],
                    'dist_param1': agg_result['dist_param1'],
                    'dist_param2': agg_result['dist_param2'],
                    'num_estimates': len(estimates)
                })

    # Write output CSV
    if aggregated_results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Always include baseline_value column (populated only if baseline_values dict exists)
            writer.writerow(['name', 'baseline_value', 'expected_value', 'units', 'distribution', 'dist_param1', 'dist_param2', 'n'])

            for result in aggregated_results:
                # Get baseline value if available
                baseline_val = ''
                if baseline_values:
                    baseline_val_raw = baseline_values.get(result['name'])
                    if baseline_val_raw is not None:
                        baseline_val = format_scientific(baseline_val_raw)

                # Write row
                writer.writerow([
                    result['name'],
                    baseline_val,
                    format_scientific(result['expected_value']),
                    result['units'],
                    result['distribution'],
                    format_scientific(result['dist_param1']),
                    format_scientific(result['dist_param2']),
                    result['num_estimates']
                ])

        print()
        print("=" * 70)
        print("SUCCESS")
        print("=" * 70)
        print(f"Output file: {output_csv}")
        print(f"Parameters with estimates: {found_params}/{total_params}")
        print(f"Total estimates aggregated: {sum(r['num_estimates'] for r in aggregated_results)}")
        print()
        print("Distribution summary:")
        dist_counts = {}
        for r in aggregated_results:
            dist = r['distribution']
            dist_counts[dist] = dist_counts.get(dist, 0) + 1
        for dist, count in sorted(dist_counts.items()):
            print(f"  {dist}: {count}")
    else:
        print()
        print("No estimates found to aggregate")
        sys.exit(1)


if __name__ == "__main__":
    main()
