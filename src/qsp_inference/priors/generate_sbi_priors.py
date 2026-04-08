#!/usr/bin/env python3
"""
Generate SBI priors from parameter list and model baseline values.

Takes a list of parameter names, instantiates the model to get baseline values,
and generates lognormal distribution parameters based on a specified coefficient
of variation (CV).

Output CSV format:
    name, expected_value, units, distribution, dist_param1, dist_param2, lower_bound, upper_bound

    Note: lower_bound and upper_bound columns are generated as empty by default.
          Edit the CSV manually to add truncation bounds if needed.

Distribution type:
    - lognormal: For positive parameters
        dist_param1=μ (log mean), dist_param2=σ (log std)
        expected_value=exp(μ + σ²/2)
        CV relationship: σ = sqrt(ln(1 + CV²))

Truncation bounds (optional):
    - lower_bound: Minimum value (inclusive) in original parameter space
    - upper_bound: Maximum value (inclusive) in original parameter space
    - Leave empty for unbounded distributions

Usage:
    python metadata/priors/generate_sbi_priors.py param_list.txt model_name output.csv [--cv CV]

Example:
    python metadata/priors/generate_sbi_priors.py \\
        projects/pdac_2025/cache/pdac_baseline_params.txt \\
        immune_oncology_model_PDAC \\
        projects/pdac_2025/cache/pdac_sbi_priors.csv \\
        --cv 2.0
"""

import csv
import argparse
import numpy as np
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Tuple


def _generate_lognormal_params(baseline_value: float, cv: float) -> Tuple[float, float, float]:
    """
    Generate lognormal distribution parameters from baseline value and CV.

    For a lognormal distribution with target expected value E[X] = baseline_value and CV:
        CV = sqrt(exp(σ²) - 1)
        => σ = sqrt(ln(1 + CV²))
        => μ = ln(E[X]) - σ²/2

    Args:
        baseline_value: Target expected value (must be positive)
        cv: Coefficient of variation (e.g., 2.0 for 200%)

    Returns:
        Tuple of (expected_value, mu, sigma) for lognormal distribution

    Raises:
        ValueError: If baseline_value <= 0
    """
    if baseline_value <= 0:
        raise ValueError(f"Baseline value must be positive for lognormal: {baseline_value}")

    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(baseline_value) - sigma**2 / 2
    expected_value = np.exp(mu + sigma**2 / 2)

    return (expected_value, mu, sigma)


def _format_scientific(value: float) -> str:
    """Format value in scientific notation."""
    return f"{value:.2e}"


def _export_baseline_values(model_name: str) -> Dict[str, Tuple[float, str]]:
    """
    Export baseline parameter values from MATLAB model.

    Args:
        model_name: Name of MATLAB model function

    Returns:
        Dict mapping parameter names to (value, units) tuples
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        baseline_csv = tmp.name

    try:
        matlab_cmd = f"export_baseline_parameters('{model_name}', '{baseline_csv}'); exit;"
        result = subprocess.run(['matlab', '-batch', matlab_cmd], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to export baseline values from MATLAB:\n{result.stderr}")

        baseline_values = {}
        with open(baseline_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                baseline_values[row['parameter_id']] = (float(row['baseline_value']), row['units'])

        return baseline_values

    finally:
        Path(baseline_csv).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate SBI priors from parameter list and model baseline values.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('param_list', type=Path,
                        help='Text file with parameter names (one per line)')
    parser.add_argument('model_name', type=str,
                        help='MATLAB model function name (e.g., immune_oncology_model_PDAC)')
    parser.add_argument('output_csv', type=Path,
                        help='Output CSV file path')
    parser.add_argument('--cv', type=float, default=2.0,
                        help='Coefficient of variation for priors (default: 2.0 = 200%%)')
    args = parser.parse_args()

    # Validate inputs
    if not args.param_list.exists():
        raise FileNotFoundError(f"Parameter list file not found: {args.param_list}")
    if args.cv <= 0:
        raise ValueError(f"CV must be positive, got {args.cv}")

    print(f"Generating SBI priors (CV={args.cv})...")
    print(f"  Parameter list: {args.param_list}")
    print(f"  Model: {args.model_name}")
    print(f"  Output: {args.output_csv}")

    # Read parameter names
    param_names = []
    with open(args.param_list, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                param_names.append(line)

    print(f"  {len(param_names)} parameters to process")

    # Export baseline values from model
    baseline_values = _export_baseline_values(args.model_name)
    print(f"  {len(baseline_values)} baseline values exported from model")

    # Generate priors for each parameter
    results = []
    skipped = []

    for param_name in param_names:
        if param_name not in baseline_values:
            skipped.append((param_name, "Not found in model"))
            continue

        baseline_val, units = baseline_values[param_name]

        try:
            expected_val, mu, sigma = _generate_lognormal_params(baseline_val, args.cv)
            results.append({
                'name': param_name,
                'expected_value': expected_val,
                'units': units,
                'distribution': 'lognormal',
                'dist_param1': mu,
                'dist_param2': sigma
            })
        except ValueError as e:
            skipped.append((param_name, str(e)))

    print(f"  ✓ Generated {len(results)} lognormal priors")
    if skipped:
        print(f"  ⚠ Skipped {len(skipped)} parameters:")
        for param, reason in skipped:
            print(f"    - {param}: {reason}")

    # Write output CSV
    if not results:
        raise ValueError("No priors generated - all parameters were skipped")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'expected_value', 'units', 'distribution', 'dist_param1', 'dist_param2', 'lower_bound', 'upper_bound'])
        for result in results:
            writer.writerow([
                result['name'],
                _format_scientific(result['expected_value']),
                result['units'],
                result['distribution'],
                _format_scientific(result['dist_param1']),
                _format_scientific(result['dist_param2']),
                '',  # lower_bound - empty by default, edit manually if needed
                ''   # upper_bound - empty by default, edit manually if needed
            ])

    sigma = np.sqrt(np.log(1 + args.cv**2))
    print(f"\n✓ Saved: {args.output_csv}")
    print(f"  {len(results)}/{len(param_names)} parameters, σ={sigma:.3f} for CV={args.cv}")


if __name__ == "__main__":
    main()
