"""SBI inference tools and diagnostics.

Heavy optional deps (torch, sbi, matplotlib-heavy paths) are grouped and
imported under try/except so submodules with lighter footprints — in
particular ``qsp_inference.inference.restriction``, which only needs
sklearn — can be imported without the sbi extra installed. If an optional
dep is missing, the symbols it provides are omitted from the package
namespace; import the submodule directly for a clear ImportError message.
"""

# Lightweight — no torch required.
from qsp_inference.inference.restriction import (
    RestrictionClassifier,
    train_restriction_classifier,
    sample_restricted,
)

# Heavy — require torch/sbi/etc. Guard so partial installs (e.g. CI without
# the sbi extra) can still import the restriction module.
try:
    from qsp_inference.inference.diagnostics import (
        sbi_recovery,
        sbi_z_score_contraction,
        compute_per_param_calibration,
        sbi_calibration_ecdf,
        sbi_coverage_check,
        sbi_boundary_piling,
        sbi_mmd_misspecification,
        sbi_prior_predictive_pvalues,
        sbi_loo_predictive_check,
        sbi_self_reference_null,
        sbi_posterior_predictive_check,
        sbi_posterior_predictive_coverage,
        sbi_posterior_correlations,
        sbi_learning_curve,
        sbi_seed_stability,
        sbi_dimensionality_sweep,
        save_diagnostics,
    )
    from qsp_inference.inference.plot_distributions import (
        plot_posterior_marginals,
        plot_posterior_pairs,
        plot_posterior_vs_prior_marginals,
    )
    from qsp_inference.inference.gaussian_copula_transform import (
        compute_quantiles,
        transform_to_normal,
        compute_quantiles_from_array,
        transform_to_normal_from_array,
    )
    from qsp_inference.inference.data_processing import (
        get_observed_data,
        add_observation_noise,
        processed_simulator,
        processed_multi_scenario_simulator,
        prepare_observed_data,
        convert_posterior_samples_to_original_space,
    )
    from qsp_inference.inference.posterior_predictive import (
        generate_prior_predictive_checks,
        generate_posterior_predictive_checks,
        generate_posterior_predictive_simulations,
        plot_ppc_histograms,
        plot_posterior_predictive_spaghetti,
    )
    from qsp_inference.inference.obed import (
        classify_mpr,
        classify_recist,
        compute_orr,
        mi_ksg,
        mi_continuous_binary,
        mi_sweep_binary,
        mi_sweep_continuous,
        loo_retrain_posterior_width,
        summarize_loo_by_observable,
        generate_tightened_theta_sets,
    )
except ImportError:
    pass

__all__ = [
    # Diagnostics
    "sbi_recovery",
    "sbi_z_score_contraction",
    "compute_per_param_calibration",
    "sbi_calibration_ecdf",
    "sbi_coverage_check",
    "sbi_boundary_piling",
    "sbi_mmd_misspecification",
    "sbi_prior_predictive_pvalues",
    "sbi_loo_predictive_check",
    "sbi_self_reference_null",
    "sbi_posterior_predictive_check",
    "sbi_posterior_predictive_coverage",
    "sbi_posterior_correlations",
    "sbi_learning_curve",
    "sbi_seed_stability",
    "sbi_dimensionality_sweep",
    "save_diagnostics",
    # Prior restriction
    "RestrictionClassifier",
    "train_restriction_classifier",
    "sample_restricted",
    # Plotting
    "plot_posterior_marginals",
    "plot_posterior_pairs",
    "plot_posterior_vs_prior_marginals",
    # Transforms
    "compute_quantiles",
    "transform_to_normal",
    "compute_quantiles_from_array",
    "transform_to_normal_from_array",
    # Data processing
    "get_observed_data",
    "add_observation_noise",
    "processed_simulator",
    "processed_multi_scenario_simulator",
    "prepare_observed_data",
    "convert_posterior_samples_to_original_space",
    # Posterior predictive
    "generate_prior_predictive_checks",
    "generate_posterior_predictive_checks",
    "generate_posterior_predictive_simulations",
    "plot_ppc_histograms",
    "plot_posterior_predictive_spaghetti",
    # OBED
    "classify_mpr",
    "classify_recist",
    "compute_orr",
    "mi_ksg",
    "mi_continuous_binary",
    "mi_sweep_binary",
    "mi_sweep_continuous",
    "loo_retrain_posterior_width",
    "summarize_loo_by_observable",
    "generate_tightened_theta_sets",
]
