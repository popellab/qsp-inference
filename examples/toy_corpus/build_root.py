#!/usr/bin/env python
"""Write a self-contained project root for the toy corpus.

Every file ``vpop_fit.ROOT_INPUTS`` names, plus the target YAMLs and the cohort
registry, for the twenty-parameter model in :mod:`model`. The point of the
exercise is that the real driver reads this root unchanged, so nothing here is a
stub: the priors are a real composite copula, the omega roles resolve through the
real role table, and the targets carry real observable bodies.

Two correlations, and they are deliberately dissimilar. ``Sigma_1``'s copula is
epistemic, over the eight parameters a stage-1 fit constrained jointly.
``R_bp`` comes from three elicited patient axes and shares only two parameters
with it. A pipeline that reconnected them would have to change one of the two
matrices, which is visible.

The statistics the targets report are left empty here; ``make_truth`` fills them
from a cohort draw off a truth cloud. A root straight out of this script is
therefore not yet fittable, and says so.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from model import ARMS, PARAMETERS, PARAM_NAMES, READOUT_TIMES, SPECIES

# ---------------------------------------------------------------------------
# Stage 1: which parameters a joint submodel fit constrained, and how
# ---------------------------------------------------------------------------
# Shrinkage of the stage-1 posterior against the CSV prior, and how far the
# posterior median sits from the CSV median in CSV sds. Both nonzero so the
# composite prior is not the CSV prior wearing a different name.
FITTED_SHRINK = 0.6
FITTED_SHIFT = 0.25

#: Epistemic factors: which stage-1 experiment constrained what. A parameter's
#: loading is its share of that experiment's information, so parameters read off
#: one curve come out correlated whether or not patients vary together in them.
EPISTEMIC_FACTORS = {
    "growth_kill_assay": {"r_T": 0.8, "K_T": 0.7, "k_kill": -0.6, "h_kill": -0.5},
    "turnover_panel": {"d_E": 0.7, "d_R": 0.7, "d_M": 0.6, "d_C": 0.4,
                       "k_kill": 0.3},
}
FITTED = tuple(sorted({n for f in EPISTEMIC_FACTORS.values() for n in f}))

# ---------------------------------------------------------------------------
# Between-patient axes. eq:crn's R, and nothing to do with the above.
# ---------------------------------------------------------------------------
BP_AXES = (
    ("immune_infiltration",
     "High patients mount and sustain an effector response; low patients do not."),
    ("tumour_aggression",
     "High patients carry a fast-growing tumour with room to grow into."),
    ("myeloid_suppression",
     "High patients accumulate suppressive myeloid and regulatory cells."),
)

#: parameter -> (axis, strength bin, sign, why). Eleven of the twenty load on
#: nothing, which is the honest default and not an omission.
BP_LOADINGS = {
    "k_rec_E": ("immune_infiltration", "primary", "+",
                "Recruitment is the rate the infiltration phenotype is defined by."),
    "s_E":     ("immune_infiltration", "secondary", "+",
                "Constitutive influx sets the floor a patient's infiltration starts from."),
    "k_exh":   ("immune_infiltration", "weak", "-",
                "Patients who sustain a response exhaust more slowly, but the link is loose."),
    "r_T":     ("tumour_aggression", "primary", "+",
                "Growth rate is what the aggression phenotype means."),
    "K_T":     ("tumour_aggression", "secondary", "+",
                "Aggressive tumours are also the ones with stromal room to expand."),
    "k_sec_C": ("tumour_aggression", "weak", "+",
                "Bulkier disease is more inflammatory, weakly."),
    "k_rec_M": ("myeloid_suppression", "primary", "+",
                "Macrophage recruitment is the suppressive phenotype's main axis."),
    "k_rec_R": ("myeloid_suppression", "primary", "+",
                "Treg recruitment travels with it in the same patients."),
    "i_M":     ("myeloid_suppression", "weak", "-",
                "Suppressive patients suppress at lower macrophage density."),
}

# ---------------------------------------------------------------------------
# Cohorts and targets
# ---------------------------------------------------------------------------
COHORTS = (
    dict(cohort_id="obs_cross_section", n_c=40, scenarios=["baseline"],
         source_tag="ToyObservational2020",
         description="Untreated resections scored once by imaging and mIHC."),
    dict(cohort_id="obs_cytokine_panel", n_c=12, scenarios=["baseline"],
         source_tag="ToyCytokine2021",
         description="Untreated tumour homogenates assayed for cytokine content."),
    dict(cohort_id="trial_vaccine_d14", n_c=24, scenarios=["vaccine"],
         source_tag="ToyTrial2023",
         description="Vaccinated patients biopsied two weeks after dosing."),
    dict(cohort_id="trial_vaccine_d28", n_c=9, scenarios=["vaccine"],
         source_tag="ToyTrial2023",
         description="The subset of the same trial resected four weeks after dosing."),
)

#: The two trial cohorts are the same nine people twice, plus fifteen biopsied
#: only. Counted, so the fit draws them jointly and their rows correlate.
BLOCKS = (
    dict(block_id="toy_trial_2023",
         description="One trial; the resected patients were all biopsied first.",
         cohorts=["trial_vaccine_d14", "trial_vaccine_d28"],
         strata=[{"cohorts": ["trial_vaccine_d14", "trial_vaccine_d28"], "n": 9},
                 {"cohorts": ["trial_vaccine_d14"], "n": 15}]),
)

_Q3 = [{"stat": "quantile", "p": 0.25}, {"stat": "quantile", "p": 0.5},
       {"stat": "quantile", "p": 0.75}]
_MED_IQR = [{"stat": "quantile", "p": 0.5}, {"stat": "iqr"}]

TARGETS = (
    dict(id="tumour_burden_baseline", arm="baseline", cohort="obs_cross_section",
         time=28.0, kind="density", modality="imaging", stats=_Q3,
         species=["tumour_burden"], units="model units",
         code="return species_dict['tumour_burden']",
         rationale="Radiographic tumour volume at resection."),
    dict(id="cd8_frac_baseline", arm="baseline", cohort="obs_cross_section",
         time=28.0, kind="fraction", modality="mihc", stats=_MED_IQR,
         species=["cd8_total", "cells_total"], units="dimensionless",
         code="return species_dict['cd8_total'] / species_dict['cells_total']",
         rationale="Pan-CD8 as a fraction of all cells in the section."),
    dict(id="treg_frac_baseline", arm="baseline", cohort="obs_cross_section",
         time=28.0, kind="fraction", modality="mihc",
         stats=[{"stat": "quantile", "p": 0.5}, {"stat": "sd"}],
         species=["treg", "cells_total"], units="dimensionless",
         code="return species_dict['treg'] / species_dict['cells_total']",
         rationale="FoxP3+ cells as a fraction of all cells."),
    dict(id="cytokine_conc_baseline", arm="baseline", cohort="obs_cytokine_panel",
         time=28.0, kind="concentration", modality="elisa", stats=_Q3,
         species=["cytokine"], units="assay units",
         aux=[{"name": "R_assay_units", "group": "unit_conversion"}],
         code="return species_dict['cytokine'] * constants['R_assay_units']",
         rationale="Homogenate cytokine in the assay's own units, which the "
                   "model does not carry, so the conversion is inferred."),
    dict(id="mac_frac_baseline", arm="baseline", cohort="obs_cytokine_panel",
         time=28.0, kind="fraction", modality="mihc",
         stats=[{"stat": "mean"}, {"stat": "sd"}],
         species=["macrophage", "immune_total"], units="dimensionless",
         code="return species_dict['macrophage'] / species_dict['immune_total']",
         rationale="Macrophages as a fraction of infiltrating immune cells."),
    dict(id="cd8_frac_vac_d14", arm="vaccine", cohort="trial_vaccine_d14",
         time=14.0, kind="fraction", modality="mihc", stats=_MED_IQR,
         species=["cd8_total", "cells_total"], units="dimensionless",
         code="return species_dict['cd8_total'] / species_dict['cells_total']",
         rationale="The same mIHC readout, on biopsy two weeks after dosing."),
    dict(id="exh_frac_vac_d14", arm="vaccine", cohort="trial_vaccine_d14",
         time=14.0, kind="fraction", modality="mihc", stats=_Q3,
         species=["cd8_exhausted", "cd8_total"], units="dimensionless",
         code="return species_dict['cd8_exhausted'] / species_dict['cd8_total']",
         rationale="PD-1-high CD8 as a fraction of all CD8."),
    dict(id="tumour_burden_vac_d28", arm="vaccine", cohort="trial_vaccine_d28",
         time=28.0, kind="density", modality="imaging", stats=_Q3,
         species=["tumour_burden"], units="model units",
         code="return species_dict['tumour_burden']",
         rationale="Radiographic tumour volume at resection, post-vaccine."),
    dict(id="cd8_frac_vac_d28", arm="vaccine", cohort="trial_vaccine_d28",
         time=28.0, kind="fraction", modality="mihc", stats=_MED_IQR,
         species=["cd8_total", "cells_total"], units="dimensionless",
         code="return species_dict['cd8_total'] / species_dict['cells_total']",
         rationale="The same mIHC readout on the resection specimen."),
    dict(id="cd8_fc_vac_d28", arm="vaccine", cohort="trial_vaccine_d28",
         time=28.0, kind="fold_change", modality="mihc", stats=_Q3,
         reference=14.0, species=["cd8_total"], units="dimensionless",
         code=("x = species_dict['cd8_total']\n"
               "    return x / x[0]"),
         rationale="Paired resection-over-biopsy CD8 fold change in the same "
                   "patients, so a per-patient staining bias cancels."),
)

ARM_OF_DIR = {arm: arm for arm in ARMS}


# ---------------------------------------------------------------------------
def _observables_module() -> str:
    """The generated-observables module, written by hand because there is no SBML."""
    return f'''"""Derived symbols for the toy model. Stands in for qsp-codegen's output."""

STATES = {SPECIES!r}

CONSTANTS: dict = {{}}


def observables(states, constants=CONSTANTS):
    """Raw species -> the symbols a target body may name."""
    C, E, M, R, T, X = (states[n] for n in STATES)
    cd8_total = E + X
    immune_total = cd8_total + R + M
    return {{
        "cytokine": C,
        "cd8_effector": E,
        "cd8_exhausted": X,
        "cd8_total": cd8_total,
        "macrophage": M,
        "treg": R,
        "tumour_burden": T,
        "immune_total": immune_total,
        "cells_total": immune_total + T,
    }}
'''


def _target_doc(t: dict) -> dict:
    observable = {
        "code": ("def compute_observable(time, species_dict, constants):\n"
                 f"    {t['code']}\n"),
        "units": t["units"],
        "readout_time": t["time"],
        "readout_time_unit": "day",
        "species": list(t["species"]),
        "constants": [],
        "mapping_rationale": t["rationale"],
        "readout": {
            "quantity_kind": t["kind"],
            "assay_modality": t["modality"],
            "reference": (None if "reference" not in t else
                          {"kind": "timepoint", "timepoint": t["reference"]}),
        },
        "aggregation": None,
    }
    if t.get("aux"):
        observable["auxiliary_parameters"] = list(t["aux"])
    return {
        "id": t["id"],
        "calibration_target_id": t["id"],
        "cohort_id": t["cohort"],
        "study_interpretation": (
            "Synthetic. The numbers below were generated by make_truth from a "
            "known phi*, so the fit's recovery of phi* is checkable."),
        "observable": observable,
        "empirical_data": {
            "units": t["units"],
            "observed_distribution": {
                # make_truth fills these in. Left empty rather than defaulted so
                # a root that has not been through it fails at row_specs instead
                # of fitting numbers nobody generated.
                "statistics": [],
                "quantile_convention": "type7",
                "spread_source": "across_patient",
            },
        },
        "primary_data_source": {"source_tag": _source_tag(t["cohort"])},
    }


def _source_tag(cohort_id: str) -> str:
    return next(c["source_tag"] for c in COHORTS if c["cohort_id"] == cohort_id)


def _epistemic_correlation() -> np.ndarray:
    """``Sigma_1``'s copula: which stage-1 experiment constrained which parameter."""
    lam = np.zeros((len(FITTED), len(EPISTEMIC_FACTORS)))
    for k, loads in enumerate(EPISTEMIC_FACTORS.values()):
        for name, value in loads.items():
            lam[FITTED.index(name), k] = value
    comm = (lam ** 2).sum(1)
    over = comm > 0.81
    if over.any():
        lam[over] *= np.sqrt(0.81 / comm[over])[:, None]
        comm = (lam ** 2).sum(1)
    R = lam @ lam.T + np.diag(1.0 - comm)
    np.fill_diagonal(R, 1.0)
    np.linalg.cholesky(R)
    return R


def write(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    median = {p[0]: p[1] for p in PARAMETERS}
    sd = {p[0]: p[2] for p in PARAMETERS}
    role = {p[0]: p[3] for p in PARAMETERS}
    scale = {p[0]: p[4] for p in PARAMETERS}
    about = {p[0]: p[5] for p in PARAMETERS}

    # -- the flat prior, and the parameter order everything else is indexed by --
    (root / "parameters").mkdir(exist_ok=True)
    with (root / "parameters/pdac_priors.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "median", "units", "distribution", "dist_param1",
                    "dist_param2", "lower_bound", "upper_bound"])
        for name in PARAM_NAMES:
            w.writerow([name, median[name], "", "lognormal",
                        float(np.log(median[name])), sd[name], "", ""])

    # -- stage 1: a posterior for eight of them, with an epistemic copula --
    R_1 = _epistemic_correlation()
    (root / "notes/calibration").mkdir(parents=True, exist_ok=True)
    (root / "notes/calibration/submodel_priors.yaml").write_text(yaml.safe_dump({
        "metadata": {"generated_by": "examples/toy_corpus/build_root.py",
                     "n_parameters": len(FITTED),
                     "note": "synthetic stage-1 posteriors; the copula is "
                             "epistemic and is not eq:crn's R"},
        "parameters": [
            {"name": name,
             "marginal": {"distribution": "lognormal",
                          "mu": float(np.log(median[name])
                                      + FITTED_SHIFT * sd[name]),
                          "sigma": float(FITTED_SHRINK * sd[name])},
             "source_targets": [f"{name}_toy"]}
            for name in FITTED
        ],
        "copula": {"type": "gaussian", "parameters": list(FITTED),
                   "correlation": [[float(v) for v in row] for row in R_1]},
    }, sort_keys=False))

    (root / "parameters/derived_priors.yaml").write_text(
        "# No derived parameters in this corpus. Declared empty rather than\n"
        "# absent: the fit requires the file, and an absent one would be\n"
        "# indistinguishable from a corpus whose children were forgotten.\n"
        "parameters: {}\n")

    # -- omega: the role per parameter, and the axes behind R_bp --
    with (root / "parameters/omega_priors.csv").open("w", newline="") as fh:
        fh.write("# Between-patient spread overrides. A parameter absent from\n"
                 "# this file runs at the global default; every one of the toy's\n"
                 "# twenty is listed so none of them does.\n")
        w = csv.writer(fh)
        w.writerow(["name", "role", "omega", "scale", "rationale"])
        for name in PARAM_NAMES:
            w.writerow([name, role[name], "", scale[name], about[name]])

    (root / "parameters/omega_axes.yaml").write_text(yaml.safe_dump({
        "axes": [{"name": n, "description": d, "proposed_by": "synthetic"}
                 for n, d in BP_AXES]}, sort_keys=False))

    drivers = root / "parameters/elicitation/drivers"
    drivers.mkdir(parents=True, exist_ok=True)
    drivers.parent.joinpath("role").mkdir(exist_ok=True)
    drivers.joinpath("loadings.json").write_text(json.dumps([
        {"name": name,
         "loadings": ([{"axis": BP_LOADINGS[name][0],
                        "strength": BP_LOADINGS[name][1],
                        "sign": BP_LOADINGS[name][2]}]
                      if name in BP_LOADINGS else []),
         "n": 5, "votes": {}, "contested": False,
         "reason": (BP_LOADINGS[name][3] if name in BP_LOADINGS
                    else "Nothing ties this rate to a patient phenotype.")}
        for name in PARAM_NAMES], indent=2))

    # Unanimous everywhere, so the fit's elicitation report has nothing to flag.
    # A corpus where it does is the interesting case, and this is not it.
    (root / "parameters/elicitation/role/votes.json").write_text(json.dumps([
        {"name": name, "role": role[name], "consensus": True,
         "votes": {f"persona_{i}": role[name] for i in range(1, 6)}}
        for name in PARAM_NAMES], indent=2))

    # -- the auxiliary group the cytokine target reads --
    (root / "auxiliary_config.yaml").write_text(yaml.safe_dump({
        "groups": {"unit_conversion": {
            "description": "Assay-to-model unit conversions, inferred rather "
                           "than known, one per assay that reports in its own "
                           "units.",
            "base_prior": {"distribution": "lognormal",
                           "mu": float(np.log(2.0)), "sigma": 0.4}}}},
        sort_keys=False))

    # -- the generated observables --
    (root / "core/generated").mkdir(parents=True, exist_ok=True)
    (root / "core/generated/observables.py").write_text(_observables_module())

    # -- cohorts and targets --
    ct = root / "calibration_targets"
    ct.mkdir(exist_ok=True)
    (ct / "cohorts.yaml").write_text(yaml.safe_dump(
        {"cohorts": [dict(c) for c in COHORTS],
         "blocks": [dict(b) for b in BLOCKS]}, sort_keys=False))
    for t in TARGETS:
        (ct / t["arm"]).mkdir(exist_ok=True)
        (ct / t["arm"] / f"{t['id']}.yaml").write_text(
            yaml.safe_dump(_target_doc(t), sort_keys=False))

    # -- the corpus's shape, for the driver --
    (root / "vpop_config.yaml").write_text(yaml.safe_dump({
        "arms": ARM_OF_DIR,
        "precomposed": {},
        "excluded_rows": [],
        "constants": {"N_CLOUD": 2000, "N_CLOUD_V": 20000, "N_DIFF": 200},
    }, sort_keys=False))

    print(f"root {root}")
    print(f"  {len(PARAM_NAMES)} parameters, {len(FITTED)} with a stage-1 posterior")
    print(f"  {len(TARGETS)} targets over {len(COHORTS)} cohorts in "
          f"{len(ARM_OF_DIR)} arms, species at {list(READOUT_TIMES)}")
    print(f"  {len(BP_LOADINGS)} parameters load on {len(BP_AXES)} patient axes, "
          f"{len(PARAM_NAMES) - len(BP_LOADINGS)} independent")
    print("  targets carry no statistics yet; run make_truth next")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    write(ap.parse_args().root)


if __name__ == "__main__":
    main()
