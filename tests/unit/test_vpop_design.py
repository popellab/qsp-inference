"""Unit tests for Z_r (qsp_inference.vpop.design).

Z decides which readouts pool their measurement correction. The properties that
matter are checkable without a fit: every column carried by at least two
readouts, rank well below M, and no two columns the same vector.
"""
import numpy as np
import pytest

from qsp_inference.vpop.design import build_Z, z_conditioning


def _t(kind, modality):
    return {"observable": {"readout": {"quantity_kind": kind,
                                       "assay_modality": modality}}}


CORPUS = {
    "d1": _t("density", "mihc"),
    "d2": _t("density", "mihc"),
    "f1": _t("fraction", "mihc"),
    "f2": _t("fraction", "flow_cytometry"),
    "f3": _t("fraction", "flow_cytometry"),
    "r1": _t("ratio", "mihc"),
}


class TestBuildZ:
    def test_references_get_no_column(self):
        d = build_Z(CORPUS)
        assert "kind:density" not in d.columns
        assert "assay:mihc" not in d.columns
        assert d.columns[0] == "intercept"

    def test_rows_are_readouts_in_sorted_order(self):
        d = build_Z(CORPUS)
        assert d.readouts == ("d1", "d2", "f1", "f2", "f3", "r1")
        assert d.shape[0] == 6

    def test_indicators_land_on_the_right_readouts(self):
        d = build_Z(CORPUS)
        col = d.Z[:, d.columns.index("kind:fraction")]
        assert list(col) == [0, 0, 1, 1, 1, 0]

    def test_intercept_can_be_dropped(self):
        assert "intercept" not in build_Z(CORPUS, intercept=False).columns

    def test_a_readout_alone_in_its_category_folds_to_the_reference(self):
        corpus = dict(CORPUS, odd=_t("time", "ct"))
        d = build_Z(corpus)
        assert "kind:time" in d.dropped and "assay:ct" in d.dropped
        assert not any(c.startswith(("kind:time", "assay:ct")) for c in d.columns)
        # It still carries the intercept, which is what "reference level" means.
        assert d.Z[d.readouts.index("odd"), d.columns.index("intercept")] == 1.0

    def test_singletons_can_be_kept(self):
        corpus = dict(CORPUS, odd=_t("time", "ct"))
        d = build_Z(corpus, drop_singletons=False, merge_aliases=False)
        assert "kind:time" in d.columns and "assay:ct" in d.columns


class TestAliases:
    """A corpus measuring one kind of quantity by one assay and nothing else."""

    ALIASED = {
        "d1": _t("density", "mihc"),
        "d2": _t("density", "mihc"),
        "c1": _t("concentration", "luminex"),
        "c2": _t("concentration", "luminex"),
        "c3": _t("concentration", "luminex"),
    }

    def test_identical_columns_merge_into_one(self):
        d = build_Z(self.ALIASED)
        assert d.merged == (("kind:concentration", "assay:luminex"),)
        assert "kind:concentration~assay:luminex" in d.columns
        assert "kind:concentration" not in d.columns
        assert "assay:luminex" not in d.columns

    def test_merging_removes_the_rank_deficiency(self):
        kept = z_conditioning(build_Z(self.ALIASED, merge_aliases=False))
        merged = z_conditioning(build_Z(self.ALIASED))
        assert kept["deficient"] and not merged["deficient"]
        assert merged["cond"] < kept["cond"]

    def test_the_merged_column_still_marks_the_right_readouts(self):
        d = build_Z(self.ALIASED)
        col = d.Z[:, d.columns.index("kind:concentration~assay:luminex")]
        marked = {r for r, v in zip(d.readouts, col) if v}
        assert marked == {"c1", "c2", "c3"}

    def test_a_within_kind_modality_contrast_breaks_the_confound(self):
        """One concentration by another assay is all it takes."""
        corpus = dict(self.ALIASED, c4=_t("concentration", "mihc"),
                      d3=_t("density", "luminex"))
        d = build_Z(corpus)
        assert d.merged == ()
        assert "kind:concentration" in d.columns and "assay:luminex" in d.columns


class TestPooling:
    def test_rare_modalities_pool_into_one_column(self):
        corpus = dict(CORPUS, x=_t("fraction", "mpm"), y=_t("fraction", "ct"))
        groups = {"mpm": "imaging_other", "ct": "imaging_other"}
        d = build_Z(corpus, modality_groups=groups)
        assert "assay:imaging_other" in d.columns
        assert z_conditioning(d)["support"]["assay:imaging_other"] == 2

    def test_without_pooling_they_would_have_folded_away(self):
        corpus = dict(CORPUS, x=_t("fraction", "mpm"), y=_t("fraction", "ct"))
        d = build_Z(corpus)
        assert "assay:mpm" in d.dropped and "assay:ct" in d.dropped


class TestConditioning:
    def test_reports_support_rank_and_M(self):
        c = z_conditioning(build_Z(CORPUS))
        assert c["M"] == 6
        assert c["support"]["intercept"] == 6
        assert c["rank"] == c["n_columns"]
        assert not c["deficient"] and not c["saturates"]

    def test_saturation_is_flagged(self):
        """rank(Z) = M makes gamma a free per-readout intercept."""
        corpus = {"a": _t("density", "m1"), "b": _t("fraction", "m2"),
                  "c": _t("ratio", "m3")}
        d = build_Z(corpus, kind_reference="density", modality_reference="m1",
                    drop_singletons=False, merge_aliases=False)
        assert z_conditioning(d)["saturates"]

    def test_a_singleton_column_is_named(self):
        corpus = dict(CORPUS, odd=_t("time", "ct"))
        d = build_Z(corpus, drop_singletons=False, merge_aliases=False)
        # kind:ratio is a singleton in this fixture too, and should be named.
        assert set(z_conditioning(d)["singletons"]) == {
            "kind:time", "assay:ct", "kind:ratio"}
