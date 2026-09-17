"""Tests for the ``sample_method`` point-subsampling option.

Covers the feature ported from the topochange working copy (2026-07):
``max_points`` can subsample either by taking the first N points in file
order ("head": streaming, low memory, spatially biased) or an unbiased
random subsample ("random": ``filters.randomize`` then ``filters.head``).

All tests here are dependency-free: PDAL pipeline *construction* is tested
by monkeypatching ``run_pdal_pipeline`` and capturing the steps list, so no
pdal installation is needed.
"""

import numpy as np
import pytest

from topochange import RegistrationConfig
from topochange import alignment_utils


# ---------------------------------------------------------------------------
# RegistrationConfig validation (pure dataclass logic)
# ---------------------------------------------------------------------------

class TestRegistrationConfigSampleMethod:
    def test_default_is_head(self):
        assert RegistrationConfig().sample_method == "head"

    def test_case_normalization(self):
        assert RegistrationConfig(sample_method="RANDOM").sample_method == "random"
        assert RegistrationConfig(sample_method="Head").sample_method == "head"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="sample_method"):
            RegistrationConfig(sample_method="middle")

    def test_to_dict_round_trip(self):
        d = RegistrationConfig(sample_method="random", max_points=1000).to_dict()
        assert d["sample_method"] == "random"
        assert d["max_points"] == 1000


# ---------------------------------------------------------------------------
# load_points_from_las pipeline construction (run_pdal_pipeline mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Capture the PDAL steps list instead of executing a pipeline."""
    captured = {}

    def _fake_run(steps, **kwargs):
        captured["steps"] = steps
        arr = np.zeros(5, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8")])
        return arr, {}

    monkeypatch.setattr(alignment_utils, "run_pdal_pipeline", _fake_run)
    las = tmp_path / "dummy.las"
    las.write_bytes(b"")  # only existence is checked before pipeline build
    return captured, las


class TestLoadPointsSampleMethod:
    @staticmethod
    def _types(steps):
        return [s["type"] for s in steps]

    def test_head_appends_only_filters_head(self, fake_pipeline):
        captured, las = fake_pipeline
        pts = alignment_utils.load_points_from_las(
            las, max_points=100, sample_method="head"
        )
        types = self._types(captured["steps"])
        assert pts.shape == (5, 3)
        assert "filters.randomize" not in types
        assert types[-1] == "filters.head"
        assert captured["steps"][-1]["count"] == 100

    def test_random_inserts_randomize_then_head(self, fake_pipeline):
        captured, las = fake_pipeline
        alignment_utils.load_points_from_las(
            las, max_points=100, sample_method="random"
        )
        types = self._types(captured["steps"])
        # order matters: shuffle the stream, THEN keep the first N
        assert types[-2:] == ["filters.randomize", "filters.head"]
        assert captured["steps"][-1]["count"] == 100

    def test_default_is_head(self, fake_pipeline):
        captured, las = fake_pipeline
        alignment_utils.load_points_from_las(las, max_points=50)
        types = self._types(captured["steps"])
        assert "filters.randomize" not in types
        assert types[-1] == "filters.head"

    def test_no_max_points_no_subsampling_stage(self, fake_pipeline):
        captured, las = fake_pipeline
        alignment_utils.load_points_from_las(las)
        types = self._types(captured["steps"])
        assert "filters.head" not in types
        assert "filters.randomize" not in types

    def test_invalid_sample_method_raises(self, fake_pipeline):
        captured, las = fake_pipeline
        with pytest.raises(ValueError, match="sample_method"):
            alignment_utils.load_points_from_las(
                las, max_points=100, sample_method="middle"
            )
        # the pipeline must not have been executed
        assert "steps" not in captured
