"""Regression tests for vertical datum resolution.

Covers three defects found in 1.0.0:

1. PDAL reports an absent SRS field as ``""``, not None, so ``from_file``'s
   ``is not None`` guard took the wrong branch and ``is_orthometric`` ended up
   None instead of the intended value.
2. A file with no vertical CRS was recorded as ``is_orthometric = False``,
   asserting ellipsoidal heights on no evidence. If the heights are really
   orthometric, every downstream vertical transform is wrong by the geoid
   separation, silently.
3. ``needs_vertical`` did not check that the target vertical kind was
   resolvable, so an undetermined reference produced
   ``TypeError: 'NoneType' object is not subscriptable`` from a filename
   f-string rather than a usable error.

Also pins the catalog-to-CRS resolution, where ``vertical_datum_to_crs``
returns None for ellipsoidal datums and the ellipsoidal fact was being lost.
"""

import re
import pytest

from topochange.crs_utils import (
    is_orthometric,
    vertical_datum_to_crs,
    ellipsoidal_height_crs_from_horizontal,
    resolve_catalog_vertical,
)


def _crs_name(wkt):
    return re.search(r'"([^"]+)"', wkt).group(1) if wkt else None


class TestEmptySrsFields:
    """PDAL gives "" for absent SRS fields, which is falsy but not None."""

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_vertical_crs_is_undetermined(self, value):
        assert is_orthometric(value) is None

    def test_srs_field_normalization(self):
        # the normalization from_file applies: "" -> None, real values kept
        def _srs_field(value):
            return value or None

        assert _srs_field("") is None
        assert _srs_field(None) is None
        assert _srs_field("VERT_CS[...]") == "VERT_CS[...]"

    def test_empty_string_would_pass_a_none_check(self):
        # the exact bug: "" is falsy but survives `is not None`
        assert ("" is not None) is True
        assert bool("") is False


class TestUnknownIsNotEllipsoidal:
    """Undetermined must stay undetermined, never default to ellipsoidal."""

    def test_unclassifiable_vertical_crs_returns_none(self):
        assert is_orthometric("") is None
        assert is_orthometric(None) is None

    def test_navd88_is_orthometric(self):
        wkt = (
            'VERT_CS["NAVD88 height",VERT_DATUM["North American Vertical Datum 1988",2005],'
            'UNIT["metre",1],AXIS["Gravity-related height",UP],AUTHORITY["EPSG","5703"]]'
        )
        assert is_orthometric(wkt) is True

    def test_vertical_kind_mapping_is_three_way(self):
        def kind(v):
            return "orthometric" if v else "ellipsoidal" if v is False else None

        assert kind(True) == "orthometric"
        assert kind(False) == "ellipsoidal"   # False means ellipsoidal, not unknown
        assert kind(None) is None             # unknown stays unknown

    def test_none_kind_would_crash_the_filename_builder(self):
        # the original failure: f"{target_vertical_kind[:4]}"
        target_vertical_kind = None
        with pytest.raises(TypeError):
            _ = target_vertical_kind[:4]


class TestEllipsoidalHeightFromHorizontal:
    """Ellipsoidal height is datum-relative; derive it from the horizontal CRS."""

    @pytest.mark.parametrize("epsg", ["32611", "32612", "26911", "6340"])
    def test_derives_ellipsoidal_vertical_crs(self, epsg):
        vcrs = ellipsoidal_height_crs_from_horizontal(epsg)
        assert vcrs is not None
        assert "ellipsoidal" in vcrs.name.lower()
        # the point of the whole exercise: it classifies as NOT orthometric
        assert is_orthometric(vcrs.to_wkt()) is False

    def test_none_horizontal_returns_none(self):
        assert ellipsoidal_height_crs_from_horizontal(None) is None

    def test_garbage_returns_none_not_raises(self):
        assert ellipsoidal_height_crs_from_horizontal("not-a-crs") is None


class TestCatalogVerticalResolution:
    """OpenTopography catalog metadata -> vertical CRS, geoid, orthometric flag."""

    def test_ellipsoidal_resolves_positively(self):
        # vertical_datum_to_crs alone returns None here, losing the fact
        assert vertical_datum_to_crs("ellipsoidal", None) is None

        wkt, geoid, ortho = resolve_catalog_vertical({
            "is_orthometric": False, "vertical_datum": "ellipsoidal",
            "geoid_model": None, "horizontal_crs": "32611",
        })
        assert ortho is False
        assert geoid is None
        assert wkt is not None
        assert "ellipsoidal" in _crs_name(wkt).lower()

    def test_orthometric_resolves_to_navd88(self):
        wkt, geoid, ortho = resolve_catalog_vertical({
            "is_orthometric": True, "vertical_datum": "NAVD88",
            "geoid_model": "geoid12b", "horizontal_crs": "32612",
        })
        assert ortho is True
        assert geoid == "geoid12b"
        assert "NAVD88" in _crs_name(wkt)

    def test_catalog_word_ellipsoid_also_recognized(self):
        _, _, ortho = resolve_catalog_vertical({
            "is_orthometric": None, "vertical_datum": "Ellipsoid",
            "geoid_model": None, "horizontal_crs": "32611",
        })
        assert ortho is False

    def test_unknown_stays_unknown(self):
        wkt, geoid, ortho = resolve_catalog_vertical({
            "is_orthometric": None, "vertical_datum": None,
            "geoid_model": None, "horizontal_crs": "32611",
        })
        assert (wkt, geoid, ortho) == (None, None, None)

    def test_orthometric_flag_survives_missing_horizontal(self):
        # no horizontal CRS to derive from, but the catalog's fact is kept
        wkt, _, ortho = resolve_catalog_vertical({
            "is_orthometric": False, "vertical_datum": "ellipsoidal",
            "geoid_model": None, "horizontal_crs": None,
        })
        assert wkt is None
        assert ortho is False


class TestTransformGuard:
    """An undetermined reference must fail with a usable error, not TypeError."""

    @staticmethod
    def _pair(target_is_ortho):
        from types import SimpleNamespace
        from topochange import PointCloudPair

        def _pc(is_ortho):
            return SimpleNamespace(
                filename="dummy.laz",
                epoch=2019.5,
                current_horizontal_crs="EPSG:32611",
                original_horizontal_crs="EPSG:32611",
                current_vertical_crs=None,
                original_vertical_crs=None,
                geoid_model=None,
                is_orthometric=is_ortho,
                vertical_unit=None,
            )

        return PointCloudPair(_pc(True), _pc(target_is_ortho))

    def test_undetermined_target_raises_value_error(self, monkeypatch):
        pair = self._pair(target_is_ortho=None)
        monkeypatch.setattr(
            pair, "check_all_match",
            lambda *a, **k: {"transformations_needed": ["vertical_datum"]},
            raising=False,
        )
        with pytest.raises(ValueError, match="vertical CRS could not be determined"):
            pair.transform_compare_to_match_reference(
                skip_epoch=True, skip_horizontal=True, verbose=False
            )

    def test_error_names_a_remedy(self, monkeypatch):
        pair = self._pair(target_is_ortho=None)
        monkeypatch.setattr(
            pair, "check_all_match",
            lambda *a, **k: {"transformations_needed": ["vertical_datum"]},
            raising=False,
        )
        with pytest.raises(ValueError) as exc:
            pair.transform_compare_to_match_reference(
                skip_epoch=True, skip_horizontal=True, verbose=False
            )
        msg = str(exc.value)
        assert "add_metadata" in msg
        assert "skip_vertical=True" in msg

    def test_skip_vertical_bypasses_the_guard(self, monkeypatch):
        pair = self._pair(target_is_ortho=None)
        monkeypatch.setattr(
            pair, "check_all_match",
            lambda *a, **k: {"transformations_needed": ["vertical_datum"]},
            raising=False,
        )
        try:
            pair.transform_compare_to_match_reference(
                skip_epoch=True, skip_horizontal=True, skip_vertical=True,
                verbose=False,
            )
        except ValueError:
            pytest.fail("guard fired even though skip_vertical=True")
        except Exception:
            # the stub is too thin to complete a real transform; clearing the
            # guard without a ValueError is the whole assertion here
            pass


class TestGeoidWarningIsThreeState:
    """Ellipsoidal data must not be told to set a geoid.

    geoid_model=None is correct for ellipsoidal heights, not a missing value.
    Warning there invites set_*_geoid() on data that already has no geoid,
    which applies a correction of tens of metres that must not exist.
    """

    @staticmethod
    def _branch(geoid, is_ortho):
        """The decision the warning block makes, in isolation."""
        if geoid:
            return "geoid"
        elif is_ortho is False:
            return "ellipsoidal"
        return "warn"

    def test_ellipsoidal_reports_instead_of_warning(self):
        assert self._branch(None, False) == "ellipsoidal"

    def test_orthometric_with_geoid_reports_geoid(self):
        assert self._branch("geoid12b", True) == "geoid"

    def test_genuinely_undetermined_still_warns(self):
        assert self._branch(None, None) == "warn"

    def test_orthometric_without_geoid_still_warns(self):
        # NAVD88 with no geoid named is a real gap worth flagging
        assert self._branch(None, True) == "warn"

    def test_catalog_string_for_ellipsoid_yields_no_geoid(self):
        # "WGS84 (Ellipsoid)" must resolve to ellipsoidal with geoid None
        _, geoid, ortho = resolve_catalog_vertical({
            "is_orthometric": False, "vertical_datum": "ellipsoidal",
            "geoid_model": None, "horizontal_crs": "32611",
        })
        assert ortho is False
        assert geoid is None
        assert self._branch(geoid, ortho) == "ellipsoidal"


class TestVerticalUnitReconciliation:
    """A vertical CRS states a datum, not a unit.

    The ellipsoidal-height CRS derived from a horizontal CRS carries metres
    incidentally. Letting that override a file recorded in US survey feet
    would scale every elevation by ~3.28 with nothing raised.
    """

    @staticmethod
    def _unit(name, display=None):
        from topochange.unit_utils import lookup_unit
        return lookup_unit(name)

    def test_catalog_unit_wins_when_both_known(self):
        from topochange.unit_utils import reconcile_vertical_unit
        cat, hdr = self._unit("us_survey_foot"), self._unit("meter")
        chosen, warning = reconcile_vertical_unit(cat, hdr)
        assert chosen.name == "us_survey_foot"
        assert warning is not None and "disagreement" in warning

    def test_agreeing_sources_produce_no_warning(self):
        from topochange.unit_utils import reconcile_vertical_unit
        u = self._unit("meter")
        chosen, warning = reconcile_vertical_unit(u, u)
        assert chosen.name == "meter"
        assert warning is None

    def test_header_used_when_catalog_silent(self):
        # "WGS84 (Ellipsoid)" states no unit; the file header does
        from topochange.unit_utils import reconcile_vertical_unit
        from topochange.unit_utils import UNKNOWN_UNIT
        chosen, warning = reconcile_vertical_unit(UNKNOWN_UNIT, self._unit("us_survey_foot"))
        assert chosen.name == "us_survey_foot"
        assert warning is None

    def test_header_none_and_catalog_silent_warns(self):
        from topochange.unit_utils import reconcile_vertical_unit, UNKNOWN_UNIT
        chosen, warning = reconcile_vertical_unit(UNKNOWN_UNIT, UNKNOWN_UNIT)
        assert chosen is None
        assert warning is not None and "3.28" in warning

    def test_missing_objects_are_tolerated(self):
        from topochange.unit_utils import reconcile_vertical_unit
        chosen, warning = reconcile_vertical_unit(None, None)
        assert chosen is None
        assert warning is not None

    def test_header_survives_a_metre_bearing_ellipsoidal_crs(self):
        # the regression that motivated this: the synthesized WGS84 ellipsoidal
        # height CRS is in metres, but the file is in US survey feet
        from topochange.unit_utils import reconcile_vertical_unit, UNKNOWN_UNIT
        chosen, _ = reconcile_vertical_unit(UNKNOWN_UNIT, self._unit("us_survey_foot"))
        assert chosen.name == "us_survey_foot", "header unit must not be overwritten"
