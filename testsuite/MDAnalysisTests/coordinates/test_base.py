# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2026 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the Lesser GNU Public Licence, v2.1 or any higher version

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_allclose

import MDAnalysis as mda
from MDAnalysis.coordinates.timestep import Timestep
from MDAnalysis.lib.mdamath import triclinic_vectors, triclinic_box


@pytest.fixture
def ts():
    return Timestep(10)


class TestTimestepDimensionsValidation:
    """Comprehensive regression tests for Issue #5534:
    Validation of Timestep.dimensions and mdamath triclinic functions.
    """

    def test_dimensions_none_resets_to_zero(self, ts):
        """Verify setting dimensions to None resets unitcell to 0 and returns None."""
        ts.dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        assert ts.dimensions is not None
        assert ts.dimensions[0] == 10.0

        ts.dimensions = None
        assert ts.dimensions is None
        assert np.all(ts._unitcell == 0.0)

    def test_dimensions_zero_box(self, ts):
        """Verify setting dimensions to all zeros makes getter return None."""
        ts.dimensions = np.zeros(6, dtype=np.float32)
        assert ts.dimensions is None
        assert np.all(ts._unitcell == 0.0)

        ts.dimensions = [0, 0, 0, 0, 0, 0]
        assert ts.dimensions is None

    @pytest.mark.parametrize("idx", range(6))
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_dimensions_rejects_nonfinite_in_all_components(self, ts, idx, value):
        """Verify NaN, +Inf, -Inf are rejected in any of the 6 dimensions."""
        dims = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        dims[idx] = value
        with pytest.raises(
            ValueError, match="Unit cell dimensions must contain only finite values"
        ):
            ts.dimensions = dims

    @pytest.mark.parametrize(
        "invalid_shape",
        [
            [],
            [10.0],
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 90.0, 90.0],
            [10.0, 10.0, 10.0, 90.0, 90.0, 90.0, 90.0],
            np.ones((2, 3)),
            np.ones((6, 1)),
            np.eye(3),
        ],
    )
    def test_dimensions_rejects_invalid_shapes(self, ts, invalid_shape):
        """Verify shapes other than (6,) are rejected."""
        with pytest.raises(
            ValueError,
            match="Unit cell dimensions must be a six-element sequence",
        ):
            ts.dimensions = invalid_shape

    @pytest.mark.parametrize(
        "valid_input",
        [
            [15.0, 20.0, 25.0, 90.0, 90.0, 90.0],
            (15.0, 20.0, 25.0, 90.0, 90.0, 90.0),
            np.array([15.0, 20.0, 25.0, 90.0, 90.0, 90.0], dtype=np.float32),
            np.array([15.0, 20.0, 25.0, 90.0, 90.0, 90.0], dtype=np.float64),
            ["15.0", "20.0", "25.0", "90.0", "90.0", "90.0"],
        ],
    )
    def test_dimensions_accepts_valid_sequences(self, ts, valid_input):
        """Verify lists, tuples, numpy arrays, and string numbers are accepted."""
        ts.dimensions = valid_input
        assert ts.dimensions is not None
        assert ts.dimensions.dtype == np.float32
        assert_allclose(
            ts.dimensions,
            [15.0, 20.0, 25.0, 90.0, 90.0, 90.0],
            rtol=1e-5,
        )

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_triclinic_dimensions_nonfinite_setter(self, ts, value):
        """Verify setting nonfinite triclinic matrix resets dimensions to None."""
        ts.triclinic_dimensions = [[value, 0, 0], [0, 10, 0], [0, 0, 10]]
        assert ts.dimensions is None
        assert ts.triclinic_dimensions is None


class TestMdaMathTriclinicValidation:
    """Regression tests for mdamath triclinic_vectors and triclinic_box finiteness checks."""

    @pytest.mark.parametrize("idx", range(6))
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_triclinic_vectors_rejects_nonfinite(self, idx, value):
        """Verify triclinic_vectors returns 3x3 zeros when any component is non-finite."""
        dim = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        dim[idx] = value
        box = triclinic_vectors(dim)
        assert_equal(box, np.zeros((3, 3)))

    def test_triclinic_vectors_inf_length_regression(self):
        """Specific regression test: +Inf length must not leak into diagonal box matrix."""
        dim = [np.inf, 10.0, 10.0, 90.0, 90.0, 90.0]
        box = triclinic_vectors(dim)
        assert_equal(box, np.zeros((3, 3)))

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_triclinic_box_rejects_nonfinite(self, value):
        """Verify triclinic_box returns zero vector when any vector component is non-finite."""
        x = [value, 0.0, 0.0]
        y = [0.0, 10.0, 0.0]
        z = [0.0, 0.0, 10.0]
        box = triclinic_box(x, y, z)
        assert_equal(box, np.zeros(6, dtype=np.float32))


class TestUniverseDimensionsIntegration:
    """End-to-end integration tests using mda.Universe reproducer from Issue #5534."""

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_universe_empty_rejects_nonfinite_dimensions(self, value):
        """Verify Universe rejects non-finite dimensions preventing downstream corruption."""
        u = mda.Universe.empty(2, trajectory=True)
        u.atoms.positions = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]

        with pytest.raises(
            ValueError, match="Unit cell dimensions must contain only finite values"
        ):
            u.dimensions = [value, 2.0, 3.0, 90.0, 90.0, 90.0]

        # Ensure u.dimensions remained None and wasn't partially overwritten
        assert u.dimensions is None

    def test_universe_pack_into_box_with_valid_dimensions(self):
        """Verify normal workflow succeeds when valid dimensions are assigned."""
        u = mda.Universe.empty(2, trajectory=True)
        u.atoms.positions = [[0.0, 0.0, 0.0], [15.0, 15.0, 15.0]]
        u.dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]

        packed = u.atoms.pack_into_box()
        assert np.all(np.isfinite(packed))
        assert np.all(packed <= 10.0)
        assert np.all(packed >= 0.0)

    def test_universe_none_dimensions_accepted(self):
        """Verify Universe accepts dimensions=None."""
        u = mda.Universe.empty(2, trajectory=True)
        u.dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        assert u.dimensions is not None

        u.dimensions = None
        assert u.dimensions is None
