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
from numpy.testing import assert_equal

from MDAnalysis.coordinates.timestep import Timestep
from MDAnalysis.lib.mdamath import triclinic_vectors, triclinic_box


@pytest.fixture
def ts():
    return Timestep(10)


class TestTimestepDimensionsValidation:
    """Regression tests for Issue #5534: Timestep.dimensions non-finite validation."""

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_dimensions_rejects_nonfinite(self, ts, value):
        with pytest.raises(ValueError, match="finite"):
            ts.dimensions = [value, 2, 3, 90, 90, 90]

    def test_dimensions_rejects_invalid_shape(self, ts):
        with pytest.raises(ValueError, match="six-element"):
            ts.dimensions = [10, 10, 10]

    def test_dimensions_accepts_valid_and_none(self, ts):
        ts.dimensions = [10, 10, 10, 90, 90, 90]
        assert ts.dimensions is not None
        assert ts.dimensions[0] == 10.0

        ts.dimensions = None
        assert ts.dimensions is None

    def test_dimensions_zero_box(self, ts):
        ts.dimensions = np.zeros(6)
        assert ts.dimensions is None

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_triclinic_vectors_rejects_nonfinite(self, value):
        dim = [value, 10, 10, 90, 90, 90]
        box = triclinic_vectors(dim)
        assert_equal(box, np.zeros((3, 3)))

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_triclinic_box_rejects_nonfinite(self, value):
        x = [value, 0, 0]
        y = [0, 10, 0]
        z = [0, 0, 10]
        box = triclinic_box(x, y, z)
        assert_equal(box, np.zeros(6, dtype=np.float32))

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_triclinic_dimensions_nonfinite_setter(self, ts, value):
        ts.triclinic_dimensions = [[value, 0, 0], [0, 10, 0], [0, 0, 10]]
        assert ts.dimensions is None
        assert ts.triclinic_dimensions is None
