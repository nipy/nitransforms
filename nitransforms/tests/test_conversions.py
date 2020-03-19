# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Conversions between formats."""
import numpy as np
from .. import linear as _l


def test_conversions(data_path):
    """Check conversions between formats."""
    lta = _l.load(data_path / "regressions" / "robust_register.lta", fmt="lta")
    itk = _l.load(data_path / "regressions" / "robust_register.tfm", fmt="itk")

    assert np.allclose(lta.matrix, itk.matrix)
