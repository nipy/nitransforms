# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Conversions between formats."""
import numpy as np
import pytest
from .. import linear as _l


@pytest.mark.parametrize('filename', [
    "from-fsnative_to-bold_mode-image",
    "from-fsnative_to-scanner_mode-image",
    "from-scanner_to-bold_mode-image",
    "from-scanner_to-fsnative_mode-image",
])
def test_lta2itk_conversions(data_path, filename):
    """Check conversions between formats."""
    lta = _l.load(data_path / "regressions" / ".".join((filename, "lta")),
                  fmt="lta")
    itk = _l.load(data_path / "regressions" / ".".join((filename, "tfm")),
                  fmt="itk")
    assert np.allclose(lta.matrix, itk.matrix)


def test_concatenation(data_path):
    """Check replacement to lta_concat."""
    lta0 = _l.load(data_path / "regressions" / "from-scanner_to-fsnative_mode-image.lta",
                   fmt="lta")
    lta1 = _l.load(data_path / "regressions" / "from-fsnative_to-bold_mode-image.lta", fmt="lta")

    lta_combined = _l.load(data_path / "regressions" / "from-scanner_to-bold_mode-image.lta",
                           fmt="lta")

    assert np.allclose(lta1.matrix.dot(lta0.matrix), lta_combined.matrix)
