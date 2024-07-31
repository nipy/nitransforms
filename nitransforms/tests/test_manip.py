# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of nonlinear transforms."""
import pytest

import numpy as np
from ..manip import TransformChain
from ..linear import Affine

FMT = {"lta": "fs", "tfm": "itk"}


@pytest.mark.parametrize("ext0", ["lta", "tfm"])
@pytest.mark.parametrize("ext1", ["lta", "tfm"])
@pytest.mark.parametrize("ext2", ["lta", "tfm"])
def test_collapse_affines(tmp_path, data_path, ext0, ext1, ext2):
    """Check whether affines are correctly collapsed."""
    chain = TransformChain(
        [
            Affine.from_filename(
                data_path
                / "regressions"
                / f"from-fsnative_to-scanner_mode-image.{ext0}",
                fmt=f"{FMT[ext0]}",
            ),
            Affine.from_filename(
                data_path / "regressions" / f"from-scanner_to-bold_mode-image.{ext1}",
                fmt=f"{FMT[ext1]}",
            ),
        ]
    )
    assert np.allclose(
        chain.asaffine().matrix,
        Affine.from_filename(
            data_path / "regressions" / f"from-fsnative_to-bold_mode-image.{ext2}",
            fmt=f"{FMT[ext2]}",
        ).matrix,
    )
