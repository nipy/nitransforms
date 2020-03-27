# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of nonlinear transforms."""
import os
import shutil
from subprocess import check_call
import pytest

import numpy as np
import nibabel as nb
from ..manip import load as _load
from .test_nonlinear import (
    TESTS_BORDER_TOLERANCE,
    APPLY_NONLINEAR_CMD,
)


def test_itk_h5(tmp_path, testdata_path):
    """Check a translation-only field on one or more axes, different image orientations."""
    os.chdir(str(tmp_path))
    img_fname = testdata_path / "T1w_scanner.nii.gz"
    xfm_fname = testdata_path / "ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"

    xfm = _load(xfm_fname)

    assert len(xfm) == 2

    ref_fname = tmp_path / "reference.nii.gz"
    nb.Nifti1Image(
        np.zeros(xfm.reference.shape, dtype='uint16'),
        xfm.reference.affine,
    ).to_filename(str(ref_fname))

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD['itk'](
        transform=xfm_fname,
        reference=ref_fname,
        moving=img_fname)

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load('resampled.nii.gz')

    nt_moved = xfm.apply(img_fname, order=0)
    nt_moved.to_filename('nt_resampled.nii.gz')
    diff = sw_moved.get_fdata() - nt_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < TESTS_BORDER_TOLERANCE
