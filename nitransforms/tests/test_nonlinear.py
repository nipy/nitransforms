# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of nonlinear transforms."""
import os
import shutil
from subprocess import check_call
import pytest

import numpy as np
import nibabel as nb
from ..nonlinear import DisplacementsFieldTransform

TESTS_BORDER_TOLERANCE = 0.05
APPLY_NONLINEAR_CMD = {
    'itk': """\
antsApplyTransforms -d 3 -r {reference} -i {moving} \
-o resampled.nii.gz -n NearestNeighbor -t {transform} --float\
""".format,
}


@pytest.mark.parametrize('sw_tool', ['itk'])
def test_displacements_field(tmp_path, data_path, sw_tool):
    os.chdir(str(tmp_path))
    img_fname = os.path.join(data_path, 'tpl-OASIS30ANTs_T1w.nii.gz')
    xfm_fname = os.path.join(
        data_path, 'ds-005_sub-01_from-OASIS_to-T1_warp.nii.gz')
    ants_warp = nb.load(xfm_fname)
    field = nb.Nifti1Image(
        np.squeeze(np.asanyarray(ants_warp.dataobj)),
        ants_warp.affine, ants_warp.header
    )

    xfm = DisplacementsFieldTransform(field)

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=img_fname,
        moving=img_fname)

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip("Command {} not found on host".format(exe))

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load('resampled.nii.gz')

    nt_moved = xfm.resample(img_fname, order=0)
    diff = sw_moved.get_fdata() - nt_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < TESTS_BORDER_TOLERANCE
