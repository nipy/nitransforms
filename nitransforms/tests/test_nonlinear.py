# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of nonlinear transforms."""
import os
import shutil
from subprocess import check_call
import pytest

import numpy as np
import nibabel as nb
from ..io.base import TransformFileError
from ..nonlinear import DisplacementsFieldTransform, load as nlload
from ..io.itk import ITKDisplacementsField

TESTS_BORDER_TOLERANCE = 0.05
APPLY_NONLINEAR_CMD = {
    'itk': """\
antsApplyTransforms -d 3 -r {reference} -i {moving} \
-o resampled.nii.gz -n NearestNeighbor -t {transform} --float\
""".format,
    'afni': """\
3dNwarpApply -nwarp {transform} -source {moving} \
-master {reference} -interp NN -prefix resampled.nii.gz
""".format,
}


@pytest.mark.parametrize('size', [(20, 20, 20), (20, 20, 20, 3)])
def test_itk_disp_load(size):
    """Checks field sizes."""
    with pytest.raises(TransformFileError):
        ITKDisplacementsField.from_image(
            nb.Nifti1Image(np.zeros(size), None, None))


@pytest.mark.parametrize('size', [(20, 20, 20), (20, 20, 20, 1, 3)])
def test_displacements_bad_sizes(size):
    """Checks field sizes."""
    with pytest.raises(ValueError):
        DisplacementsFieldTransform(
            nb.Nifti1Image(np.zeros(size), None, None))


def test_itk_disp_load_intent():
    """Checks whether the NIfTI intent is fixed."""
    with pytest.warns(UserWarning):
        field = ITKDisplacementsField.from_image(
            nb.Nifti1Image(np.zeros((20, 20, 20, 1, 3)), None, None))

    assert field.header.get_intent()[0] == 'vector'


@pytest.mark.xfail(reason="Oblique datasets not fully implemented")
@pytest.mark.parametrize('image_orientation', ['RAS', 'LAS', 'LPS', 'oblique'])
@pytest.mark.parametrize('sw_tool', ['itk', 'afni'])
@pytest.mark.parametrize('axis', [0, 1, 2, (0, 1), (1, 2), (0, 1, 2)])
def test_displacements_field1(tmp_path, get_testdata, image_orientation, sw_tool, axis):
    """Check a translation-only field on one or more axes, different image orientations."""
    os.chdir(str(tmp_path))
    nii = get_testdata[image_orientation]
    nii.to_filename('reference.nii.gz')
    fieldmap = np.zeros((*nii.shape[:3], 1, 3), dtype='float32')
    fieldmap[..., axis] = -10.0

    _hdr = nii.header.copy()
    if sw_tool in ('itk', ):
        _hdr.set_intent('vector')
    _hdr.set_data_dtype('float32')

    xfm_fname = 'warp.nii.gz'
    field = nb.Nifti1Image(fieldmap, nii.affine, _hdr)
    field.to_filename(xfm_fname)

    xfm = nlload(xfm_fname, fmt=sw_tool)

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=tmp_path / 'reference.nii.gz',
        moving=tmp_path / 'reference.nii.gz')

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip("Command {} not found on host".format(exe))

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load('resampled.nii.gz')

    nt_moved = xfm.apply(nii, order=0)
    nt_moved.to_filename('nt_resampled.nii.gz')
    diff = sw_moved.get_fdata() - nt_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < TESTS_BORDER_TOLERANCE


@pytest.mark.parametrize('sw_tool', ['itk', 'afni'])
def test_displacements_field2(tmp_path, testdata_path, sw_tool):
    """Check a translation-only field on one or more axes, different image orientations."""
    os.chdir(str(tmp_path))
    img_fname = testdata_path / 'tpl-OASIS30ANTs_T1w.nii.gz'
    xfm_fname = testdata_path / 'ds-005_sub-01_from-OASIS_to-T1_warp_{}.nii.gz'.format(sw_tool)

    xfm = nlload(xfm_fname, fmt=sw_tool)

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD[sw_tool](
        transform=xfm_fname,
        reference=img_fname,
        moving=img_fname)

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip("Command {} not found on host".format(exe))

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load('resampled.nii.gz')

    nt_moved = xfm.apply(img_fname, order=0)
    nt_moved.to_filename('nt_resampled.nii.gz')
    diff = sw_moved.get_fdata() - nt_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < TESTS_BORDER_TOLERANCE
