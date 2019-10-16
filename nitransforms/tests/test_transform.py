# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of the transform module."""
import os
import pytest
import numpy as np
from subprocess import check_call
import shutil

import nibabel as nb
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from nibabel.tmpdirs import InTemporaryDirectory
from .. import linear as nbl
from .utils import assert_affines_by_filename

TESTS_BORDER_TOLERANCE = 0.05
APPLY_LINEAR_CMD = {
    'fsl': """\
flirt -setbackground 0 -interp nearestneighbour -in {moving} -ref {reference} \
-applyxfm -init {transform} -out resampled.nii.gz\
""".format,
    'itk': """\
antsApplyTransforms -d 3 -r {reference} -i {moving} \
-o resampled.nii.gz -n NearestNeighbor -t {transform} --float\
""".format,
    'afni': """\
3dAllineate -base {reference} -input {moving} \
-prefix resampled.nii.gz -1Dmatrix_apply {transform} -final NN\
""".format,
}


@pytest.mark.xfail(reason="Not fully implemented")
@pytest.mark.parametrize('image_orientation', [
    'RAS', 'LAS', 'LPS',  # 'oblique',
])
@pytest.mark.parametrize('sw_tool', ['itk', 'fsl', 'afni'])
def test_linear_load(tmpdir, data_path, get_data, image_orientation, sw_tool):
    """Check implementation of loading affines from formats."""
    tmpdir.chdir()

    img = get_data[image_orientation]
    img.to_filename('img.nii.gz')

    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    xfm = nbl.Affine(T)
    xfm.reference = img

    ext = ''
    if sw_tool == 'itk':
        ext = '.tfm'

    fname = 'affine-%s.%s%s' % (image_orientation, sw_tool, ext)
    xfm_fname = os.path.join(data_path, fname)
    fmt = fname.split('.')[-1]

    if sw_tool == 'fsl':
        with pytest.raises(ValueError):
            loaded = nbl.load(xfm_fname, fmt=fmt)
        with pytest.raises(ValueError):
            loaded = nbl.load(xfm_fname, fmt=fmt, reference='img.nii.gz')
        with pytest.raises(ValueError):
            loaded = nbl.load(xfm_fname, fmt=fmt, moving='img.nii.gz')

        loaded = nbl.load(
            xfm_fname, fmt=fmt, moving='img.nii.gz', reference='img.nii.gz'
        )
    elif sw_tool == 'afni':
        with pytest.raises(ValueError):
            loaded = nbl.load(xfm_fname, fmt=fmt)

        loaded = nbl.load(xfm_fname, fmt=fmt, reference='img.nii.gz')
    elif sw_tool == 'itk':
        loaded = nbl.load(xfm_fname, fmt=fmt)

    assert loaded == xfm


@pytest.mark.parametrize('image_orientation', [
    'RAS', 'LAS', 'LPS',  # 'oblique',
])
@pytest.mark.parametrize('sw_tool', ['itk', 'fsl', 'afni'])
def test_linear_save(data_path, get_data, image_orientation, sw_tool):
    """Check implementation of exporting affines to formats."""
    img = get_data[image_orientation]
    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    xfm = nbl.Affine(T)
    xfm.reference = img

    ext = ''
    if sw_tool == 'itk':
        ext = '.tfm'

    with InTemporaryDirectory():
        xfm_fname1 = 'M.%s%s' % (sw_tool, ext)
        xfm.to_filename(xfm_fname1, fmt=sw_tool)

        xfm_fname2 = os.path.join(
            data_path, 'affine-%s.%s%s' % (image_orientation, sw_tool, ext))
        assert_affines_by_filename(xfm_fname1, xfm_fname2)


@pytest.mark.parametrize('image_orientation', [
    'RAS', 'LAS', 'LPS',  # 'oblique',
])
@pytest.mark.parametrize('sw_tool', ['itk', 'fsl', 'afni'])
def test_apply_linear_transform(
        tmpdir,
        data_path,
        get_data,
        image_orientation,
        sw_tool
):
    """Check implementation of exporting affines to formats."""
    tmpdir.chdir()

    img = get_data[image_orientation]
    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    xfm = nbl.Affine(T)
    xfm.reference = img

    ext = ''
    if sw_tool == 'itk':
        ext = '.tfm'

    img.to_filename('img.nii.gz')
    xfm_fname = 'M.%s%s' % (sw_tool, ext)
    xfm.to_filename(xfm_fname, fmt=sw_tool)

    cmd = APPLY_LINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=os.path.abspath('img.nii.gz'),
        moving=os.path.abspath('img.nii.gz'))

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip("Command {} not found on host".format(exe))

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load('resampled.nii.gz')

    nt_moved = xfm.resample(img, order=0)
    diff = sw_moved.get_fdata() - nt_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < TESTS_BORDER_TOLERANCE
