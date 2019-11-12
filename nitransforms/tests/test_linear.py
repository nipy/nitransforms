# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of linear transforms."""
import os
import pytest
import numpy as np
from subprocess import check_call
import shutil
import h5py

import nibabel as nb
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from .. import linear as ntl
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


@pytest.mark.xfail(raises=(FileNotFoundError, NotImplementedError))
@pytest.mark.parametrize('image_orientation', ['RAS', 'LAS', 'LPS', 'oblique'])
@pytest.mark.parametrize('sw_tool', ['itk', 'fsl', 'afni', 'fs'])
def test_linear_save(tmpdir, data_path, get_testdata, image_orientation, sw_tool):
    """Check implementation of exporting affines to formats."""
    tmpdir.chdir()
    img = get_testdata[image_orientation]
    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    xfm = ntl.Affine(T)
    xfm.reference = img

    ext = ''
    if sw_tool == 'itk':
        ext = '.tfm'
    elif sw_tool == 'fs':
        ext = '.lta'

    xfm_fname1 = 'M.%s%s' % (sw_tool, ext)
    xfm.to_filename(xfm_fname1, fmt=sw_tool)

    xfm_fname2 = str(data_path / 'affine-%s.%s%s') % (
        image_orientation, sw_tool, ext)
    assert_affines_by_filename(xfm_fname1, xfm_fname2)


@pytest.mark.parametrize('image_orientation', [
    'RAS', 'LAS', 'LPS',  'oblique',
])
@pytest.mark.parametrize('sw_tool', ['itk', 'fsl', 'afni'])
def test_apply_linear_transform(
        tmpdir,
        get_testdata,
        image_orientation,
        sw_tool
):
    """Check implementation of exporting affines to formats."""
    tmpdir.chdir()

    img = get_testdata[image_orientation]
    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])

    # AFNI's implicitly deobliques datasets
    # if sw_tool == "afni":
    #     T = _afni_deoblique()
    xfm = ntl.Affine(T)
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

    nt_moved = xfm.apply(img, order=0)
    diff = sw_moved.get_fdata() - nt_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < TESTS_BORDER_TOLERANCE


def test_Affine_to_x5(tmpdir, data_path):
    """Test affine's operations."""
    tmpdir.chdir()
    aff = ntl.Affine()
    with h5py.File('xfm.x5', 'w') as f:
        aff._to_hdf5(f.create_group('Affine'))

    aff.reference = data_path / 'someones_anatomy.nii.gz'
    with h5py.File('withref-xfm.x5', 'w') as f:
        aff._to_hdf5(f.create_group('Affine'))


def test_concatenation(data_path):
    """Check concatenation of affines."""
    aff = ntl.Affine(reference=data_path / 'someones_anatomy.nii.gz')
    x = [(0., 0., 0.), (1., 1., 1.), (-1., -1., -1.)]
    assert np.all((aff + ntl.Affine())(x) == x)
    assert np.all((aff + ntl.Affine())(x, inverse=True) == x)
