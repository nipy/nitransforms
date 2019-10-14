# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of the transform module."""
import os
import pytest

from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from nibabel.tmpdirs import InTemporaryDirectory
from .. import linear as nbl
from .checkaffines import assert_affines_by_filename


@pytest.mark.parametrize('image_orientation', [
    'RAS', 'LAS', 'LPS',
    # 'oblique',
])
@pytest.mark.parametrize('sw_tool', ['itk', 'fsl', 'afni'])
def test_affines_save(data_path, get_data, image_orientation, sw_tool):
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
