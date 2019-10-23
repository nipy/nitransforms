"""Tests of the base module."""
import numpy as np
import pytest

from ..base import ImageGrid


@pytest.mark.parametrize('image_orientation', ['RAS', 'LAS', 'LPS', 'oblique'])
def test_ImageGrid(get_testdata, image_orientation):
    """Check the grid object."""
    im = get_testdata[image_orientation]

    img = ImageGrid(im)
    assert np.allclose(img.affine, np.linalg.inv(img.inverse))

    # Test ras2vox and vox2ras conversions
    ijk = [[10, 10, 10], [40, 4, 20], [0, 0, 0], [s - 1 for s in im.shape[:3]]]
    xyz = [img._affine.dot(idx + [1])[:-1] for idx in ijk]

    assert np.allclose(img.ras(ijk[0]), xyz[0])
    assert np.allclose(np.round(img.index(xyz[0])), ijk[0])
    assert np.allclose(img.ras(ijk), xyz)
    assert np.allclose(np.round(img.index(xyz)), ijk)

    # nd index / coords
    idxs = img.ndindex
    coords = img.ndcoords
    assert len(idxs.shape) == len(coords.shape) == 2
    assert idxs.shape[0] == coords.shape[0] == img.ndim == 3
    assert idxs.shape[1] == coords.shape[1] == img.nvox == np.prod(im.shape)


def test_ImageGrid_load(data_path, get_testdata):
    """Check that images can be objects or paths and equality."""
    im1 = get_testdata['RAS']
    im2 = data_path / 'someones_anatomy.nii.gz'

    assert ImageGrid(im1) == ImageGrid(im2)
