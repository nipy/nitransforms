"""Tests of the base module."""
import numpy as np
import pytest

from ..base import ImageGrid


@pytest.mark.parametrize('image_orientation', ['RAS', 'LAS', 'LPS', 'oblique'])
def test_ImageGrid(get_data, image_orientation):
    """Check the grid object."""
    im = get_data[image_orientation]

    img = ImageGrid(im)
    assert np.all(img.affine == np.linalg.inv(img.inverse))

    # nd index / coords
    idxs = img.ndindex
    coords = img.ndcoords
    assert len(idxs.shape) == len(coords.shape) == 2
    assert idxs.shape[0] == coords.shape[0] == img.ndim == 3
    assert idxs.shape[1] == coords.shape[1] == img.nvox == np.prod(im.shape)
