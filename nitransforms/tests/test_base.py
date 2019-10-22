"""Tests of the base module"""
import numpy as np

from ..base import ImageSpace


def test_ImageSpace(get_data):
    im = get_data['RAS']

    img = ImageSpace(im)
    assert np.all(img.affine == np.linalg.inv(img.inverse))

    # nd index / coords
    idxs = img.ndindex
    coords = img.ndcoords
    assert len(idxs.shape) == len(coords.shape) == 2
    assert idxs.shape[0] == coords.shape[0] == img.ndim == 3
    assert idxs.shape[1] == coords.shape[1] == img.nvox == np.prod(im.shape)
