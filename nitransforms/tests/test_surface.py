import os
import tempfile

import numpy as np
import scipy.sparse as sparse

from nitransforms.surface import SurfaceTransform


def test_surface_transform_x5():
    mat = sparse.random(10, 10, density=0.5)
    xfm = SurfaceTransform(mat)
    fn = tempfile.mktemp(suffix=".h5")
    print(fn)
    xfm.to_filename(fn)

    xfm2 = SurfaceTransform.from_filename(fn)
    try:
        assert xfm.mat.shape == xfm2.mat.shape
        np.testing.assert_array_equal(xfm.mat.data, xfm2.mat.data)
        np.testing.assert_array_equal(xfm.mat.indices, xfm2.mat.indices)
        np.testing.assert_array_equal(xfm.mat.indptr, xfm2.mat.indptr)
    except Exception:
        os.remove(fn)
        raise
    os.remove(fn)


def test_surface_transform_npz():
    mat = sparse.random(10, 10, density=0.5)
    xfm = SurfaceTransform(mat)
    fn = tempfile.mktemp(suffix=".npz")
    print(fn)
    xfm.to_filename(fn)

    xfm2 = SurfaceTransform.from_filename(fn)
    try:
        assert xfm.mat.shape == xfm2.mat.shape
        np.testing.assert_array_equal(xfm.mat.data, xfm2.mat.data)
        np.testing.assert_array_equal(xfm.mat.indices, xfm2.mat.indices)
        np.testing.assert_array_equal(xfm.mat.indptr, xfm2.mat.indptr)
    except Exception:
        os.remove(fn)
        raise
    os.remove(fn)
