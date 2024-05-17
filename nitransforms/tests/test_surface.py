import os
import tempfile

import numpy as np
import scipy.sparse as sparse

from nitransforms.surface import SurfaceCoordinateTransform


def test_surface_transform_x5():
    mat = sparse.random(10, 10, density=0.5)
    xfm = SurfaceCoordinateTransform(mat)
    fn = tempfile.mktemp(suffix=".h5")
    print(fn)
    xfm.to_filename(fn)

    xfm2 = SurfaceCoordinateTransform.from_filename(fn)
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
    xfm = SurfaceCoordinateTransform(mat)
    fn = tempfile.mktemp(suffix=".npz")
    print(fn)
    xfm.to_filename(fn)

    xfm2 = SurfaceCoordinateTransform.from_filename(fn)
    try:
        assert xfm.mat.shape == xfm2.mat.shape
        np.testing.assert_array_equal(xfm.mat.data, xfm2.mat.data)
        np.testing.assert_array_equal(xfm.mat.indices, xfm2.mat.indices)
        np.testing.assert_array_equal(xfm.mat.indptr, xfm2.mat.indptr)
    except Exception:
        os.remove(fn)
        raise
    os.remove(fn)


def test_surface_transform_normalization():
    mat = np.random.uniform(size=(20, 10))
    xfm = SurfaceCoordinateTransform(mat)
    x = np.random.uniform(size=(5, 20))
    y_element = xfm.apply(x, normalize="element")
    np.testing.assert_array_less(y_element.sum(axis=1), x.sum(axis=1))
    y_sum = xfm.apply(x, normalize="sum")
    np.testing.assert_allclose(y_sum.sum(axis=1), x.sum(axis=1))
    y_none = xfm.apply(x, normalize="none")
    assert y_none.sum() != y_element.sum()
    assert y_none.sum() != y_sum.sum()
