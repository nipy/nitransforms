import os
import tempfile

import numpy as np
import nibabel as nb
from nitransforms.base import SurfaceMesh
from nitransforms.surface import SurfaceCoordinateTransform, SurfaceResampler


# def test_surface_transform_npz():
#     mat = sparse.random(10, 10, density=0.5)
#     xfm = SurfaceCoordinateTransform(mat)
#     fn = tempfile.mktemp(suffix=".npz")
#     print(fn)
#     xfm.to_filename(fn)
#
#     xfm2 = SurfaceCoordinateTransform.from_filename(fn)
#     try:
#         assert xfm.mat.shape == xfm2.mat.shape
#         np.testing.assert_array_equal(xfm.mat.data, xfm2.mat.data)
#         np.testing.assert_array_equal(xfm.mat.indices, xfm2.mat.indices)
#         np.testing.assert_array_equal(xfm.mat.indptr, xfm2.mat.indptr)
#     except Exception:
#         os.remove(fn)
#         raise
#     os.remove(fn)


# def test_surface_transform_normalization():
#     mat = np.random.uniform(size=(20, 10))
#     xfm = SurfaceCoordinateTransform(mat)
#     x = np.random.uniform(size=(5, 20))
#     y_element = xfm.apply(x, normalize="element")
#     np.testing.assert_array_less(y_element.sum(axis=1), x.sum(axis=1))
#     y_sum = xfm.apply(x, normalize="sum")
#     np.testing.assert_allclose(y_sum.sum(axis=1), x.sum(axis=1))
#     y_none = xfm.apply(x, normalize="none")
#     assert y_none.sum() != y_element.sum()
#     assert y_none.sum() != y_sum.sum()


def test_SurfaceResampler(testdata_path):
    dif_tol = 0.001
    sphere_reg_path = testdata_path / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsLR_desc-reg_sphere.surf.gii"
    fslr_sphere_path = testdata_path / "tpl-fsLR_hemi-R_den-32k_sphere.surf.gii"
    shape_path = testdata_path / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_thickness.shape.gii"
    ref_resampled_thickness_path = testdata_path / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsLR_thickness.shape.gii"

    fslr_sphere = SurfaceMesh(nb.load(fslr_sphere_path))
    sphere_reg = SurfaceMesh(nb.load(sphere_reg_path))
    subj_thickness = nb.load(shape_path)

    reference = fslr_sphere
    moving = sphere_reg
    # compare results to what connecome workbench produces
    resampling = SurfaceResampler(reference, moving)
    resampled_thickness = resampling.apply(subj_thickness.agg_data(), normalize='element')
    ref_resampled = nb.load(ref_resampled_thickness_path).agg_data()

    max_dif = np.abs(resampled_thickness.astype(np.float32) - ref_resampled).max()
    assert max_dif < dif_tol

    # test file io
    fn = tempfile.mktemp(suffix=".h5")
    try:
        resampling.to_filename(fn)
        resampling2 = SurfaceResampler.from_filename(fn)

        assert resampling2 == resampling2
        assert np.all(resampling2.reference._coords == resampling.reference._coords)
        assert np.all(resampling2.reference._triangles == resampling.reference._triangles)
        assert np.all(resampling2.reference._coords == resampling.reference._coords)
        assert np.all(resampling2.moving._triangles == resampling.moving._triangles)

        resampled_thickness2 = resampling2.apply(subj_thickness.agg_data(), normalize='element')
        assert np.all(resampled_thickness2 == resampled_thickness)
    except Exception:
        os.remove(fn)
        raise
    os.remove(fn)