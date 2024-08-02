import tempfile

import numpy as np
import nibabel as nb
import pytest
from scipy import sparse
from nitransforms.base import SurfaceMesh
from nitransforms.surface import (
    SurfaceTransformBase,
    SurfaceCoordinateTransform,
    SurfaceResampler
)

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

def test_SurfaceTransformBase(testdata_path):
    # note these transformations are a bit of a weird use of surface transformation, but I'm
    # just testing the base class and the io
    sphere_reg_path = (
        testdata_path
        / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsLR_desc-reg_sphere.surf.gii"
    )
    pial_path = testdata_path / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_pial.surf.gii"

    sphere_reg = SurfaceMesh(nb.load(sphere_reg_path))
    pial = SurfaceMesh(nb.load(pial_path))
    stfb = SurfaceTransformBase(sphere_reg, pial)

    # test loading from filenames
    stfb_ff = SurfaceTransformBase.from_filename(sphere_reg_path, pial_path)
    assert stfb_ff == stfb

    # test inversion and setting
    stfb_i = ~stfb
    stfb.reference = pial
    stfb.moving = sphere_reg
    assert np.all(stfb_i._reference._coords == stfb._reference._coords)
    assert np.all(stfb_i._reference._triangles == stfb._reference._triangles)
    assert np.all(stfb_i._moving._coords == stfb._moving._coords)
    assert np.all(stfb_i._moving._triangles == stfb._moving._triangles)
    # test equality
    assert stfb_i == stfb


def test_SurfaceCoordinateTransform(testdata_path):
    # note these transformations are a bit of a weird use of surface transformation, but I'm
    # just testing the class and the io
    sphere_reg_path = (
        testdata_path
        / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsLR_desc-reg_sphere.surf.gii"
    )
    pial_path = testdata_path / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_pial.surf.gii"
    fslr_sphere_path = testdata_path / "tpl-fsLR_hemi-R_den-32k_sphere.surf.gii"

    sphere_reg = SurfaceMesh(nb.load(sphere_reg_path))
    pial = SurfaceMesh(nb.load(pial_path))
    fslr_sphere = SurfaceMesh(nb.load(fslr_sphere_path))

    # test mesh correspondence test
    with pytest.raises(ValueError):
        sct = SurfaceCoordinateTransform(fslr_sphere, pial)

    # test loading from filenames
    sct = SurfaceCoordinateTransform(pial, sphere_reg)
    sctf = SurfaceCoordinateTransform.from_filename(reference_path=pial_path,
                                                    moving_path=sphere_reg_path)
    assert sct == sctf

    # test mapping
    assert np.all(sct.map(sct.moving._coords[:100], inverse=True) == sct.reference._coords[:100])
    assert np.all(sct.map(sct.reference._coords[:100]) == sct.moving._coords[:100])
    with pytest.raises(NotImplementedError):
        sct.map(sct.moving._coords[0])

    # test inversion and addition
    scti = ~sct

    assert sct + scti == SurfaceCoordinateTransform(pial, pial)
    assert scti + sct == SurfaceCoordinateTransform(sphere_reg, sphere_reg)

    sct.reference = sphere_reg
    sct.moving = pial
    assert np.all(scti.reference._coords == sct.reference._coords)
    assert np.all(scti.reference._triangles == sct.reference._triangles)
    assert scti == sct


def test_SurfaceCoordinateTransformIO(testdata_path, tmpdir):
    sphere_reg_path = (
        testdata_path
        / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsLR_desc-reg_sphere.surf.gii"
    )
    pial_path = testdata_path / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_pial.surf.gii"

    sct = SurfaceCoordinateTransform(pial_path, sphere_reg_path)
    fn = tempfile.mktemp(suffix=".h5")
    sct.to_filename(fn)
    sct2 = SurfaceCoordinateTransform.from_filename(fn)
    assert sct == sct2


def test_ProjectUnproject(testdata_path):

    sphere_reg_path = (
        testdata_path
        / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsLR_desc-reg_sphere.surf.gii"
    )
    fslr_sphere_path = testdata_path / "tpl-fsLR_hemi-R_den-32k_sphere.surf.gii"
    subj_fsaverage_sphere_path = (
        testdata_path
        / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsaverage_desc-reg_sphere.surf.gii"
    )
    fslr_fsaverage_sphere_path = (
        testdata_path
        / "tpl-fsLR_space-fsaverage_hemi-R_den-32k_sphere.surf.gii"
    )
    pial_path = testdata_path / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_pial.surf.gii"

    # test project-unproject funcitonality
    projunproj = SurfaceResampler(sphere_reg_path, fslr_sphere_path)
    with pytest.raises(ValueError):
        projunproj.apply(pial_path)
    transformed = projunproj.apply(fslr_fsaverage_sphere_path)
    projunproj_ref = nb.load(subj_fsaverage_sphere_path)
    assert (projunproj_ref.agg_data()[0] - transformed._coords).max() < 0.0005
    assert np.all(transformed._triangles == projunproj_ref.agg_data()[1])


def test_SurfaceResampler(testdata_path, tmpdir):
    dif_tol = 0.001
    fslr_sphere_path = (
        testdata_path
        / "tpl-fsLR_hemi-R_den-32k_sphere.surf.gii"
    )
    shape_path = (
        testdata_path
        / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_thickness.shape.gii"
    )
    ref_resampled_thickness_path = (
        testdata_path
        / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsLR_thickness.shape.gii"
    )
    pial_path = (
        testdata_path / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_pial.surf.gii"
    )
    sphere_reg_path = (
        testdata_path
        / "sub-sid000005_ses-budapest_acq-MPRAGE_hemi-R_space-fsLR_desc-reg_sphere.surf.gii"
    )

    fslr_sphere = SurfaceMesh(nb.load(fslr_sphere_path))
    sphere_reg = SurfaceMesh(nb.load(sphere_reg_path))
    subj_thickness = nb.load(shape_path)

    with pytest.raises(ValueError):
        SurfaceResampler(sphere_reg_path, pial_path)
    with pytest.raises(ValueError):
        SurfaceResampler(pial_path, sphere_reg_path)

    reference = fslr_sphere
    moving = sphere_reg
    # compare results to what connectome workbench produces
    resampling = SurfaceResampler(reference, moving)
    resampled_thickness = resampling.apply(subj_thickness.agg_data(), normalize='element')
    ref_resampled = nb.load(ref_resampled_thickness_path).agg_data()

    max_dif = np.abs(resampled_thickness.astype(np.float32) - ref_resampled).max()
    assert max_dif < dif_tol

    with pytest.raises(ValueError):
        SurfaceResampler(reference, moving, mat=resampling.mat[:, :10000])
    with pytest.raises(ValueError):
        SurfaceResampler(reference, moving, mat=resampling.mat[:10000, :])
    with pytest.raises(ValueError):
        resampling.reference = reference
    with pytest.raises(ValueError):
        resampling.moving = moving
    with pytest.raises(NotImplementedError):
        _ = SurfaceResampler(reference, moving, "foo")

    # test file io
    fn = tempfile.mktemp(suffix=".h5")
    resampling.to_filename(fn)
    resampling2 = SurfaceResampler.from_filename(fn)

    # assert resampling2 == resampling
    assert np.allclose(resampling2.reference._coords, resampling.reference._coords)
    assert np.all(resampling2.reference._triangles == resampling.reference._triangles)
    assert np.allclose(resampling2.reference._coords, resampling.reference._coords)
    assert np.all(resampling2.moving._triangles == resampling.moving._triangles)

    resampled_thickness2 = resampling2.apply(subj_thickness.agg_data(), normalize='element')
    assert np.all(resampled_thickness2 == resampled_thickness)

    # test loading with a csr
    assert isinstance(resampling.mat, sparse.csr_array)
    resampling2a = SurfaceResampler(reference, moving, mat=resampling.mat)
    resampled_thickness2a = resampling2a.apply(subj_thickness.agg_data(), normalize='element')
    assert np.all(resampled_thickness2a == resampled_thickness)

    with pytest.raises(ValueError):
        _ = SurfaceResampler(moving, reference, mat=resampling.mat)

    # test map
    assert np.all(resampling.map(np.array([[0, 0, 0]])) == np.array([[0, 0, 0]]))

    # test loading from surfaces
    resampling3 = SurfaceResampler.from_filename(reference_path=fslr_sphere_path,
                                                 moving_path=sphere_reg_path)
    assert resampling3 == resampling
    resampled_thickness3 = resampling3.apply(subj_thickness.agg_data(), normalize='element')
    assert np.all(resampled_thickness3 == resampled_thickness)
