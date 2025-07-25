# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of nonlinear transforms."""

import os
import pytest

import numpy as np
import nibabel as nb
from nitransforms.resampling import apply
from nitransforms.base import TransformError, ImageGrid
from nitransforms.io.base import TransformFileError
from nitransforms.nonlinear import (
    BSplineFieldTransform,
    DenseFieldTransform,
)
from ..io.itk import ITKDisplacementsField


SOME_TEST_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 2.0, 3.0],
    [10.0, -10.0, 5.0],
    [-5.0, 7.0, -2.0],
    [12.0, 0.0, -11.0],
])

@pytest.mark.parametrize("size", [(20, 20, 20), (20, 20, 20, 3)])
def test_itk_disp_load(size):
    """Checks field sizes."""
    with pytest.raises(TransformFileError):
        ITKDisplacementsField.from_image(
            nb.Nifti1Image(np.zeros(size), np.eye(4), None)
        )


@pytest.mark.parametrize("size", [(20, 20, 20), (20, 20, 20, 2, 3), (20, 20, 20, 1, 4)])
def test_displacements_bad_sizes(size):
    """Checks field sizes."""
    with pytest.raises(TransformError):
        DenseFieldTransform(nb.Nifti1Image(np.zeros(size), np.eye(4), None))


def test_itk_disp_load_intent():
    """Checks whether the NIfTI intent is fixed."""
    with pytest.warns(UserWarning):
        field = ITKDisplacementsField.from_image(
            nb.Nifti1Image(np.zeros((20, 20, 20, 1, 3)), np.eye(4), None)
        )

    assert field.header.get_intent()[0] == "vector"


def test_displacements_init():
    identity1 = DenseFieldTransform(
        np.zeros((10, 10, 10, 3)),
        reference=nb.Nifti1Image(np.zeros((10, 10, 10, 3)), np.eye(4), None),
    )
    identity2 = DenseFieldTransform(
        reference=nb.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4), None),
    )

    assert np.array_equal(identity1._field, identity2._field)
    assert np.array_equal(identity1._deltas, identity2._deltas)

    with pytest.raises(TransformError):
        DenseFieldTransform()
    with pytest.raises(TransformError):
        DenseFieldTransform(np.zeros((10, 10, 10, 3)))
    with pytest.raises(TransformError):
        DenseFieldTransform(
            np.zeros((10, 10, 10, 3)),
            reference=np.zeros((10, 10, 10, 3)),
        )


def test_bsplines_init():
    with pytest.raises(TransformError):
        BSplineFieldTransform(
            nb.Nifti1Image(np.zeros((10, 10, 10, 4)), np.eye(4), None),
            reference=nb.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4), None),
        )


def test_bsplines_references(testdata_path):
    with pytest.raises(TransformError):
        BSplineFieldTransform(
            testdata_path / "someones_bspline_coefficients.nii.gz"
        ).to_field()

    with pytest.raises(TransformError):
        apply(
            BSplineFieldTransform(
                testdata_path / "someones_bspline_coefficients.nii.gz"
            ),
            testdata_path / "someones_anatomy.nii.gz",
        )

    apply(
        BSplineFieldTransform(testdata_path / "someones_bspline_coefficients.nii.gz"),
        testdata_path / "someones_anatomy.nii.gz",
        reference=testdata_path / "someones_anatomy.nii.gz",
    )


@pytest.mark.xfail(
    reason="GH-267: disabled while debugging",
    strict=False,
)
def test_bspline(tmp_path, testdata_path):
    """Cross-check B-Splines and deformation field."""
    os.chdir(str(tmp_path))

    img_name = testdata_path / "someones_anatomy.nii.gz"
    disp_name = testdata_path / "someones_displacement_field.nii.gz"
    bs_name = testdata_path / "someones_bspline_coefficients.nii.gz"

    bsplxfm = BSplineFieldTransform(bs_name, reference=img_name)
    dispxfm = DenseFieldTransform(disp_name)

    out_disp = apply(dispxfm, img_name)
    out_bspl = apply(bsplxfm, img_name)

    out_disp.to_filename("resampled_field.nii.gz")
    out_bspl.to_filename("resampled_bsplines.nii.gz")

    assert (
        np.sqrt(
            (out_disp.get_fdata(dtype="float32") - out_bspl.get_fdata(dtype="float32"))
            ** 2
        ).mean()
        < 0.2
    )

@pytest.mark.parametrize("image_orientation", ["RAS", "LAS", "LPS", "oblique"])
@pytest.mark.parametrize("ongrid", [True, False])
def test_densefield_map(tmp_path, get_testdata, image_orientation, ongrid):
    """Create a constant displacement field and compare mappings."""

    nii = get_testdata[image_orientation]

    # Create a reference centered at the origin with various axis orders/flips
    shape = nii.shape
    ref_affine = nii.affine.copy()
    reference = ImageGrid(nb.Nifti1Image(np.zeros(shape), ref_affine, None))
    indices = reference.ndindex

    gridpoints = reference.ras(indices)
    points = gridpoints if ongrid else SOME_TEST_POINTS

    coordinates = gridpoints.reshape(*shape, 3)
    deltas = np.stack((
        np.zeros(np.prod(shape), dtype="float32").reshape(shape),
        np.linspace(-80, 80, num=np.prod(shape), dtype="float32").reshape(shape),
        np.linspace(-50, 50, num=np.prod(shape), dtype="float32").reshape(shape),
    ), axis=-1)

    atol = 1e-4 if image_orientation == "oblique" else 1e-7

    # Build an identity transform (deltas)
    id_xfm_deltas = DenseFieldTransform(reference=reference)
    np.testing.assert_array_equal(coordinates, id_xfm_deltas._field)
    np.testing.assert_allclose(points, id_xfm_deltas.map(points), atol=atol)

    # Build an identity transform (deformation)
    id_xfm_field = DenseFieldTransform(coordinates, is_deltas=False, reference=reference)
    np.testing.assert_array_equal(coordinates, id_xfm_field._field)
    np.testing.assert_allclose(points, id_xfm_field.map(points), atol=atol)

    # Collapse to zero transform (deltas)
    zero_xfm_deltas = DenseFieldTransform(-coordinates, reference=reference)
    np.testing.assert_array_equal(np.zeros_like(zero_xfm_deltas._field), zero_xfm_deltas._field)
    np.testing.assert_allclose(np.zeros_like(points), zero_xfm_deltas.map(points), atol=atol)

    # Collapse to zero transform (deformation)
    zero_xfm_field = DenseFieldTransform(np.zeros_like(deltas), is_deltas=False, reference=reference)
    np.testing.assert_array_equal(np.zeros_like(zero_xfm_field._field), zero_xfm_field._field)
    np.testing.assert_allclose(np.zeros_like(points), zero_xfm_field.map(points), atol=atol)

    # Now let's apply a transform
    xfm = DenseFieldTransform(deltas, reference=reference)
    np.testing.assert_array_equal(deltas, xfm._deltas)
    np.testing.assert_array_equal(coordinates + deltas, xfm._field)

    mapped = xfm.map(points)
    nit_deltas = mapped - points

    if ongrid:
        mapped_image = mapped.reshape(*shape, 3)
        np.testing.assert_allclose(deltas + coordinates, mapped_image)
        np.testing.assert_allclose(deltas, nit_deltas.reshape(*shape, 3), atol=1e-4)
        np.testing.assert_allclose(xfm._field, mapped_image)
        


@pytest.mark.parametrize("is_deltas", [True, False])
def test_densefield_oob_resampling(is_deltas):
    """Ensure mapping outside the field returns input coordinates."""
    ref = nb.Nifti1Image(np.zeros((2, 2, 2), dtype="uint8"), np.eye(4))

    if is_deltas:
        field = nb.Nifti1Image(np.ones((2, 2, 2, 3), dtype="float32"), np.eye(4))
    else:
        grid = np.stack(
            np.meshgrid(*[np.arange(2) for _ in range(3)], indexing="ij"),
            axis=-1,
        ).astype("float32")
        field = nb.Nifti1Image(grid + 1.0, np.eye(4))

    xfm = DenseFieldTransform(field, is_deltas=is_deltas, reference=ref)

    points = np.array([[-1.0, -1.0, -1.0], [0.5, 0.5, 0.5], [3.0, 3.0, 3.0]])
    mapped = xfm.map(points)

    assert np.allclose(mapped[0], points[0])
    assert np.allclose(mapped[2], points[2])
    assert np.allclose(mapped[1], points[1] + 1)


def test_bspline_map_gridpoints():
    """BSpline mapping matches dense field on grid points."""
    ref = nb.Nifti1Image(np.zeros((5, 5, 5), dtype="uint8"), np.eye(4))
    coeff = nb.Nifti1Image(
        np.random.RandomState(0).rand(9, 9, 9, 3).astype("float32"), np.eye(4)
    )

    bspline = BSplineFieldTransform(coeff, reference=ref)
    dense = bspline.to_field()

    # Use a couple of voxel centers from the reference grid
    ijk = np.array([[1, 1, 1], [2, 3, 0]])
    pts = nb.affines.apply_affine(ref.affine, ijk)

    assert np.allclose(bspline.map(pts), dense.map(pts), atol=1e-6)


def test_bspline_map_manual():
    """BSpline interpolation agrees with manual computation."""
    ref = nb.Nifti1Image(np.zeros((5, 5, 5), dtype="uint8"), np.eye(4))
    rng = np.random.RandomState(0)
    coeff = nb.Nifti1Image(rng.rand(9, 9, 9, 3).astype("float32"), np.eye(4))

    bspline = BSplineFieldTransform(coeff, reference=ref)

    from nitransforms.base import _as_homogeneous
    from nitransforms.interp.bspline import _cubic_bspline

    def manual_map(x):
        ijk = (bspline._knots.inverse @ _as_homogeneous(x).squeeze())[:3]
        w_start = np.floor(ijk).astype(int) - 1
        w_end = w_start + 3
        w_start = np.maximum(w_start, 0)
        w_end = np.minimum(w_end, np.array(bspline._coeffs.shape[:3]) - 1)

        window = []
        for i in range(w_start[0], w_end[0] + 1):
            for j in range(w_start[1], w_end[1] + 1):
                for k in range(w_start[2], w_end[2] + 1):
                    window.append([i, j, k])
        window = np.array(window)

        dist = np.abs(window - ijk)
        weights = _cubic_bspline(dist).prod(1)
        coeffs = bspline._coeffs[window[:, 0], window[:, 1], window[:, 2]]

        return x + coeffs.T @ weights

    pts = np.array([[1.2, 1.5, 2.0], [3.3, 1.7, 2.4]])
    expected = np.vstack([manual_map(p) for p in pts])
    assert np.allclose(bspline.map(pts), expected, atol=1e-6)
