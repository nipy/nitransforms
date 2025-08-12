import numpy as np
import nibabel as nb
import pytest
from h5py import File as H5File

from nitransforms.nonlinear import DenseFieldTransform, BSplineFieldTransform
from nitransforms.io.x5 import X5Transform, X5Domain, to_filename, from_filename


def test_x5_transform_defaults():
    xf = X5Transform(
        type="linear",
        transform=np.eye(4),
        dimension_kinds=("space", "space", "space", "vector"),
    )
    assert xf.domain is None
    assert xf.subtype is None
    assert xf.representation is None
    assert xf.metadata is None
    assert xf.inverse is None
    assert xf.jacobian is None
    assert xf.array_length == 1
    # Disabled for now
    # assert xf.additional_parameters is None


def test_to_filename(tmp_path):
    domain = X5Domain(grid=True, size=(10, 10, 10), mapping=np.eye(4))
    node = X5Transform(
        type="linear",
        transform=np.eye(4),
        dimension_kinds=("space", "space", "space", "vector"),
        domain=domain,
    )
    fname = tmp_path / "test.x5"
    to_filename(fname, [node])

    with H5File(fname, "r") as f:
        assert f.attrs["Format"] == "X5"
        assert f.attrs["Version"] == 1
        grp = f["TransformGroup"]
        assert "0" in grp
        assert grp["0"].attrs["Type"] == "linear"
        assert grp["0"].attrs["ArrayLength"] == 1


def test_from_filename_roundtrip(tmp_path):
    domain = X5Domain(grid=False, size=(5, 5, 5), mapping=np.eye(4))
    node = X5Transform(
        type="linear",
        transform=np.eye(4),
        dimension_kinds=("space", "space", "space", "vector"),
        domain=domain,
        metadata={"foo": "bar"},
        inverse=np.eye(4),
    )
    fname = tmp_path / "test.x5"
    to_filename(fname, [node])

    x5_list = from_filename(fname)
    assert len(x5_list) == 1
    x5 = x5_list[0]
    assert x5.type == node.type
    assert np.allclose(x5.transform, node.transform)
    assert x5.dimension_kinds == list(node.dimension_kinds)
    assert x5.domain.grid == domain.grid
    assert x5.domain.size == tuple(domain.size)
    assert np.allclose(x5.domain.mapping, domain.mapping)
    assert x5.metadata == node.metadata
    assert np.allclose(x5.inverse, node.inverse)


def test_from_filename_invalid(tmp_path):
    fname = tmp_path / "invalid.h5"
    with H5File(fname, "w") as f:
        f.attrs["Format"] = "NOTX5"

    with pytest.raises(TypeError):
        from_filename(fname)


@pytest.mark.parametrize("is_deltas", [True, False])
def test_densefield_x5_roundtrip(tmp_path, is_deltas):
    """Ensure dense field transforms roundtrip via X5."""
    ref = nb.Nifti1Image(np.zeros((2, 2, 2), dtype="uint8"), np.eye(4))
    disp = nb.Nifti1Image(np.random.rand(2, 2, 2, 3).astype("float32"), np.eye(4))

    xfm = DenseFieldTransform(disp, is_deltas=is_deltas, reference=ref)

    node = xfm.to_x5(metadata={"GeneratedBy": "pytest"})
    assert node.type == "nonlinear"
    assert node.subtype == "densefield"
    assert node.representation == "displacements" if is_deltas else "deformations"
    assert node.domain.size == ref.shape
    assert node.metadata["GeneratedBy"] == "pytest"

    fname = tmp_path / "test.x5"
    to_filename(fname, [node])

    xfm2 = DenseFieldTransform.from_filename(fname, fmt="X5")

    assert xfm2.reference.shape == ref.shape
    assert np.allclose(xfm2.reference.affine, ref.affine)
    assert xfm == xfm2


def test_bspline_to_x5(tmp_path):
    """Check BSpline transforms export to X5."""
    coeff = nb.Nifti1Image(np.zeros((2, 2, 2, 3), dtype="float32"), np.eye(4))
    ref = nb.Nifti1Image(np.zeros((2, 2, 2), dtype="uint8"), np.eye(4))

    xfm = BSplineFieldTransform(coeff, reference=ref)
    node = xfm.to_x5(metadata={"tool": "pytest"})
    assert node.type == "nonlinear"
    assert node.subtype == "bspline"
    assert node.representation == "coefficients"
    assert node.metadata["tool"] == "pytest"

    fname = tmp_path / "bspline.x5"
    to_filename(fname, [node])

    xfm2 = BSplineFieldTransform.from_filename(fname, fmt="X5")
    assert np.allclose(xfm._coeffs, xfm2._coeffs)
    assert xfm2.reference.shape == ref.shape
    assert np.allclose(xfm2.reference.affine, ref.affine)
