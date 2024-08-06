# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of linear transforms."""

import pytest
import numpy as np
import h5py

from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from nitransforms import linear as nitl
from nitransforms import io
from .utils import assert_affines_by_filename


@pytest.mark.parametrize(
    "matrix",
    [
        [0.0],
        np.ones((3, 3, 3)),
        np.ones((3, 4)),
    ],
)
def test_linear_typeerrors1(matrix):
    """Exercise errors in Affine creation."""
    with pytest.raises(TypeError):
        nitl.Affine(matrix)


def test_linear_typeerrors2(data_path):
    """Exercise errors in Affine creation."""
    with pytest.raises(io.TransformFileError):
        nitl.Affine.from_filename(data_path / "itktflist.tfm", fmt="itk")


def test_linear_filenotfound(data_path):
    """Exercise errors in Affine creation."""
    with pytest.raises(FileNotFoundError):
        nitl.Affine.from_filename("doesnotexist.tfm", fmt="itk")

    with pytest.raises(FileNotFoundError):
        nitl.LinearTransformsMapping.from_filename("doesnotexist.tfm", fmt="itk")

    with pytest.raises(FileNotFoundError):
        nitl.LinearTransformsMapping.from_filename("doesnotexist.mat", fmt="fsl")


def test_linear_valueerror():
    """Exercise errors in Affine creation."""
    with pytest.raises(ValueError):
        nitl.Affine(np.ones((4, 4)))


def test_linear_load_unsupported(data_path):
    """Exercise loading transform without I/O implementation."""
    with pytest.raises(TypeError):
        nitl.load(data_path / "itktflist2.tfm", fmt="X5")


def test_linear_load_mistaken(data_path):
    """Exercise loading transform without I/O implementation."""
    with pytest.raises(io.TransformFileError):
        nitl.load(data_path / "itktflist2.tfm", fmt="afni")


def test_loadsave_itk(tmp_path, data_path, testdata_path):
    """Test idempotency."""
    ref_file = testdata_path / "someones_anatomy.nii.gz"
    xfm = nitl.load(data_path / "itktflist2.tfm", fmt="itk")
    assert isinstance(xfm, nitl.LinearTransformsMapping)
    xfm.reference = ref_file
    xfm.to_filename(tmp_path / "transform-mapping.tfm", fmt="itk")

    assert (data_path / "itktflist2.tfm").read_text() == (
        tmp_path / "transform-mapping.tfm"
    ).read_text()

    single_xfm = nitl.load(data_path / "affine-LAS.itk.tfm", fmt="itk")
    assert isinstance(single_xfm, nitl.Affine)
    assert single_xfm == nitl.Affine.from_filename(
        data_path / "affine-LAS.itk.tfm", fmt="itk"
    )


@pytest.mark.parametrize(
    "image_orientation",
    [
        "RAS",
        "LAS",
        "LPS",
        "oblique",
    ],
)
def test_itkmat_loadsave(tmpdir, data_path, image_orientation):
    tmpdir.chdir()

    io.itk.ITKLinearTransform.from_filename(
        data_path / f"affine-{image_orientation}.itk.tfm"
    ).to_filename(f"affine-{image_orientation}.itk.mat")

    xfm = nitl.load(data_path / f"affine-{image_orientation}.itk.tfm", fmt="itk")
    mat1 = nitl.load(f"affine-{image_orientation}.itk.mat", fmt="itk")

    assert xfm == mat1

    mat2 = nitl.Affine.from_filename(f"affine-{image_orientation}.itk.mat", fmt="itk")

    assert xfm == mat2

    mat3 = nitl.LinearTransformsMapping.from_filename(
        f"affine-{image_orientation}.itk.mat", fmt="itk"
    )

    assert xfm == mat3


@pytest.mark.parametrize("autofmt", (False, True))
@pytest.mark.parametrize("fmt", ["itk", "fsl", "afni", "lta"])
def test_loadsave(tmp_path, data_path, testdata_path, autofmt, fmt):
    """Test idempotency."""
    supplied_fmt = None if autofmt else fmt

    # Load reference transform
    ref_file = testdata_path / "someones_anatomy.nii.gz"
    xfm = nitl.load(data_path / "itktflist2.tfm", fmt="itk")
    xfm.reference = ref_file

    fname = tmp_path / ".".join(("transform-mapping", fmt))
    xfm.to_filename(fname, fmt=fmt)

    if fmt == "fsl":
        # FSL should not read a transform without reference
        with pytest.raises(io.TransformIOError):
            nitl.load(fname, fmt=supplied_fmt)
            nitl.load(fname, fmt=supplied_fmt, moving=ref_file)

        with pytest.warns(UserWarning):
            assert np.allclose(
                xfm.matrix,
                nitl.load(fname, fmt=supplied_fmt, reference=ref_file).matrix,
            )

        assert np.allclose(
            xfm.matrix,
            nitl.load(
                fname, fmt=supplied_fmt, reference=ref_file, moving=ref_file
            ).matrix,
        )
    else:
        assert xfm == nitl.load(fname, fmt=supplied_fmt, reference=ref_file)

    xfm.to_filename(fname, fmt=fmt, moving=ref_file)

    if fmt == "fsl":
        assert np.allclose(
            xfm.matrix,
            nitl.load(
                fname, fmt=supplied_fmt, reference=ref_file, moving=ref_file
            ).matrix,
            rtol=1e-2,  # FSL incurs into large errors due to rounding
        )
    else:
        assert xfm == nitl.load(fname, fmt=supplied_fmt, reference=ref_file)

    ref_file = testdata_path / "someones_anatomy.nii.gz"
    xfm = nitl.load(data_path / "affine-LAS.itk.tfm", fmt="itk")
    xfm.reference = ref_file
    fname = tmp_path / ".".join(("single-transform", fmt))
    xfm.to_filename(fname, fmt=fmt)
    if fmt == "fsl":
        assert np.allclose(
            xfm.matrix,
            nitl.load(
                fname, fmt=supplied_fmt, reference=ref_file, moving=ref_file
            ).matrix,
            rtol=1e-2,  # FSL incurs into large errors due to rounding
        )
    else:
        assert xfm == nitl.load(fname, fmt=supplied_fmt, reference=ref_file)

    xfm.to_filename(fname, fmt=fmt, moving=ref_file)
    if fmt == "fsl":
        assert np.allclose(
            xfm.matrix,
            nitl.load(
                fname, fmt=supplied_fmt, reference=ref_file, moving=ref_file
            ).matrix,
            rtol=1e-2,  # FSL incurs into large errors due to rounding
        )
    else:
        assert xfm == nitl.load(fname, fmt=supplied_fmt, reference=ref_file)


@pytest.mark.parametrize("image_orientation", ["RAS", "LAS", "LPS", "oblique"])
@pytest.mark.parametrize("sw_tool", ["itk", "fsl", "afni", "fs"])
def test_linear_save(tmpdir, data_path, get_testdata, image_orientation, sw_tool):
    """Check implementation of exporting affines to formats."""
    tmpdir.chdir()
    img = get_testdata[image_orientation]
    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    if sw_tool == "fs":
        # Account for the fact that FS defines LTA transforms reversed
        T = np.linalg.inv(T)

    xfm = (
        nitl.Affine(T)
        if (sw_tool, image_orientation) != ("afni", "oblique")
        # AFNI is special when moving or reference are oblique - let io do the magic
        else nitl.Affine(
            io.afni.AFNILinearTransform.from_ras(T).to_ras(
                reference=img,
                moving=img,
            )
        )
    )
    xfm.reference = img

    ext = ""
    if sw_tool == "itk":
        ext = ".tfm"
    elif sw_tool == "fs":
        ext = ".lta"

    xfm_fname1 = f"M.{sw_tool}{ext}"
    xfm.to_filename(xfm_fname1, fmt=sw_tool)

    xfm_fname2 = str(data_path / "affine-%s.%s%s") % (image_orientation, sw_tool, ext)
    assert_affines_by_filename(xfm_fname1, xfm_fname2)


def test_Affine_to_x5(tmpdir, testdata_path):
    """Test affine's operations."""
    tmpdir.chdir()
    aff = nitl.Affine()
    with h5py.File("xfm.x5", "w") as f:
        aff._to_hdf5(f.create_group("Affine"))

    aff.reference = testdata_path / "someones_anatomy.nii.gz"
    with h5py.File("withref-xfm.x5", "w") as f:
        aff._to_hdf5(f.create_group("Affine"))


def test_mulmat_operator(testdata_path):
    """Check the @ operator."""
    ref = testdata_path / "someones_anatomy.nii.gz"
    mat1 = np.diag([2.0, 2.0, 2.0, 1.0])
    mat2 = from_matvec(np.eye(3), (4, 2, -1))
    aff = nitl.Affine(mat1, reference=ref)

    composed = aff @ mat2
    assert composed.reference is None
    assert composed == nitl.Affine(mat1.dot(mat2))

    composed = nitl.Affine(mat2) @ aff
    assert composed.reference == aff.reference
    assert composed == nitl.Affine(mat2.dot(mat1), reference=ref)
