# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Test io module for ITK transforms."""

import os
import shutil
from subprocess import check_call

import pytest

import numpy as np
import nibabel as nb
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from scipy.io import loadmat
from h5py import File as H5File


from nitransforms.io.base import TransformIOError, TransformFileError
from nitransforms.io.itk import (
    ITKLinearTransform,
    ITKLinearTransformArray,
    ITKDisplacementsField,
    ITKCompositeH5,
)
from nitransforms import nonlinear as nitnl
from nitransforms.conftest import _datadir, _testdir
from nitransforms.tests.utils import get_points

RNG_SEED = 202508140819
LPS = np.diag([-1, -1, 1, 1])
ITK_MAT = LPS.dot(np.ones((4, 4)).dot(LPS))


def test_ITKLinearTransform(tmpdir, testdata_path):
    tmpdir.chdir()

    matlabfile = testdata_path / "ds-005_sub-01_from-T1_to-OASIS_affine.mat"
    mat = loadmat(str(matlabfile))
    with open(str(matlabfile), "rb") as f:
        itkxfm = ITKLinearTransform.from_fileobj(f)
    assert np.allclose(
        itkxfm["parameters"][:3, :3].flatten(),
        mat["AffineTransform_float_3_3"][:-3].flatten(),
    )
    assert np.allclose(itkxfm["offset"], mat["fixed"].reshape((3,)))

    itkxfm = ITKLinearTransform.from_filename(matlabfile)
    assert np.allclose(
        itkxfm["parameters"][:3, :3].flatten(),
        mat["AffineTransform_float_3_3"][:-3].flatten(),
    )
    assert np.allclose(itkxfm["offset"], mat["fixed"].reshape((3,)))

    # Test to_filename(textfiles)
    itkxfm.to_filename("textfile.tfm")
    with open("textfile.tfm") as f:
        itkxfm2 = ITKLinearTransform.from_fileobj(f)
    assert np.allclose(itkxfm["parameters"], itkxfm2["parameters"])

    # Test to_filename(matlab)
    itkxfm.to_filename("copy.mat")
    with open("copy.mat", "rb") as f:
        itkxfm3 = ITKLinearTransform.from_fileobj(f)
    assert np.all(itkxfm["parameters"] == itkxfm3["parameters"])

    rasmat = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    itkxfm = ITKLinearTransform.from_ras(rasmat)
    assert np.allclose(itkxfm["parameters"], ITK_MAT * rasmat)
    assert np.allclose(itkxfm.to_ras(), rasmat)


def test_ITKLinearTransformArray(tmpdir, data_path):
    tmpdir.chdir()

    with open(str(data_path / "itktflist.tfm")) as f:
        text = f.read()
        f.seek(0)
        itklist = ITKLinearTransformArray.from_fileobj(f)

    itklistb = ITKLinearTransformArray.from_filename(data_path / "itktflist.tfm")
    assert itklist["nxforms"] == itklistb["nxforms"]
    assert all(
        [
            np.allclose(x1["parameters"], x2["parameters"])
            for x1, x2 in zip(itklist.xforms, itklistb.xforms)
        ]
    )

    tmpdir.join("empty.mat").write("")
    with pytest.raises(TransformFileError):
        itklistb.from_filename("empty.mat")

    assert itklist["nxforms"] == 9
    assert text == itklist.to_string()
    with pytest.raises(TransformFileError):
        ITKLinearTransformArray.from_string("\n".join(text.splitlines()[1:]))

    itklist.to_filename("copy.tfm")
    with open("copy.tfm") as f:
        copytext = f.read()
    assert text == copytext

    itklist = ITKLinearTransformArray(
        xforms=[np.random.normal(size=(4, 4)) for _ in range(4)]
    )

    assert itklist["nxforms"] == 4
    assert itklist["xforms"][1].structarr["index"] == 1

    with pytest.raises(KeyError):
        itklist["invalid_key"]

    xfm = itklist["xforms"][1]
    xfm["index"] = 1
    with open("extracted.tfm", "w") as f:
        f.write(xfm.to_string())

    with open("extracted.tfm") as f:
        xfm2 = ITKLinearTransform.from_fileobj(f)
    assert np.allclose(
        xfm.structarr["parameters"][:3, ...], xfm2.structarr["parameters"][:3, ...]
    )

    # ITK does not handle transforms lists in Matlab format
    with pytest.raises(TransformFileError):
        itklist.to_filename("matlablist.mat")

    with pytest.raises(TransformFileError):
        xfm2 = ITKLinearTransformArray.from_binary({})

    with open("filename.mat", "ab") as f:
        with pytest.raises(TransformFileError):
            xfm2 = ITKLinearTransformArray.from_fileobj(f)


@pytest.mark.parametrize("size", [(20, 20, 20), (20, 20, 20, 3)])
def test_itk_disp_load(size):
    """Checks field sizes."""
    with pytest.raises(TransformFileError):
        ITKDisplacementsField.from_image(
            nb.Nifti1Image(np.zeros(size), np.eye(4), None)
        )


def test_itk_disp_load_intent():
    """Checks whether the NIfTI intent is fixed."""
    with pytest.warns(UserWarning):
        field = ITKDisplacementsField.from_image(
            nb.Nifti1Image(np.zeros((20, 20, 20, 1, 3)), np.eye(4), None)
        )

    assert field.header.get_intent()[0] == "vector"


@pytest.mark.parametrize("only_linear", [True, False])
@pytest.mark.parametrize(
    "h5_path,nxforms",
    [
        (_datadir / "affine-antsComposite.h5", 1),
        (
            _testdir
            / "ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
            2,
        ),
    ],
)
def test_itk_h5(tmpdir, only_linear, h5_path, nxforms):
    """Test displacements fields."""
    assert (
        len(
            list(
                ITKCompositeH5.from_filename(
                    h5_path,
                    only_linear=only_linear,
                )
            )
        )
        == nxforms
        if not only_linear
        else 1
    )

    with pytest.raises(TransformFileError):
        list(
            ITKCompositeH5.from_filename(
                h5_path.absolute().name.replace(".h5", ".x5"),
                only_linear=only_linear,
            )
        )

    tmpdir.chdir()
    shutil.copy(h5_path, "test.h5")
    os.chmod("test.h5", 0o666)

    with H5File("test.h5", "r+") as h5file:
        h5group = h5file["TransformGroup"]
        xfm = h5group[list(h5group.keys())[1]]
        xfm["TransformType"][0] = b"InventTransform"

    with pytest.raises(TransformIOError):
        ITKCompositeH5.from_filename("test.h5")


def test_itk_linear_h5(tmpdir, data_path, testdata_path):
    """Check different lower-level loading options."""

    # File loadable with transform array
    h5xfm = ITKLinearTransformArray.from_filename(data_path / "affine-antsComposite.h5")
    assert len(h5xfm.xforms) == 1

    with open(data_path / "affine-antsComposite.h5", "rb") as f:
        h5xfm = ITKLinearTransformArray.from_fileobj(f)
    assert len(h5xfm.xforms) == 1

    # File loadable with single affine object
    ITKLinearTransform.from_filename(data_path / "affine-antsComposite.h5")

    with open(data_path / "affine-antsComposite.h5", "rb") as f:
        ITKLinearTransform.from_fileobj(f)

    # Exercise only_linear
    ITKCompositeH5.from_filename(
        testdata_path
        / "ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
        only_linear=True,
    )

    tmpdir.chdir()
    shutil.copy(data_path / "affine-antsComposite.h5", "test.h5")
    os.chmod("test.h5", 0o666)

    with H5File("test.h5", "r+") as h5file:
        h5group = h5file["TransformGroup"]
        xfm = h5group.create_group("2")
        xfm["TransformType"] = (b"AffineTransform", b"")
        xfm["TransformParameters"] = np.zeros(12, dtype=float)
        xfm["TransformFixedParameters"] = np.zeros(3, dtype=float)

    # File loadable with transform array
    h5xfm = ITKLinearTransformArray.from_filename("test.h5")
    assert len(h5xfm.xforms) == 2

    # File loadable with generalistic object (NOTE we directly access the list)
    h5xfm = ITKCompositeH5.from_filename("test.h5")
    assert len(h5xfm) == 2

    # Error raised if the we try to use the single affine loader
    with pytest.raises(TransformIOError):
        ITKLinearTransform.from_filename("test.h5")

    shutil.copy(data_path / "affine-antsComposite.h5", "test.h5")
    os.chmod("test.h5", 0o666)

    # Generate an empty h5 file
    with H5File("test.h5", "r+") as h5file:
        h5group = h5file["TransformGroup"]
        del h5group["1"]

    # File loadable with generalistic object
    h5xfm = ITKCompositeH5.from_filename("test.h5")
    assert len(h5xfm) == 0

    # Error raised if the we try to use the single affine loader
    with pytest.raises(TransformIOError):
        ITKLinearTransform.from_filename("test.h5")


# Added tests for h5 orientation bug
@pytest.mark.xfail(
    reason="GH-137/GH-171: displacement field dimension order is wrong",
    strict=False,
)
def test_itk_h5_field_order(tmp_path):
    """Displacement fields stored in row-major order should fail to round-trip."""
    shape = (3, 4, 5)
    vals = np.arange(np.prod(shape), dtype=float).reshape(shape)
    field = np.stack([vals, vals + 100, vals + 200], axis=0)

    params = field.reshape(-1, order="C")
    fixed = np.array(
        list(shape) + [0, 0, 0] + [1, 1, 1] + list(np.eye(3).ravel()), dtype=float
    )
    fname = tmp_path / "field.h5"
    with H5File(fname, "w") as f:
        grp = f.create_group("TransformGroup")
        grp.create_group("0")["TransformType"] = np.array(
            [b"CompositeTransform_double_3_3"]
        )
        g1 = grp.create_group("1")
        g1["TransformType"] = np.array([b"DisplacementFieldTransform_float_3_3"])
        g1["TransformFixedParameters"] = fixed
        g1["TransformParameters"] = params

    img = ITKCompositeH5.from_filename(fname)[0]
    expected = np.moveaxis(field, 0, -1)
    expected[..., (0, 1)] *= -1
    assert np.allclose(img.get_fdata(), expected)


def _load_composite_testdata(data_path):
    """Return the composite HDF5 and displacement field from regressions."""
    h5file = data_path / "regressions" / "ants_t1_to_mniComposite.h5"
    # Generated using
    # CompositeTransformUtil --disassemble ants_t1_to_mniComposite.h5 \
    #     ants_t1_to_mniComposite
    warpfile = (
        data_path
        / "regressions"
        / ("01_ants_t1_to_mniComposite_DisplacementFieldTransform.nii.gz")
    )
    if not (h5file.exists() and warpfile.exists()):
        pytest.skip("Composite transform test data not available")
    return h5file, warpfile


@pytest.mark.xfail(
    reason="GH-137/GH-171: displacement field dimension order is wrong",
    strict=False,
)
def test_itk_h5_displacement_mismatch(testdata_path):
    """Composite displacements should match the standalone field"""
    h5file, warpfile = _load_composite_testdata(testdata_path)
    xforms = ITKCompositeH5.from_filename(h5file)
    field_h5 = xforms[1]
    field_img = ITKDisplacementsField.from_filename(warpfile)

    np.testing.assert_array_equal(
        np.asanyarray(field_h5.dataobj), np.asanyarray(field_img.dataobj)
    )


def test_itk_h5_transpose_fix(testdata_path):
    """Check the displacement field orientation explicitly.

    ITK stores displacement fields with the vector dimension leading in
    Fortran (column-major) order [1]_. Transposing the parameters from the HDF5
    composite file accordingly should match the standalone displacement image.

    References
    ----------
    .. [1] ITK Software Guide. https://itk.org/ItkSoftwareGuide.pdf
    """
    h5file, warpfile = _load_composite_testdata(testdata_path)

    with H5File(h5file, "r") as f:
        group = f["TransformGroup"]["2"]
        size = group["TransformFixedParameters"][:3].astype(int)
        params = group["TransformParameters"][:].reshape(*size, 3)

    img = nb.load(warpfile)
    ref = np.squeeze(np.asanyarray(img.dataobj))

    np.testing.assert_array_equal(params.transpose(2, 1, 0, 3), ref)


def test_itk_h5_field_order_fortran(tmp_path):
    """Verify Fortran-order displacement fields load correctly"""
    shape = (3, 4, 5)
    vals = np.arange(np.prod(shape), dtype=float).reshape(shape)
    field = np.stack([vals, vals + 100, vals + 200], axis=0)

    params = field.reshape(-1, order="F")
    fixed = np.array(
        list(shape) + [0, 0, 0] + [1, 1, 1] + list(np.eye(3).ravel()), dtype=float
    )
    fname = tmp_path / "field_f.h5"
    with H5File(fname, "w") as f:
        grp = f.create_group("TransformGroup")
        grp.create_group("0")["TransformType"] = np.array(
            [b"CompositeTransform_double_3_3"]
        )
        g1 = grp.create_group("1")
        g1["TransformType"] = np.array([b"DisplacementFieldTransform_float_3_3"])
        g1["TransformFixedParameters"] = fixed
        g1["TransformParameters"] = params

    img = ITKCompositeH5.from_filename(fname)[0]
    expected = np.moveaxis(field, 0, -1)
    expected[..., (0, 1)] *= -1
    assert np.allclose(img.get_fdata(), expected)


# Tests against ANTs' ``antsApplyTransformsToPoints``
@pytest.mark.parametrize("ongrid", [True, False])
def test_densefield_map_vs_ants(testdata_path, tmp_path, ongrid):
    """Map points with DenseFieldTransform and compare to ANTs."""

    rng = np.random.default_rng(RNG_SEED)
    warpfile = (
        testdata_path
        / "regressions"
        / ("01_ants_t1_to_mniComposite_DisplacementFieldTransform.nii.gz")
    )
    if not warpfile.exists():
        pytest.skip("Composite transform test data not available")

    nii = ITKDisplacementsField.from_filename(warpfile)

    # Get sampling indices
    coords_xyz, points_ijk, grid_xyz, shape, ref_affine, reference, subsample = (
        get_points(nii, ongrid, npoints=5, rng=rng)
    )
    coords_map = grid_xyz.reshape(*shape, 3)

    csvin = tmp_path / "fixed_coords.csv"
    csvout = tmp_path / "moving_coords.csv"

    # antsApplyTransformsToPoints wants LPS coordinates, see last post at
    # http://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/
    lps_xyz = coords_xyz.copy() * (-1, -1, 1)
    np.savetxt(csvin, lps_xyz, delimiter=",", header="x,y,z", comments="")

    cmd = f"antsApplyTransformsToPoints -d 3 -i {csvin} -o {csvout} -t {warpfile}"
    exe = cmd.split()[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")
    check_call(cmd, shell=True)

    ants_res = np.genfromtxt(csvout, delimiter=",", names=True)

    # antsApplyTransformsToPoints writes LPS coordinates, see last post at
    # http://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/
    ants_pts = np.vstack([ants_res[n] for n in ("x", "y", "z")]).T * (-1, -1, 1)

    xfm = nitnl.DenseFieldTransform(nii, reference=reference)
    mapped = xfm.map(coords_xyz)

    if ongrid:
        ants_mapped_xyz = ants_pts.reshape(*shape, 3)
        nit_mapped_xyz = mapped.reshape(*shape, 3)

        nb.load(warpfile).to_filename(tmp_path / "original_ants_deltas.nii.gz")

        nb.Nifti1Image(coords_map, ref_affine, None).to_filename(
            tmp_path / "baseline_positions.nii.gz"
        )

        nii.to_filename(tmp_path / "original_interpreted_deltas.nii.gz")

        nb.Nifti1Image(nit_mapped_xyz, ref_affine, None).to_filename(
            tmp_path / "nit_deformation_xyz.nii.gz"
        )
        nb.Nifti1Image(ants_mapped_xyz - coords_map, ref_affine, None).to_filename(
            tmp_path / "ants_deltas_xyz.nii.gz"
        )
        nb.Nifti1Image(nit_mapped_xyz - coords_map, ref_affine, None).to_filename(
            tmp_path / "nit_deltas_xyz.nii.gz"
        )

    # When transforming off-grid points, rounding errors are large
    atol = 0 if ongrid else 1e-1
    rtol = 1e-4 if ongrid else 1e-3
    np.testing.assert_allclose(mapped, ants_pts, atol=atol, rtol=rtol)


@pytest.mark.parametrize("image_orientation", ["RAS", "LAS", "LPS", "oblique"])
@pytest.mark.parametrize("ongrid", [True, False])
def test_constant_field_vs_ants(tmp_path, get_testdata, image_orientation, ongrid):
    """Create a constant displacement field and compare mappings."""

    rng = np.random.default_rng(RNG_SEED)

    nii = get_testdata[image_orientation]

    # Get sampling indices
    coords_xyz, points_ijk, grid_xyz, shape, ref_affine, reference, subsample = (
        get_points(nii, ongrid, npoints=5, rng=rng)
    )

    tol = (
        {"atol": 0, "rtol": 1e-4}
        if image_orientation != "oblique"
        else {"atol": 1e-4, "rtol": 1e-2}
    )
    coords_map = grid_xyz.reshape(*shape, 3)

    deltas = np.hstack(
        (
            np.zeros(np.prod(shape)),
            np.linspace(-80, 80, num=np.prod(shape)),
            np.linspace(-50, 50, num=np.prod(shape)),
        )
    ).reshape(shape + (3,))
    gold_mapped_xyz = coords_map + deltas

    fieldnii = nb.Nifti1Image(deltas, ref_affine, None)
    warpfile = tmp_path / "itk_transform.nii.gz"

    # Ensure direct (xfm) and ITK roundtrip (itk_xfm) are equivalent
    xfm = nitnl.DenseFieldTransform(fieldnii)
    xfm.to_filename(warpfile, fmt="itk")
    itk_xfm = nitnl.DenseFieldTransform(ITKDisplacementsField.from_filename(warpfile))

    np.testing.assert_allclose(xfm.reference.affine, itk_xfm.reference.affine)
    np.testing.assert_allclose(ref_affine, itk_xfm.reference.affine)
    np.testing.assert_allclose(xfm.reference.shape, itk_xfm.reference.shape)
    np.testing.assert_allclose(xfm._field, itk_xfm._field, **tol)
    if image_orientation != "oblique":
        assert xfm == itk_xfm

    # Ensure deltas and mapped grid are equivalent
    orig_grid_mapped_xyz = xfm.map(grid_xyz).reshape(*shape, -1)
    np.testing.assert_allclose(gold_mapped_xyz, orig_grid_mapped_xyz)

    # Test ANTs mapping
    grid_mapped_xyz = itk_xfm.map(grid_xyz).reshape(*shape, -1)

    # Check apparent healthiness of mapping
    np.testing.assert_allclose(gold_mapped_xyz, grid_mapped_xyz, **tol)
    np.testing.assert_allclose(orig_grid_mapped_xyz, grid_mapped_xyz, **tol)

    csvout = tmp_path / "mapped_xyz.csv"
    csvin = tmp_path / "coords_xyz.csv"
    # antsApplyTransformsToPoints wants LPS coordinates, see last post at
    # http://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/
    lps_xyz = coords_xyz.copy() * (-1, -1, 1)
    np.savetxt(csvin, lps_xyz, delimiter=",", header="x,y,z", comments="")

    cmd = f"antsApplyTransformsToPoints -d 3 -i {csvin} -o {csvout} -t {warpfile}"
    exe = cmd.split()[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")
    check_call(cmd, shell=True)

    ants_res = np.genfromtxt(csvout, delimiter=",", names=True)
    # antsApplyTransformsToPoints writes LPS coordinates, see last post at
    # http://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/
    ants_pts = np.vstack([ants_res[n] for n in ("x", "y", "z")]).T * (-1, -1, 1)

    nb.Nifti1Image(grid_mapped_xyz, ref_affine, None).to_filename(
        tmp_path / "grid_mapped.nii.gz"
    )
    nb.Nifti1Image(coords_map, ref_affine, None).to_filename(
        tmp_path / "baseline_field.nii.gz"
    )
    nb.Nifti1Image(gold_mapped_xyz, ref_affine, None).to_filename(
        tmp_path / "gold_mapped_xyz.nii.gz"
    )

    if ongrid:
        ants_pts = ants_pts.reshape(*shape, 3)

        nb.Nifti1Image(ants_pts, ref_affine, None).to_filename(
            tmp_path / "ants_mapped_xyz.nii.gz"
        )
        np.testing.assert_allclose(gold_mapped_xyz, ants_pts, rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(deltas, ants_pts - coords_map, rtol=1e-2, atol=1e-3)
    else:
        # TODO Change test to norms and investigate extreme cases
        # We're likely hitting OBB points (see gh-188)
        # https://github.com/nipy/nitransforms/pull/188
        np.testing.assert_allclose(
            xfm.map(coords_xyz) - coords_xyz, ants_pts - coords_xyz, rtol=1, atol=1
        )
