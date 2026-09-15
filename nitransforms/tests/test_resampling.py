# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Exercise the standalone ``apply()`` implementation."""

import os
import pytest
import numpy as np
from subprocess import check_call
import shutil

import nibabel as nb
from nibabel.eulerangles import euler2mat
from nibabel.affines import from_matvec
from nitransforms import linear as nitl
from nitransforms import nonlinear as nitnl
from nitransforms import manip as nitm
from nitransforms import io
from nitransforms.resampling import apply

RMSE_TOL_LINEAR = 0.09
RMSE_TOL_NONLINEAR = 0.05
APPLY_LINEAR_CMD = {
    "fsl": """\
flirt -setbackground 0 -interp nearestneighbour -in {moving} -ref {reference} \
-applyxfm -init {transform} -out {resampled}\
""".format,
    "itk": """\
antsApplyTransforms -d 3 -r {reference} -i {moving} \
-o {resampled} -n NearestNeighbor -t {transform} --float\
""".format,
    "afni": """\
3dAllineate -base {reference} -input {moving} \
-prefix {resampled} -1Dmatrix_apply {transform} -final NN\
""".format,
    "fs": """\
mri_vol2vol --mov {moving} --targ {reference} --lta {transform} \
--o {resampled} --nearest""".format,
}
APPLY_NONLINEAR_CMD = {
    "itk": """\
antsApplyTransforms -d 3 -r {reference} -i {moving} \
-o {output} -n NearestNeighbor -t {transform} {extra}\
""".format,
    "afni": """\
3dNwarpApply -nwarp {transform} -source {moving} \
-master {reference} -interp NN -prefix {output} {extra}\
""".format,
    "fsl": """\
applywarp -i {moving} -r {reference} -o {output} {extra}\
-w {transform} --interp=nn""".format,
}


def test_apply_singleton_time_dimension():
    """Resampling fails when the input image has a trailing singleton dimension (gh-270)."""

    data = np.reshape(np.arange(27, dtype=np.uint8), (3, 3, 3, 1))
    nii = nb.Nifti1Image(data, np.eye(4))
    ref = nb.Nifti1Image(np.zeros((4, 4, 4)), np.eye(4))
    xfm = nitl.Affine(np.eye(4), reference=ref)
    movednii = apply(xfm, nii)
    assert movednii.shape == ref.shape + (1, )

    movednii = apply(xfm, nii, reference=ref)
    assert movednii.shape == ref.shape + (1, )


@pytest.mark.parametrize(
    "image_orientation",
    [
        "RAS",
        "LAS",
        "LPS",
        "oblique",
    ],
)
@pytest.mark.parametrize("sw_tool", ["itk", "fsl", "afni", "fs"])
def test_apply_linear_transform(
    tmpdir, get_testdata, get_testmask, image_orientation, sw_tool
):
    """Check implementation of exporting affines to formats."""
    tmpdir.chdir()

    img = get_testdata[image_orientation]
    msk = get_testmask[image_orientation]

    # Generate test transform
    T = from_matvec(euler2mat(x=0.9, y=0.001, z=0.001), [4.0, 2.0, -1.0])
    xfm = nitl.Affine(T)
    xfm.reference = img

    ext = ""
    if sw_tool == "itk":
        ext = ".tfm"
    elif sw_tool == "fs":
        ext = ".lta"

    img.to_filename("img.nii.gz")
    msk.to_filename("mask.nii.gz")

    # Write out transform file (software-dependent)
    xfm_fname = f"M.{sw_tool}{ext}"
    # Change reference dataset for AFNI & oblique
    if (sw_tool, image_orientation) == ("afni", "oblique"):
        io.afni.AFNILinearTransform.from_ras(
            T,
            moving=img,
            reference=img,
        ).to_filename(xfm_fname)
    else:
        xfm.to_filename(xfm_fname, fmt=sw_tool)

    cmd = APPLY_LINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=os.path.abspath("mask.nii.gz"),
        moving=os.path.abspath("mask.nii.gz"),
        resampled=os.path.abspath("resampled_brainmask.nii.gz"),
    )

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")

    # resample mask
    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved_mask = nb.load("resampled_brainmask.nii.gz")

    nt_moved_mask = apply(xfm, msk, order=0)
    nt_moved_mask.set_data_dtype(msk.get_data_dtype())
    nt_moved_mask.to_filename("ntmask.nii.gz")
    diff = np.asanyarray(sw_moved_mask.dataobj) - np.asanyarray(nt_moved_mask.dataobj)

    assert np.sqrt((diff**2).mean()) < RMSE_TOL_LINEAR
    brainmask = np.asanyarray(nt_moved_mask.dataobj, dtype=bool)

    cmd = APPLY_LINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=os.path.abspath("img.nii.gz"),
        moving=os.path.abspath("img.nii.gz"),
        resampled=os.path.abspath("resampled.nii.gz"),
    )

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load("resampled.nii.gz")
    sw_moved.set_data_dtype(img.get_data_dtype())

    nt_moved = apply(xfm, img, order=0)
    diff = np.asanyarray(
        sw_moved.dataobj, dtype=sw_moved.get_data_dtype()
    ) - np.asanyarray(nt_moved.dataobj, dtype=nt_moved.get_data_dtype())

    # A certain tolerance is necessary because of resampling at borders
    assert np.sqrt((diff[brainmask] ** 2).mean()) < RMSE_TOL_LINEAR

    nt_moved = apply(xfm, "img.nii.gz", order=0)
    diff = np.asanyarray(
        sw_moved.dataobj, dtype=sw_moved.get_data_dtype()
    ) - np.asanyarray(nt_moved.dataobj, dtype=nt_moved.get_data_dtype())
    # A certain tolerance is necessary because of resampling at borders
    assert np.sqrt((diff[brainmask] ** 2).mean()) < RMSE_TOL_LINEAR


@pytest.mark.parametrize("image_orientation", ["RAS", "LAS", "LPS", "oblique"])
@pytest.mark.parametrize("sw_tool", ["itk", "afni"])
@pytest.mark.parametrize("axis", [0, 1, 2, (0, 1), (1, 2), (0, 1, 2)])
def test_displacements_field1(
    tmp_path,
    get_testdata,
    get_testmask,
    image_orientation,
    sw_tool,
    axis,
):
    """Check a translation-only field on one or more axes, different image orientations."""
    if (image_orientation, sw_tool) == ("oblique", "afni"):
        pytest.skip("AFNI obliques are not yet implemented for displacements fields")

    os.chdir(str(tmp_path))
    nii = get_testdata[image_orientation]
    msk = get_testmask[image_orientation]
    nii.to_filename("reference.nii.gz")
    msk.to_filename("mask.nii.gz")

    fieldmap = np.zeros((*nii.shape[:3], 3), dtype="float32")
    fieldmap[..., axis] = -10.0

    # Generate a transform file for the particular software
    xfm_fname = "warp.nii.gz"
    xfm = nitnl.DenseFieldTransform(
        fieldmap,
        reference=nii,
    )
    xfm.to_filename(xfm_fname, fmt=sw_tool)

    tool_output = tmp_path / f"{sw_tool}_brainmask.nii.gz"
    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=tmp_path / "mask.nii.gz",
        moving=tmp_path / "mask.nii.gz",
        output=tool_output,
        extra="--output-data-type uchar" if sw_tool == "itk" else "",
    )

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")

    # resample mask
    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved_mask = np.asanyarray(nb.load(tool_output).dataobj, dtype=bool)
    nt_moved_mask = apply(xfm, msk, order=0)
    nt_moved_mask.to_filename(tmp_path / "nit_brainmask.nii.gz")
    brainmask = np.asanyarray(nt_moved_mask.dataobj, dtype=bool)
    percent_diff = (sw_moved_mask != brainmask)[5:-5, 5:-5, 5:-5].sum() / brainmask.size

    assert percent_diff < 1e-8, (
        f"Resampled masks differed by {percent_diff * 100:0.2f}%."
    )

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=tmp_path / "reference.nii.gz",
        moving=tmp_path / "reference.nii.gz",
        output=tmp_path / f"{sw_tool}_resampled.nii.gz",
        extra="--output-data-type uchar" if sw_tool == "itk" else "",
    )

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load(f"{sw_tool}_resampled.nii.gz")

    nt_moved = apply(xfm, nii, order=0)
    nt_moved.set_data_dtype(nii.get_data_dtype())
    nt_moved.to_filename("nt_resampled.nii.gz")
    sw_moved.set_data_dtype(nt_moved.get_data_dtype())
    diff = np.asanyarray(
        sw_moved.dataobj, dtype=sw_moved.get_data_dtype()
    ) - np.asanyarray(nt_moved.dataobj, dtype=nt_moved.get_data_dtype())
    # A certain tolerance is necessary because of resampling at borders
    assert np.sqrt((diff[brainmask] ** 2).mean()) < RMSE_TOL_LINEAR


@pytest.mark.parametrize("sw_tool", ["itk", "afni"])
def test_displacements_field2(tmp_path, testdata_path, sw_tool):
    """Check a translation-only field on one or more axes, different image orientations."""
    os.chdir(str(tmp_path))
    img_fname = testdata_path / "tpl-OASIS30ANTs_T1w.nii.gz"
    xfm_fname = testdata_path / "ds-005_sub-01_from-OASIS_to-T1_warp_{}.nii.gz".format(
        sw_tool
    )

    xfm = nitnl.load(xfm_fname, fmt=sw_tool)

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD[sw_tool](
        transform=xfm_fname,
        reference=img_fname,
        moving=img_fname,
        output="resampled.nii.gz",
        extra="",
    )

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load("resampled.nii.gz")

    nt_moved = apply(xfm, img_fname, order=0)
    nt_moved.to_filename("nt_resampled.nii.gz")
    sw_moved.set_data_dtype(nt_moved.get_data_dtype())
    diff = np.asanyarray(
        sw_moved.dataobj, dtype=sw_moved.get_data_dtype()
    ) - np.asanyarray(nt_moved.dataobj, dtype=nt_moved.get_data_dtype())
    # A certain tolerance is necessary because of resampling at borders
    assert np.sqrt((diff**2).mean()) < RMSE_TOL_LINEAR


@pytest.mark.slow
def test_apply_transformchain(tmp_path, testdata_path):
    """Check a translation-only field on one or more axes, different image orientations."""
    os.chdir(str(tmp_path))
    img_fname = testdata_path / "T1w_scanner.nii.gz"
    xfm_fname = (
        testdata_path
        / "ds-005_sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    )

    xfm = nitm.load(xfm_fname, fmt="itk")

    assert len(xfm) == 2

    ref_fname = tmp_path / "reference.nii.gz"
    nb.Nifti1Image(
        np.zeros(xfm.reference.shape, dtype="uint16"),
        xfm.reference.affine,
    ).to_filename(str(ref_fname))

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD["itk"](
        transform=xfm_fname,
        reference=ref_fname,
        moving=img_fname,
        output="resampled.nii.gz",
        extra="",
    )

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip(f"Command {exe} not found on host")

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load("resampled.nii.gz")

    nt_moved = apply(xfm, img_fname, order=0)
    nt_moved.to_filename("nt_resampled.nii.gz")
    diff = sw_moved.get_fdata() - nt_moved.get_fdata()
    # A certain tolerance is necessary because of resampling at borders
    assert (np.abs(diff) > 1e-3).sum() / diff.size < RMSE_TOL_LINEAR


@pytest.mark.parametrize("serialize_4d", [True, False])
def test_LinearTransformsMapping_apply(
    tmp_path, data_path, testdata_path, serialize_4d
):
    """Apply transform mappings."""
    hmc = nitl.load(
        data_path / "hmc-itk.tfm", fmt="itk", reference=testdata_path / "sbref.nii.gz"
    )
    assert isinstance(hmc, nitl.LinearTransformsMapping)

    # Test-case: realign functional data on to sbref
    nii = apply(
        hmc,
        testdata_path / "func.nii.gz",
        order=1,
        reference=testdata_path / "sbref.nii.gz",
        serialize_nvols=2 if serialize_4d else np.inf,
    )
    assert nii.dataobj.shape[-1] == len(hmc)

    # Test-case: write out a fieldmap moved with head
    hmcinv = nitl.LinearTransformsMapping(
        np.linalg.inv(hmc.matrix), reference=testdata_path / "func.nii.gz"
    )

    nii = apply(
        hmcinv,
        testdata_path / "fmap.nii.gz",
        order=1,
        serialize_nvols=2 if serialize_4d else np.inf,
    )
    assert nii.dataobj.shape[-1] == len(hmc)

    # Ensure a ValueError is issued when trying to apply mismatched transforms
    # (e.g., in this case, two transforms while the functional has 8 volumes)
    hmc = nitl.LinearTransformsMapping(hmc.matrix[:2, ...])
    with pytest.raises(ValueError):
        apply(
            hmc,
            testdata_path / "func.nii.gz",
            order=1,
            reference=testdata_path / "sbref.nii.gz",
            serialize_nvols=2 if serialize_4d else np.inf,
        )


def _assert_apply(img, xfm, expected, **kwargs):
    """Apply ``xfm`` to ``img`` and check each output volume.

    ``expected`` is a list of ``((i, j, k), value)`` pairs, one per output volume.
    """
    data = np.asanyarray(apply(xfm, img, order=0, reference=img, **kwargs).dataobj)
    is_4d = data.ndim == 4
    assert data.shape == img.shape[:3] + ((len(expected),) if is_4d else ())
    for vol, (loc, value) in enumerate(expected):
        volume = data[..., vol] if is_4d else data
        assert tuple(np.argwhere(volume)[0]) == loc
        assert volume[loc] == value


def test_apply_3d():
    """Resample a single 3D volume through a single 3D transform."""
    base = np.zeros((10, 5, 5), dtype=np.float32)
    base[9, 2, 2] = 7
    img = nb.Nifti1Image(base, np.eye(4))

    mat = np.eye(4)
    mat[0, 3] = 1  # shift by one voxel along x
    _assert_apply(img, nitl.Affine(mat), [((8, 2, 2), 7)])


@pytest.mark.parametrize("serialize_4d", [True, False])
def test_apply_3d_transform_4d_data(serialize_4d):
    """Regression test for a single 3D transform applied to 4D data."""
    nvols = 9
    base = np.zeros((10, 5, 5), dtype=np.float32)
    base[9, 2, 2] = 1
    # Distinguish volumes so a per-volume mix-up would be caught
    img = nb.Nifti1Image(
        np.stack([(vol + 1) * base for vol in range(nvols)], axis=-1), np.eye(4)
    )
    kwargs = {} if serialize_4d else {"serialize_nvols": nvols + 1}

    mat = np.eye(4)
    mat[0, 3] = 1  # single shift broadcast over all volumes
    expected = [((8, 2, 2), vol + 1) for vol in range(nvols)]
    _assert_apply(img, nitl.Affine(mat), expected, **kwargs)


@pytest.mark.parametrize("serialize_4d", [True, False])
def test_apply_4d_transform_3d_data(serialize_4d):
    """Regression test for a 4D transform applied to a single 3D volume."""
    nvols = 9
    base = np.zeros((10, 5, 5), dtype=np.float32)
    base[9, 2, 2] = 1
    img = nb.Nifti1Image(base, np.eye(4))
    kwargs = {} if serialize_4d else {"serialize_nvols": nvols + 1}

    # A distinct shift per output volume, all sampling the same 3D image
    transforms = []
    for vol in range(nvols):
        mat = np.eye(4)
        mat[0, 3] = vol
        transforms.append(nitl.Affine(mat))
    xfm = nitl.LinearTransformsMapping(transforms, reference=img)

    expected = [((9 - vol, 2, 2), 1) for vol in range(nvols)]
    _assert_apply(img, xfm, expected, **kwargs)


@pytest.mark.parametrize("serialize_4d", [True, False])
def test_apply_4d(serialize_4d):
    """Regression test for per-volume transforms with serialized resampling."""
    nvols = 9
    base = np.zeros((10, 5, 5), dtype=np.float32)
    base[9, 2, 2] = 1
    img = nb.Nifti1Image(np.stack([base] * nvols, axis=-1), np.eye(4))
    kwargs = {} if serialize_4d else {"serialize_nvols": nvols + 1}

    transforms = []
    for vol in range(nvols):
        mat = np.eye(4)
        mat[0, 3] = vol
        transforms.append(nitl.Affine(mat))
    xfm = nitl.LinearTransformsMapping(transforms, reference=img)

    expected = [((9 - vol, 2, 2), 1) for vol in range(nvols)]
    _assert_apply(img, xfm, expected, **kwargs)
