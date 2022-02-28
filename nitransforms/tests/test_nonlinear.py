# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tests of nonlinear transforms."""
import os
import shutil
from subprocess import check_call
import pytest

import numpy as np
import nibabel as nb
from nitransforms.base import TransformError
from nitransforms.io.base import TransformFileError
from nitransforms.nonlinear import (
    BSplineFieldTransform,
    DisplacementsFieldTransform,
    load as nlload,
)
from ..io.itk import ITKDisplacementsField


RMSE_TOL = 0.05
APPLY_NONLINEAR_CMD = {
    "itk": """\
antsApplyTransforms -d 3 -r {reference} -i {moving} \
-o {output} -n NearestNeighbor -t {transform} {extra}\
""".format,
    "afni": """\
3dNwarpApply -nwarp {transform} -source {moving} \
-master {reference} -interp NN -prefix {output} {extra}\
""".format,
    'fsl': """\
applywarp -i {moving} -r {reference} -o {output} {extra}\
-w {transform} --interp=nn""".format,
}


@pytest.mark.parametrize("size", [(20, 20, 20), (20, 20, 20, 3)])
def test_itk_disp_load(size):
    """Checks field sizes."""
    with pytest.raises(TransformFileError):
        ITKDisplacementsField.from_image(nb.Nifti1Image(np.zeros(size), np.eye(4), None))


@pytest.mark.parametrize("size", [(20, 20, 20), (20, 20, 20, 2, 3), (20, 20, 20, 1, 4)])
def test_displacements_bad_sizes(size):
    """Checks field sizes."""
    with pytest.raises(TransformError):
        DisplacementsFieldTransform(nb.Nifti1Image(np.zeros(size), np.eye(4), None))


def test_itk_disp_load_intent():
    """Checks whether the NIfTI intent is fixed."""
    with pytest.warns(UserWarning):
        field = ITKDisplacementsField.from_image(
            nb.Nifti1Image(np.zeros((20, 20, 20, 1, 3)), np.eye(4), None)
        )

    assert field.header.get_intent()[0] == "vector"


def test_displacements_init():
    DisplacementsFieldTransform(
        np.zeros((10, 10, 10, 3)),
        reference=nb.Nifti1Image(np.zeros((10, 10, 10, 3)), np.eye(4), None),
    )

    with pytest.raises(TransformError):
        DisplacementsFieldTransform(np.zeros((10, 10, 10, 3)))
    with pytest.raises(TransformError):
        DisplacementsFieldTransform(
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
        BSplineFieldTransform(
            testdata_path / "someones_bspline_coefficients.nii.gz"
        ).apply(testdata_path / "someones_anatomy.nii.gz")

    BSplineFieldTransform(
        testdata_path / "someones_bspline_coefficients.nii.gz"
    ).apply(
        testdata_path / "someones_anatomy.nii.gz",
        reference=testdata_path / "someones_anatomy.nii.gz"
    )


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

    fieldmap = np.zeros(
        (*nii.shape[:3], 1, 3) if sw_tool != "fsl" else (*nii.shape[:3], 3),
        dtype="float32",
    )
    fieldmap[..., axis] = -10.0

    _hdr = nii.header.copy()
    if sw_tool in ("itk",):
        _hdr.set_intent("vector")
    _hdr.set_data_dtype("float32")

    xfm_fname = "warp.nii.gz"
    field = nb.Nifti1Image(fieldmap, nii.affine, _hdr)
    field.to_filename(xfm_fname)

    xfm = nlload(xfm_fname, fmt=sw_tool)

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=tmp_path / "mask.nii.gz",
        moving=tmp_path / "mask.nii.gz",
        output=tmp_path / "resampled_brainmask.nii.gz",
        extra="--output-data-type uchar" if sw_tool == "itk" else "",
    )

    # skip test if command is not available on host
    exe = cmd.split(" ", 1)[0]
    if not shutil.which(exe):
        pytest.skip("Command {} not found on host".format(exe))

    # resample mask
    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved_mask = nb.load("resampled_brainmask.nii.gz")
    nt_moved_mask = xfm.apply(msk, order=0)
    nt_moved_mask.set_data_dtype(msk.get_data_dtype())
    diff = np.asanyarray(sw_moved_mask.dataobj) - np.asanyarray(nt_moved_mask.dataobj)

    assert np.sqrt((diff ** 2).mean()) < RMSE_TOL
    brainmask = np.asanyarray(nt_moved_mask.dataobj, dtype=bool)

    # Then apply the transform and cross-check with software
    cmd = APPLY_NONLINEAR_CMD[sw_tool](
        transform=os.path.abspath(xfm_fname),
        reference=tmp_path / "reference.nii.gz",
        moving=tmp_path / "reference.nii.gz",
        output=tmp_path / "resampled.nii.gz",
        extra="--output-data-type uchar" if sw_tool == "itk" else ""
    )

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load("resampled.nii.gz")

    nt_moved = xfm.apply(nii, order=0)
    nt_moved.set_data_dtype(nii.get_data_dtype())
    nt_moved.to_filename("nt_resampled.nii.gz")
    sw_moved.set_data_dtype(nt_moved.get_data_dtype())
    diff = (
        np.asanyarray(sw_moved.dataobj, dtype=sw_moved.get_data_dtype())
        - np.asanyarray(nt_moved.dataobj, dtype=nt_moved.get_data_dtype())
    )
    # A certain tolerance is necessary because of resampling at borders
    assert np.sqrt((diff[brainmask] ** 2).mean()) < RMSE_TOL


@pytest.mark.parametrize("sw_tool", ["itk", "afni"])
def test_displacements_field2(tmp_path, testdata_path, sw_tool):
    """Check a translation-only field on one or more axes, different image orientations."""
    os.chdir(str(tmp_path))
    img_fname = testdata_path / "tpl-OASIS30ANTs_T1w.nii.gz"
    xfm_fname = testdata_path / "ds-005_sub-01_from-OASIS_to-T1_warp_{}.nii.gz".format(
        sw_tool
    )

    xfm = nlload(xfm_fname, fmt=sw_tool)

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
        pytest.skip("Command {} not found on host".format(exe))

    exit_code = check_call([cmd], shell=True)
    assert exit_code == 0
    sw_moved = nb.load("resampled.nii.gz")

    nt_moved = xfm.apply(img_fname, order=0)
    nt_moved.to_filename("nt_resampled.nii.gz")
    sw_moved.set_data_dtype(nt_moved.get_data_dtype())
    diff = (
        np.asanyarray(sw_moved.dataobj, dtype=sw_moved.get_data_dtype())
        - np.asanyarray(nt_moved.dataobj, dtype=nt_moved.get_data_dtype())
    )
    # A certain tolerance is necessary because of resampling at borders
    assert np.sqrt((diff ** 2).mean()) < RMSE_TOL


def test_bspline(tmp_path, testdata_path):
    """Cross-check B-Splines and deformation field."""
    os.chdir(str(tmp_path))

    img_name = testdata_path / "someones_anatomy.nii.gz"
    disp_name = testdata_path / "someones_displacement_field.nii.gz"
    bs_name = testdata_path / "someones_bspline_coefficients.nii.gz"

    bsplxfm = BSplineFieldTransform(bs_name, reference=img_name)
    dispxfm = DisplacementsFieldTransform(disp_name)

    out_disp = dispxfm.apply(img_name)
    out_bspl = bsplxfm.apply(img_name)

    out_disp.to_filename("resampled_field.nii.gz")
    out_bspl.to_filename("resampled_bsplines.nii.gz")

    assert np.sqrt(
        (out_disp.get_fdata(dtype="float32") - out_bspl.get_fdata(dtype="float32")) ** 2
    ).mean() < 0.2
