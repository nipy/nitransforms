# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Conversions between formats."""

import numpy as np
import pytest
from .. import linear as _l
from ..io.lta import FSLinearTransformArray as LTA


def test_concatenation(data_path):
    """Check replacement to lta_concat."""
    lta0 = _l.load(
        data_path / "regressions" / "from-scanner_to-fsnative_mode-image.lta", fmt="lta"
    )
    lta1 = _l.load(
        data_path / "regressions" / "from-fsnative_to-bold_mode-image.lta", fmt="lta"
    )

    lta_combined = _l.load(
        data_path / "regressions" / "from-scanner_to-bold_mode-image.lta", fmt="lta"
    )

    assert np.allclose(lta1.matrix.dot(lta0.matrix), lta_combined.matrix)


@pytest.mark.parametrize(
    "filename",
    [
        "from-fsnative_to-bold_mode-image",
        "from-fsnative_to-scanner_mode-image",
        "from-scanner_to-bold_mode-image",
        "from-scanner_to-fsnative_mode-image",
    ],
)
def test_lta2itk_conversions(data_path, filename):
    """Check conversions between formats."""
    lta = _l.load(data_path / "regressions" / f"{filename}.lta", fmt="lta")
    itk = _l.load(data_path / "regressions" / f"{filename}.tfm", fmt="itk")
    assert np.allclose(lta.matrix, itk.matrix)


@pytest.mark.parametrize(
    "filename,moving,reference",
    [
        ("from-fsnative_to-bold_mode-image", "T1w_fsnative.nii.gz", "bold.nii.gz"),
        (
            "from-fsnative_to-scanner_mode-image",
            "T1w_fsnative.nii.gz",
            "T1w_scanner.nii.gz",
        ),
        ("from-scanner_to-bold_mode-image", "T1w_scanner.nii.gz", "bold.nii.gz"),
        (
            "from-scanner_to-fsnative_mode-image",
            "T1w_scanner.nii.gz",
            "T1w_fsnative.nii.gz",
        ),
    ],
)
def test_itk2lta_conversions(
    data_path, testdata_path, tmp_path, filename, moving, reference
):
    """Check conversions between formats."""
    itk = _l.load(data_path / "regressions" / "".join((filename, ".tfm")), fmt="itk")
    itk.reference = testdata_path / reference
    itk.to_filename(tmp_path / "test.lta", fmt="fs", moving=testdata_path / moving)

    converted_lta = LTA.from_filename(tmp_path / "test.lta")
    expected_fname = (
        data_path / "regressions" / "".join((filename, "_type-ras2ras.lta"))
    )
    if not expected_fname.exists():
        expected_fname = data_path / "regressions" / "".join((filename, ".lta"))

    exp_lta = LTA.from_filename(expected_fname)
    assert np.allclose(converted_lta["xforms"][0]["m_L"], exp_lta["xforms"][0]["m_L"])


@pytest.mark.parametrize(
    "fromto",
    [
        ("fsnative", "bold"),
        ("fsnative", "scanner"),
        ("scanner", "bold"),
        ("scanner", "fsnative"),
    ],
)
def test_lta2fsl_conversions(data_path, fromto, testdata_path):
    """Check conversions between formats."""
    filename = f"from-{fromto[0]}_to-{fromto[1]}_mode-image"
    movname = "bold.nii.gz" if fromto[1] == "bold" else f"T1w_{fromto[1]}.nii.gz"

    lta = _l.load(data_path / "regressions" / f"{filename}.lta", fmt="lta")
    fsl = _l.load(
        data_path / "regressions" / f"{filename}.fsl",
        moving=testdata_path / movname,
        reference=testdata_path / f"T1w_{fromto[0]}.nii.gz",
        fmt="fsl",
    )
    assert np.allclose(lta.matrix, fsl.matrix, atol=1e-4)


@pytest.mark.parametrize(
    "fromto",
    [
        ("fsnative", "bold"),
        ("fsnative", "scanner"),
        ("scanner", "bold"),
        ("scanner", "fsnative"),
    ],
)
def test_fsl2lta_conversions(
    data_path,
    testdata_path,
    tmp_path,
    fromto,
):
    """Check conversions between formats."""
    filename = f"from-{fromto[0]}_to-{fromto[1]}_mode-image"
    refname = "bold.nii.gz" if fromto[1] == "bold" else f"T1w_{fromto[1]}.nii.gz"

    fsl = _l.load(
        data_path / "regressions" / f"{filename}.fsl",
        reference=testdata_path / f"T1w_{fromto[0]}.nii.gz",
        moving=testdata_path / refname,
        fmt="fsl",
    )
    fsl.to_filename(
        tmp_path / "test.lta",
        fmt="fs",
    )

    converted_lta = LTA.from_filename(tmp_path / "test.lta")
    expected_fname = (
        data_path / "regressions" / "".join((filename, "_type-ras2ras.lta"))
    )
    if not expected_fname.exists():
        expected_fname = data_path / "regressions" / "".join((filename, ".lta"))

    exp_lta = LTA.from_filename(expected_fname)
    assert np.allclose(
        converted_lta["xforms"][0]["m_L"], exp_lta["xforms"][0]["m_L"], atol=1e-4
    )
